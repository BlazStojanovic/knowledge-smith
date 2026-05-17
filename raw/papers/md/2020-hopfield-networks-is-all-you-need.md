---
arxiv: '2008.02217'
authors:
- Hubert Ramsauer
- Bernhard Schäfl
- Johannes Lehner
- Philipp Seidl
- Michael Widrich
- Thomas Adler
- Lukas Gruber
- Markus Holzleitner
- Milena Pavlović
- Geir Kjetil Sandve
- Victor Greiff
- David Kreil
- Michael Kopp
- Günter Klambauer
- Johannes Brandstetter
- Sepp Hochreiter
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Hopfield Networks is All You Need
url: http://arxiv.org/abs/2008.02217v3
year: 2020
---

# Hopfield Networks is All You Need

Hubert Ramsauer11footnotemark: 1  Bernhard Schäfl11footnotemark: 1  Johannes Lehner11footnotemark: 1  Philipp Seidl11footnotemark: 1  
  
 Michael Widrich11footnotemark: 1  Thomas Adler11footnotemark: 1  Lukas Gruber11footnotemark: 1  Markus Holzleitner11footnotemark: 1  
  
 Milena Pavlović33footnotemark: 3 ,,~{}^{,}44footnotemark: 4  Geir Kjetil Sandve44footnotemark: 4  Victor Greiff33footnotemark: 3  David Kreil22footnotemark: 2  
  
 Michael Kopp22footnotemark: 2  Günter Klambauer11footnotemark: 1  Johannes Brandstetter11footnotemark: 1  Sepp Hochreiter11footnotemark: 1 ,,~{}^{,}22footnotemark: 2
  
11footnotemark: 1  ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning,
  
  Johannes Kepler University Linz, Austria
  
22footnotemark: 2  Institute of Advanced Research in
Artificial Intelligence (IARAI)
  
33footnotemark: 3  Department of Immunology, University of Oslo, Norway
  
44footnotemark: 4  Department of Informatics, University of Oslo, Norway

###### Abstract

We introduce a modern Hopfield network
with continuous states and a corresponding update rule.
The new Hopfield network can store
exponentially (with the dimension of the
associative space) many patterns,
retrieves the pattern with
one update, and has exponentially small retrieval errors.
It has three types of energy minima (fixed points of the update):
(1) global fixed point averaging over all patterns,
(2) metastable states averaging over a subset of patterns, and
(3) fixed points which store a single pattern.
The new update rule
is equivalent to the attention mechanism used in transformers.
This equivalence enables a
characterization of the heads of transformer models.
These heads perform in the first layers preferably
global averaging and in higher layers partial averaging
via metastable states.
The new modern Hopfield network can be integrated
into deep learning architectures
as layers to allow the storage of and access to
raw input data, intermediate results, or learned prototypes.
These Hopfield layers enable new ways of deep learning,
beyond
fully-connected, convolutional, or recurrent networks,
and provide pooling, memory, association, and attention mechanisms.
We demonstrate the broad applicability of the Hopfield layers
across various domains.
Hopfield layers improved state-of-the-art on three out of four
considered multiple instance learning problems as well as
on immune repertoire
classification with several hundreds of thousands of instances.
On the UCI benchmark collections of small classification tasks,
where deep learning methods typically struggle,
Hopfield layers yielded a new state-of-the-art
when compared to different machine learning methods.
Finally, Hopfield layers achieved state-of-the-art
on two drug design datasets.
The implementation is available at: <https://github.com/ml-jku/hopfield-layers>

## 1 Introduction

The deep learning community
has been looking for alternatives to recurrent neural networks (RNNs)
for storing information.
For example, linear memory networks use a linear autoencoder for sequences
as a memory (Carta et al., [2020](#bib.bib20)).
Additional memories for RNNs like
holographic reduced representations (Danihelka et al., [2016](#bib.bib28)),
tensor product representations (Schlag & Schmidhuber, [2018](#bib.bib80); Schlag et al., [2019](#bib.bib81))
and classical associative memories
(extended to fast weight approaches) (Schmidhuber, [1992](#bib.bib83); Ba et al., [2016a](#bib.bib6); [b](#bib.bib7); Zhang & Zhou, [2017](#bib.bib118); Schlag et al., [2021](#bib.bib82))
have been suggested.
Most approaches to new memories are based on attention.
The neural Turing machine (NTM) is equipped with an
external memory and an attention process (Graves et al., [2014](#bib.bib41)).
Memory networks (Weston et al., [2014](#bib.bib104)) use an arg⁡max\arg\max attention
by first mapping a query and patterns into a space and
then retrieving the pattern with the largest dot product.
End to end memory networks (EMN) make this attention scheme differentiable
by replacing arg⁡max\arg\max through a softmaxsoftmax\mathrm{softmax} (Sukhbaatar et al., [2015a](#bib.bib88); [b](#bib.bib89)).
EMN with dot products became very popular and implement a key-value
attention (Daniluk et al., [2017](#bib.bib29)) for self-attention.
An enhancement of EMN is the transformer (Vaswani et al., [2017a](#bib.bib96); [b](#bib.bib97))
and its extensions (Dehghani et al., [2018](#bib.bib30)).
The transformer has had a great impact on the natural language processing
(NLP) community, in particular via the BERT models (Devlin et al., [2018](#bib.bib32); [2019](#bib.bib33)).

Contribution of this work:
(i) introducing novel deep learning layers that are equipped
with a memory via modern Hopfield networks,
(ii) introducing a novel energy function and a novel update rule for
continuous modern Hopfield networks that are differentiable and
typically retrieve patterns after one update.
Differentiability is required for gradient descent parameter updates and
retrieval with one update is compatible with activating the layers of deep networks.

We suggest using modern Hopfield networks
to store information or learned prototypes in different
layers of neural networks.
Binary Hopfield networks
were introduced as associative memories
that can store and retrieve patterns (Hopfield, [1982](#bib.bib47)).
A query pattern can retrieve the pattern to which it is most similar
or an average over similar patterns.
Hopfield networks seem to be an ancient technique,
however, new energy functions improved
their properties.
The stability of spurious states or metastable states
was sensibly reduced (Barra et al., [2018](#bib.bib10)).
The largest and most impactful successes are reported
on increasing the storage capacity of Hopfield networks.
In a d𝑑d-dimensional space,
the standard Hopfield model can store d𝑑d uncorrelated patterns
without errors but only
C​d/log⁡(d)𝐶𝑑𝑑Cd/\log(d) random patterns with
C<1/2𝐶12C<1/2 for a fixed stable pattern or C<1/4𝐶14C<1/4 if all patterns
are stable (McEliece et al., [1987](#bib.bib70)).
The same bound holds for nonlinear learning rules (Mazza, [1997](#bib.bib69)).
Using tricks-of-trade and allowing
small retrieval errors, the storage capacity
is about 0.138​d0.138𝑑0.138d (Crisanti et al., [1986](#bib.bib27); Hertz et al., [1991](#bib.bib43); Torres et al., [2002](#bib.bib95)).
If the learning rule is not related to the Hebb rule, then up to d𝑑d
patterns can be stored (Abu-Mostafa & StJacques, [1985](#bib.bib1)).
For Hopfield networks with non-zero diagonal matrices,
the storage can be
increased to C​d​log⁡(d)𝐶𝑑𝑑Cd\log(d) (Folli et al., [2017](#bib.bib37)).
In contrast to the storage capacity, the number of energy minima
(spurious states, stable states) of Hopfield networks
is exponential in d𝑑d (Tanaka & Edwards, [1980](#bib.bib91); Bruck & Roychowdhury, [1990](#bib.bib15); Wainrib & Touboul, [2013](#bib.bib100)).

The standard binary Hopfield network has
an energy function that can be expressed as
the sum of interaction functions F𝐹F with F​(x)=x2𝐹𝑥superscript𝑥2F(x)=x^{2}.
Modern Hopfield networks, also called
“dense associative memory” (DAM) models,
use an energy function with interaction functions
of the form F​(x)=xn𝐹𝑥superscript𝑥𝑛F(x)=x^{n} and, thereby, achieve
a storage capacity proportional to dn−1superscript𝑑𝑛1d^{n-1}
(Krotov & Hopfield, [2016](#bib.bib59); [2018](#bib.bib60)).
The energy function of modern Hopfield networks
makes them robust against adversarial attacks (Krotov & Hopfield, [2018](#bib.bib60)).
Modern binary Hopfield networks with energy functions based on
interaction functions of the form F​(x)=exp⁡(x)𝐹𝑥𝑥F(x)=\exp(x) even lead
to storage capacity of 2d/2superscript2𝑑22^{d/2},
where all stored binary patterns are fixed points but the radius of
attraction vanishes (Demircigil et al., [2017](#bib.bib31)).
However, in order to integrate Hopfield networks into deep learning
architectures, it is necessary to make them differentiable, that is,
we require continuous Hopfield networks (Hopfield, [1984](#bib.bib48); Koiran, [1994](#bib.bib57)).

Therefore, we generalize the
energy function of Demircigil et al. ([2017](#bib.bib31)) that builds on exponential interaction functions
to continuous patterns and states and obtain
a new modern Hopfield network.
We also propose a new update rule which
ensures global
convergence to stationary points of the energy (local minima or saddle points).
We prove that our new modern Hopfield network typically
retrieves patterns in one update step (ϵitalic-ϵ\epsilon-close to the fixed point)
with an exponentially low error
and has a storage capacity proportional to cd−14superscript𝑐𝑑14c^{\frac{d-1}{4}} (reasonable settings for
c=1.37𝑐1.37c=1.37 and c=3.15𝑐3.15c=3.15 are given in Theorem [3](#Thmtheorem3 "Theorem 3. ‣ New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need")).
The retrieval of patterns with one update is important to integrate
Hopfield networks in deep learning architectures,
where layers are activated only once.
Surprisingly, our new update rule is also
the key-value attention
as used in transformer and BERT models (see Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Hopfield Networks is All You Need")).
Our modern Hopfield networks can be integrated as a new layer
in deep learning architectures for pooling, memory, prototype learning, and attention.
We test these new layers on different benchmark datasets and tasks like
immune repertoire classification.

!(/html/2008.02217/assets/x1.png)

Figure 1: We generalize
the energy
of binary modern Hopfield networks to continuous states
while keeping fast convergence and
storage capacity properties.
We also propose a new update
rule that minimizes the energy.
The new update rule is the attention mechanism of the transformer.
Formulae are modified to express softmaxsoftmax\mathrm{softmax} as row vector.
“==”-sign means “keeps the properties”.

## 2 Modern Hopfield Nets with Continuous States

##### New energy function for continuous state Hopfield networks.

In order to integrate modern Hopfield networks into deep learning
architectures, we have to make them continuous.
To allow for continuous states, we propose a new energy function that is
a modification of the energy of modern Hopfield networks (Demircigil et al., [2017](#bib.bib31)).
We also propose a new update rule which
can be proven to converge to
stationary points of the energy (local minima or saddle points).

We have N𝑁N stored (key) patterns 𝒙i∈ℝdsubscript𝒙𝑖superscriptℝ𝑑\bm{x}\_{i}\in\mathbb{R}^{d} represented by the matrix
𝑿=(𝒙1,…,𝒙N)𝑿subscript𝒙1…subscript𝒙𝑁\bm{X}=\left(\bm{x}\_{1},\ldots,\bm{x}\_{N}\right) with
the largest pattern M=maxi⁡‖𝒙i‖𝑀subscript𝑖normsubscript𝒙𝑖M=\max\_{i}{{\left\|\bm{x}\_{i}\right\|}}.
The state (query) pattern is 𝝃∈ℝd𝝃superscriptℝ𝑑\bm{\xi}\in\mathbb{R}^{d}.
For exponential interaction functions,
we need the log-sum-exp function (lselse\mathrm{lse}) for 0<β0𝛽0<\beta

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | lse​(β,𝒙)lse𝛽𝒙\displaystyle\mathrm{lse}(\beta,\bm{x})\ | =β−1​log⁡(∑i=1Nexp⁡(β​xi)),absentsuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽subscript𝑥𝑖\displaystyle=\ \beta^{-1}\log\left(\sum\_{i=1}^{N}\exp(\beta x\_{i})\right)\ , |  | (1) |

which is convex (see appendix Eq. ([461](#A1.E461 "In Definition A2 (Log-Sum-Exp Function). ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")),
and Lemma [A22](#ThmlemmaA22 "Lemma A22. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
The energy function EE\mathrm{E} of the
modern Hopfield networks for binary patterns 𝒙isubscript𝒙𝑖\bm{x}\_{i} and a binary state pattern 𝝃𝝃\bm{\xi}
is E=−∑i=1NF​(𝝃T​𝒙i)Esuperscriptsubscript𝑖1𝑁𝐹superscript𝝃𝑇subscript𝒙𝑖\mathrm{E}=-\sum\_{i=1}^{N}F\left(\bm{\xi}^{T}\bm{x}\_{i}\right) (Krotov & Hopfield, [2016](#bib.bib59)).
Here, F​(x)=xn𝐹𝑥superscript𝑥𝑛F(x)=x^{n} is the interaction function, where n=2𝑛2n=2 gives the classical
Hopfield network.
The storage capacity is proportional to dn−1superscript𝑑𝑛1d^{n-1} (Krotov & Hopfield, [2016](#bib.bib59)).
This model was generalized by Demircigil et al. ([2017](#bib.bib31))
to exponential interaction functions
F​(x)=exp⁡(x)𝐹𝑥𝑥F(x)=\exp(x) which gives the energy
E=−exp⁡(lse​(1,𝑿T​𝝃))Else1superscript𝑿𝑇𝝃\mathrm{E}=-\exp(\mathrm{lse}(1,\bm{X}^{T}\bm{\xi})).
This energy leads to an exponential
storage capacity of N=2d/2𝑁superscript2𝑑2N=2^{d/2} for binary patterns.
Furthermore, with a single update, the fixed point
is recovered with high probability for random patterns.
However, still this modern Hopfield network has binary states.

We generalize this energy function to continuous-valued patterns
while keeping the properties of the modern Hopfield networks like
the exponential storage capacity and the extremely fast convergence
(see Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Hopfield Networks is All You Need")).
For the new energy we take the logarithm
of the negative energy of modern Hopfield networks
and add a quadratic term of the current state.
The quadratic term ensures that the
norm of the state vector 𝝃𝝃\bm{\xi} remains finite and the energy is bounded.
Classical Hopfield networks do not require to bound the norm of their state vector,
since it is binary and has fixed length.
We define the novel energy function EE\mathrm{E} as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EE\displaystyle\mathrm{E}\ | =−lse​(β,𝑿T​𝝃)+12​𝝃T​𝝃+β−1​log⁡N+12​M2.absentlse𝛽superscript𝑿𝑇𝝃12superscript𝝃𝑇𝝃superscript𝛽1𝑁12superscript𝑀2\displaystyle=\ -\ \mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\ +\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \beta^{-1}\log N\ +\ \frac{1}{2}M^{2}\ . |  | (2) |

We have 0⩽E⩽2​M20E2superscript𝑀20\leqslant\mathrm{E}\leqslant 2M^{2} (see appendix Lemma [A1](#ThmlemmaA1 "Lemma A1. ‣ A.1.2 New Energy Function ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
Using 𝒑=softmax​(β​𝑿T​𝝃)𝒑softmax𝛽superscript𝑿𝑇𝝃\bm{p}=\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi}), we define a novel update rule
(see Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Hopfield Networks is All You Need")):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =f​(𝝃)=𝑿​𝒑=𝑿​softmax​(β​𝑿T​𝝃).absent𝑓𝝃𝑿𝒑𝑿softmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ f(\bm{\xi})\ =\ \bm{X}\bm{p}\ =\ \bm{X}\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ . |  | (3) |

The next theorem states that the
update rule Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need"))
converges globally.
The proof uses the Concave-Convex Procedure (CCCP) (Yuille & Rangarajan, [2002](#bib.bib113); [2003](#bib.bib114)),
which is equivalent
to Legendre minimization (Rangarajan et al., [1996](#bib.bib77); [1999](#bib.bib78))
algorithms (Yuille & Rangarajan, [2003](#bib.bib114)).

###### Theorem 1.

The update rule Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need"))
converges globally:
For 𝛏t+1=f​(𝛏t)superscript𝛏𝑡1𝑓superscript𝛏𝑡\bm{\xi}^{t+1}=f(\bm{\xi}^{t}),
the energy E​(𝛏t)→E​(𝛏∗)→Esuperscript𝛏𝑡Esuperscript𝛏\mathrm{E}(\bm{\xi}^{t})\to\mathrm{E}(\bm{\xi}^{\*}) for t→∞→𝑡t\to\infty
and a fixed point 𝛏∗superscript𝛏\bm{\xi}^{\*}.

###### Proof.

The update rule in Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need"))
is the CCCP for minimizing the energy EE\mathrm{E}, which is the sum of the convex
1/2​𝝃T​𝝃12superscript𝝃𝑇𝝃1/2\bm{\xi}^{T}\bm{\xi} and concave −lselse-\mathrm{lse} (see details in appendix Theorem [1](#Thmtheorem1 "Theorem 1. ‣ New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need")).
Theorem 2 in Yuille & Rangarajan ([2002](#bib.bib113)) yields the global convergence property.
Also, in Theorem 2 in Sriperumbudur & Lanckriet ([2009](#bib.bib86)) the global convergence of CCCP is proven
via a rigorous analysis using Zangwill’s global convergence theory
of iterative algorithms.
∎

The global convergence theorem only assures that
for the energy E​(𝝃t)→E​(𝝃∗)→Esuperscript𝝃𝑡Esuperscript𝝃\mathrm{E}(\bm{\xi}^{t})\to\mathrm{E}(\bm{\xi}^{\*}) for t→∞→𝑡t\to\infty
but not 𝝃t→𝝃∗→superscript𝝃𝑡superscript𝝃\bm{\xi}^{t}\to\bm{\xi}^{\*}.
The next theorem strengthens
Zangwill’s global convergence theorem (Meyer, [1976](#bib.bib71))
and gives convergence results similar to
those known for expectation maximization (Wu, [1983](#bib.bib108)).

###### Theorem 2.

For the iteration Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need"))
we have E​(𝛏t)→E​(𝛏∗)=E∗→Esuperscript𝛏𝑡Esuperscript𝛏superscriptE\mathrm{E}\left(\bm{\xi}^{t}\right)\to\mathrm{E}\left(\bm{\xi}^{\*}\right)=\mathrm{E}^{\*}
as t→∞→𝑡t\to\infty, for some stationary point 𝛏∗superscript𝛏\bm{\xi}^{\*}.
Furthermore, ‖𝛏t+1−𝛏t‖→0→normsuperscript𝛏𝑡1superscript𝛏𝑡0{{\left\|\bm{\xi}^{t+1}-\bm{\xi}^{t}\right\|}}\to 0 and
either {𝛏t}t=0∞superscriptsubscriptsuperscript𝛏𝑡𝑡0\{\bm{\xi}^{t}\}\_{t=0}^{\infty} converges
or, in the other case, the set of limit points of {𝛏t}t=0∞superscriptsubscriptsuperscript𝛏𝑡𝑡0\{\bm{\xi}^{t}\}\_{t=0}^{\infty}
is a connected and compact subset of ℒ​(E∗)ℒsuperscriptE\mathcal{L}\left(\mathrm{E}^{\*}\right), where
ℒ​(a)={𝛏∈ℒ∣E​(𝛏)=a}ℒ𝑎conditional-set𝛏ℒE𝛏𝑎\mathcal{L}\left(a\right)=\{\bm{\xi}\in\mathcal{L}\mid\mathrm{E}\left(\bm{\xi}\right)=a\}
and ℒℒ\mathcal{L} is the set of stationary points of the iteration Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need")).
If ℒ​(E∗)ℒsuperscriptE\mathcal{L}\left(\mathrm{E}^{\*}\right) is finite,
then any sequence {𝛏t}t=0∞superscriptsubscriptsuperscript𝛏𝑡𝑡0\{\bm{\xi}^{t}\}\_{t=0}^{\infty}
generated by the iteration Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need"))
converges to some 𝛏∗∈ℒ​(E∗)superscript𝛏ℒsuperscriptE\bm{\xi}^{\*}\in\mathcal{L}\left(\mathrm{E}^{\*}\right).

For a proof, see appendix Theorem [2](#Thmtheorem2 "Theorem 2. ‣ New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need").
Therefore, all the limit points of any sequence generated by
the iteration Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need")) are
stationary points (local minima or saddle points) of the
energy function EE\mathrm{E}. Either the iteration converges or,
otherwise, the set of limit points
is a connected and compact set.

The next theorem gives the results on the storage capacity
of our new continuous state modern Hopfield network.
We first define what we mean by storing and retrieving patterns
using a modern Hopfield network with continuous states.

###### Definition 1 (Pattern Stored and Retrieved).

We assume that around every pattern 𝐱isubscript𝐱𝑖\bm{x}\_{i} a sphere SisubscriptS𝑖\mathrm{S}\_{i} is given.
We say 𝐱isubscript𝐱𝑖\bm{x}\_{i} is stored if there is a single fixed point 𝐱i∗∈Sisuperscriptsubscript𝐱𝑖subscriptS𝑖\bm{x}\_{i}^{\*}\in\mathrm{S}\_{i} to
which all points 𝛏∈Si𝛏subscriptS𝑖\bm{\xi}\in\mathrm{S}\_{i} converge,
and Si∩Sj=∅subscriptS𝑖subscriptS𝑗\mathrm{S}\_{i}\cap\mathrm{S}\_{j}=\emptyset for i≠j𝑖𝑗i\not=j.
We say 𝐱isubscript𝐱𝑖\bm{x}\_{i} is retrieved for a given ϵitalic-ϵ\epsilon if
iteration (update rule) Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need")) gives
a point 𝐱~isubscript~𝐱𝑖\tilde{\bm{x}}\_{i} that is at least
ϵitalic-ϵ\epsilon-close to the single fixed point 𝐱i∗∈Sisuperscriptsubscript𝐱𝑖subscriptS𝑖\bm{x}\_{i}^{\*}\in\mathrm{S}\_{i}.
The retrieval error is ‖𝐱~i−𝐱i‖normsubscript~𝐱𝑖subscript𝐱𝑖{{\left\|\tilde{\bm{x}}\_{i}-\bm{x}\_{i}\right\|}}.

As with classical Hopfield networks, we consider patterns on the sphere,
i.e. patterns with a fixed norm.
For randomly chosen patterns, the number of patterns that can be stored
is exponential in the dimension d𝑑d of the space of the patterns (𝒙i∈ℝdsubscript𝒙𝑖superscriptℝ𝑑\bm{x}\_{i}\in\mathbb{R}^{d}).

###### Theorem 3.

We assume a failure probability 0<p⩽10𝑝10<p\leqslant 1 and randomly chosen patterns
on the sphere with radius M:=K​d−1assign𝑀𝐾𝑑1M:=K\sqrt{d-1}.
We define a:=2d−1​(1+ln⁡(2​β​K2​p​(d−1)))assign𝑎2𝑑112𝛽superscript𝐾2𝑝𝑑1a:=\frac{2}{d-1}(1+\ln(2\beta K^{2}p(d-1))),
b:=2​K2​β5assign𝑏2superscript𝐾2𝛽5b:=\frac{2K^{2}\beta}{5},
and c:=bW0(exp(a+ln(b))c:=\frac{b}{W\_{0}(\exp(a+\ln(b))},
where W0subscript𝑊0W\_{0} is the upper branch of the Lambert W𝑊W function (Olver et al., [2010](#bib.bib72), [(4.13)](http://dlmf.nist.gov/4.13)),
and ensure c≥(2p)4d−1𝑐superscript2𝑝4𝑑1c\geq\left(\frac{2}{\sqrt{p}}\right)^{\frac{4}{d-1}}.
Then with probability 1−p1𝑝1-p, the number of random patterns
that can be stored is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | ≥p​cd−14.absent𝑝superscript𝑐𝑑14\displaystyle\geq\ \sqrt{p}\ c^{\frac{d-1}{4}}\ . |  | (4) |

Therefore it is proven for c≥3.1546𝑐3.1546c\geq 3.1546 with
β=1𝛽1\beta=1, K=3𝐾3K=3, d=20𝑑20d=20 and p=0.001𝑝0.001p=0.001 (a+ln⁡(b)>1.27𝑎𝑏1.27a+\ln(b)>1.27)
and proven for c≥1.3718𝑐1.3718c\geq 1.3718 with β=1𝛽1\beta=1, K=1𝐾1K=1, d=75𝑑75d=75, and p=0.001𝑝0.001p=0.001
(a+ln⁡(b)<−0.94𝑎𝑏0.94a+\ln(b)<-0.94).

For a proof, see appendix Theorem [A5](#ThmtheoremA5 "Theorem A5 (Storage Capacity (Main): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").

The next theorem states that the update rule typically
retrieves patterns after one update.
Retrieval of a pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} for fixed point 𝒙i∗superscriptsubscript𝒙𝑖\bm{x}\_{i}^{\*} and query 𝝃𝝃\bm{\xi}
is defined via an ϵitalic-ϵ\epsilon by ‖f​(𝝃)−𝒙i∗‖<ϵnorm𝑓𝝃superscriptsubscript𝒙𝑖italic-ϵ{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}<\epsilon,
that is, the update is ϵitalic-ϵ\epsilon-close to the fixed point.
Retrieval with one update
is crucial to integrate modern Hopfield networks into
deep learning architectures, where layers are activated only once.
First we need the concept of separation of a pattern.
For pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} we define its separation ΔisubscriptΔ𝑖\Delta\_{i}
to other patterns by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | :=minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)=𝒙iT​𝒙i−maxj,j≠i⁡𝒙iT​𝒙j.assignabsentsubscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖subscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗\displaystyle:=\ \min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right)\ =\ \bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \max\_{j,j\not=i}\bm{x}\_{i}^{T}\bm{x}\_{j}\ . |  | (5) |

The update rule retrieves patterns
with one update for well separated patterns, that is, patterns with large
ΔisubscriptΔ𝑖\Delta\_{i}.

###### Theorem 4.

With query 𝛏𝛏\bm{\xi}, after one update the distance of the new point f​(𝛏)𝑓𝛏f(\bm{\xi})
to the fixed point 𝐱i∗superscriptsubscript𝐱𝑖\bm{x}\_{i}^{\*} is exponentially small in the separation ΔisubscriptΔ𝑖\Delta\_{i}.
The precise bounds using the Jacobian J=∂f​(𝛏)∂𝛏J𝑓𝛏𝛏\mathrm{J}=\frac{\partial f(\bm{\xi})}{\partial\bm{\xi}} and its value JmsuperscriptJ𝑚\mathrm{J}^{m} in the mean value
theorem are:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i∗‖norm𝑓𝝃superscriptsubscript𝒙𝑖\displaystyle{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}\ | ⩽‖Jm‖2​‖𝝃−𝒙i∗‖,absentsubscriptnormsuperscriptJ𝑚2norm𝝃superscriptsubscript𝒙𝑖\displaystyle\leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}^{\*}\right\|}}\ , |  | (6) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jm‖2subscriptnormsuperscriptJ𝑚2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ | ⩽ 2​β​N​M2​(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M)).absent2𝛽𝑁superscript𝑀2𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle\leqslant\ 2\ \beta\ N\ M^{2}\ (N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ . |  | (7) |

For given ϵitalic-ϵ\epsilon and
sufficient large ΔisubscriptΔ𝑖\Delta\_{i}, we have ‖f​(𝛏)−𝐱i∗‖<ϵnorm𝑓𝛏superscriptsubscript𝐱𝑖italic-ϵ{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}<\epsilon,
that is, retrieval with one update.

See proof in appendix Theorem [A8](#ThmtheoremA8 "Theorem A8 (Pattern Retrieval with One Update). ‣ A.1.6.2 Retrieval of Patterns with One Update and Small Retrieval Error. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").

At the same time,
the retrieval error decreases exponentially with the separation ΔisubscriptΔ𝑖\Delta\_{i}.

###### Theorem 5 (Exponentially Small Retrieval Error).

The retrieval error ‖f​(𝛏)−𝐱i‖norm𝑓𝛏subscript𝐱𝑖{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}\right\|}} of pattern 𝐱isubscript𝐱𝑖\bm{x}\_{i}
is bounded by

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i‖norm𝑓𝝃subscript𝒙𝑖\displaystyle{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}\right\|}}\ | ⩽ 2​(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M))​Mabsent2𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀𝑀\displaystyle\leqslant\ 2\ (N-1)\ \exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ M |  | (8) |

and for
‖𝐱i−𝐱i∗‖⩽12​β​Mnormsubscript𝐱𝑖superscriptsubscript𝐱𝑖12𝛽𝑀{{\left\|\bm{x}\_{i}-\bm{x}\_{i}^{\*}\right\|}}\leqslant\frac{1}{2\ \beta\ M}
together with ‖𝐱i−𝛏‖⩽12​β​Mnormsubscript𝐱𝑖𝛏12𝛽𝑀{{\left\|\bm{x}\_{i}-\bm{\xi}\right\|}}\leqslant\frac{1}{2\ \beta\ M}
by

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒙i−𝒙i∗‖normsubscript𝒙𝑖superscriptsubscript𝒙𝑖\displaystyle{{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{i}^{\*}\right\|}}\ | ⩽ 2​e​(N−1)​M​exp⁡(−β​Δi).absent2𝑒𝑁1𝑀𝛽subscriptΔ𝑖\displaystyle\leqslant\ 2\ e\ (N-1)\ M\ \exp(-\ \beta\ \Delta\_{i})\ . |  | (9) |

See proof in appendix Theorem [A9](#ThmtheoremA9 "Theorem A9 (Exponentially Small Retrieval Error). ‣ A.1.6.2 Retrieval of Patterns with One Update and Small Retrieval Error. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").

##### Metastable states and one global fixed point.

So far, we considered patterns 𝒙isubscript𝒙𝑖\bm{x}\_{i} that are well separated
and the iteration converges to a fixed point which is
near a pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i}.
If no pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} is well separated from the others,
then the iteration converges to a global fixed point close to
the arithmetic mean of the vectors.
In this case the softmaxsoftmax\mathrm{softmax} vector 𝒑𝒑\bm{p} is close to uniform, that is, pi=1/Nsubscript𝑝𝑖1𝑁p\_{i}=1/N.
If some vectors are similar to each other and well separated from all
other vectors, then a metastable state near the similar
vectors exists. Iterations that start near the metastable state converge
to this metastable state, also if initialized by one of the similar patterns.
For convergence proofs to one global fixed point
and to metastable states see appendix
Lemma [A7](#ThmlemmaA7 "Lemma A7. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") and Lemma [A12](#ThmlemmaA12 "Lemma A12. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), respectively.

##### Hopfield update rule is attention of the transformer.

The Hopfield network update rule
is the attention mechanism used
in transformer and BERT models
(see Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Hopfield Networks is All You Need")).
To see this, we assume N𝑁N stored (key) patterns 𝒚isubscript𝒚𝑖\bm{y}\_{i}
and S𝑆S state (query) patterns 𝒓isubscript𝒓𝑖\bm{r}\_{i} that are mapped to the
Hopfield space of dimension dksubscript𝑑𝑘d\_{k}.
We set 𝒙i=𝑾KT​𝒚isubscript𝒙𝑖superscriptsubscript𝑾𝐾𝑇subscript𝒚𝑖\bm{x}\_{i}=\bm{W}\_{K}^{T}\bm{y}\_{i}, 𝝃i=𝑾QT​𝒓isubscript𝝃𝑖superscriptsubscript𝑾𝑄𝑇subscript𝒓𝑖\bm{\xi}\_{i}=\bm{W}\_{Q}^{T}\bm{r}\_{i},
and multiply the result of our update rule with 𝑾Vsubscript𝑾𝑉\bm{W}\_{V}.
The matrices 𝒀=(𝒚1,…,𝒚N)T𝒀superscriptsubscript𝒚1…subscript𝒚𝑁𝑇\bm{Y}=(\bm{y}\_{1},\ldots,\bm{y}\_{N})^{T} and 𝑹=(𝒓1,…,𝒓S)T𝑹superscriptsubscript𝒓1…subscript𝒓𝑆𝑇\bm{R}=(\bm{r}\_{1},\ldots,\bm{r}\_{S})^{T} combine the 𝒚isubscript𝒚𝑖\bm{y}\_{i} and 𝒓isubscript𝒓𝑖\bm{r}\_{i}
as row vectors.
We define the matrices 𝑿T=𝑲=𝒀​𝑾Ksuperscript𝑿𝑇𝑲𝒀subscript𝑾𝐾\bm{X}^{T}=\bm{K}=\bm{Y}\bm{W}\_{K}, 𝚵T=𝑸=𝑹​𝑾Qsuperscript𝚵𝑇𝑸𝑹subscript𝑾𝑄\bm{\Xi}^{T}=\bm{Q}=\bm{R}\bm{W}\_{Q},
and 𝑽=𝒀​𝑾K​𝑾V=𝑿T​𝑾V𝑽𝒀subscript𝑾𝐾subscript𝑾𝑉superscript𝑿𝑇subscript𝑾𝑉\bm{V}=\bm{Y}\bm{W}\_{K}\bm{W}\_{V}=\bm{X}^{T}\bm{W}\_{V}, where
𝑾K∈ℝdy×dk,𝑾Q∈ℝdr×dk,𝑾V∈ℝdk×dvformulae-sequencesubscript𝑾𝐾superscriptℝsubscript𝑑𝑦subscript𝑑𝑘formulae-sequencesubscript𝑾𝑄superscriptℝsubscript𝑑𝑟subscript𝑑𝑘subscript𝑾𝑉superscriptℝsubscript𝑑𝑘subscript𝑑𝑣\bm{W}\_{K}\in\mathbb{R}^{d\_{y}\times d\_{k}},\bm{W}\_{Q}\in\mathbb{R}^{d\_{r}\times d\_{k}},\bm{W}\_{V}\in\mathbb{R}^{d\_{k}\times d\_{v}}.
If β=1/dk𝛽1subscript𝑑𝑘\beta=1/\sqrt{d\_{k}} and softmax∈ℝNsoftmaxsuperscriptℝ𝑁\mathrm{softmax}\in\mathbb{R}^{N} is changed to a row vector, we obtain
for the update rule Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need")) multiplied by 𝑾Vsubscript𝑾𝑉\bm{W}\_{V}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒁=softmax​(1/dk​𝑸​𝑲T)​𝑽=softmax​(β​𝑹​𝑾𝑸​𝑾𝑲T​𝒀T)​𝒀​𝑾𝑲​𝑾𝑽.𝒁softmax1subscript𝑑𝑘𝑸superscript𝑲𝑇𝑽softmax𝛽𝑹subscript𝑾𝑸superscriptsubscript𝑾𝑲𝑇superscript𝒀𝑇𝒀subscript𝑾𝑲subscript𝑾𝑽\displaystyle\bm{Z}\ =\ \mathrm{softmax}\left(1/\sqrt{d\_{k}}\ \bm{Q}\ \bm{K}^{T}\right)\ \bm{V}\ =\ \mathrm{softmax}\left(\beta\ \bm{R}\ \bm{W}\_{\bm{Q}}\ \bm{W}\_{\bm{K}}^{T}\bm{Y}^{T}\right)\ \bm{Y}\ \bm{W}\_{\bm{K}}\bm{W}\_{\bm{V}}\ . |  | (10) |

The left part of Eq. ([10](#S2.E10 "In Hopfield update rule is attention of the transformer. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need"))
is the transformer attention. In the transformer self-attention 𝑹=𝒀𝑹𝒀\bm{R}=\bm{Y}, and
𝑾𝑲​𝑾𝑽subscript𝑾𝑲subscript𝑾𝑽\bm{W}\_{\bm{K}}\bm{W}\_{\bm{V}} replaced by just 𝑾𝑽subscript𝑾𝑽\bm{W}\_{\bm{V}}.
Besides the attention mechanism,
Hopfield networks allow for other functionalities
in deep network architectures,
which we introduce via specific layers
in the next section.
The right part of Eq. ([10](#S2.E10 "In Hopfield update rule is attention of the transformer. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need"))
serves to explain these specific layers.

## 3 New Hopfield Layers for Deep Learning

Modern Hopfield networks with continuous states
can be integrated into deep learning
architectures, because they are continuous and
differentiable with respect to their parameters.
Furthermore, they typically retrieve patterns with
one update, which is conform to deep learning layers
that are activated only once.
For these two reasons, modern Hopfield networks
can serve as specialized layers in
deep networks to equip them with memories.
Below, we introduce three types of Hopfield layers:
Hopfield, HopfieldPooling, and
HopfieldLayer.
Possible applications of Hopfield layers
in deep network architectures comprise:

* •

  multiple instance learning (MIL) (Dietterich et al., [1997](#bib.bib34)),
* •

  processing of and learning with point sets (Qi et al., [2017a](#bib.bib75); [b](#bib.bib76); Xu et al., [2018](#bib.bib112)),
* •

  set-based and permutation invariant learning (Guttenberg et al., [2016](#bib.bib42); Ravanbakhsh et al., [2016](#bib.bib79); Zaheer et al., [2017](#bib.bib115); Korshunova et al., [2018](#bib.bib58); Ilse et al., [2018](#bib.bib49); Zhai et al., [2020](#bib.bib117)),
* •

  attention-based learning (Vaswani et al., [2017a](#bib.bib96)),
* •

  deep learning with associative memories (Graves et al., [2014](#bib.bib41); Weston et al., [2014](#bib.bib104); Ba et al., [2016a](#bib.bib6); [b](#bib.bib7); Schlag & Schmidhuber, [2018](#bib.bib80); Schlag et al., [2019](#bib.bib81)),
* •

  natural language processing (Devlin et al., [2018](#bib.bib32); [2019](#bib.bib33)),
* •

  sequence analysis and time series prediction (Hochreiter, [1991](#bib.bib44); Hochreiter & Schmidhuber, [1997](#bib.bib45); Cho et al., [2014](#bib.bib24)), and
* •

  storing and retrieving reference data, e.g. the training data, outliers, high error data points,
  prototypes or cluster centers, support vectors & border cases.

Hopfield network layers can substitute
existing layers like
pooling layers,
permutation equivariant layers (Guttenberg et al., [2016](#bib.bib42); Ravanbakhsh et al., [2016](#bib.bib79)),
GRU (Cho et al., [2014](#bib.bib24)) &
LSTM (Hochreiter, [1991](#bib.bib44); Hochreiter & Schmidhuber, [1997](#bib.bib45)) layers, and
attention layers (Vaswani et al., [2017a](#bib.bib96); [b](#bib.bib97); Bahdanau et al., [2014](#bib.bib8)).

!(/html/2008.02217/assets/x2.png)

Figure 2: Left: A standard deep network with layers
( ■■\blacksquare) propagates
either a vector or a set of vectors from the input to the output.
Right: A deep network, where layers ( ■■\blacksquare)
are equipped with associative memories
via Hopfield layers ( ■■\blacksquare).

##### Types of neural networks.

We consider two types of feed-forward neural networks:
(I) Neural networks that propagate an activation vector from the input layer
to the output layer. Examples are fully-connected or convolutional neural networks.
(II) Neural networks that propagate a set of vectors from the input layer to
the output layer, where each layer applies the same operation to each element of the set
and the output layer may summarize the set via a vector.
An example is the transformer.
Recurrent neural networks are networks of type (I),
which are iteratively applied to a set or a sequence,
where intermediate results are stored in a memory and can be reused.
Modern Hopfield networks can be integrated into both types of
neural network architectures and enable to equip each of their layers
with associative memories. See Fig. [2](#S3.F2 "Figure 2 ‣ 3 New Hopfield Layers for Deep Learning ‣ Hopfield Networks is All You Need").

##### Types of new Hopfield layers.

We introduce three types of Hopfield layers:
Hopfield, HopfieldPooling, and
HopfieldLayer.
The continuous modern Hopfield network
results in a plethora of new deep learning architectures,
since we can
(a) propagate sets or single vectors,
(b) propagate queries, stored patterns, or both,
(c) learn static queries or stored patterns,
(d) fill the memory by training sets, prototypes, or external data.
Next, we provide three useful types of Hopfield layers.
The implementation is available at: <https://github.com/ml-jku/hopfield-layers>

(1) Layer Hopfield
for networks that propagate sets of vectors via
state (query) patterns R𝑅\bm{R} and stored (key) patterns Y𝑌\bm{Y}.
The layer Hopfield is the realization of
formula ([10](#S2.E10 "In Hopfield update rule is attention of the transformer. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need")).
The memory of the Hopfield layer can be
filled with sets from the input or previous layers,
see Fig. [3](#S3.F3 "Figure 3 ‣ Types of new Hopfield layers. ‣ 3 New Hopfield Layers for Deep Learning ‣ Hopfield Networks is All You Need").
The memory may be filled with a reference set,
which is covered by providing the reference set as additional input.
Thus, the layer Hopfield allows the association of two sets.
A prominent example of a layer that performs such association
is the transformer attention mechanism,
which associates keys and queries, e.g. two point sets that have to be compared.
This layer allows for different kinds of
sequence-to-sequence learning,
point set operations,
and retrieval-based methods.
The layer Hopfield with skip connections in a ResNet architecture
is identical to the popular transformer and BERT models.
In the experiments, we analyzed these Hopfield layers in transformer architectures.
In our experiments in which we compare machine learning methods
on small datasets of the UCI benchmark collection
the layer Hopfield is also used.

!(/html/2008.02217/assets/x3.png)

Figure 3: The layer Hopfield allows the association of two sets 𝑹𝑹\bm{R} ( ■■\blacksquare) and 𝒀𝒀\bm{Y} ( ■■\blacksquare).
It can be integrated into deep networks that propagate sets of vectors.
The Hopfield memory
is filled
with a set from either the input or previous layers.
The output is a set of vectors 𝒁𝒁\bm{Z} ( ■■\blacksquare).

(2) Layer HopfieldPooling
for networks that propagate patterns via the stored (key) patterns Y𝑌\bm{Y}.
This layer performs
a pooling or summarization of sets 𝒀𝒀\bm{Y} obtained
from queries in previous layers or the input.
The memory of the HopfieldPooling layer is
filled with sets from the input or previous layers.
The HopfieldPooling layer uses the queries to search for
patterns in the memory, the stored set.
If more patterns are similar to
a particular search pattern (query), then the result is an average
over these patterns.
The state (query) patterns of each layer are static and can be learned.
Multiple queries supply a set to the next layer, where each query
corresponds to one element of the set.
Thus, the layer HopfieldPooling enables
fixed pattern search,
pooling operations,
and memories like LSTMs or GRUs.
The static pattern functionality is typically needed if particular
patterns must be identified in the data.
  
A single HopfieldPooling layer allows for
multiple instance learning.
Static state (query) patterns together with position encoding in the
keys allows for performing pooling operations. The position encoding can
be two-dimensional, where standard convolutional filters can be
constructed as in convolutional neural networks (CNNs).
The HopfieldPooling layer can substitute pooling, averaging, LSTM,
and permutation equivariant layers.
See Fig. [4](#S3.F4 "Figure 4 ‣ Types of new Hopfield layers. ‣ 3 New Hopfield Layers for Deep Learning ‣ Hopfield Networks is All You Need").
The layer HopfieldPooling is used for
experiments with multiple instance learning tasks, e.g. for immune repertoire classification in the experiments.

!(/html/2008.02217/assets/x4.png)

Figure 4: The layer HopfieldPooling enables
pooling or summarization of sets,
which are obtained from the input or from previous layers.
The input 𝒀𝒀\bm{Y} ( ■■\blacksquare)
can be either a set or a sequence.
The query patterns of each layer are static and can be learned.
The output is a set of vectors 𝒁𝒁\bm{Z} ( ■■\blacksquare),
where the number of vectors equals the number of query patterns.
The layer HopfieldPooling can realize multiple instance learning.

(3) Layer HopfieldLayer
for networks that propagate a vector or a set of vectors via state (query) patterns R𝑅\bm{R}.
The queries 𝑹𝑹\bm{R} can be input vectors or queries
that are computed from the output of previous layers.
The memory of the HopfieldLayer layer is
filled with a fixed set, which can be the training set,
a reference set, prototype set, or a learned set (a learned matrix).
The stored (key) patterns are static and can be learned.
If the training set is stored in the memory,
then each layer constructs a new set of queries based on the
query results of previous layers.
The stored patterns can be initialized by the training set or
a reference set and then learned, in which case they deviate from the training set.
The stored patterns can be interpreted as weights from the state (query)
to hidden neurons that have a softmax activation function (Krotov & Hopfield, [2020](#bib.bib61)).
The layer HopfieldLayer can substitute a fully connected layer,
see Fig. [5](#S3.F5 "Figure 5 ‣ Types of new Hopfield layers. ‣ 3 New Hopfield Layers for Deep Learning ‣ Hopfield Networks is All You Need").
A single HopfieldLayer layer also allows for
approaches similar to support vector machines (SVMs),
approaches similar to k𝑘k-nearest neighbor,
approaches similar to learning vector quantization,
and pattern search.
For classification, the raw data
𝒚i=(𝒛i,𝒕i)subscript𝒚𝑖subscript𝒛𝑖subscript𝒕𝑖\bm{y}\_{i}=(\bm{z}\_{i},\bm{t}\_{i}) can be the concatenation of input 𝒛isubscript𝒛𝑖\bm{z}\_{i} and target 𝒕isubscript𝒕𝑖\bm{t}\_{i}.
In this case, the matrices 𝑾Ksubscript𝑾𝐾\bm{W}\_{K} and 𝑾Vsubscript𝑾𝑉\bm{W}\_{V}
can be designed such that
inside the softmax the input 𝒛isubscript𝒛𝑖\bm{z}\_{i} is used and outside the softmax
the target 𝒕isubscript𝒕𝑖\bm{t}\_{i}.
Thus, the softmax provides a weighted average of the target vectors
based on the similarity between the query and the inputs.
Also SVM models, k𝑘k-nearest neighbor, and learning vector quantization can
be considered as weighted averages of the targets.
The encoder-decoder attention layer of the transformers
are a HopfieldLayer layer, where the memory is filled
with the encoder output set.
In our experiments with the drug design benchmark datasets,
the layer HopfieldLayer has been applied and compared to
other machine learning methods.

!(/html/2008.02217/assets/x5.png)

Figure 5: The layer HopfieldLayer enables
multiple queries of the training set,
a reference set, prototype set, or a learned set (a learned matrix).
The queries for each layer are computed from the results of previous layers.
The input is a set of vectors 𝑹𝑹\bm{R} ( ■■\blacksquare). The output is also a set of vectors 𝒁𝒁\bm{Z} ( ■■\blacksquare),
where the number of output vectors equals the number of input vectors.
The layer HopfieldLayer can realize
SVM models, k𝑘k-nearest neighbor, and LVQ.

##### Additional functionality of new Hopfield layers.

The insights about energy, convergence, and storage properties
provide all new Hopfield layers with additional functionalities:
i) multiple updates to control how
precise fixed points are found without additional parameters needed.
ii) variable β𝛽\beta to determine the kind of fixed points
such as the size of metastable states.
The variable β𝛽\beta controls over how many patterns is averaged.
As observed in the experiments,
the variable is relevant in combination with the learning rate to steer the learning
dynamics.
The parameter β𝛽\beta governs the fixed point dynamics
and can be learned, too.
iii) controlling the storage capacity via the dimension of the
associative space. The storage capacity can be relevant
for tasks with a huge number of instances as in
the immune repertoire classification experiment.
iv) pattern normalization controls, like the layernorm,
the fixed point dynamics by the norm and shift of the patterns.
For more details see appendix, Section [A.6](#A1.SS6 "A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").

## 4 Experiments

We show that our proposed Hopfield layers can be applied successfully
to a wide range of tasks.
The tasks are from natural language processing,
contain multiple instance learning problems,
a collection of small classification tasks,
and drug design problems.

##### Analysis of transformer and BERT models.

Transformer and BERT models can be implemented by the layer Hopfield.
The kind of fixed point of the Hopfield net is determined by
how the pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} is separated from others patterns.
(a) a global fixed point: no separation of a pattern from the others,
(b) a fixed point close to a single pattern: pattern is
separated from other patterns,
(c) metastable state: some patterns are similar
to each other and well separated from all other vectors.
We observed that the attention heads of transformer and BERT models
are predominantly in metastable states, which
are categorized into four classes:
(I) averaging over a very large number of patterns (very large metastable state or fixed point (a)),
(II) averaging over a large number of patterns (large metastable state),
(III) averaging over a medium number of patterns (medium metastable state),
(IV) averaging over a small number of patterns (small metastable state or fixed point (c)).
For analyzing the metastable states, we calculated
the minimal number k𝑘k of softmaxsoftmax\mathrm{softmax} values required to sum up to 0.900.900.90.
Hence, k𝑘k indicates the size of a metastable state.
To determine in which of the four classes a head is mainly operating,
we computed the distribution of k𝑘k across sequences.
Concretely, for N𝑁N tokens and for k¯¯𝑘\bar{k} as the median of the distribution,
a head is classified
as operating in class (I) if 1/2​N<k¯12𝑁¯𝑘1/2N<\bar{k},
as operating in class (II) if 1/8​N<k¯⩽1/2​N18𝑁¯𝑘12𝑁1/8N<\bar{k}\leqslant 1/2N,
as operating in class (III) if 1/32​N<k¯⩽1/8​N132𝑁¯𝑘18𝑁1/32N<\bar{k}\leqslant 1/8N, and
as operating in class (IV) if k¯⩽1/32​N¯𝑘132𝑁\bar{k}\leqslant 1/32N.
We analyzed pre-trained BERT models from Hugging
Face Inc. (Wolf et al., [2019](#bib.bib107)) according to these operating classes.
In Fig. [A.3](#A1.F3 "Figure A.3 ‣ A.5.1.1 Analysis of operating modes of the heads of a pre-trained BERT model. ‣ A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") in the appendix the distribution of the pre-trained
bert-base-cased
model is depicted (for other models see
appendix Section [A.5.1.4](#A1.SS5.SSS1.P4 "A.5.1.4 Learning Dynamics of Transformer and BERT Models. ‣ A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
Operating classes (II) (large metastable states) and
(IV) (small metastable states)
are often observed in the middle layers.
Operating class (I) (averaging over a very large number of patterns)
is abundant in lower layers.
Similar observations have been reported in other studies (Toneva & Wehbe, [2019a](#bib.bib93); [b](#bib.bib94); Tay et al., [2020](#bib.bib92)).
Operating class (III) (medium metastable states) is predominant in the last layers.

##### Multiple Instance Learning Datasets.

For multiple instance learning (MIL) (Dietterich et al., [1997](#bib.bib34)),
we integrate our new Hopfield network
via the layer HopfieldPooling into deep learning architectures.
Recently, deep learning methods
have been applied to MIL problems (Ilse et al., [2018](#bib.bib49)), but still
the performance on many datasets lacks improvement.
Thus, MIL datasets still pose an
interesting challenge, in which Hopfield layers equipped with memory
are a promising approach.

•Immune Repertoire Classification.
The first MIL task is immune repertoire classification,
where a deep learning architecture with HopfieldPooling
(DeepRC) was used (Widrich et al., [2020a](#bib.bib105); [b](#bib.bib106)).
Immune repertoire classification (Emerson et al., [2017](#bib.bib35))
typically requires
to extract few patterns from a large set of sequences, the repertoire,
that are indicative for the respective immune status.
The datasets contain ≈\approx
300,000 instances per immune repertoire,
which represents one of the largest multiple instance
learning experiments ever conducted (Carbonneau et al., [2018](#bib.bib17)).
Most MIL methods fail due the large number of instances.
This experiment comprises real-world and simulated datasets.
Simulated datasets are generated by implanting
sequence motifs (Akbar et al., [2019](#bib.bib3); Weber et al., [2020](#bib.bib103)) with low
frequency into simulated or experimentally-observed immune receptor sequences.
The performance of
DeepRC was compared with other machine learning methods:
(i) known motif,
(ii) SVM using k𝑘k-mers
and MinMax or Jaccard kernel,
(iii) K𝐾K-Nearest Neighbor (KNN) with k𝑘k-mers,
(iv) logistic regression with k𝑘k-mers,
(v) burden test with k𝑘k-mers, and
(vi) logistic multiple instance learning (lMIL).
On the real-world dataset DeepRC achieved
an AUC of 0.832±0.022plus-or-minus0.8320.0220.832\pm 0.022, followed by
the SVM with MinMax kernel (AUC 0.825±0.022plus-or-minus0.8250.0220.825\pm 0.022) and the burden
test with an AUC of 0.699±0.041plus-or-minus0.6990.0410.699\pm 0.041.
Across datasets, DeepRC outperformed all competing methods
with respect to average AUC (Widrich et al., [2020a](#bib.bib105); [b](#bib.bib106)).

•MIL benchmark datasets.
We apply Hopfield layers to further
MIL datasets (Ilse et al., [2018](#bib.bib49); Küçükaşcı &
Baydoğan, [2018](#bib.bib62); Cheplygina et al., [2016](#bib.bib23)):
Elephant, Fox and Tiger
for image annotation (Andrews et al., [2003](#bib.bib5)).
These datasets consist of color images from the Corel dataset
that have been preprocessed and
segmented. An image consists
of a set of segments (or blobs),
each characterized by color, texture and shape
descriptors.
The datasets have 100 positive and 100 negative
example images.
The latter have been randomly drawn from a pool of photos of
other animals.
Elephant comprises 1,391 instances and 230 features,
Fox 1,320 instances and 230 features, and
Tiger has 1,220 instances and 230 features.
Furthermore, we use the
UCSB breast cancer classification (Kandemir et al., [2014](#bib.bib52))
dataset, which consists of 2,002 instances across 58 input objects. An instance
represents a patch of a histopathological image of cancerous or normal tissue.
The layer HopfieldPooling is used, which allows for
computing a per-input-object representation by extracting an
average of instances that are indicative for one of the two classes.
The input to the layer HopfieldPooling
is a set of embedded instances 𝒀𝒀\bm{Y}.
A trainable but fixed state (query) pattern 𝑸𝑸\bm{Q} is used for averaging
over class-indicative instances.
This averaging enables a compression of variable-sized bags to a
fixed-sized representation to discriminate the bags.
More details in appendix Sec. [A.5.2](#A1.SS5.SSS2 "A.5.2 Experiment 2: Multiple Instance Learning Datasets. ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
Our approach has set a new state-of-the-art and has
outperformed other methods (Küçükaşcı &
Baydoğan, [2018](#bib.bib62); Carbonneau et al., [2016](#bib.bib18))
on the datasets Tiger, Elephant and UCSB Breast Cancer
(see Table [1](#S4.T1 "Table 1 ‣ Multiple Instance Learning Datasets. ‣ 4 Experiments ‣ Hopfield Networks is All You Need")).

| Method | tiger | fox | elephant | UCSB |
| --- | --- | --- | --- | --- |
| Hopfield (ours) | 91.3±0.5plus-or-minus91.30.5\mathbf{91.3\pm 0.5} | 64.05±0.4plus-or-minus64.050.464.05\pm 0.4 | 94.9±0.3plus-or-minus94.90.3\mathbf{94.9\pm 0.3} | 89.5±0.8plus-or-minus89.50.8\mathbf{89.5\pm 0.8} |
| Path encoding (Küçükaşcı & Baydoğan, [2018](#bib.bib62)) | 91.0±1.0plus-or-minus91.01.091.0\pm 1.0a | 71.2±1.4plus-or-minus71.21.471.2\pm 1.4a | 94.4±0.7plus-or-minus94.40.794.4\pm 0.7a | 88.0±2.2plus-or-minus88.02.288.0\pm 2.2a |
| MInD (Cheplygina et al., [2016](#bib.bib23)) | 85.3±1.1plus-or-minus85.31.185.3\pm 1.1a | 70.4±1.6plus-or-minus70.41.670.4\pm 1.6a | 93.6±0.9plus-or-minus93.60.993.6\pm 0.9a | 83.1±2.7plus-or-minus83.12.783.1\pm 2.7a |
| MILES (Chen et al., [2006](#bib.bib22)) | 87.2±1.7plus-or-minus87.21.787.2\pm 1.7b | 73.8±1.6plus-or-minus73.81.6\mathbf{73.8\pm 1.6}a | 92.7±0.7plus-or-minus92.70.792.7\pm 0.7a | 83.3±2.6plus-or-minus83.32.683.3\pm 2.6a |
| APR (Dietterich et al., [1997](#bib.bib34)) | 77.8±0.7plus-or-minus77.80.777.8\pm 0.7b | 54.1±0.9plus-or-minus54.10.954.1\pm 0.9b | 55.0±1.0plus-or-minus55.01.055.0\pm 1.0b | — |
| Citation-kNN (Wang, [2000](#bib.bib101)) | 85.5±0.9plus-or-minus85.50.985.5\pm 0.9b | 63.5±1.5plus-or-minus63.51.563.5\pm 1.5b | 89.6±0.9plus-or-minus89.60.989.6\pm 0.9b | 70.6±3.2plus-or-minus70.63.270.6\pm 3.2a |
| DD (Maron & Lozano-Pérez, [1998](#bib.bib67)) | 84.184.184.1b | 63.163.163.1b | 90.790.790.7b | — |

Table 1: Results for MIL datasets Tiger, Fox, Elephant, and
UCSB Breast Cancer in terms of AUC.
Results for all methods except the first are taken from either
a(Küçükaşcı &
Baydoğan, [2018](#bib.bib62)) or b(Carbonneau et al., [2016](#bib.bib18)),
depending on which reports the higher AUC.

##### UCI Benchmark Collection.

So far deep learning struggled with small datasets.
However, Hopfield networks are promising for handling small datasets,
since they can store the training data points or their representations
to perform similarity-based, nearest neighbor, or
learning vector quantization methods.
Therefore, we test the Hopfield layer Hopfield on the small datasets of
the UC Irvine (UCI) Machine Learning Repository that
have been used to benchmark
supervised learning methods (Fernández-Delgado et al., [2014](#bib.bib36); Wainberg et al., [2016](#bib.bib99); Khan et al., [2018](#bib.bib53))
and also feed-forward neural networks (Klambauer et al., [2017a](#bib.bib55); Wu et al., [2018](#bib.bib109)), where
our Hopfield networks could exploit their memory.
The whole 121 datasets in the collection
vary strongly with respect to their size, number of
features, and difficulties (Fernández-Delgado et al., [2014](#bib.bib36)), such that they have
been divided into 75 “small datasets” with less than 1,000 samples and
45 “large datasets” with more than or equal
to 1,000 samples in Klambauer et al. ([2017a](#bib.bib55)).

|  |  |  |
| --- | --- | --- |
| Method | avg. rank diff. | p𝑝p-value |
| Hopfield (ours) | −3.923.92\mathbf{-3.92} | — |
| SVM | −3.233.23-3.23 | 0.150.150.15 |
| SNN | −2.852.85-2.85 | 0.100.100.10 |
| RandomForest | −2.792.79-2.79 | 0.050.050.05 |
| … | … | … |
| Stacking | 8.738.738.73 | 1.21.21.2e−1111-11 |

Table 2: Results on 75 small datasets of the UCI benchmarks
given as difference to average rank.

On the 75 small datasets,
Random Forests (RFs) and Support Vector Machines (SVM) are highly accurate, whereas
on the large datasets, deep learning methods and neural networks are in the lead (Klambauer et al., [2017a](#bib.bib55); [b](#bib.bib56); Wu et al., [2018](#bib.bib109)).
We applied a modern Hopfield network via the layer HopfieldLayer,
where a self-normalizing net (SNN) maps the
input vector to 𝒀𝒀\bm{Y} and 𝑹𝑹\bm{R}.
The output 𝒁𝒁\bm{Z} of HopfieldLayer enters
a softmax output.
We compared our modern Hopfield networks against deep learning methods
(e.g. SNNs, resnet),
RFs, SVMs, boosting, bagging, and many other machine
learning methods of Fernández-Delgado et al. ([2014](#bib.bib36)).
Since for each method, multiple variants and implementations had been included, we
used method groups and representatives as defined by Klambauer et al. ([2017a](#bib.bib55)).
For each dataset, a ranking of the methods was calculated which is presented in
Table [2](#S4.T2 "Table 2 ‣ UCI Benchmark Collection. ‣ 4 Experiments ‣ Hopfield Networks is All You Need").
We found that Hopfield networks outperform all other methods on the
small datasets, setting a new state-of-the-art for 10 datasets.
The difference is significant except for the first three runner-up methods
(Wilcoxon signed rank test).
See appendix Section [A.5.3](#A1.SS5.SSS3 "A.5.3 Experiment 3: Classification on Small UCI Benchmark Datasets ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") for details.

Drug Design Benchmark Datasets.
We test the Hopfield layer HopfieldLayer, on four drug design datasets.
These datasets represent four main areas of modeling tasks in drug design,
concretely to develop accurate models for predicting
a) new anti-virals (HIV) by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen,
b) new protein inhibitors, concretely human β𝛽\beta-secretase (BACE) inhibitors by Subramanian et al. ([2016](#bib.bib87)),
c) metabolic effects as blood-brain barrier permeability (BBBP) (Martins et al., [2012](#bib.bib68)) and
d) side effects of a chemical compound from the Side Effect Resource (SIDER) Kuhn et al. ([2016](#bib.bib63)).
We applied the Hopfield layer HopfieldLayer,
where the training data is used
as stored patterns 𝒀𝒀\bm{Y}, the input vector as state pattern 𝑹𝑹\bm{R}, and
the corresponding training label to project the output of
the Hopfield layer 𝒀​𝑾V𝒀subscript𝑾𝑉\bm{Y}\bm{W}\_{V}.
Our architecture with HopfieldLayer has reached state-of-the-art
for predicting side
effects on SIDER 0.672±0.019plus-or-minus0.6720.0190.672\pm 0.019
as well as for predicting β𝛽\beta-secretase BACE 0.902±0.023plus-or-minus0.9020.0230.902\pm 0.023.
For details, see Table [A.5](#A1.T5 "Table A.5 ‣ A.5.4.2 Results. ‣ A.5.4 Experiment 4: Drug Design Benchmark Datasets ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") in the appendix.

Conclusion.
We have introduced a modern Hopfield network with continuous states
and the corresponding new update rule.
This network can store exponentially many patterns, retrieves patterns with
one update, and has exponentially small retrieval errors.
We analyzed the attention heads of BERT models.
The new modern Hopfield networks have been integrated
into deep learning architectures
as layers to allow the storage of and access to
raw input data, intermediate results, or learned prototypes.
These Hopfield layers enable new ways of deep learning, beyond
fully-connected, convolutional, or recurrent networks,
and provide pooling, memory, association, and attention mechanisms.
Hopfield layers that equip neural network layers with memories
improved state-of-the-art
in three out of four
considered multiple instance learning problems and
on immune repertoire classification, and
on two drug design dataset.
They yielded the best results among
different machine learning methods on
the UCI benchmark collections of small classification tasks.

## Acknowledgments

The ELLIS Unit Linz, the LIT AI Lab and the
Institute for Machine Learning are supported by
the Land Oberösterreich,
LIT grants DeepToxGen (LIT-2017-3-YOU-003),
and AI-SNN (LIT-2018-6-YOU-214), the Medical Cognitive Computing Center (MC3),
Janssen Pharmaceutica,
UCB Biopharma,
Merck Group,
Audi.JKU Deep Learning Center, Audi Electronic Venture GmbH,
TGW,
Primal, S3AI (FFG-872172),
Silicon Austria Labs (SAL),
Anyline,
FILL,
EnliteAI,
Google Brain,
ZF Friedrichshafen AG,
Robert Bosch GmbH,
TÜV Austria,
DCS,
and the NVIDIA Corporation.
IARAI is supported by Here Technologies.

## Appendix A Appendix

This appendix consists of six sections (A.1–A.6).
Section A.1 introduces the new modern Hopfield network with continuous
states and its update rule. Furthermore, Section A.1 provides
a thorough and profound theoretical analysis of
this new Hopfield network.
Section A.2 provides the mathematical background for Section A.1.
Section A.3 reviews binary Modern Hopfield Networks of Krotov &
Hopfield.
Section A.4 shows that the Hopfield update rule is the attention
mechanism of the transformer.
Section A.5 gives details on the
experiments.
Section A.6 describes the PyTorch implementation of
layers based on the new Hopfield networks and how to use them.

###### Contents of the appendix

1. [1 Introduction](#S1 "In Hopfield Networks is All You Need")
2. [2 Modern Hopfield Nets with Continuous States](#S2 "In Hopfield Networks is All You Need")
3. [3 New Hopfield Layers for Deep Learning](#S3 "In Hopfield Networks is All You Need")
4. [4 Experiments](#S4 "In Hopfield Networks is All You Need")
5. [A Appendix](#A1 "In Hopfield Networks is All You Need")
   1. [A.1 Continuous State Modern Hopfield Networks (A New Concept)](#A1.SS1 "In Appendix A Appendix ‣ Hopfield Networks is All You Need")
      1. [A.1.1 Introduction](#A1.SS1.SSS1 "In A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      2. [A.1.2 New Energy Function](#A1.SS1.SSS2 "In A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      3. [A.1.3 New Update Rule](#A1.SS1.SSS3 "In A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      4. [A.1.4 Global Convergence of the Update Rule](#A1.SS1.SSS4 "In A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      5. [A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration](#A1.SS1.SSS5 "In A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      6. [A.1.6 Properties of Fixed Points Near Stored Pattern](#A1.SS1.SSS6 "In A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      7. [A.1.7 Learning Associations](#A1.SS1.SSS7 "In A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      8. [A.1.8 Infinite Many Patterns and Forgetting Patterns](#A1.SS1.SSS8 "In A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      9. [A.1.9 Number of Spurious States](#A1.SS1.SSS9 "In A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
   2. [A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function](#A1.SS2 "In Appendix A Appendix ‣ Hopfield Networks is All You Need")
   3. [A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield)](#A1.SS3 "In Appendix A Appendix ‣ Hopfield Networks is All You Need")
      1. [A.3.1 Modern Hopfield Networks: Introduction](#A1.SS3.SSS1 "In A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      2. [A.3.2 Energy and Update Rule for Binary Modern Hopfield Networks](#A1.SS3.SSS2 "In A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
   4. [A.4 Hopfield Update Rule is Attention of The Transformer](#A1.SS4 "In Appendix A Appendix ‣ Hopfield Networks is All You Need")
   5. [A.5 Experiments](#A1.SS5 "In Appendix A Appendix ‣ Hopfield Networks is All You Need")
      1. [A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics](#A1.SS5.SSS1 "In A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      2. [A.5.2 Experiment 2: Multiple Instance Learning Datasets.](#A1.SS5.SSS2 "In A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      3. [A.5.3 Experiment 3: Classification on Small UCI Benchmark Datasets](#A1.SS5.SSS3 "In A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      4. [A.5.4 Experiment 4: Drug Design Benchmark Datasets](#A1.SS5.SSS4 "In A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
   6. [A.6 PyTorch Implementation of Hopfield Layers](#A1.SS6 "In Appendix A Appendix ‣ Hopfield Networks is All You Need")
      1. [A.6.1 Introduction](#A1.SS6.SSS1 "In A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      2. [A.6.2 Functionality](#A1.SS6.SSS2 "In A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
      3. [A.6.3 Usage](#A1.SS6.SSS3 "In A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")

###### List of theorems

1. [Theorem A1 (Global Convergence (Zangwill): Energy).](#ThmtheoremA1 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
2. [Theorem A2 (Global Convergence: Stationary Points).](#ThmtheoremA2 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
3. [Theorem A3 (Storage Capacity (M=2): Placed Patterns).](#ThmtheoremA3 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
4. [Theorem A4 (Storage Capacity (M=5): Placed Patterns).](#ThmtheoremA4 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
5. [Theorem A5 (Storage Capacity (Main): Random Patterns).](#ThmtheoremA5 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
6. [Theorem A6 (Storage Capacity (d computed): Random Patterns).](#ThmtheoremA6 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
7. [Theorem A7 (Storage Capacity (expected separation): Random Patterns).](#ThmtheoremA7 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
8. [Theorem A8 (Pattern Retrieval with One Update).](#ThmtheoremA8 "In A.1.6.2 Retrieval of Patterns with One Update and Small Retrieval Error. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
9. [Theorem A9 (Exponentially Small Retrieval Error).](#ThmtheoremA9 "In A.1.6.2 Retrieval of Patterns with One Update and Small Retrieval Error. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
10. [Theorem A10 (Storage Capacity for Binary Modern Hopfield Nets (Demircigil et al. 2017)).](#ThmtheoremA10 "In A.3.2 Energy and Update Rule for Binary Modern Hopfield Networks ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")

###### List of definitions

1. [Definition A1 (Softmax).](#ThmdefinitionA1 "In A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
2. [Definition A2 (Log-Sum-Exp Function).](#ThmdefinitionA2 "In A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
3. [Definition A3 (Convex Conjugate).](#ThmdefinitionA3 "In A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
4. [Definition A4 (Legendre Transform).](#ThmdefinitionA4 "In A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
5. [Definition A5 (Epi-Sum).](#ThmdefinitionA5 "In A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
6. [Definition A6 (Lambert Function).](#ThmdefinitionA6 "In A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")

###### List of figures

1. [1 We generalize
   the energy
   of binary modern Hopfield networks to continuous states
   while keeping fast convergence and
   storage capacity properties.
   We also propose a new update
   rule that minimizes the energy.
   The new update rule is the attention mechanism of the transformer.
   Formulae are modified to express softmaxsoftmax\mathrm{softmax} as row vector.
   “==”-sign means “keeps the properties”.](#S1.F1 "Figure 1In Hopfield Networks is All You Need")
2. [2 Left: A standard deep network with layers
   ( ■■\blacksquare) propagates
   either a vector or a set of vectors from the input to the output.
   Right: A deep network, where layers ( ■■\blacksquare)
   are equipped with associative memories
   via Hopfield layers ( ■■\blacksquare).](#S3.F2 "Figure 2In Hopfield Networks is All You Need")
3. [3 The layer Hopfield allows the association of two sets 𝑹𝑹\bm{R} ( ■■\blacksquare) and 𝒀𝒀\bm{Y} ( ■■\blacksquare).
   It can be integrated into deep networks that propagate sets of vectors.
   The Hopfield memory
   is filled
   with a set from either the input or previous layers.
   The output is a set of vectors 𝒁𝒁\bm{Z} ( ■■\blacksquare).](#S3.F3 "Figure 3In 3 New Hopfield Layers for Deep Learning ‣ Hopfield Networks is All You Need")
4. [4 The layer HopfieldPooling enables
   pooling or summarization of sets,
   which are obtained from the input or from previous layers.
   The input 𝒀𝒀\bm{Y} ( ■■\blacksquare)
   can be either a set or a sequence.
   The query patterns of each layer are static and can be learned.
   The output is a set of vectors 𝒁𝒁\bm{Z} ( ■■\blacksquare),
   where the number of vectors equals the number of query patterns.
   The layer HopfieldPooling can realize multiple instance learning.](#S3.F4 "Figure 4In 3 New Hopfield Layers for Deep Learning ‣ Hopfield Networks is All You Need")
5. [5 The layer HopfieldLayer enables
   multiple queries of the training set,
   a reference set, prototype set, or a learned set (a learned matrix).
   The queries for each layer are computed from the results of previous layers.
   The input is a set of vectors 𝑹𝑹\bm{R} ( ■■\blacksquare). The output is also a set of vectors 𝒁𝒁\bm{Z} ( ■■\blacksquare),
   where the number of output vectors equals the number of input vectors.
   The layer HopfieldLayer can realize
   SVM models, k𝑘k-nearest neighbor, and LVQ.](#S3.F5 "Figure 5In 3 New Hopfield Layers for Deep Learning ‣ Hopfield Networks is All You Need")
6. [A.1 The three cases of fixed points](#A1.F1 "Figure A.1In A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
7. [A.2 From binary Hopfield network to transformer](#A1.F2 "Figure A.2In Appendix A Appendix ‣ Hopfield Networks is All You Need")
8. [A.3 Analysis of operating modes of the heads of a pre-trained BERT model.
   For each head in each layer, the distribution of the minimal number
   k𝑘k of patterns required to sum up the softmaxsoftmax\mathrm{softmax} values to 0.900.900.90 is displayed as a violin plot in a panel.
   k𝑘k indicates the size of a metastable state.
   The bold number in the center of each panel gives the median k¯¯𝑘\bar{k} of the distribution.
   The heads in each layer are sorted according to k¯¯𝑘\bar{k}.
   Attention heads belong to the class they mainly operate in.
   Class (IV) in blue:
   Small metastable state or fixed point close to a single pattern, which
   is abundant in the middle layers (6, 7, and 8).
   Class (II) in orange: Large metastable state, which is
   prominent in middle layers (3, 4, and 5).
   Class (I) in red: Very large metastable state or global fixed point,
   which is predominant in the first layer.
   These heads can potentially be replaced by averaging operations.
   Class (III) in green: Medium metastable state,
   which is frequently observed in higher layers.
   We hypothesize that these heads are used to collect information required to
   perform the respective task.
   These heads should be the main target to improve transformer and BERT models.](#A1.F3 "Figure A.3In A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
9. [A.4 Ridge plots of the distribution of counts](#A1.F4 "Figure A.4In A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
10. [A.5 Change of count density during training](#A1.F5 "Figure A.5In A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
11. [A.6 Attentions of a Gaussian averaging heads](#A1.F6 "Figure A.6In A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
12. [A.7 A flowchart of the Hopfield layer](#A1.F7 "Figure A.7In A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")

###### List of tables

1. [1 Results for MIL datasets Tiger, Fox, Elephant, and
   UCSB Breast Cancer in terms of AUC.
   Results for all methods except the first are taken from either
   a(Küçükaşcı &
   Baydoğan, 2018) or b(Carbonneau et al., 2016),
   depending on which reports the higher AUC.](#S4.T1 "Table 1In 4 Experiments ‣ Hopfield Networks is All You Need")
2. [2 Results on 75 small datasets of the UCI benchmarks
   given as difference to average rank.](#S4.T2 "Table 2In 4 Experiments ‣ Hopfield Networks is All You Need")
3. [A.1 Results of immune repertoire classification across all datasets](#A1.T1 "Table A.1In A.5.2 Experiment 2: Multiple Instance Learning Datasets. ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
4. [A.2 Hyperparameter selection for MIL datasets](#A1.T2 "Table A.2In A.5.2 Experiment 2: Multiple Instance Learning Datasets. ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
5. [A.3 Hyperparameter selection for small UCI benchmark datasets](#A1.T3 "Table A.3In A.5.3 Experiment 3: Classification on Small UCI Benchmark Datasets ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
6. [A.4 Hyperparameter selection for drug design datasets](#A1.T4 "Table A.4In A.5.4 Experiment 4: Drug Design Benchmark Datasets ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
7. [A.5 Results on drug design benchmark datasets](#A1.T5 "Table A.5In A.5.4 Experiment 4: Drug Design Benchmark Datasets ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")

### A.1 Continuous State Modern Hopfield Networks (A New Concept)

#### A.1.1 Introduction

In Section [A.1](#A1.SS1 "A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
our new modern Hopfield network is introduced.
In Subsection [A.1.2](#A1.SS1.SSS2 "A.1.2 New Energy Function ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") we present the new energy function.
Then in Subsection [A.1.3](#A1.SS1.SSS3 "A.1.3 New Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"),
our new update rule is introduced.
In Subsection [A.1.4](#A1.SS1.SSS4 "A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we show that this update rule
ensures global convergence.
We show that all the limit points of any sequence generated by
the update rule are the
stationary points (local minima or saddle points) of the
energy function.
In Section [A.1.5](#A1.SS1.SSS5 "A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we consider the local convergence
of the update rule and see that patterns are retrieved with one update.
In Subsection [A.1.6](#A1.SS1.SSS6 "A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we consider the
properties of the fixed points that are associated with the
stored patterns.
In Subsection [A.1.6.1](#A1.SS1.SSS6.P1 "A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"),
we show that exponentially many patterns can be stored.
The main result is given in Theorem [A5](#ThmtheoremA5 "Theorem A5 (Storage Capacity (Main): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"):
For random patterns on a sphere we can store and retrieve
exponentially (in the dimension of the Hopfield space) many patterns.
Subsection [A.1.6.2](#A1.SS1.SSS6.P2 "A.1.6.2 Retrieval of Patterns with One Update and Small Retrieval Error. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") reports that
patterns are typically retrieved with one update step and that the
retrieval error is exponentially small.

In Subsection [A.1.7](#A1.SS1.SSS7 "A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we consider how
associations for the new Hopfield networks can be learned.
In Subsection [A.1.7.2](#A1.SS1.SSS7.P2 "A.1.7.2 Learning an Association Matrix – Only One Set is Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we analyze if the
association is learned directly by a bilinear form.
In Subsection [A.1.7.3](#A1.SS1.SSS7.P3 "A.1.7.3 Learning Two Association Mappings – Both Sets are Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we analyze if
stored patterns and query patterns are mapped to the space of
the Hopfield network. Therefore, we treat the architecture of
the transformer and BERT.
In Subsection [A.1.8](#A1.SS1.SSS8 "A.1.8 Infinite Many Patterns and Forgetting Patterns ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we introduce
a temporal component into the new Hopfield network that
leads to a forgetting behavior.
The forgetting allows us to treat infinite memory capacity
in Subsection [A.1.8.1](#A1.SS1.SSS8.P1 "A.1.8.1 Infinite Many Patterns. ‣ A.1.8 Infinite Many Patterns and Forgetting Patterns ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
In Subsection [A.1.8.2](#A1.SS1.SSS8.P2 "A.1.8.2 Forgetting Patterns. ‣ A.1.8 Infinite Many Patterns and Forgetting Patterns ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we consider the controlled
forgetting behavior.

In Section [A.2](#A1.SS2 "A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we provide the mathematical
background that is needed for our proofs.
In particular we give lemmas on
properties of the softmax,
the log-sum-exponential, the Legendre transform, and the
Lambert W𝑊W function.

In Section [A.3](#A1.SS3 "A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we review the new Hopfield network
as introduced by Krotov and Hopfield in 2016.
However in contrast to our new Hopfield network,
the Hopfield network of Krotov and Hopfield is binary, that is, a network with binary states.
In Subsection [A.3.1](#A1.SS3.SSS1 "A.3.1 Modern Hopfield Networks: Introduction ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we give an introduction to
neural networks equipped with associative memories and new Hopfield networks.
In Subsection [A.3.1.1](#A1.SS3.SSS1.P1 "A.3.1.1 Additional Memory and Attention for Neural Networks. ‣ A.3.1 Modern Hopfield Networks: Introduction ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we discuss neural networks that are
enhanced by an additional external memory and by attention mechanisms.
In Subsection [A.3.1.2](#A1.SS3.SSS1.P2 "A.3.1.2 Modern Hopfield networks: Overview. ‣ A.3.1 Modern Hopfield Networks: Introduction ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we give an overview over
the modern Hopfield networks.
Finally, in Subsection [A.3.2](#A1.SS3.SSS2 "A.3.2 Energy and Update Rule for Binary Modern Hopfield Networks ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we present
the energy function and the update rule for the modern, binary
Hopfield networks.

#### A.1.2 New Energy Function

We have patterns 𝒙1,…,𝒙N

subscript𝒙1…subscript𝒙𝑁\bm{x}\_{1},\ldots,\bm{x}\_{N}
that are represented by the matrix

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑿𝑿\displaystyle\bm{X}\ | =(𝒙1,…,𝒙N).absentsubscript𝒙1…subscript𝒙𝑁\displaystyle=\ \left(\bm{x}\_{1},\ldots,\bm{x}\_{N}\right)\ . |  | (11) |

The largest norm of a pattern is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M𝑀\displaystyle M\ | =maxi⁡‖𝒙i‖.absentsubscript𝑖normsubscript𝒙𝑖\displaystyle=\ \max\_{i}{{\left\|\bm{x}\_{i}\right\|}}\ . |  | (12) |

The query or state of the Hopfield network is 𝝃𝝃\bm{\xi}.

The energy function EE\mathrm{E}
in the new type of Hopfield models of Krotov and Hopfield
is E=−∑i=1NF​(𝝃T​𝒙i)Esuperscriptsubscript𝑖1𝑁𝐹superscript𝝃𝑇subscript𝒙𝑖\mathrm{E}=-\sum\_{i=1}^{N}F\left(\bm{\xi}^{T}\bm{x}\_{i}\right)
for binary patterns 𝒙isubscript𝒙𝑖\bm{x}\_{i} and binary state 𝝃𝝃\bm{\xi}
with interaction function F​(x)=xn𝐹𝑥superscript𝑥𝑛F(x)=x^{n}, where n=2𝑛2n=2 gives classical
Hopfield model (Krotov & Hopfield, [2016](#bib.bib59)).
The storage capacity is proportional to dn−1superscript𝑑𝑛1d^{n-1} (Krotov & Hopfield, [2016](#bib.bib59)).
This model was generalized by Demircigil et al. (Demircigil et al., [2017](#bib.bib31))
to exponential interaction functions
F​(x)=exp⁡(x)𝐹𝑥𝑥F(x)=\exp(x), which gives the energy
E=−exp⁡(lse​(1,𝑿T​𝝃))Else1superscript𝑿𝑇𝝃\mathrm{E}=-\exp(\mathrm{lse}(1,\bm{X}^{T}\bm{\xi})).
This energy leads to an exponential
storage capacity of N=2d/2𝑁superscript2𝑑2N=2^{d/2} for binary patterns.
Furthermore, with a single update the fixed point
is recovered with high probability. See more details in Section [A.3](#A1.SS3 "A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").

In contrast to the these binary modern Hopfield networks,
we focus on modern Hopfield networks
with continuous states that can store continuous patterns.
We generalize the energy of Demircigil et al. (Demircigil et al., [2017](#bib.bib31))
to continuous states while keeping the lselse\mathrm{lse} properties which
ensure high storage capacity and fast convergence.
Our new energy EE\mathrm{E} for a continuous query or state 𝝃𝝃\bm{\xi} is defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EE\displaystyle\mathrm{E}\ | =−lse​(β,𝑿T​𝝃)+12​𝝃T​𝝃+β−1​ln⁡N+12​M2absentlse𝛽superscript𝑿𝑇𝝃12superscript𝝃𝑇𝝃superscript𝛽1𝑁12superscript𝑀2\displaystyle=\ -\ \mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\ +\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}M^{2} |  | (13) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−β−1​ln⁡(∑i=1Nexp⁡(β​𝒙iT​𝝃))+β−1​ln⁡N+12​𝝃T​𝝃+12​M2absentsuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃superscript𝛽1𝑁12superscript𝝃𝑇𝝃12superscript𝑀2\displaystyle=\ -\ \beta^{-1}\ln\left(\sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}M^{2} |  | (14) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−β−1​ln⁡(1N​∑i=1Nexp⁡(−12​β​(M2−‖𝒙i‖2))​exp⁡(−12​β​‖𝒙i−𝝃‖2)).absentsuperscript𝛽11𝑁superscriptsubscript𝑖1𝑁12𝛽superscript𝑀2superscriptnormsubscript𝒙𝑖212𝛽superscriptnormsubscript𝒙𝑖𝝃2\displaystyle=\ -\ \beta^{-1}\ln\left(\frac{1}{N}\ \sum\_{i=1}^{N}\exp\left(-\ \frac{1}{2}\ \beta\ \left(M^{2}\ -\ {{\left\|\bm{x}\_{i}\right\|}}^{2}\right)\right)\ \exp\left(-\ \frac{1}{2}\ \beta\ {{\left\|\bm{x}\_{i}\ -\ \bm{\xi}\right\|}}^{2}\right)\right)\ . |  | (15) |

First let us collect and prove some properties of EE\mathrm{E}.
The next lemma gives bounds on the energy EE\mathrm{E}.

###### Lemma A1.

The energy EE\mathrm{E} is larger than zero:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 00\displaystyle 0\ | ⩽E.absentE\displaystyle\leqslant\ \mathrm{E}\ . |  | (16) |

For 𝛏𝛏\bm{\xi} in the simplex defined by the patterns,
the energy EE\mathrm{E} is upper bounded by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EE\displaystyle\mathrm{E}\ | ⩽β−1​ln⁡N+12​M2,absentsuperscript𝛽1𝑁12superscript𝑀2\displaystyle\leqslant\ \beta^{-1}\ln N\ +\ \frac{1}{2}\ M^{2}\ , |  | (17) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EE\displaystyle\mathrm{E}\ | ⩽ 2​M2.absent2superscript𝑀2\displaystyle\leqslant\ 2\ M^{2}\ . |  | (18) |

###### Proof.

We start by deriving the lower bound of zero.
The pattern most similar to query or state 𝝃𝝃\bm{\xi} is
𝒙𝝃subscript𝒙𝝃\bm{x}\_{\bm{\xi}}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒙𝝃subscript𝒙𝝃\displaystyle\bm{x}\_{\bm{\xi}}\ | =𝒙k,k=arg⁡maxi⁡𝝃T​𝒙i.formulae-sequenceabsentsubscript𝒙𝑘𝑘subscript𝑖superscript𝝃𝑇subscript𝒙𝑖\displaystyle=\ \bm{x}\_{k}\ ,\quad k\ =\ \arg\max\_{i}\bm{\xi}^{T}\bm{x}\_{i}\ . |  | (19) |

We obtain

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EE\displaystyle\mathrm{E}\ | =−β−1​ln⁡(∑i=1Nexp⁡(β​𝒙iT​𝝃))+β−1​ln⁡N+12​𝝃T​𝝃+12​M2absentsuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃superscript𝛽1𝑁12superscript𝝃𝑇𝝃12superscript𝑀2\displaystyle=\ -\ \beta^{-1}\ln\left(\sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}\ M^{2} |  | (20) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−β−1​ln⁡(1N​∑i=1Nexp⁡(β​𝒙iT​𝝃))+12​𝝃T​𝝃+12​M2absentsuperscript𝛽11𝑁superscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃12superscript𝝃𝑇𝝃12superscript𝑀2\displaystyle=\ -\ \beta^{-1}\ln\left(\frac{1}{N}\ \sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)\ +\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}\ M^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥−β−1​ln⁡(1N​∑i=1Nexp⁡(β​𝒙iT​𝝃))+12​𝝃T​𝝃+12​𝒙𝝃T​𝒙𝝃absentsuperscript𝛽11𝑁superscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃12superscript𝝃𝑇𝝃12superscriptsubscript𝒙𝝃𝑇subscript𝒙𝝃\displaystyle\geq\ -\ \beta^{-1}\ln\left(\frac{1}{N}\ \sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)\ +\ \frac{1}{2}\ \bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}\ \bm{x}\_{\bm{\xi}}^{T}\bm{x}\_{\bm{\xi}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥−β−1​ln⁡(exp⁡(β​𝒙𝝃T​𝝃))+12​𝝃T​𝝃+12​𝒙𝝃T​𝒙𝝃absentsuperscript𝛽1𝛽superscriptsubscript𝒙𝝃𝑇𝝃12superscript𝝃𝑇𝝃12superscriptsubscript𝒙𝝃𝑇subscript𝒙𝝃\displaystyle\geq\ -\ \beta^{-1}\ln\left(\exp(\beta\bm{x}\_{\bm{\xi}}^{T}\bm{\xi})\right)\ +\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}\ \bm{x}\_{\bm{\xi}}^{T}\bm{x}\_{\bm{\xi}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−𝒙𝝃T​𝝃+12​𝝃T​𝝃+12​𝒙𝝃T​𝒙𝝃absentsuperscriptsubscript𝒙𝝃𝑇𝝃12superscript𝝃𝑇𝝃12superscriptsubscript𝒙𝝃𝑇subscript𝒙𝝃\displaystyle=\ -\ \bm{x}\_{\bm{\xi}}^{T}\bm{\xi}\ +\ \frac{1}{2}\ \bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}\ \bm{x}\_{\bm{\xi}}^{T}\bm{x}\_{\bm{\xi}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =12​(𝝃−𝒙𝝃)T​(𝝃−𝒙𝝃)=12​‖𝝃−𝒙𝝃‖2≥ 0.absent12superscript𝝃subscript𝒙𝝃𝑇𝝃subscript𝒙𝝃12superscriptnorm𝝃subscript𝒙𝝃2 0\displaystyle=\frac{1}{2}\ \left(\bm{\xi}\ -\ \bm{x}\_{\bm{\xi}}\right)^{T}\left(\bm{\xi}\ -\ \bm{x}\_{\bm{\xi}}\right)\ =\ \frac{1}{2}\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{\bm{\xi}}\right\|}}^{2}\ \geq\ 0\ . |  |

The energy is zero and, therefore, the bound attained, if all 𝒙isubscript𝒙𝑖\bm{x}\_{i} are equal,
that is, 𝒙i=𝒙subscript𝒙𝑖𝒙\bm{x}\_{i}=\bm{x} for all i𝑖i and
𝝃=𝒙𝝃𝒙\bm{\xi}=\bm{x}.

For deriving upper bounds on the energy EE\mathrm{E},
we require the the query 𝝃𝝃\bm{\xi}
to be in the simplex defined by the patterns, that is,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝝃=∑i=1Npi​𝒙i,∑i=1Npi= 1,∀i: 0⩽pi.:formulae-sequence𝝃superscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝑖1𝑁subscript𝑝𝑖  1subscriptfor-all𝑖 0subscript𝑝𝑖\displaystyle\bm{\xi}\ =\ \sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i}\ ,\quad\sum\_{i=1}^{N}p\_{i}\ =\ 1\ ,\quad\forall\_{i}:\ 0\ \leqslant\ p\_{i}\ . |  | (21) |

The first upper bound is.

|  |  |  |  |
| --- | --- | --- | --- |
|  | E=−β−1​ln⁡(∑i=1Nexp⁡(β​𝒙iT​𝝃))+12​𝝃T​𝝃+β−1​ln⁡N+12​M2Esuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃12superscript𝝃𝑇𝝃superscript𝛽1𝑁12superscript𝑀2\displaystyle\mathrm{E}\ =\ -\ \beta^{-1}\ln\left(\sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)\ +\ \frac{1}{2}\ \bm{\xi}^{T}\bm{\xi}\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}\ M^{2} |  | (22) |
|  |  |  |
| --- | --- | --- |
|  | ⩽−∑i=1Npi​(𝒙iT​𝝃)+12​𝝃T​𝝃+β−1​ln⁡N+12​M2absentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖superscriptsubscript𝒙𝑖𝑇𝝃12superscript𝝃𝑇𝝃superscript𝛽1𝑁12superscript𝑀2\displaystyle\leqslant\ -\sum\_{i=1}^{N}p\_{i}\ (\bm{x}\_{i}^{T}\bm{\xi})\ +\ \frac{1}{2}\ \bm{\xi}^{T}\bm{\xi}\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}\ M^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | =−12​𝝃T​𝝃+β−1​ln⁡N+12​M2⩽β−1​ln⁡N+12​M2.absent12superscript𝝃𝑇𝝃superscript𝛽1𝑁12superscript𝑀2superscript𝛽1𝑁12superscript𝑀2\displaystyle=\ -\ \frac{1}{2}\ \bm{\xi}^{T}\bm{\xi}\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}\ M^{2}\ \leqslant\ \beta^{-1}\ln N\ +\ \frac{1}{2}\ M^{2}\ . |  |

For the first inequality we applied
Lemma [A19](#ThmlemmaA19 "Lemma A19. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") to −lse​(β,𝑿T​𝝃)lse𝛽superscript𝑿𝑇𝝃-\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi}) with 𝒛=𝒑𝒛𝒑\bm{z}=\bm{p} giving

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −lse​(β,𝑿T​𝝃)lse𝛽superscript𝑿𝑇𝝃\displaystyle-\ \mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\ | ⩽−∑i=1Npi​(𝒙iT​𝝃)+β−1​∑i=1Npi​ln⁡pi⩽−∑i=1Npi​(𝒙iT​𝝃),absentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖superscriptsubscript𝒙𝑖𝑇𝝃superscript𝛽1superscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝑝𝑖superscriptsubscript𝑖1𝑁subscript𝑝𝑖superscriptsubscript𝒙𝑖𝑇𝝃\displaystyle\leqslant\ -\ \sum\_{i=1}^{N}p\_{i}\ (\bm{x}\_{i}^{T}\bm{\xi})\ +\ \beta^{-1}\sum\_{i=1}^{N}p\_{i}\ln p\_{i}\ \leqslant\ -\ \sum\_{i=1}^{N}p\_{i}\ (\bm{x}\_{i}^{T}\bm{\xi})\ , |  | (23) |

as the term involving the logarithm is non-positive.

Next we derive the second upper bound, for which
we need the mean 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} of the patterns

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒎𝒙subscript𝒎𝒙\displaystyle\bm{m}\_{\bm{x}}\ | =1N​∑i=1N𝒙i.absent1𝑁superscriptsubscript𝑖1𝑁subscript𝒙𝑖\displaystyle=\ \frac{1}{N}\ \sum\_{i=1}^{N}\bm{x}\_{i}\ . |  | (24) |

We obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | E=−β−1​ln⁡(∑i=1Nexp⁡(β​𝒙iT​𝝃))+12​𝝃T​𝝃+β−1​ln⁡N+12​M2Esuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃12superscript𝝃𝑇𝝃superscript𝛽1𝑁12superscript𝑀2\displaystyle\mathrm{E}\ =\ -\ \beta^{-1}\ln\left(\sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)\ +\ \frac{1}{2}\ \bm{\xi}^{T}\bm{\xi}\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}\ M^{2} |  | (25) |
|  |  |  |
| --- | --- | --- |
|  | ⩽−∑i=1N1N​𝒙iT​𝝃+12​𝝃T​𝝃+12​M2absentsuperscriptsubscript𝑖1𝑁1𝑁superscriptsubscript𝒙𝑖𝑇𝝃12superscript𝝃𝑇𝝃12superscript𝑀2\displaystyle\leqslant\ -\sum\_{i=1}^{N}\frac{1}{N}\ \bm{x}\_{i}^{T}\bm{\xi}\ +\ \frac{1}{2}\ \bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}\ M^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | =−𝒎𝒙T​𝝃+12​𝝃T​𝝃+12​M2absentsuperscriptsubscript𝒎𝒙𝑇𝝃12superscript𝝃𝑇𝝃12superscript𝑀2\displaystyle=\ -\ \bm{m}\_{\bm{x}}^{T}\bm{\xi}\ +\ \frac{1}{2}\ \bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}\ M^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | ⩽‖𝒎𝒙‖​‖𝝃‖+12​‖𝝃‖2+12​M2absentnormsubscript𝒎𝒙norm𝝃12superscriptnorm𝝃212superscript𝑀2\displaystyle\leqslant\ {{\left\|\bm{m}\_{\bm{x}}\right\|}}\ {{\left\|\bm{\xi}\right\|}}\ +\ \frac{1}{2}\ {{\left\|\bm{\xi}\right\|}}^{2}\ +\ \frac{1}{2}\ M^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | ⩽ 2​M2,absent2superscript𝑀2\displaystyle\leqslant\ 2\ M^{2}\ , |  |

where for the first inequality we again applied
Lemma [A19](#ThmlemmaA19 "Lemma A19. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") with 𝒛=(1/N,…,1/N)𝒛1𝑁…1𝑁\bm{z}=(1/N,\ldots,1/N)
and β−1​∑i1/N​ln⁡(1/N)=−β−1​ln⁡(N)superscript𝛽1subscript𝑖1𝑁1𝑁superscript𝛽1𝑁\beta^{-1}\sum\_{i}1/N\ln(1/N)=-\beta^{-1}\ln(N).
This inequality also follows from Jensen’s inequality.
The second inequality uses the Cauchy-Schwarz inequality.
The last inequality uses

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝝃‖norm𝝃\displaystyle{{\left\|\bm{\xi}\right\|}}\ | =‖∑ipi​𝒙i‖⩽∑ipi​‖𝒙i‖⩽∑ipi​M=Mabsentnormsubscript𝑖subscript𝑝𝑖subscript𝒙𝑖subscript𝑖subscript𝑝𝑖normsubscript𝒙𝑖subscript𝑖subscript𝑝𝑖𝑀𝑀\displaystyle=\ {{\left\|\sum\_{i}p\_{i}\ \bm{x}\_{i}\right\|}}\ \leqslant\ \sum\_{i}p\_{i}\ {{\left\|\bm{x}\_{i}\right\|}}\ \leqslant\ \sum\_{i}p\_{i}M\ =\ M |  | (26) |

and

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒎𝒙‖normsubscript𝒎𝒙\displaystyle{{\left\|\bm{m}\_{\bm{x}}\right\|}}\ | =‖∑i(1/N)​𝒙i‖⩽∑i(1/N)​‖𝒙i‖⩽∑i(1/N)​M=M.absentnormsubscript𝑖1𝑁subscript𝒙𝑖subscript𝑖1𝑁normsubscript𝒙𝑖subscript𝑖1𝑁𝑀𝑀\displaystyle=\ {{\left\|\sum\_{i}(1/N)\ \bm{x}\_{i}\right\|}}\ \leqslant\ \sum\_{i}(1/N)\ {{\left\|\bm{x}\_{i}\right\|}}\ \leqslant\ \sum\_{i}(1/N)\ M\ =\ M\ . |  | (27) |

∎

#### A.1.3 New Update Rule

We now introduce an update rule for minimizing the energy function EE\mathrm{E}.
The new update rule is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =𝑿​𝒑=𝑿​softmax​(β​𝑿T​𝝃),absent𝑿𝒑𝑿softmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ \bm{X}\bm{p}\ =\ \bm{X}\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ , |  | (28) |

where we used

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝑿T​𝝃).absentsoftmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ \mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ . |  | (29) |

The new state 𝝃newsuperscript𝝃new\bm{\xi}^{\mathrm{new}} is in the simplex defined by the patterns, no matter
what the previous state 𝝃𝝃\bm{\xi} was.
For comparison, the synchronous update rule for the classical Hopfield network with threshold zero is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =sgn(𝑿​𝑿T​𝝃).absentsgn𝑿superscript𝑿𝑇𝝃\displaystyle=\ \mathop{\mathrm{sgn}\,}(\bm{X}\bm{X}^{T}\bm{\xi})\ . |  | (30) |

Therefore, instead of using the vector 𝑿T​𝝃superscript𝑿𝑇𝝃\bm{X}^{T}\bm{\xi} as in the
classical Hopfield network, its softmax version softmax​(β​𝑿T​𝝃)softmax𝛽superscript𝑿𝑇𝝃\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi}) is used.

In the next section (Section [A.1.4](#A1.SS1.SSS4 "A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) we show
that the update rule Eq. ([28](#A1.E28 "In A.1.3 New Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) ensures global convergence.
We show that all the limit points of any sequence generated by
the update rule are the
stationary points (local minima or saddle points) of the
energy function EE\mathrm{E}.
In Section [A.1.5](#A1.SS1.SSS5 "A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") we consider the local convergence
of the update rule Eq. ([28](#A1.E28 "In A.1.3 New Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
and see that patterns are retrieved with one update.

#### A.1.4 Global Convergence of the Update Rule

We are interested in the global convergence,
that is, convergence from each initial point,
of the iteration

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =f​(𝝃)=𝑿​𝒑=𝑿​softmax​(β​𝑿T​𝝃),absent𝑓𝝃𝑿𝒑𝑿softmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ f(\bm{\xi})\ =\ \bm{X}\bm{p}\ =\ \bm{X}\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ , |  | (31) |

where we used

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝑿T​𝝃).absentsoftmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ \mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ . |  | (32) |

We defined the energy function

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EE\displaystyle\mathrm{E}\ | =−lse​(β,𝑿T​𝝃)+12​𝝃T​𝝃+β−1​ln⁡N+12​M2absentlse𝛽superscript𝑿𝑇𝝃12superscript𝝃𝑇𝝃superscript𝛽1𝑁12superscript𝑀2\displaystyle=\ -\ \mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\ +\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}M^{2} |  | (33) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−β−1​ln⁡(∑i=1Nexp⁡(β​𝒙iT​𝝃))+β−1​ln⁡N+12​𝝃T​𝝃+12​M2.absentsuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃superscript𝛽1𝑁12superscript𝝃𝑇𝝃12superscript𝑀2\displaystyle=\ -\ \beta^{-1}\ln\left(\sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}M^{2}\ . |  | (34) |

We will show that
the update rule in Eq. ([31](#A1.E31 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
is the Concave-Convex Procedure (CCCP) for minimizing the energy EE\mathrm{E}.
The CCCP is proven to converge globally.

###### Theorem A1 (Global Convergence (Zangwill): Energy).

The update rule Eq. ([31](#A1.E31 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
converges globally:
For 𝛏t+1=f​(𝛏t)superscript𝛏𝑡1𝑓superscript𝛏𝑡\bm{\xi}^{t+1}=f(\bm{\xi}^{t}),
the energy E​(𝛏t)→E​(𝛏∗)→Esuperscript𝛏𝑡Esuperscript𝛏\mathrm{E}(\bm{\xi}^{t})\to\mathrm{E}(\bm{\xi}^{\*}) for t→∞→𝑡t\to\infty
and a fixed point 𝛏∗superscript𝛏\bm{\xi}^{\*}.

###### Proof.

The Concave-Convex Procedure (CCCP) (Yuille & Rangarajan, [2002](#bib.bib113); [2003](#bib.bib114))
minimizes a function that is the sum of a concave function and a convex function.
CCCP is equivalent to Legendre minimization (Rangarajan et al., [1996](#bib.bib77); [1999](#bib.bib78))
algorithms (Yuille & Rangarajan, [2003](#bib.bib114)).
The Jacobian of the
softmax is positive semi-definite
according to Lemma [A22](#ThmlemmaA22 "Lemma A22. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
The Jacobian of the softmax is the
Hessian of the lselse\mathrm{lse}, therefore lselse\mathrm{lse} is a convex and
−lselse-\mathrm{lse} a concave function.
Therefore, the energy function E​(𝝃)E𝝃\mathrm{E}(\bm{\xi}) is the sum of the convex function
E1​(𝝃)=1/2​𝝃T​𝝃+C1subscriptE1𝝃12superscript𝝃𝑇𝝃subscript𝐶1\mathrm{E}\_{1}(\bm{\xi})=1/2\bm{\xi}^{T}\bm{\xi}+C\_{1}
and the concave function E2​(𝝃)=−lsesubscriptE2𝝃lse\mathrm{E}\_{2}(\bm{\xi})=-\mathrm{lse}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E​(𝝃)E𝝃\displaystyle\mathrm{E}(\bm{\xi})\ | =E1​(𝝃)+E2​(𝝃),absentsubscriptE1𝝃subscriptE2𝝃\displaystyle=\ \mathrm{E}\_{1}(\bm{\xi})\ +\ \mathrm{E}\_{2}(\bm{\xi})\ , |  | (35) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E1​(𝝃)subscriptE1𝝃\displaystyle\mathrm{E}\_{1}(\bm{\xi})\ | =12​𝝃T​𝝃+β−1​ln⁡N+12​M2=12​𝝃T​𝝃+C1,absent12superscript𝝃𝑇𝝃superscript𝛽1𝑁12superscript𝑀212superscript𝝃𝑇𝝃subscript𝐶1\displaystyle=\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}M^{2}\ =\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ C\_{1}\ , |  | (36) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E2​(𝝃)subscriptE2𝝃\displaystyle\mathrm{E}\_{2}(\bm{\xi})\ | =−lse​(β,𝑿T​𝝃),absentlse𝛽superscript𝑿𝑇𝝃\displaystyle=\ -\ \mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\ , |  | (37) |

where C1subscript𝐶1C\_{1} does not depend on 𝝃𝝃\bm{\xi}.

The Concave-Convex Procedure (CCCP) (Yuille & Rangarajan, [2002](#bib.bib113); [2003](#bib.bib114))
applied to EE\mathrm{E} is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇ξE1​(𝝃t+1)subscript∇𝜉subscriptE1superscript𝝃𝑡1\displaystyle\nabla\_{\xi}\mathrm{E}\_{1}\left(\bm{\xi}^{t+1}\right)\ | =−∇ξE2​(𝝃t),absentsubscript∇𝜉subscriptE2superscript𝝃𝑡\displaystyle=\ -\ \nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ , |  | (38) |

which is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇ξ(12​𝝃T​𝝃+C1)⁡(𝝃t+1)subscript∇𝜉12superscript𝝃𝑇𝝃subscript𝐶1superscript𝝃𝑡1\displaystyle\nabla\_{\xi}\left(\frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ C\_{1}\right)\left(\bm{\xi}^{t+1}\right)\ | =∇ξlse​(β,𝑿T​𝝃t).absentsubscript∇𝜉lse𝛽superscript𝑿𝑇superscript𝝃𝑡\displaystyle=\ \nabla\_{\xi}\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi}^{t})\ . |  | (39) |

The resulting update rule is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃t+1superscript𝝃𝑡1\displaystyle\bm{\xi}^{t+1}\ | =𝑿​𝒑t=𝑿​softmax​(β​𝑿T​𝝃t)absent𝑿superscript𝒑𝑡𝑿softmax𝛽superscript𝑿𝑇superscript𝝃𝑡\displaystyle=\ \bm{X}\bm{p}^{t}\ =\ \bm{X}\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi}^{t}) |  | (40) |

using

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑tsuperscript𝒑𝑡\displaystyle\bm{p}^{t}\ | =softmax​(β​𝑿T​𝝃t).absentsoftmax𝛽superscript𝑿𝑇superscript𝝃𝑡\displaystyle=\ \mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi}^{t})\ . |  | (41) |

This is the update rule in Eq. ([31](#A1.E31 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).

Theorem 2 in Yuille & Rangarajan ([2002](#bib.bib113)) and Theorem 2 in Yuille & Rangarajan ([2003](#bib.bib114)) state that
the update rule Eq. ([31](#A1.E31 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
is guaranteed to monotonically decrease the energy EE\mathrm{E}
as a function of time.
See also Theorem 2 in Sriperumbudur & Lanckriet ([2009](#bib.bib86)).
∎

Although the objective converges in all cases,
it does not necessarily converge to a local minimum (Lipp & Boyd, [2016](#bib.bib65)).

However the convergence proof of CCCP in Yuille & Rangarajan ([2002](#bib.bib113); [2003](#bib.bib114))
was not as rigorous as required.
In Sriperumbudur & Lanckriet ([2009](#bib.bib86)) a rigorous analysis of the convergence of CCCP
is performed using Zangwill’s global convergence theory
of iterative algorithms.

In Sriperumbudur & Lanckriet ([2009](#bib.bib86)) the minimization problem

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | min𝝃subscript𝝃\displaystyle\min\_{\bm{\xi}} | ​E1+E2 subscriptE1subscriptE2\displaystyle{\mbox{\ ~{}}}\mathrm{E}\_{1}\ +\ \mathrm{E}\_{2} |  | (42) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | s.t. | ​𝒄​(𝝃)⩽𝟎,𝒅​(𝝃)= 0formulae-sequence 𝒄𝝃0𝒅𝝃 0\displaystyle{\mbox{\ ~{}}}\bm{c}(\bm{\xi})\leqslant\bm{0}\ ,\quad\bm{d}(\bm{\xi})\ =\ \bm{0} |  |

is considered with E1subscriptE1\mathrm{E}\_{1} convex, −E2subscriptE2-\mathrm{E}\_{2} convex, 𝒄𝒄\bm{c}
component-wise convex function, and 𝒅𝒅\bm{d} an affine function.
The CCCP algorithm solves this minimization problem by linearization of the
concave part and is defined in Sriperumbudur & Lanckriet ([2009](#bib.bib86)) as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃t+1∈arg⁡min𝝃superscript𝝃𝑡1subscript𝝃\displaystyle\bm{\xi}^{t+1}\ \in\ \arg\min\_{\bm{\xi}} | ​E1​(𝝃)+𝝃T​∇ξE2​(𝝃t) subscriptE1𝝃superscript𝝃𝑇subscript∇𝜉subscriptE2superscript𝝃𝑡\displaystyle{\mbox{\ ~{}}}\mathrm{E}\_{1}\left(\bm{\xi}\right)\ +\ \bm{\xi}^{T}\nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right) |  | (43) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | s.t. | ​𝒄​(𝝃)⩽𝟎,𝒅​(𝝃)= 0.formulae-sequence 𝒄𝝃0𝒅𝝃 0\displaystyle{\mbox{\ ~{}}}\bm{c}(\bm{\xi})\leqslant\bm{0}\ ,\quad\bm{d}(\bm{\xi})\ =\ \bm{0}\ . |  |

We define the upper bound ECsubscriptEC\mathrm{E}\_{\mathrm{C}} on the energy:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EC​(𝝃,𝝃t)subscriptEC𝝃superscript𝝃𝑡\displaystyle\mathrm{E}\_{\mathrm{C}}\left(\bm{\xi},\bm{\xi}^{t}\right)\ | :=E1​(𝝃)+E2​(𝝃t)+(𝝃−𝝃t)T​∇ξE2​(𝝃t).assignabsentsubscriptE1𝝃subscriptE2superscript𝝃𝑡superscript𝝃superscript𝝃𝑡𝑇subscript∇𝜉subscriptE2superscript𝝃𝑡\displaystyle:=\ \mathrm{E}\_{1}\left(\bm{\xi}\right)\ +\ \mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ +\ \left(\bm{\xi}\ -\ \bm{\xi}^{t}\right)^{T}\nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ . |  | (44) |

ECsubscriptEC\mathrm{E}\_{\mathrm{C}} is equal to the energy E​(𝝃t)Esuperscript𝝃𝑡\mathrm{E}\left(\bm{\xi}^{t}\right) for 𝝃=𝝃t𝝃superscript𝝃𝑡\bm{\xi}=\bm{\xi}^{t}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EC​(𝝃t,𝝃t)subscriptECsuperscript𝝃𝑡superscript𝝃𝑡\displaystyle\mathrm{E}\_{\mathrm{C}}\left(\bm{\xi}^{t},\bm{\xi}^{t}\right)\ | =E1​(𝝃t)+E2​(𝝃t)=E​(𝝃t).absentsubscriptE1superscript𝝃𝑡subscriptE2superscript𝝃𝑡Esuperscript𝝃𝑡\displaystyle=\ \mathrm{E}\_{1}\left(\bm{\xi}^{t}\right)\ +\ \mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ =\ \mathrm{E}\left(\bm{\xi}^{t}\right)\ . |  | (45) |

Since −E2subscriptE2-\mathrm{E}\_{2} is convex, the first order characterization of convexity
holds (Eq. 3.2 in Boyd & Vandenberghe ([2009](#bib.bib12))):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −E2​(𝝃)subscriptE2𝝃\displaystyle-\ \mathrm{E}\_{2}\left(\bm{\xi}\right)\ | ≥−E2​(𝝃t)−(𝝃−𝝃t)T​∇ξE2​(𝝃t),absentsubscriptE2superscript𝝃𝑡superscript𝝃superscript𝝃𝑡𝑇subscript∇𝜉subscriptE2superscript𝝃𝑡\displaystyle\geq\ -\ \mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ -\ \left(\bm{\xi}\ -\ \bm{\xi}^{t}\right)^{T}\nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ , |  | (46) |

that is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E2​(𝝃)subscriptE2𝝃\displaystyle\mathrm{E}\_{2}\left(\bm{\xi}\right)\ | ⩽E2​(𝝃t)+(𝝃−𝝃t)T​∇ξE2​(𝝃t).absentsubscriptE2superscript𝝃𝑡superscript𝝃superscript𝝃𝑡𝑇subscript∇𝜉subscriptE2superscript𝝃𝑡\displaystyle\leqslant\ \mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ +\ \left(\bm{\xi}\ -\ \bm{\xi}^{t}\right)^{T}\nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ . |  | (47) |

Therefore, for 𝝃≠𝝃t𝝃superscript𝝃𝑡\bm{\xi}\not=\bm{\xi}^{t} the function ECsubscriptEC\mathrm{E}\_{\mathrm{C}} is an upper bound on the energy:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E​(𝝃)E𝝃\displaystyle\mathrm{E}\left(\bm{\xi}\right)\ | ⩽EC​(𝝃,𝝃t)=E1​(𝝃)+E2​(𝝃t)+(𝝃−𝝃t)T​∇ξE2​(𝝃t)absentsubscriptEC𝝃superscript𝝃𝑡subscriptE1𝝃subscriptE2superscript𝝃𝑡superscript𝝃superscript𝝃𝑡𝑇subscript∇𝜉subscriptE2superscript𝝃𝑡\displaystyle\leqslant\ \mathrm{E}\_{\mathrm{C}}\left(\bm{\xi},\bm{\xi}^{t}\right)\ =\ \mathrm{E}\_{1}\left(\bm{\xi}\right)\ +\ \mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ +\ \left(\bm{\xi}\ -\ \bm{\xi}^{t}\right)^{T}\nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right) |  | (48) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =E1​(𝝃)+𝝃T​∇ξE2​(𝝃t)+C2,absentsubscriptE1𝝃superscript𝝃𝑇subscript∇𝜉subscriptE2superscript𝝃𝑡subscript𝐶2\displaystyle=\ \mathrm{E}\_{1}\left(\bm{\xi}\right)\ +\ \bm{\xi}^{T}\nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ +\ C\_{2}\ , |  |

where C2subscript𝐶2C\_{2} does not depend on 𝝃𝝃\bm{\xi}.
Since we do not have constraints, 𝝃t+1superscript𝝃𝑡1\bm{\xi}^{t+1} is defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃t+1∈arg⁡min𝝃superscript𝝃𝑡1subscript𝝃\displaystyle\bm{\xi}^{t+1}\ \in\ \arg\min\_{\bm{\xi}} | ​EC​(𝝃,𝝃t), subscriptEC𝝃superscript𝝃𝑡\displaystyle{\mbox{\ ~{}}}\mathrm{E}\_{\mathrm{C}}\left(\bm{\xi},\bm{\xi}^{t}\right)\ , |  | (49) |

hence
EC​(𝝃t+1,𝝃t)⩽EC​(𝝃t,𝝃t)subscriptECsuperscript𝝃𝑡1superscript𝝃𝑡subscriptECsuperscript𝝃𝑡superscript𝝃𝑡\mathrm{E}\_{\mathrm{C}}\left(\bm{\xi}^{t+1},\bm{\xi}^{t}\right)\leqslant\ \mathrm{E}\_{\mathrm{C}}\left(\bm{\xi}^{t},\bm{\xi}^{t}\right).
Combining the inequalities gives:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E​(𝝃t+1)Esuperscript𝝃𝑡1\displaystyle\mathrm{E}\left(\bm{\xi}^{t+1}\right)\ | ⩽EC​(𝝃t+1,𝝃t)⩽EC​(𝝃t,𝝃t)=E​(𝝃t).absentsubscriptECsuperscript𝝃𝑡1superscript𝝃𝑡subscriptECsuperscript𝝃𝑡superscript𝝃𝑡Esuperscript𝝃𝑡\displaystyle\leqslant\ \mathrm{E}\_{\mathrm{C}}\left(\bm{\xi}^{t+1},\bm{\xi}^{t}\right)\ \leqslant\ \mathrm{E}\_{\mathrm{C}}\left(\bm{\xi}^{t},\bm{\xi}^{t}\right)\ =\ \mathrm{E}\left(\bm{\xi}^{t}\right)\ . |  | (50) |

Since we do not have constraints, 𝝃t+1superscript𝝃𝑡1\bm{\xi}^{t+1} is the minimum of

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EC​(𝝃,𝝃t)subscriptEC𝝃superscript𝝃𝑡\displaystyle\mathrm{E}\_{\mathrm{C}}\left(\bm{\xi},\bm{\xi}^{t}\right)\ | =E1​(𝝃)+𝝃T​∇ξE2​(𝝃t)+C2absentsubscriptE1𝝃superscript𝝃𝑇subscript∇𝜉subscriptE2superscript𝝃𝑡subscript𝐶2\displaystyle=\ \mathrm{E}\_{1}\left(\bm{\xi}\right)\ +\ \bm{\xi}^{T}\nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ +\ C\_{2} |  | (51) |

as a function of 𝝃𝝃\bm{\xi}.

For a minimum not at the border, the derivative has to be the zero vector

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂EC​(𝝃,𝝃t)∂𝝃subscriptEC𝝃superscript𝝃𝑡𝝃\displaystyle\frac{\partial\mathrm{E}\_{\mathrm{C}}\left(\bm{\xi},\bm{\xi}^{t}\right)}{\partial\bm{\xi}}\ | =𝝃+∇ξE2​(𝝃t)=𝝃−𝑿​softmax​(β​𝑿T​𝝃t)= 0absent𝝃subscript∇𝜉subscriptE2superscript𝝃𝑡𝝃𝑿softmax𝛽superscript𝑿𝑇superscript𝝃𝑡 0\displaystyle=\ \bm{\xi}\ +\ \nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ =\ \bm{\xi}\ -\ \bm{X}\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi}^{t})\ =\ \bm{0} |  | (52) |

and the Hessian must be positive semi-definite

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂2EC​(𝝃,𝝃t)∂𝝃2superscript2subscriptEC𝝃superscript𝝃𝑡superscript𝝃2\displaystyle\frac{\partial^{2}\mathrm{E}\_{\mathrm{C}}\left(\bm{\xi},\bm{\xi}^{t}\right)}{\partial\bm{\xi}^{2}}\ | =𝑰.absent𝑰\displaystyle=\ \bm{I}\ . |  | (53) |

The Hessian is strict positive definite everywhere, therefore the optimization
problem is strict convex (if the domain is convex) and there exist only one minimum,
which is a global minimum.
ECsubscriptEC\mathrm{E}\_{\mathrm{C}} can even be written as a quadratic form:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EC​(𝝃,𝝃t)subscriptEC𝝃superscript𝝃𝑡\displaystyle\mathrm{E}\_{\mathrm{C}}\left(\bm{\xi},\bm{\xi}^{t}\right)\ | =12​(𝝃+∇ξE2​(𝝃t))T​(𝝃+∇ξE2​(𝝃t))+C3,absent12superscript𝝃subscript∇𝜉subscriptE2superscript𝝃𝑡𝑇𝝃subscript∇𝜉subscriptE2superscript𝝃𝑡subscript𝐶3\displaystyle=\ \frac{1}{2}\ \left(\bm{\xi}\ +\ \nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\right)^{T}\left(\bm{\xi}\ +\ \nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\right)\ +\ C\_{3}\ , |  | (54) |

where C3subscript𝐶3C\_{3} does not depend on 𝝃𝝃\bm{\xi}.

Therefore, the minimum is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃t+1superscript𝝃𝑡1\displaystyle\bm{\xi}^{t+1}\ | =−∇ξE2​(𝝃t)=𝑿​softmax​(β​𝑿T​𝝃t)absentsubscript∇𝜉subscriptE2superscript𝝃𝑡𝑿softmax𝛽superscript𝑿𝑇superscript𝝃𝑡\displaystyle=\ -\ \nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)\ =\ \bm{X}\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi}^{t}) |  | (55) |

if it is in the domain as we assume.

Using M=maxi⁡‖𝒙i‖𝑀subscript𝑖normsubscript𝒙𝑖M=\max\_{i}{{\left\|\bm{x}\_{i}\right\|}},
𝝃t+1superscript𝝃𝑡1\bm{\xi}^{t+1} is in the sphere S={𝒙∣‖𝒙‖⩽M}Sconditional-set𝒙norm𝒙𝑀\mathrm{S}=\{\bm{x}\mid{{\left\|\bm{x}\right\|}}\leqslant M\} which is a convex and compact set.
Hence, if 𝝃0∈Ssuperscript𝝃0S\bm{\xi}^{0}\in\mathrm{S}, then the iteration is a mapping from SS\mathrm{S} to SS\mathrm{S}.
Therefore, the point-set-map defined by the iteration Eq. ([55](#A1.E55 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
is uniformly compact on SS\mathrm{S} according to Remark 7 in Sriperumbudur & Lanckriet ([2009](#bib.bib86)).
Theorem 2 and Theorem 4 in (Sriperumbudur & Lanckriet, [2009](#bib.bib86)) states that
all the limit points of the iteration Eq. ([55](#A1.E55 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
are stationary points.
These theorems follow from
Zangwill’s global convergence theorem:
Convergence Theorem A, page 91 in Zangwill ([1969](#bib.bib116)) and page 3
in Wu ([1983](#bib.bib108)).

The global convergence theorem only assures that for the sequence
𝝃t+1=f​(𝝃t)superscript𝝃𝑡1𝑓superscript𝝃𝑡\bm{\xi}^{t+1}=f(\bm{\xi}^{t}) and
a function ΦΦ\Phi we have Φ​(𝝃t)→Φ​(𝝃∗)→Φsuperscript𝝃𝑡Φsuperscript𝝃\Phi(\bm{\xi}^{t})\to\Phi(\bm{\xi}^{\*}) for t→∞→𝑡t\to\infty
but not 𝝃t→𝝃∗→superscript𝝃𝑡superscript𝝃\bm{\xi}^{t}\to\bm{\xi}^{\*}.
However, if f𝑓f is strictly monotone with respect to ΦΦ\Phi, then
we can strengthen Zangwill’s global convergence theorem (Meyer, [1976](#bib.bib71)).
We set Φ=EΦE\Phi=\mathrm{E} and show E​(𝝃t+1)<E​(𝝃t)Esuperscript𝝃𝑡1Esuperscript𝝃𝑡\mathrm{E}(\bm{\xi}^{t+1})<\mathrm{E}(\bm{\xi}^{t}) if 𝝃tsuperscript𝝃𝑡\bm{\xi}^{t} is
not a stationary point of EE\mathrm{E}, that is, f𝑓f is strictly monotone with respect
to EE\mathrm{E}.
The following theorem is similar to the convergence results for the
expectation maximization (EM) algorithm in Wu ([1983](#bib.bib108)) which
are given in theorems 1 to 6 in Wu ([1983](#bib.bib108)).
The following theorem is also very similar to Theorem 8 in Sriperumbudur & Lanckriet ([2009](#bib.bib86)).

###### Theorem A2 (Global Convergence: Stationary Points).

For the iteration Eq. ([55](#A1.E55 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
we have E​(𝛏t)→E​(𝛏∗)=E∗→Esuperscript𝛏𝑡Esuperscript𝛏superscriptE\mathrm{E}\left(\bm{\xi}^{t}\right)\to\mathrm{E}\left(\bm{\xi}^{\*}\right)=\mathrm{E}^{\*}
as t→∞→𝑡t\to\infty, for some stationary point 𝛏∗superscript𝛏\bm{\xi}^{\*}.
Furthermore ‖𝛏t+1−𝛏t‖→0→normsuperscript𝛏𝑡1superscript𝛏𝑡0{{\left\|\bm{\xi}^{t+1}-\bm{\xi}^{t}\right\|}}\to 0 and
either {𝛏t}t=0∞superscriptsubscriptsuperscript𝛏𝑡𝑡0\{\bm{\xi}^{t}\}\_{t=0}^{\infty} converges
or, in the other case, the set of limit points of {𝛏t}t=0∞superscriptsubscriptsuperscript𝛏𝑡𝑡0\{\bm{\xi}^{t}\}\_{t=0}^{\infty}
is a connected and compact subset of ℒ​(E∗)ℒsuperscriptE\mathcal{L}\left(\mathrm{E}^{\*}\right), where
ℒ​(a)={𝛏∈ℒ∣E​(𝛏)=a}ℒ𝑎conditional-set𝛏ℒE𝛏𝑎\mathcal{L}\left(a\right)=\{\bm{\xi}\in\mathcal{L}\mid\mathrm{E}\left(\bm{\xi}\right)=a\}
and ℒℒ\mathcal{L} is the set of stationary points of the iteration Eq. ([55](#A1.E55 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
If ℒ​(E∗)ℒsuperscriptE\mathcal{L}\left(\mathrm{E}^{\*}\right) is finite, then any sequence {𝛏t}t=0∞superscriptsubscriptsuperscript𝛏𝑡𝑡0\{\bm{\xi}^{t}\}\_{t=0}^{\infty}
generated by the iteration Eq. ([55](#A1.E55 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
converges to some 𝛏∗∈ℒ​(E∗)superscript𝛏ℒsuperscriptE\bm{\xi}^{\*}\in\mathcal{L}\left(\mathrm{E}^{\*}\right).

###### Proof.

We have E​(𝝃t)=E1​(𝝃t)+E2​(𝝃t)Esuperscript𝝃𝑡subscriptE1superscript𝝃𝑡subscriptE2superscript𝝃𝑡\mathrm{E}\left(\bm{\xi}^{t}\right)=\mathrm{E}\_{1}\left(\bm{\xi}^{t}\right)+\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right).
The gradient
∇ξE2​(𝝃t)=−∇ξlse​(β,𝑿T​𝝃)subscript∇𝜉subscriptE2superscript𝝃𝑡subscript∇𝜉lse𝛽superscript𝑿𝑇𝝃\nabla\_{\xi}\mathrm{E}\_{2}\left(\bm{\xi}^{t}\right)=-\nabla\_{\xi}\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi}) is
continuous. Therefore, Eq. ([51](#A1.E51 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) has minimum in the sphere SS\mathrm{S}, which is
a convex and compact set.
If 𝝃t+1≠𝝃tsuperscript𝝃𝑡1superscript𝝃𝑡\bm{\xi}^{t+1}\not=\bm{\xi}^{t},
then 𝝃tsuperscript𝝃𝑡\bm{\xi}^{t} was not the minimum of
Eq. ([48](#A1.E48 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) as the derivative at 𝝃tsuperscript𝝃𝑡\bm{\xi}^{t} is not equal to zero.
Eq. ([53](#A1.E53 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) shows that the optimization problem Eq. ([48](#A1.E48 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
is strict convex, hence it has only one minimum, which is a global minimum.
Eq. ([54](#A1.E54 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) shows that the optimization problem Eq. ([48](#A1.E48 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
is even a quadratic form.
Therefore, we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E​(𝝃t+1)Esuperscript𝝃𝑡1\displaystyle\mathrm{E}\left(\bm{\xi}^{t+1}\right)\ | ⩽EC​(𝝃t+1,𝝃t)<EC​(𝝃t,𝝃t)=E​(𝝃t).absentsubscriptECsuperscript𝝃𝑡1superscript𝝃𝑡subscriptECsuperscript𝝃𝑡superscript𝝃𝑡Esuperscript𝝃𝑡\displaystyle\leqslant\ \mathrm{E}\_{\mathrm{C}}\left(\bm{\xi}^{t+1},\bm{\xi}^{t}\right)\ <\ \mathrm{E}\_{\mathrm{C}}\left(\bm{\xi}^{t},\bm{\xi}^{t}\right)\ =\ \mathrm{E}\left(\bm{\xi}^{t}\right)\ . |  | (56) |

Therefore, the point-set-map defined by the iteration Eq. ([55](#A1.E55 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
(for definitions see (Sriperumbudur & Lanckriet, [2009](#bib.bib86)))
is strictly monotonic with respect to EE\mathrm{E}.
Therefore, we can apply Theorem 3 in Sriperumbudur & Lanckriet ([2009](#bib.bib86)) or
Theorem 3.1 and Corollary 3.2 in Meyer ([1976](#bib.bib71)),
which give the statements of the theorem.

∎

We showed global convergence of the iteration Eq. ([31](#A1.E31 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
We have shown that all the limit points of any sequence generated by
the iteration Eq. ([31](#A1.E31 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) are the
stationary points (critical points; local minima or saddle points) of the
energy function EE\mathrm{E}.
Local maxima as stationary points are only possible
if the iterations exactly hits a local maximum.
However, convergence to a local maximum without being there
is not possible because
Eq. ([56](#A1.E56 "In Proof. ‣ A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) ensures a strict decrease of the
energy EE\mathrm{E}. Therefore, almost sure local maxima are not
obtained as stationary points.
Either the iteration converges or,
in the second case, the set of limit points
is a connected and compact set.
But what happens if 𝝃0superscript𝝃0\bm{\xi}^{0} is in an ϵitalic-ϵ\epsilon-neighborhood
around a local minimum 𝝃∗superscript𝝃\bm{\xi}^{\*}?
Will the iteration Eq. ([31](#A1.E31 "In A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) converge to 𝝃∗superscript𝝃\bm{\xi}^{\*}?
What is the rate of convergence?
These questions are about
local convergence which will be treated in detail in next section.

#### A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration

For the proof of local convergence to a fixed point
we will apply Banach fixed point theorem.
For the rate of convergence we will rely on properties of a contraction mapping.

##### A.1.5.1 General Bound on the Jacobian of the Iteration.

We consider the iteration

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =f​(𝝃)=𝑿​𝒑=𝑿​softmax​(β​𝑿T​𝝃)absent𝑓𝝃𝑿𝒑𝑿softmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ f(\bm{\xi})\ =\ \bm{X}\bm{p}\ =\ \bm{X}\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi}) |  | (57) |

using

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝑿T​𝝃).absentsoftmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ \mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ . |  | (58) |

The Jacobian JJ\mathrm{J} is symmetric and has the following form:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | JJ\displaystyle\mathrm{J}\ | =∂f​(𝝃)∂𝝃=β​𝑿​(diag​(𝒑)−𝒑​𝒑T)​𝑿T=𝑿​Js​𝑿T,absent𝑓𝝃𝝃𝛽𝑿diag𝒑𝒑superscript𝒑𝑇superscript𝑿𝑇𝑿subscriptJ𝑠superscript𝑿𝑇\displaystyle=\ \frac{\partial f(\bm{\xi})}{\partial\bm{\xi}}\ =\ \beta\ \bm{X}\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\bm{X}^{T}\ =\ \bm{X}\mathrm{J}\_{s}\bm{X}^{T}\ , |  | (59) |

where JssubscriptJ𝑠\mathrm{J}\_{s} is Jacobian of the softmax.

To analyze the local convergence of the iteration,
we distinguish between the following three cases (see also Fig. [A.1](#A1.F1 "Figure A.1 ‣ A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
Here we only provide an informal discussion to give the reader some
intuition.
A rigorous formulation of the results can be found in the corresponding subsections.

1. a)

   If the patterns 𝒙isubscript𝒙𝑖\bm{x}\_{i} are not well separated,
   the iteration goes to a fixed point close to
   the arithmetic mean of the vectors.
   In this case 𝒑𝒑\bm{p} is close to pi=1/Nsubscript𝑝𝑖1𝑁p\_{i}=1/N.
2. b)

   If the patterns 𝒙isubscript𝒙𝑖\bm{x}\_{i} are well separated,
   then the iteration goes to the pattern to which the initial
   𝝃𝝃\bm{\xi} is similar.
   If the initial 𝝃𝝃\bm{\xi} is similar to a vector 𝒙isubscript𝒙𝑖\bm{x}\_{i} then it will
   converge to a vector close to 𝒙isubscript𝒙𝑖\bm{x}\_{i} and 𝒑𝒑\bm{p} will converge to a vector
   close to 𝒆isubscript𝒆𝑖\bm{e}\_{i}.
3. c)

   If some vectors are similar to each other but well separated from all
   other vectors, then a so called metastable state between the similar
   vectors exists.
   Iterations that start near the metastable state converge
   to this metastable state.

!(/html/2008.02217/assets/x6.png)

Figure A.1: The three cases of fixed points.
a) Stored patterns (fixed point is single pattern):
patterns are stored if they are well separated.
Each pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} has a single fixed point 𝒙i∗superscriptsubscript𝒙𝑖\bm{x}\_{i}^{\*} close to it.
In the sphere SisubscriptS𝑖\mathrm{S}\_{i}, pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} is the only pattern and 𝒙i∗superscriptsubscript𝒙𝑖\bm{x}\_{i}^{\*} the only fixed point.
b) Metastable state (fixed point is average of similar patterns):
𝒙isubscript𝒙𝑖\bm{x}\_{i} and 𝒙jsubscript𝒙𝑗\bm{x}\_{j} are similar to each other and not
well separated. The fixed point 𝒎𝒙∗superscriptsubscript𝒎𝒙\bm{m}\_{\bm{x}}^{\*} is a metastable state
that is close to the mean 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} of the similar patterns.
c) Global fixed point (fixed point is average of all patterns):
no pattern is well separated from the others.
A single global fixed point 𝒎𝒙∗superscriptsubscript𝒎𝒙\bm{m}\_{\bm{x}}^{\*} exists that is close to
the arithmetic mean 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} of all patterns.

We begin with a bound on the Jacobian of the iteration, thereby
heavily relying on
the Jacobian of the softmax from Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").

###### Lemma A2.

For N𝑁N patterns 𝐗=(𝐱1,…,𝐱N)𝐗subscript𝐱1…subscript𝐱𝑁\bm{X}=(\bm{x}\_{1},\ldots,\bm{x}\_{N}),
𝐩=softmax​(β​𝐗T​𝛏)𝐩softmax𝛽superscript𝐗𝑇𝛏\bm{p}=\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi}), M=maxi⁡‖𝐱i‖𝑀subscript𝑖normsubscript𝐱𝑖M=\max\_{i}{{\left\|\bm{x}\_{i}\right\|}}, and
m=maxi⁡pi​(1−pi)𝑚subscript𝑖subscript𝑝𝑖1subscript𝑝𝑖m=\max\_{i}p\_{i}(1-p\_{i}), the spectral norm of the Jacobian
JJ\mathrm{J} of the fixed point iteration
is bounded:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽ 2​β​‖𝑿‖22​m⩽ 2​β​N​M2​m.absent2𝛽superscriptsubscriptnorm𝑿22𝑚2𝛽𝑁superscript𝑀2𝑚\displaystyle\leqslant\ 2\ \beta\ {{\left\|\bm{X}\right\|}}\_{2}^{2}\ m\ \leqslant\ 2\ \beta\ N\ M^{2}\ m\ . |  | (60) |

If pmax=maxi⁡pi≥1−ϵsubscript𝑝subscript𝑖subscript𝑝𝑖1italic-ϵp\_{\max}=\max\_{i}p\_{i}\geq 1-\epsilon, then for the spectral norm of
the Jacobian holds

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽ 2​β​N​M2​ϵ− 2​ϵ2​β​N​M2< 2​β​N​M2​ϵ.absent2𝛽𝑁superscript𝑀2italic-ϵ2superscriptitalic-ϵ2𝛽𝑁superscript𝑀22𝛽𝑁superscript𝑀2italic-ϵ\displaystyle\leqslant\ 2\ \beta\ N\ M^{2}\ \epsilon\ -\ 2\ \epsilon^{2}\ \beta\ N\ M^{2}\ <\ 2\ \beta\ N\ M^{2}\ \epsilon\ . |  | (61) |

###### Proof.

With

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝑿T​𝝃),absentsoftmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ \mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ , |  | (62) |

the symmetric Jacobian JJ\mathrm{J} is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | JJ\displaystyle\mathrm{J}\ | =∂f​(𝝃)∂𝝃=β​𝑿​(diag​(𝒑)−𝒑​𝒑T)​𝑿T=𝑿​Js​𝑿T,absent𝑓𝝃𝝃𝛽𝑿diag𝒑𝒑superscript𝒑𝑇superscript𝑿𝑇𝑿subscriptJ𝑠superscript𝑿𝑇\displaystyle=\ \frac{\partial f(\bm{\xi})}{\partial\bm{\xi}}\ =\ \beta\ \bm{X}\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\bm{X}^{T}\ =\ \bm{X}\mathrm{J}\_{s}\bm{X}^{T}\ , |  | (63) |

where JssubscriptJ𝑠\mathrm{J}\_{s} is Jacobian of the softmax.

With m=maxi⁡pi​(1−pi)𝑚subscript𝑖subscript𝑝𝑖1subscript𝑝𝑖m=\max\_{i}p\_{i}(1-p\_{i}),
Eq. ([476](#A1.E476 "In Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) from
Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js‖2subscriptnormsubscriptJ𝑠2\displaystyle{{\left\|\mathrm{J}\_{s}\right\|}}\_{2}\ | =β​‖diag​(𝒑)−𝒑​𝒑T‖2⩽ 2​m​β.absent𝛽subscriptnormdiag𝒑𝒑superscript𝒑𝑇22𝑚𝛽\displaystyle=\ \beta\ {{\left\|\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right\|}}\_{2}\ \leqslant\ 2\ m\ \beta\ . |  | (64) |

Using this bound on ‖Js‖2subscriptnormsubscriptJ𝑠2{{\left\|\mathrm{J}\_{s}\right\|}}\_{2}, we obtain

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽β​‖𝑿T‖2​‖Js‖2​‖𝑿‖2⩽ 2​m​β​‖𝑿‖22.absent𝛽subscriptnormsuperscript𝑿𝑇2subscriptnormsubscriptJ𝑠2subscriptnorm𝑿22𝑚𝛽superscriptsubscriptnorm𝑿22\displaystyle\leqslant\ \beta\ {{\left\|\bm{X}^{T}\right\|}}\_{2}\ {{\left\|\mathrm{J}\_{s}\right\|}}\_{2}\ {{\left\|\bm{X}\right\|}}\_{2}\ \leqslant\ 2\ m\ \beta\ {{\left\|\bm{X}\right\|}}\_{2}^{2}\ . |  | (65) |

The spectral norm ∥.∥2{{\left\|.\right\|}}\_{2} is bounded by the Frobenius norm
∥.∥F{{\left\|.\right\|}}\_{F} which can be expressed by the norm squared of its
column vectors:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝑿‖2subscriptnorm𝑿2\displaystyle{{\left\|\bm{X}\right\|}}\_{2}\ | ⩽‖𝑿‖F=∑i‖𝒙i‖2.absentsubscriptnorm𝑿𝐹subscript𝑖superscriptnormsubscript𝒙𝑖2\displaystyle\leqslant\ {{\left\|\bm{X}\right\|}}\_{F}\ =\ \sqrt{\sum\_{i}{{\left\|\bm{x}\_{i}\right\|}}^{2}}\ . |  | (66) |

Therefore, we obtain the first statement of the lemma:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽ 2​β​‖𝑿‖22​m⩽ 2​β​N​M2​m.absent2𝛽superscriptsubscriptnorm𝑿22𝑚2𝛽𝑁superscript𝑀2𝑚\displaystyle\leqslant\ 2\ \beta\ {{\left\|\bm{X}\right\|}}\_{2}^{2}\ m\ \leqslant\ 2\ \beta\ N\ M^{2}\ m\ . |  | (67) |

With pmax=maxi⁡pi≥1−ϵsubscript𝑝subscript𝑖subscript𝑝𝑖1italic-ϵp\_{\max}=\max\_{i}p\_{i}\geq 1-\epsilon
Eq. ([480](#A1.E480 "In Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
in Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js‖2subscriptnormsubscriptJ𝑠2\displaystyle{{\left\|\mathrm{J}\_{s}\right\|}}\_{2}\ | ⩽ 2​β​ϵ− 2​ϵ2​β< 2​β​ϵ.absent2𝛽italic-ϵ2superscriptitalic-ϵ2𝛽2𝛽italic-ϵ\displaystyle\leqslant\ 2\ \beta\ \epsilon\ -\ 2\ \epsilon^{2}\ \beta\ <\ 2\ \beta\ \epsilon\ . |  | (68) |

Using this inequality,
we obtain the second statement of the lemma:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽ 2​β​N​M2​ϵ− 2​ϵ2​β​N​M2< 2​β​N​M2​ϵ.absent2𝛽𝑁superscript𝑀2italic-ϵ2superscriptitalic-ϵ2𝛽𝑁superscript𝑀22𝛽𝑁superscript𝑀2italic-ϵ\displaystyle\leqslant\ 2\ \beta\ N\ M^{2}\ \epsilon\ -\ 2\ \epsilon^{2}\ \beta\ N\ M^{2}\ <\ 2\ \beta\ N\ M^{2}\ \epsilon\ . |  | (69) |

∎

We now define the “separation” ΔisubscriptΔ𝑖\Delta\_{i} of a pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} from
data 𝑿=(𝒙1,…,𝒙N)𝑿subscript𝒙1…subscript𝒙𝑁\bm{X}=(\bm{x}\_{1},\ldots,\bm{x}\_{N}) here, since it has an important role
for the convergence properties of the iteration.

###### Definition 2 (Separation of Patterns).

We define ΔisubscriptΔ𝑖\Delta\_{i}, i.e. the separation of pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} from data 𝑿=(𝒙1,…,𝒙N)𝑿subscript𝒙1…subscript𝒙𝑁\bm{X}=(\bm{x}\_{1},\ldots,\bm{x}\_{N}) as:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | =minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)=𝒙iT​𝒙i−maxj,j≠i⁡𝒙iT​𝒙j.absentsubscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖subscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right)\ =\ \bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \max\_{j,j\not=i}\bm{x}\_{i}^{T}\bm{x}\_{j}\ . |  | (70) |

The pattern is separated from the other data if 0<Δi0subscriptΔ𝑖0<\Delta\_{i}.
Using the parallelogram identity, ΔisubscriptΔ𝑖\Delta\_{i} can also be expressed as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | =minj,j≠i⁡12​(‖𝒙i‖2−‖𝒙j‖2+‖𝒙i−𝒙j‖2)absentsubscript  𝑗𝑗 𝑖12superscriptnormsubscript𝒙𝑖2superscriptnormsubscript𝒙𝑗2superscriptnormsubscript𝒙𝑖subscript𝒙𝑗2\displaystyle=\ \min\_{j,j\not=i}\frac{1}{2}\ \left({{\left\|\bm{x}\_{i}\right\|}}^{2}\ -\ {{\left\|\bm{x}\_{j}\right\|}}^{2}\ +\ {{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}^{2}\right) |  | (71) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =12​‖𝒙i‖2−12​maxj,j≠i⁡(‖𝒙j‖2−‖𝒙i−𝒙j‖2).absent12superscriptnormsubscript𝒙𝑖212subscript  𝑗𝑗 𝑖superscriptnormsubscript𝒙𝑗2superscriptnormsubscript𝒙𝑖subscript𝒙𝑗2\displaystyle=\ \frac{1}{2}{{\left\|\bm{x}\_{i}\right\|}}^{2}\ -\ \frac{1}{2}\ \max\_{j,j\not=i}\left({{\left\|\bm{x}\_{j}\right\|}}^{2}\ -\ {{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}^{2}\right)\ . |  |

For ‖𝐱i‖=‖𝐱j‖normsubscript𝐱𝑖normsubscript𝐱𝑗{{\left\|\bm{x}\_{i}\right\|}}={{\left\|\bm{x}\_{j}\right\|}} we have Δi=1/2​minj,j≠i⁡‖𝐱i−𝐱j‖2subscriptΔ𝑖12subscript

𝑗𝑗
𝑖superscriptnormsubscript𝐱𝑖subscript𝐱𝑗2\Delta\_{i}=1/2\min\_{j,j\not=i}{{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}^{2}.

Analog we say for a query 𝛏𝛏\bm{\xi} and data 𝐗=(𝐱1,…,𝐱N)𝐗subscript𝐱1…subscript𝐱𝑁\bm{X}=(\bm{x}\_{1},\ldots,\bm{x}\_{N}),
that 𝐱isubscript𝐱𝑖\bm{x}\_{i} is least separated from 𝛏𝛏\bm{\xi} while
being separated from other 𝐱jsubscript𝐱𝑗\bm{x}\_{j} with j≠i𝑗𝑖j\not=i if

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | i𝑖\displaystyle i\ | =arg⁡maxk⁡minj,j≠k⁡(𝝃T​𝒙k−𝝃T​𝒙j)=arg⁡maxk⁡(𝝃T​𝒙k−maxj,j≠k⁡𝝃T​𝒙j)absentsubscript𝑘subscript  𝑗𝑗 𝑘superscript𝝃𝑇subscript𝒙𝑘superscript𝝃𝑇subscript𝒙𝑗subscript𝑘superscript𝝃𝑇subscript𝒙𝑘subscript  𝑗𝑗 𝑘superscript𝝃𝑇subscript𝒙𝑗\displaystyle=\ \arg\max\_{k}\min\_{j,j\not=k}\left(\bm{\xi}^{T}\bm{x}\_{k}\ -\ \bm{\xi}^{T}\bm{x}\_{j}\right)\ =\ \arg\max\_{k}\left(\bm{\xi}^{T}\bm{x}\_{k}\ -\ \max\_{j,j\not=k}\bm{\xi}^{T}\bm{x}\_{j}\right) |  | (72) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 00\displaystyle 0\ | ⩽c=maxk⁡minj,j≠k⁡(𝝃T​𝒙k−𝝃T​𝒙j)=maxk⁡(𝝃T​𝒙k−maxj,j≠k⁡𝝃T​𝒙j).absent𝑐subscript𝑘subscript  𝑗𝑗 𝑘superscript𝝃𝑇subscript𝒙𝑘superscript𝝃𝑇subscript𝒙𝑗subscript𝑘superscript𝝃𝑇subscript𝒙𝑘subscript  𝑗𝑗 𝑘superscript𝝃𝑇subscript𝒙𝑗\displaystyle\leqslant\ c\ =\ \max\_{k}\min\_{j,j\not=k}\left(\bm{\xi}^{T}\bm{x}\_{k}\ -\ \bm{\xi}^{T}\bm{x}\_{j}\right)\ =\ \max\_{k}\left(\bm{\xi}^{T}\bm{x}\_{k}\ -\ \max\_{j,j\not=k}\bm{\xi}^{T}\bm{x}\_{j}\right)\ . |  | (73) |

Next we consider the case where the iteration has only
one stable fixed point.

##### A.1.5.2 One Stable State: Fixed Point Near the Mean of the Patterns.

We start with the case where
no pattern is well separated from the others.

•Global fixed point near the global mean: Analysis using the data center.

We revisit the bound on the Jacobian of the iteration by
utilizing properties of pattern distributions.
We begin with a probabilistic interpretation where
we consider pisubscript𝑝𝑖p\_{i} as the probability of selecting
the vector 𝒙isubscript𝒙𝑖\bm{x}\_{i}.
Consequently, we define
expectations as E𝒑​[f​(𝒙)]=∑i=1Npi​f​(𝒙i)subscriptE𝒑delimited-[]𝑓𝒙superscriptsubscript𝑖1𝑁subscript𝑝𝑖𝑓subscript𝒙𝑖\mathbf{\mathrm{E}}\_{\bm{p}}[f(\bm{x})]=\sum\_{i=1}^{N}p\_{i}f(\bm{x}\_{i}).
In this setting the matrix

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑿​(diag​(𝒑)−𝒑​𝒑T)​𝑿T𝑿diag𝒑𝒑superscript𝒑𝑇superscript𝑿𝑇\displaystyle\bm{X}\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\bm{X}^{T} |  | (74) |

is the covariance matrix of data 𝑿𝑿\bm{X} when its
vectors are selected according
to the probability 𝒑𝒑\bm{p}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑿​(diag​(𝒑)−𝒑​𝒑T)​𝑿T=𝑿​diag​(𝒑)​𝑿T−𝑿​𝒑​𝒑T​𝑿T𝑿diag𝒑𝒑superscript𝒑𝑇superscript𝑿𝑇𝑿diag𝒑superscript𝑿𝑇𝑿𝒑superscript𝒑𝑇superscript𝑿𝑇\displaystyle\bm{X}\left(\mathrm{diag}(\bm{p})\ -\ \bm{p}\bm{p}^{T}\right)\bm{X}^{T}\ =\ \bm{X}\mathrm{diag}(\bm{p})\bm{X}^{T}\ -\ \bm{X}\bm{p}\bm{p}^{T}\bm{X}^{T} |  | (75) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =∑i=1Npi​𝒙i​𝒙iT−(∑i=1Npi​𝒙i)​(∑i=1Npi​𝒙i)Tabsentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i}\right)\left(\sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i}\right)^{T} |  | (76) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =E𝒑​[𝒙​𝒙T]−E𝒑​[𝒙]​E𝒑​[𝒙]T=Var𝒑​[𝒙],absentsubscriptE𝒑delimited-[]𝒙superscript𝒙𝑇subscriptE𝒑delimited-[]𝒙subscriptE𝒑superscriptdelimited-[]𝒙𝑇subscriptVar𝒑delimited-[]𝒙\displaystyle=\ \mathbf{\mathrm{E}}\_{\bm{p}}[\bm{x}\ \bm{x}^{T}]\ -\ \mathbf{\mathrm{E}}\_{\bm{p}}[\bm{x}]\ \mathbf{\mathrm{E}}\_{\bm{p}}[\bm{x}]^{T}\ =\ \mathbf{\mathrm{Var}}\_{\bm{p}}[\bm{x}]\ , |  | (77) |

therefore we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | JJ\displaystyle\mathrm{J}\ | =β​Var𝒑​[𝒙].absent𝛽subscriptVar𝒑delimited-[]𝒙\displaystyle=\ \beta\ \mathbf{\mathrm{Var}}\_{\bm{p}}[\bm{x}]\ . |  | (78) |

The largest eigenvalue of the covariance matrix
(equal to the largest singular value)
is the variance in the direction of the eigenvector
associated with the largest eigenvalue.

We define:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒎𝒙subscript𝒎𝒙\displaystyle\bm{m}\_{\bm{x}}\ | =1N​∑i=1N𝒙i,absent1𝑁superscriptsubscript𝑖1𝑁subscript𝒙𝑖\displaystyle=\ \frac{1}{N}\ \sum\_{i=1}^{N}\ \bm{x}\_{i}\ , |  | (79) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | mmaxsubscript𝑚\displaystyle m\_{\max}\ | =max1⩽i⩽N⁡‖𝒙i−𝒎𝒙‖2.absentsubscript1𝑖𝑁subscriptnormsubscript𝒙𝑖subscript𝒎𝒙2\displaystyle=\ \max\_{1\leqslant i\leqslant N}{{\left\|\bm{x}\_{i}\ -\ \bm{m}\_{\bm{x}}\right\|}}\_{2}\ . |  | (80) |

𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} is the arithmetic mean (the center) of the patterns.
mmaxsubscript𝑚m\_{\max} is the maximal distance of the patterns to the center 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} .

The variance of the patterns is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Var𝒑​[𝒙]subscriptVar𝒑delimited-[]𝒙\displaystyle\mathbf{\mathrm{Var}}\_{\bm{p}}[\bm{x}]\ | =∑i=1Npi​𝒙i​𝒙iT−(∑i=1Npi​𝒙i)​(∑i=1Npi​𝒙i)Tabsentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i}\right)^{T} |  | (81) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1Npi​(𝒙i−∑i=1Npi​𝒙i)​(𝒙i−∑i=1Npi​𝒙i)T.absentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖superscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{N}p\_{i}\ \left(\bm{x}\_{i}\ -\ \sum\_{i=1}^{N}p\_{i}\bm{x}\_{i}\right)\ \left(\bm{x}\_{i}\ -\ \sum\_{i=1}^{N}p\_{i}\bm{x}\_{i}\right)^{T}\ . |  |

The maximal distance to the center mmaxsubscript𝑚m\_{\max} allows the derivation of a
bound on the norm of the Jacobian.

Next lemma gives a condition for a global fixed point.

###### Lemma A3.

The following bound on the norm ‖J‖2subscriptnormJ2{{\left\|\mathrm{J}\right\|}}\_{2} of the
Jacobian of the fixed point iteration f𝑓f holds independent of 𝐩𝐩\bm{p} or
the query 𝛏𝛏\bm{\xi}.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽β​mmax2.absent𝛽superscriptsubscript𝑚2\displaystyle\leqslant\ \beta\ m\_{\max}^{2}\ . |  | (82) |

For β​mmax2<1𝛽superscriptsubscript𝑚21\beta\ m\_{\max}^{2}<1 there exists
a unique fixed point (global fixed point) of iteration f𝑓f in each compact set.

###### Proof.

In order to bound the variance
we compute the vector 𝒂𝒂\bm{a} that minimizes

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(𝒂)𝑓𝒂\displaystyle f(\bm{a})\ | =∑i=1Npi​‖𝒙i−𝒂‖2=∑i=1Npi​(𝒙i−𝒂)T​(𝒙i−𝒂).absentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖superscriptnormsubscript𝒙𝑖𝒂2superscriptsubscript𝑖1𝑁subscript𝑝𝑖superscriptsubscript𝒙𝑖𝒂𝑇subscript𝒙𝑖𝒂\displaystyle=\ \sum\_{i=1}^{N}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bm{a}\right\|}}^{2}\ =\ \sum\_{i=1}^{N}p\_{i}(\bm{x}\_{i}\ -\ \bm{a})^{T}(\bm{x}\_{i}\ -\ \bm{a})\ . |  | (83) |

The solution to

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂f​(𝒂)∂𝒂𝑓𝒂𝒂\displaystyle\frac{\partial f(\bm{a})}{\partial\bm{a}}\ | = 2​∑i=1Npi​(𝒂−𝒙i)= 0absent2superscriptsubscript𝑖1𝑁subscript𝑝𝑖𝒂subscript𝒙𝑖 0\displaystyle=\ 2\ \sum\_{i=1}^{N}p\_{i}(\bm{a}\ -\ \bm{x}\_{i})\ =\ 0 |  | (84) |

is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒂𝒂\displaystyle\bm{a}\ | =∑i=1Npi​𝒙i.absentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖\displaystyle=\ \sum\_{i=1}^{N}p\_{i}\bm{x}\_{i}\ . |  | (85) |

The Hessian of f𝑓f is positive definite since

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂2f​(𝒂)∂𝒂2superscript2𝑓𝒂superscript𝒂2\displaystyle\frac{\partial^{2}f(\bm{a})}{\partial\bm{a}^{2}}\ | = 2​∑i=1Npi​𝑰= 2​𝑰absent2superscriptsubscript𝑖1𝑁subscript𝑝𝑖𝑰2𝑰\displaystyle=\ 2\ \sum\_{i=1}^{N}p\_{i}\ \bm{I}\ =\ 2\ \bm{I} |  | (86) |

and f𝑓f is a convex function.
Hence, the mean

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒙¯¯𝒙\displaystyle\bar{\bm{x}}\ | :=∑i=1Npi​𝒙iassignabsentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖\displaystyle:=\ \sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i} |  | (87) |

minimizes
∑i=1Npi​‖𝒙i−𝒂‖2superscriptsubscript𝑖1𝑁subscript𝑝𝑖superscriptnormsubscript𝒙𝑖𝒂2\sum\_{i=1}^{N}p\_{i}{{\left\|\bm{x}\_{i}-\bm{a}\right\|}}^{2}.
Therefore, we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i=1Npi​‖𝒙i−𝒙¯‖2superscriptsubscript𝑖1𝑁subscript𝑝𝑖superscriptnormsubscript𝒙𝑖¯𝒙2\displaystyle\sum\_{i=1}^{N}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bar{\bm{x}}\right\|}}^{2}\ | ⩽∑i=1Npi​‖𝒙i−𝒎𝒙‖2⩽mmax2.absentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖superscriptnormsubscript𝒙𝑖subscript𝒎𝒙2superscriptsubscript𝑚2\displaystyle\leqslant\ \sum\_{i=1}^{N}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bm{m}\_{\bm{x}}\right\|}}^{2}\ \leqslant\ m\_{\max}^{2}\ . |  | (88) |

Let us quickly recall that the spectral norm of an outer product of two vectors
is the product of the Euclidean norms of the vectors:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒂​𝒃T‖2subscriptnorm𝒂superscript𝒃𝑇2\displaystyle{{\left\|\bm{a}\bm{b}^{T}\right\|}}\_{2}\ | =λmax​(𝒃​𝒂T​𝒂​𝒃T)=‖𝒂‖​λmax​(𝒃​𝒃T)=‖𝒂‖​‖𝒃‖,absentsubscript𝜆𝒃superscript𝒂𝑇𝒂superscript𝒃𝑇norm𝒂subscript𝜆𝒃superscript𝒃𝑇norm𝒂norm𝒃\displaystyle=\ \sqrt{\lambda\_{\max}(\bm{b}\bm{a}^{T}\bm{a}\bm{b}^{T})}\ =\ {{\left\|\bm{a}\right\|}}\ \sqrt{\lambda\_{\max}(\bm{b}\bm{b}^{T})}\ =\ {{\left\|\bm{a}\right\|}}\ {{\left\|\bm{b}\right\|}}\ , |  | (89) |

since 𝒃​𝒃T𝒃superscript𝒃𝑇\bm{b}\bm{b}^{T} has eigenvector 𝒃/‖𝒃‖𝒃norm𝒃\bm{b}/{{\left\|\bm{b}\right\|}} with
eigenvalue ‖𝒃‖2superscriptnorm𝒃2{{\left\|\bm{b}\right\|}}^{2} and otherwise zero eigenvalues.

We now bound the variance of the patterns:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Var𝒑​[𝒙]‖2subscriptnormsubscriptVar𝒑delimited-[]𝒙2\displaystyle{{\left\|\mathbf{\mathrm{Var}}\_{\bm{p}}[\bm{x}]\right\|}}\_{2}\ | ⩽∑i=1Npi​‖(𝒙i−𝒙¯)​(𝒙i−𝒙¯)T‖2absentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscriptnormsubscript𝒙𝑖¯𝒙superscriptsubscript𝒙𝑖¯𝒙𝑇2\displaystyle\leqslant\ \sum\_{i=1}^{N}p\_{i}{{\left\|\left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)\ \left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)^{T}\right\|}}\_{2} |  | (90) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1Npi∥𝒙i−𝒙¯∥2⩽∑i=1Npi∥𝒙i−𝒎𝒙∥2⩽mmax2.\displaystyle=\ \sum\_{i=1}^{N}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bar{\bm{x}}\right\|}}^{2}\ \leqslant\ \sum\_{i=1}^{N}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bm{m}\_{\bm{x}}\right\|}}^{2}\ \leqslant\ \ m\_{\max}^{2}\ . |  |

The bound of the lemma on ‖J‖2subscriptnormJ2{{\left\|\mathrm{J}\right\|}}\_{2}
follows from Eq. ([78](#A1.E78 "In A.1.5.2 One Stable State: Fixed Point Near the Mean of the Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).

For ‖J‖2⩽β​mmax2<1subscriptnormJ2𝛽superscriptsubscript𝑚21{{\left\|\mathrm{J}\right\|}}\_{2}\leqslant\beta\ m\_{\max}^{2}<1 we have a
contraction mapping on each compact set. Banach fixed point theorem
says there is a unique fixed point in the compact set.

∎

Now let us further investigate the tightness of the bound on ‖Var𝒑​[𝒙]‖2subscriptnormsubscriptVar𝒑delimited-[]𝒙2{{\left\|\mathbf{\mathrm{Var}}\_{\bm{p}}[\bm{x}]\right\|}}\_{2} via
‖𝒙i−𝒙¯‖2superscriptnormsubscript𝒙𝑖¯𝒙2{{\left\|\bm{x}\_{i}-\bar{\bm{x}}\right\|}}^{2}: we
consider the trace, which
is the sum ∑k=1deksuperscriptsubscript𝑘1𝑑subscript𝑒𝑘\sum\_{k=1}^{d}e\_{k} of the w.l.o.g. ordered nonnegative eigenvalues eksubscript𝑒𝑘e\_{k} of Var𝒑​[𝒙]subscriptVar𝒑delimited-[]𝒙\mathbf{\mathrm{Var}}\_{\bm{p}}[\bm{x}]
The spectral norm is equal to the
largest eigenvalue e1subscript𝑒1e\_{1}, which is equal to the largest singular
value, as we have positive semidefinite matrices.
We obtain:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Var𝒑​[𝒙]‖2subscriptnormsubscriptVar𝒑delimited-[]𝒙2\displaystyle{{\left\|\mathbf{\mathrm{Var}}\_{\bm{p}}[\bm{x}]\right\|}}\_{2}\ | =Tr​(∑i=1Npi​(𝒙i−𝒙¯)​(𝒙i−𝒙¯)T)−∑k=2dekabsentTrsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖¯𝒙superscriptsubscript𝒙𝑖¯𝒙𝑇superscriptsubscript𝑘2𝑑subscript𝑒𝑘\displaystyle=\ \mathbf{\mathrm{Tr}}\left(\sum\_{i=1}^{N}p\_{i}\left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)\ \left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)^{T}\right)\ -\ \sum\_{k=2}^{d}e\_{k} |  | (91) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1Npi​Tr​((𝒙i−𝒙¯)​(𝒙i−𝒙¯)T)−∑k=2dekabsentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖Trsubscript𝒙𝑖¯𝒙superscriptsubscript𝒙𝑖¯𝒙𝑇superscriptsubscript𝑘2𝑑subscript𝑒𝑘\displaystyle=\ \sum\_{i=1}^{N}p\_{i}\mathbf{\mathrm{Tr}}\left(\left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)\ \left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)^{T}\right)\ -\ \sum\_{k=2}^{d}e\_{k} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1Npi​‖𝒙i−𝒙¯‖2−∑k=2dek.absentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖superscriptnormsubscript𝒙𝑖¯𝒙2superscriptsubscript𝑘2𝑑subscript𝑒𝑘\displaystyle=\ \sum\_{i=1}^{N}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bar{\bm{x}}\right\|}}^{2}\ -\ \sum\_{k=2}^{d}e\_{k}\ . |  |

Therefore, the tightness of the bound depends on eigenvalues
which are not the largest. Hence variations which are not
along the largest variation weaken the bound.

Next we investigate the location of
fixed points which existence is ensured by the global convergence
stated in Theorem [A2](#ThmtheoremA2 "Theorem A2 (Global Convergence: Stationary Points). ‣ A.1.4 Global Convergence of the Update Rule ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
For N𝑁N patterns 𝑿=(𝒙1,…,𝒙N)𝑿subscript𝒙1…subscript𝒙𝑁\bm{X}=(\bm{x}\_{1},\ldots,\bm{x}\_{N}),
we consider the iteration

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =f​(𝝃)=𝑿​𝒑=𝑿​softmax​(β​𝑿T​𝝃)absent𝑓𝝃𝑿𝒑𝑿softmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ f(\bm{\xi})\ =\ \bm{X}\bm{p}\ =\ \bm{X}\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi}) |  | (92) |

using

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝑿T​𝝃).absentsoftmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ \mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ . |  | (93) |

𝝃newsuperscript𝝃new\bm{\xi}^{\mathrm{new}} is in the simplex of the patterns, that is,
𝝃new=∑ipi​𝒙isuperscript𝝃newsubscript𝑖subscript𝑝𝑖subscript𝒙𝑖\bm{\xi}^{\mathrm{new}}=\sum\_{i}p\_{i}\bm{x}\_{i} with
∑ipi=1subscript𝑖subscript𝑝𝑖1\sum\_{i}p\_{i}=1 and 0⩽pi0subscript𝑝𝑖0\leqslant p\_{i}.
Hence, after one update 𝝃𝝃\bm{\xi} is in the simplex of the pattern and stays there.
If the center 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} is the zero vector 𝒎𝒙=𝟎subscript𝒎𝒙0\bm{m}\_{\bm{x}}=\bm{0}, that is, the data is centered,
then the mean is a fixed point of the iteration.
For 𝝃=𝒎𝒙=𝟎𝝃subscript𝒎𝒙0\bm{\xi}=\bm{m}\_{\bm{x}}=\bm{0} we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | = 1/N​ 1absent1𝑁1\displaystyle=\ 1/N\ \bm{1} |  | (94) |

and

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | = 1/N​𝑿​ 1=𝒎𝒙=𝝃.absent1𝑁𝑿1subscript𝒎𝒙𝝃\displaystyle=\ 1/N\ \bm{X}\ \bm{1}\ =\ \bm{m}\_{\bm{x}}\ =\ \bm{\xi}\ . |  | (95) |

In particular normalization methods like batch normalization would
promote the mean as a fixed point.

We consider the differences of dot products
for 𝒙isubscript𝒙𝑖\bm{x}\_{i}: 𝒙iT​𝒙i−𝒙iT​𝒙j=𝒙iT​(𝒙i−𝒙j)superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖subscript𝒙𝑗\bm{x}\_{i}^{T}\bm{x}\_{i}-\bm{x}\_{i}^{T}\bm{x}\_{j}=\bm{x}\_{i}^{T}(\bm{x}\_{i}-\bm{x}\_{j}),
for fixed point 𝒎𝒙∗superscriptsubscript𝒎𝒙\bm{m}\_{\bm{x}}^{\*}:
(𝒎𝒙∗)T​𝒙i−(𝒎𝒙∗)T​𝒙j=(𝒎𝒙∗)T​(𝒙i−𝒙j)superscriptsuperscriptsubscript𝒎𝒙𝑇subscript𝒙𝑖superscriptsuperscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗superscriptsuperscriptsubscript𝒎𝒙𝑇subscript𝒙𝑖subscript𝒙𝑗(\bm{m}\_{\bm{x}}^{\*})^{T}\bm{x}\_{i}-(\bm{m}\_{\bm{x}}^{\*})^{T}\bm{x}\_{j}=(\bm{m}\_{\bm{x}}^{\*})^{T}(\bm{x}\_{i}-\bm{x}\_{j}),
and for the center 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}}:
𝒎𝒙T​𝒙i−𝒎𝒙T​𝒙j=𝒎𝒙T​(𝒙i−𝒙j)superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑖superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑖subscript𝒙𝑗\bm{m}\_{\bm{x}}^{T}\bm{x}\_{i}-\bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}=\bm{m}\_{\bm{x}}^{T}(\bm{x}\_{i}-\bm{x}\_{j}).
Using the Cauchy-Schwarz inequality, we get

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |𝝃T​(𝒙i−𝒙j)|superscript𝝃𝑇subscript𝒙𝑖subscript𝒙𝑗\displaystyle\left|\bm{\xi}^{T}(\bm{x}\_{i}\ -\ \bm{x}\_{j})\right|\ | ⩽‖𝝃‖​‖𝒙i−𝒙j‖⩽‖𝝃‖​(‖𝒙i−𝒎𝒙‖+‖𝒙j−𝒎𝒙‖)absentnorm𝝃normsubscript𝒙𝑖subscript𝒙𝑗norm𝝃normsubscript𝒙𝑖subscript𝒎𝒙normsubscript𝒙𝑗subscript𝒎𝒙\displaystyle\leqslant\ {{\left\|\bm{\xi}\right\|}}\ {{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}\ \leqslant\ {{\left\|\bm{\xi}\right\|}}\ ({{\left\|\bm{x}\_{i}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ +\ {{\left\|\bm{x}\_{j}\ -\ \bm{m}\_{\bm{x}}\right\|}}) |  | (96) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽ 2​mmax​‖𝝃‖.absent2subscript𝑚norm𝝃\displaystyle\leqslant\ 2\ m\_{\max}\ {{\left\|\bm{\xi}\right\|}}\ . |  |

This inequality gives:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |𝝃T​(𝒙i−𝒙j)|superscript𝝃𝑇subscript𝒙𝑖subscript𝒙𝑗\displaystyle\left|\bm{\xi}^{T}(\bm{x}\_{i}\ -\ \bm{x}\_{j})\right|\ | ⩽ 2​mmax​(mmax+‖𝒎𝒙‖),absent2subscript𝑚subscript𝑚normsubscript𝒎𝒙\displaystyle\leqslant\ 2\ m\_{\max}\ (m\_{\max}\ +\ {{\left\|\bm{m}\_{\bm{x}}\right\|}})\ , |  | (97) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |𝝃T​(𝒙i−𝒙j)|superscript𝝃𝑇subscript𝒙𝑖subscript𝒙𝑗\displaystyle\left|\bm{\xi}^{T}(\bm{x}\_{i}\ -\ \bm{x}\_{j})\right|\ | ⩽ 2​mmax​M,absent2subscript𝑚𝑀\displaystyle\leqslant\ 2\ m\_{\max}\ M\ , |  |

where we used ‖𝝃−𝟎‖⩽‖𝝃−𝒎𝒙‖+‖𝒎𝒙−𝟎‖norm𝝃0norm𝝃subscript𝒎𝒙normsubscript𝒎𝒙0{{\left\|\bm{\xi}-\bm{0}\right\|}}\leqslant{{\left\|\bm{\xi}-\bm{m}\_{\bm{x}}\right\|}}+{{\left\|\bm{m}\_{\bm{x}}-\bm{0}\right\|}},
‖𝝃−𝒎𝒙‖=‖∑ipi​𝒙i−𝒎𝒙‖⩽∑ipi​‖𝒙i−𝒎𝒙‖⩽mmaxnorm𝝃subscript𝒎𝒙normsubscript𝑖subscript𝑝𝑖subscript𝒙𝑖subscript𝒎𝒙subscript𝑖subscript𝑝𝑖normsubscript𝒙𝑖subscript𝒎𝒙subscript𝑚{{\left\|\bm{\xi}-\bm{m}\_{\bm{x}}\right\|}}={{\left\|\sum\_{i}p\_{i}\bm{x}\_{i}-\bm{m}\_{\bm{x}}\right\|}}\leqslant\sum\_{i}p\_{i}{{\left\|\bm{x}\_{i}-\bm{m}\_{\bm{x}}\right\|}}\leqslant m\_{\max},
and M=maxi⁡‖𝒙i‖𝑀subscript𝑖normsubscript𝒙𝑖M=\max\_{i}{{\left\|\bm{x}\_{i}\right\|}}.
In particular

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | β​|𝒎𝒙T​(𝒙i−𝒙j)|𝛽superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑖subscript𝒙𝑗\displaystyle\beta\ \left|\bm{m}\_{\bm{x}}^{T}(\bm{x}\_{i}\ -\ \bm{x}\_{j})\right|\ | ⩽ 2​β​mmax​‖𝒎𝒙‖,absent2𝛽subscript𝑚normsubscript𝒎𝒙\displaystyle\leqslant\ 2\ \beta\ m\_{\max}\ {{\left\|\bm{m}\_{\bm{x}}\right\|}}\ , |  | (98) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | β​|(𝒎𝒙∗)T​(𝒙i−𝒙j)|𝛽superscriptsuperscriptsubscript𝒎𝒙𝑇subscript𝒙𝑖subscript𝒙𝑗\displaystyle\beta\ \left|(\bm{m}\_{\bm{x}}^{\*})^{T}(\bm{x}\_{i}\ -\ \bm{x}\_{j})\right|\ | ⩽ 2​β​mmax​‖𝒎𝒙∗‖⩽ 2​β​mmax​(mmax+‖𝒎𝒙‖),absent2𝛽subscript𝑚normsuperscriptsubscript𝒎𝒙2𝛽subscript𝑚subscript𝑚normsubscript𝒎𝒙\displaystyle\leqslant\ 2\ \beta\ m\_{\max}\ {{\left\|\bm{m}\_{\bm{x}}^{\*}\right\|}}\ \leqslant\ 2\ \beta\ m\_{\max}\ (m\_{\max}\ +\ {{\left\|\bm{m}\_{\bm{x}}\right\|}})\ , |  | (99) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | β​|𝒙iT​(𝒙i−𝒙j)|𝛽superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖subscript𝒙𝑗\displaystyle\beta\ \left|\bm{x}\_{i}^{T}(\bm{x}\_{i}\ -\ \bm{x}\_{j})\right|\ | ⩽ 2​β​mmax​‖𝒙i‖⩽ 2​β​mmax​(mmax+‖𝒎𝒙‖).absent2𝛽subscript𝑚normsubscript𝒙𝑖2𝛽subscript𝑚subscript𝑚normsubscript𝒎𝒙\displaystyle\leqslant\ 2\ \beta\ m\_{\max}\ {{\left\|\bm{x}\_{i}\right\|}}\ \leqslant\ 2\ \beta\ m\_{\max}\ (m\_{\max}\ +\ {{\left\|\bm{m}\_{\bm{x}}\right\|}})\ . |  | (100) |

Let i=arg⁡maxj⁡𝝃T​𝒙j𝑖subscript𝑗superscript𝝃𝑇subscript𝒙𝑗i=\arg\max\_{j}\bm{\xi}^{T}\bm{x}\_{j}, therefore the maximal softmax component
is i𝑖i.
For the maximal softmax component i𝑖i we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | [softmax​(β​𝑿T​𝝃)]i=11+∑j≠iexp⁡(−β​(𝝃T​𝒙i−𝝃T​𝒙j))subscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑖11subscript𝑗𝑖𝛽superscript𝝃𝑇subscript𝒙𝑖superscript𝝃𝑇subscript𝒙𝑗\displaystyle[\mathrm{softmax}(\beta\ \bm{X}^{T}\bm{\xi})]\_{i}\ =\ \frac{1}{1\ +\ \sum\_{j\not=i}\exp(-\ \beta\ (\bm{\xi}^{T}\bm{x}\_{i}\ -\ \bm{\xi}^{T}\bm{x}\_{j}))} |  | (101) |
|  |  |  |
| --- | --- | --- |
|  | ⩽11+∑j≠iexp⁡(− 2​β​mmax​(mmax+‖𝒎𝒙‖))absent11subscript𝑗𝑖2𝛽subscript𝑚subscript𝑚normsubscript𝒎𝒙\displaystyle\leqslant\ \frac{1}{1\ +\ \sum\_{j\not=i}\exp(-\ 2\ \beta\ m\_{\max}\ (m\_{\max}\ +\ {{\left\|\bm{m}\_{\bm{x}}\right\|}}))} |  |
|  |  |  |
| --- | --- | --- |
|  | =11+(N−1)​exp⁡(− 2​β​mmax​(mmax+‖𝒎𝒙‖))absent11𝑁12𝛽subscript𝑚subscript𝑚normsubscript𝒎𝒙\displaystyle=\ \frac{1}{1\ +\ (N-1)\exp(-\ 2\ \beta\ m\_{\max}\ (m\_{\max}\ +\ {{\left\|\bm{m}\_{\bm{x}}\right\|}}))} |  |
|  |  |  |
| --- | --- | --- |
|  | =exp⁡(2​β​mmax​(mmax+‖𝒎𝒙‖))exp⁡(2​β​mmax​(mmax+‖𝒎𝒙‖))+(N−1)absent2𝛽subscript𝑚subscript𝑚normsubscript𝒎𝒙2𝛽subscript𝑚subscript𝑚normsubscript𝒎𝒙𝑁1\displaystyle=\ \frac{\exp(2\ \beta\ m\_{\max}\ (m\_{\max}\ +\ {{\left\|\bm{m}\_{\bm{x}}\right\|}}))}{\exp(2\ \beta\ m\_{\max}\ (m\_{\max}\ +\ {{\left\|\bm{m}\_{\bm{x}}\right\|}}))\ +\ (N-1)} |  |
|  |  |  |
| --- | --- | --- |
|  | ⩽ 1/N​exp⁡(2​β​mmax​(mmax+‖𝒎𝒙‖)).absent1𝑁2𝛽subscript𝑚subscript𝑚normsubscript𝒎𝒙\displaystyle\leqslant\ 1/N\ \exp(2\ \beta\ m\_{\max}\ (m\_{\max}\ +\ {{\left\|\bm{m}\_{\bm{x}}\right\|}}))\ . |  |

Analogously we obtain for i=arg⁡maxj⁡𝒎𝒙T​𝒙j𝑖subscript𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗i=\arg\max\_{j}\bm{m}\_{\bm{x}}^{T}\bm{x}\_{j},
a bound on the maximal softmax component i𝑖i if the center is put into
the iteration:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | [softmax​(β​𝑿T​𝒎𝒙)]isubscriptdelimited-[]softmax𝛽superscript𝑿𝑇subscript𝒎𝒙𝑖\displaystyle[\mathrm{softmax}(\beta\ \bm{X}^{T}\bm{m}\_{\bm{x}})]\_{i}\ | ⩽ 1/N​exp⁡(2​β​mmax​‖𝒎𝒙‖).absent1𝑁2𝛽subscript𝑚normsubscript𝒎𝒙\displaystyle\leqslant\ 1/N\ \exp(2\ \beta\ m\_{\max}\ {{\left\|\bm{m}\_{\bm{x}}\right\|}})\ . |  | (102) |

Analog we obtain a bound for i=argmaxj(𝒎𝒙∗)T𝒙ji=\arg\max\_{j}(\bm{m}\_{\bm{x}}^{\*})^{T}\bm{x}\_{j}
on the maximal softmax component i𝑖i of the fixed point:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | [softmax​(β​𝑿T​𝒎𝒙∗)]isubscriptdelimited-[]softmax𝛽superscript𝑿𝑇superscriptsubscript𝒎𝒙𝑖\displaystyle[\mathrm{softmax}(\beta\ \bm{X}^{T}\bm{m}\_{\bm{x}}^{\*})]\_{i}\ | ⩽ 1/N​exp⁡(2​β​mmax​‖𝒎𝒙∗‖)absent1𝑁2𝛽subscript𝑚normsuperscriptsubscript𝒎𝒙\displaystyle\leqslant\ 1/N\ \exp(2\ \beta\ m\_{\max}\ {{\left\|\bm{m}\_{\bm{x}}^{\*}\right\|}}) |  | (103) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽ 1/N​exp⁡(2​β​mmax​(mmax+‖𝒎𝒙‖)).absent1𝑁2𝛽subscript𝑚subscript𝑚normsubscript𝒎𝒙\displaystyle\leqslant\ 1/N\ \exp(2\ \beta\ m\_{\max}\ (m\_{\max}\ +\ {{\left\|\bm{m}\_{\bm{x}}\right\|}}))\ . |  |

The two important terms are
mmaxsubscript𝑚m\_{\max}, the variance or spread of
the data and ‖𝒎𝒙‖normsubscript𝒎𝒙{{\left\|\bm{m}\_{\bm{x}}\right\|}}, which tells how well
the data is centered.
For a contraction mapping we already required β​mmax2<1𝛽superscriptsubscript𝑚21\beta m\_{\max}^{2}<1, therefore
the first term in the exponent is 2​β​mmax2<22𝛽superscriptsubscript𝑚222\beta m\_{\max}^{2}<2.
The second term 2​β​mmax​‖𝒎𝒙‖2𝛽subscript𝑚normsubscript𝒎𝒙2\beta m\_{\max}{{\left\|\bm{m}\_{\bm{x}}\right\|}} is small if the data is centered.

•Global fixed point near the global mean: Analysis using softmax values.

If 𝝃T​𝒙i≈𝝃T​𝒙jsuperscript𝝃𝑇subscript𝒙𝑖superscript𝝃𝑇subscript𝒙𝑗\bm{\xi}^{T}\bm{x}\_{i}\approx\bm{\xi}^{T}\bm{x}\_{j} for all i𝑖i and j𝑗j,
then pi≈1/Nsubscript𝑝𝑖1𝑁p\_{i}\approx 1/N and we have m=maxi⁡pi​(1−pi)<1/N𝑚subscript𝑖subscript𝑝𝑖1subscript𝑝𝑖1𝑁m=\max\_{i}p\_{i}(1-p\_{i})<1/N.
For M⩽1/2​β𝑀12𝛽M\leqslant 1/\sqrt{2\beta} we obtain from
Lemma [A2](#ThmlemmaA2 "Lemma A2. ‣ A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | < 1.absent1\displaystyle<\ 1\ . |  | (104) |

The local fixed point is 𝒎𝒙∗≈𝒎𝒙=(1/N)​∑i=1N𝒙isuperscriptsubscript𝒎𝒙subscript𝒎𝒙1𝑁superscriptsubscript𝑖1𝑁subscript𝒙𝑖\bm{m}\_{\bm{x}}^{\*}\approx\bm{m}\_{\bm{x}}=(1/N)\sum\_{i=1}^{N}\bm{x}\_{i}
with pi≈1/Nsubscript𝑝𝑖1𝑁p\_{i}\approx 1/N.

We now treat this case more formally.
First we discuss conditions that ensure that the iteration is a
contraction mapping.
We consider the iteration Eq. ([57](#A1.E57 "In A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) in the variable 𝒑𝒑\bm{p}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑newsuperscript𝒑new\displaystyle\bm{p}^{\mathrm{new}}\ | =g​(𝒑)=softmax​(β​𝑿T​𝑿​𝒑).absent𝑔𝒑softmax𝛽superscript𝑿𝑇𝑿𝒑\displaystyle=\ g(\bm{p})\ =\ \mathrm{softmax}(\beta\bm{X}^{T}\bm{X}\bm{p})\ . |  | (105) |

The Jacobian is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | J​(𝒑)J𝒑\displaystyle\mathrm{J}(\bm{p})\ | =∂g​(𝒑)∂𝒑=𝑿T​𝑿​Jsabsent𝑔𝒑𝒑superscript𝑿𝑇𝑿subscriptJ𝑠\displaystyle=\ \frac{\partial g(\bm{p})}{\partial\bm{p}}\ =\ \bm{X}^{T}\bm{X}\ \mathrm{J}\_{s} |  | (106) |

with

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Js​(𝒑new)subscriptJ𝑠superscript𝒑new\displaystyle\mathrm{J}\_{s}(\bm{p}^{\mathrm{new}})\ | =β​(diag​(𝒑new)−𝒑new​(𝒑new)T).absent𝛽diagsuperscript𝒑newsuperscript𝒑newsuperscriptsuperscript𝒑new𝑇\displaystyle=\ \beta\left(\mathrm{diag}(\bm{p}^{\mathrm{new}})\ -\ \bm{p}^{\mathrm{new}}(\bm{p}^{\mathrm{new}})^{T}\right)\ . |  | (107) |

The version of the mean value theorem in Lemma [A32](#ThmlemmaA32 "Lemma A32 (Mean Value Theorem). ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") states for
Jm=∫01J​(λ​𝒑)​dλ=𝑿T​𝑿​JsmsuperscriptJ𝑚superscriptsubscript01J𝜆𝒑differential-d𝜆superscript𝑿𝑇𝑿superscriptsubscriptJ𝑠𝑚\mathrm{J}^{m}=\int\_{0}^{1}\mathrm{J}(\lambda\bm{p})\ \mathrm{d}\lambda=\bm{X}^{T}\bm{X}\mathrm{J}\_{s}^{m} with the symmetric matrix
Jsm=∫01Js​(λ​𝒑)​dλsuperscriptsubscriptJ𝑠𝑚superscriptsubscript01subscriptJ𝑠𝜆𝒑differential-d𝜆\mathrm{J}\_{s}^{m}=\int\_{0}^{1}\mathrm{J}\_{s}(\lambda\bm{p})\ \mathrm{d}\lambda:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑newsuperscript𝒑new\displaystyle\bm{p}^{\mathrm{new}}\ | =g​(𝒑)=g​(𝟎)+(Jm)T​𝒑=g​(𝟎)+Jsm​𝑿T​𝑿​𝒑= 1/N​ 1+Jsm​𝑿T​𝑿​𝒑.absent𝑔𝒑𝑔0superscriptsuperscriptJ𝑚𝑇𝒑𝑔0superscriptsubscriptJ𝑠𝑚superscript𝑿𝑇𝑿𝒑1𝑁1superscriptsubscriptJ𝑠𝑚superscript𝑿𝑇𝑿𝒑\displaystyle=\ g(\bm{p})\ =\ g(\bm{0})\ +\ (\mathrm{J}^{m})^{T}\bm{p}\ =\ g(\bm{0})\ +\ \mathrm{J}\_{s}^{m}\ \bm{X}^{T}\bm{X}\ \bm{p}\ =\ 1/N\ \bm{1}\ +\ \mathrm{J}\_{s}^{m}\ \bm{X}^{T}\bm{X}\ \bm{p}\ . |  | (108) |

With m=maxi⁡pi​(1−pi)𝑚subscript𝑖subscript𝑝𝑖1subscript𝑝𝑖m=\max\_{i}p\_{i}(1-p\_{i}),
Eq. ([476](#A1.E476 "In Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) from
Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js​(𝒑)‖2subscriptnormsubscriptJ𝑠𝒑2\displaystyle{{\left\|\mathrm{J}\_{s}(\bm{p})\right\|}}\_{2}\ | =β​‖diag​(𝒑)−𝒑​𝒑T‖2⩽ 2​m​β.absent𝛽subscriptnormdiag𝒑𝒑superscript𝒑𝑇22𝑚𝛽\displaystyle=\ \beta\ {{\left\|\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right\|}}\_{2}\ \leqslant\ 2\ m\ \beta\ . |  | (109) |

First observe that
λ​pi​(1−λ​pi)⩽pi​(1−pi)𝜆subscript𝑝𝑖1𝜆subscript𝑝𝑖subscript𝑝𝑖1subscript𝑝𝑖\lambda p\_{i}(1-\lambda p\_{i})\leqslant p\_{i}(1-p\_{i}) for pi⩽0.5subscript𝑝𝑖0.5p\_{i}\leqslant 0.5 and λ∈[0,1]𝜆01\lambda\in[0,1], since
pi​(1−pi)−λ​pi​(1−λ​pi)=(1−λ)​pi​(1−(1+λ)​pi)≥0subscript𝑝𝑖1subscript𝑝𝑖𝜆subscript𝑝𝑖1𝜆subscript𝑝𝑖1𝜆subscript𝑝𝑖11𝜆subscript𝑝𝑖0p\_{i}(1-p\_{i})-\lambda p\_{i}(1-\lambda p\_{i})=(1-\lambda)p\_{i}(1-(1+\lambda)p\_{i})\geq 0.
For maxi⁡pi⩽0.5subscript𝑖subscript𝑝𝑖0.5\max\_{i}p\_{i}\leqslant 0.5 this observation leads to the following bound for JsmsuperscriptsubscriptJ𝑠𝑚\mathrm{J}\_{s}^{m}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jsm‖2subscriptnormsuperscriptsubscriptJ𝑠𝑚2\displaystyle{{\left\|\mathrm{J}\_{s}^{m}\right\|}}\_{2}\ | ⩽ 2​m​β.absent2𝑚𝛽\displaystyle\leqslant\ 2\ m\ \beta\ . |  | (110) |

Eq. ([479](#A1.E479 "In Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) in Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") states that every JssubscriptJ𝑠\mathrm{J}\_{s} is bounded by 1/2​β12𝛽1/2\beta,
therefore also the mean:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jsm‖2subscriptnormsuperscriptsubscriptJ𝑠𝑚2\displaystyle{{\left\|\mathrm{J}\_{s}^{m}\right\|}}\_{2}\ | ⩽ 0.5​β.absent0.5𝛽\displaystyle\leqslant\ 0.5\ \beta\ . |  | (111) |

Since
m=maxi⁡pi​(1−pi)<maxi⁡pi=pmax𝑚subscript𝑖subscript𝑝𝑖1subscript𝑝𝑖subscript𝑖subscript𝑝𝑖subscript𝑝m=\max\_{i}p\_{i}(1-p\_{i})<\max\_{i}p\_{i}=p\_{\max}, the previous bounds can be combined as follows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jsm‖2subscriptnormsuperscriptsubscriptJ𝑠𝑚2\displaystyle{{\left\|\mathrm{J}\_{s}^{m}\right\|}}\_{2}\ | ⩽ 2​min⁡{0.25,pmax}​β.absent20.25subscript𝑝𝛽\displaystyle\leqslant\ 2\ \min\{0.25,p\_{\max}\}\ \beta\ . |  | (112) |

Consequently,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jm‖2subscriptnormsuperscriptJ𝑚2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ | ⩽N​M2​ 2​min⁡{0.25,pmax}​β,absent𝑁superscript𝑀220.25subscript𝑝𝛽\displaystyle\leqslant\ N\ M^{2}\ 2\ \min\{0.25,p\_{\max}\}\ \beta\ , |  | (113) |

where we used Eq. ([170](#A1.E170 "In A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
‖𝑿T​𝑿‖2=‖𝑿​𝑿T‖2subscriptnormsuperscript𝑿𝑇𝑿2subscriptnorm𝑿superscript𝑿𝑇2{{\left\|\bm{X}^{T}\bm{X}\right\|}}\_{2}={{\left\|\bm{X}\bm{X}^{T}\right\|}}\_{2},
therefore ‖𝑿T​𝑿‖2subscriptnormsuperscript𝑿𝑇𝑿2{{\left\|\bm{X}^{T}\bm{X}\right\|}}\_{2} is N𝑁N times the maximal second moment of the data squared.

Obviously, g​(𝒑)𝑔𝒑g(\bm{p}) is a contraction mapping in compact sets, where

|  |  |  |  |
| --- | --- | --- | --- |
|  | N​M2​ 2​min⁡{0.25,pmax}​β< 1.𝑁superscript𝑀220.25subscript𝑝𝛽1\displaystyle N\ M^{2}\ 2\ \min\{0.25,p\_{\max}\}\ \beta\ <\ 1\ . |  | (114) |

SS\mathrm{S} is the sphere around the origin 𝟎0\bm{0} with radius one.
For

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑newsuperscript𝒑new\displaystyle\bm{p}^{\mathrm{new}}\ | =g​(𝒑)= 1/N​ 1+Jm​𝒑,absent𝑔𝒑1𝑁1superscriptJ𝑚𝒑\displaystyle=\ g(\bm{p})\ =\ 1/N\ \bm{1}\ +\ \mathrm{J}^{m}\ \bm{p}\ , |  | (115) |

we have ‖𝒑‖⩽‖𝒑‖1=1norm𝒑subscriptnorm𝒑11{{\left\|\bm{p}\right\|}}\leqslant{{\left\|\bm{p}\right\|}}\_{1}=1 and
‖𝒑new‖⩽‖𝒑new‖1=1normsuperscript𝒑newsubscriptnormsuperscript𝒑new11{{\left\|\bm{p}^{\mathrm{new}}\right\|}}\leqslant{{\left\|\bm{p}^{\mathrm{new}}\right\|}}\_{1}=1.
Therefore, g𝑔g maps points from SS\mathrm{S} into SS\mathrm{S}.
g𝑔g is a contraction mapping for

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jm‖2subscriptnormsuperscriptJ𝑚2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ | ⩽N​M2​ 2​min⁡{0.25,pmax}​β=c< 1.absent𝑁superscript𝑀220.25subscript𝑝𝛽𝑐1\displaystyle\leqslant\ N\ M^{2}\ 2\ \min\{0.25,p\_{\max}\}\ \beta\ =\ c\ <\ 1\ . |  | (116) |

According to Banach fixed point theorem g𝑔g has
a fixed point in the sphere SS\mathrm{S}.

Hölder’s inequality gives:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒑‖2superscriptnorm𝒑2\displaystyle{{\left\|\bm{p}\right\|}}^{2}\ | =𝒑T​𝒑⩽‖𝒑‖1​‖𝒑‖∞=‖𝒑‖∞=pmax.absentsuperscript𝒑𝑇𝒑subscriptnorm𝒑1subscriptnorm𝒑subscriptnorm𝒑subscript𝑝\displaystyle=\ \bm{p}^{T}\bm{p}\ \leqslant\ {{\left\|\bm{p}\right\|}}\_{1}{{\left\|\bm{p}\right\|}}\_{\infty}\ =\ {{\left\|\bm{p}\right\|}}\_{\infty}\ =\ p\_{\max}\ . |  | (117) |

Alternatively:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒑‖2superscriptnorm𝒑2\displaystyle{{\left\|\bm{p}\right\|}}^{2}\ | =∑ipi2=pmax​∑ipipmax​pi⩽pmax​∑ipi=pmax.absentsubscript𝑖superscriptsubscript𝑝𝑖2subscript𝑝subscript𝑖subscript𝑝𝑖subscript𝑝subscript𝑝𝑖subscript𝑝subscript𝑖subscript𝑝𝑖subscript𝑝\displaystyle=\ \sum\_{i}p\_{i}^{2}\ =\ p\_{\max}\sum\_{i}\frac{p\_{i}}{p\_{\max}}\ p\_{i}\ \leqslant\ p\_{\max}\sum\_{i}p\_{i}\ =\ p\_{\max}\ . |  | (118) |

Let now SS\mathrm{S} be the sphere around the origin 𝟎0\bm{0}
with radius 1/N+pmax1𝑁subscript𝑝1/\sqrt{N}+\sqrt{p\_{\max}}
and let ‖Jm​(𝒑)‖2⩽c<1subscriptnormsuperscriptJ𝑚𝒑2𝑐1{{\left\|\mathrm{J}^{m}(\bm{p})\right\|}}\_{2}\leqslant c<1 for 𝒑∈S𝒑S\bm{p}\in\mathrm{S}.
The old 𝒑𝒑\bm{p} is in the sphere SS\mathrm{S} (𝒑∈S𝒑S\bm{p}\in\mathrm{S}) since pmax<pmaxsubscript𝑝subscript𝑝p\_{\max}<\sqrt{p\_{\max}} for pmax<1subscript𝑝1p\_{\max}<1.
We have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒑new‖normsuperscript𝒑new\displaystyle{{\left\|\bm{p}^{\mathrm{new}}\right\|}}\ | ⩽ 1/N+‖Jm‖2​‖𝒑‖⩽ 1/N+pmax.absent1𝑁subscriptnormsuperscriptJ𝑚2norm𝒑1𝑁subscript𝑝\displaystyle\leqslant\ 1/\sqrt{N}\ +\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{p}\right\|}}\ \leqslant\ 1/\sqrt{N}\ +\ \sqrt{p\_{\max}}\ . |  | (119) |

Therefore, g𝑔g is a mapping from SS\mathrm{S} into SS\mathrm{S} and a contraction mapping.
According to Banach fixed point theorem, a fixed point exists in SS\mathrm{S}.

For the 1-norm, we use Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") and ‖𝒑‖1=1subscriptnorm𝒑11{{\left\|\bm{p}\right\|}}\_{1}=1
to obtain from Eq. ([115](#A1.E115 "In A.1.5.2 One Stable State: Fixed Point Near the Mean of the Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒑new− 1/N​ 1‖1subscriptnormsuperscript𝒑new1𝑁11\displaystyle{{\left\|\bm{p}^{\mathrm{new}}\ -\ 1/N\ \bm{1}\right\|}}\_{1}\ | ⩽‖Jm‖1⩽ 2​β​m​‖𝑿‖∞​M1,absentsubscriptnormsuperscriptJ𝑚12𝛽𝑚subscriptnorm𝑿subscript𝑀1\displaystyle\leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{1}\ \leqslant\ 2\ \beta\ m\ {{\left\|\bm{X}\right\|}}\_{\infty}\ M\_{1}\ , |  | (120) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒑new− 1/N​ 1‖1subscriptnormsuperscript𝒑new1𝑁11\displaystyle{{\left\|\bm{p}^{\mathrm{new}}\ -\ 1/N\ \bm{1}\right\|}}\_{1}\ | ⩽‖Jm‖1⩽ 2​β​m​N​M∞​M1,absentsubscriptnormsuperscriptJ𝑚12𝛽𝑚𝑁subscript𝑀subscript𝑀1\displaystyle\leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{1}\ \leqslant\ 2\ \beta\ m\ N\ M\_{\infty}\ M\_{1}\ , |  | (121) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒑new− 1/N​ 1‖1subscriptnormsuperscript𝒑new1𝑁11\displaystyle{{\left\|\bm{p}^{\mathrm{new}}\ -\ 1/N\ \bm{1}\right\|}}\_{1}\ | ⩽‖Jm‖1⩽ 2​β​m​N​M2,absentsubscriptnormsuperscriptJ𝑚12𝛽𝑚𝑁superscript𝑀2\displaystyle\leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{1}\ \leqslant\ 2\ \beta\ m\ N\ M^{2}\ , |  | (122) |

where m=maxi⁡pi​(1−pi)𝑚subscript𝑖subscript𝑝𝑖1subscript𝑝𝑖m=\max\_{i}p\_{i}(1-p\_{i}), M1=‖𝑿‖1=maxi⁡‖𝒙i‖1subscript𝑀1subscriptnorm𝑿1subscript𝑖subscriptnormsubscript𝒙𝑖1M\_{1}={{\left\|\bm{X}\right\|}}\_{1}=\max\_{i}{{\left\|\bm{x}\_{i}\right\|}}\_{1},
M=maxi⁡‖𝒙i‖𝑀subscript𝑖normsubscript𝒙𝑖M=\max\_{i}{{\left\|\bm{x}\_{i}\right\|}},
‖𝑿‖∞=‖𝑿T‖1=maxi⁡‖[XT]i‖1subscriptnorm𝑿subscriptnormsuperscript𝑿𝑇1subscript𝑖subscriptnormsubscriptdelimited-[]superscript𝑋𝑇𝑖1{{\left\|\bm{X}\right\|}}\_{\infty}={{\left\|\bm{X}^{T}\right\|}}\_{1}=\max\_{i}{{\left\|[X^{T}]\_{i}\right\|}}\_{1} (maximal absolute row sum norm),
and M∞=maxi⁡‖𝒙i‖∞subscript𝑀subscript𝑖subscriptnormsubscript𝒙𝑖M\_{\infty}=\max\_{i}{{\left\|\bm{x}\_{i}\right\|}}\_{\infty}.
Let us quickly mention some auxiliary estimates related to 𝑿T​𝑿superscript𝑿𝑇𝑿\bm{X}^{T}\bm{X}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝑿T​𝑿‖1subscriptnormsuperscript𝑿𝑇𝑿1\displaystyle{{\left\|\bm{X}^{T}\bm{X}\right\|}}\_{1}\ | =maxi​∑j=1N|𝒙iT​𝒙j|⩽maxi​∑j=1N‖𝒙i‖∞​‖𝒙j‖1absentsubscript𝑖superscriptsubscript𝑗1𝑁superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗subscript𝑖superscriptsubscript𝑗1𝑁subscriptnormsubscript𝒙𝑖subscriptnormsubscript𝒙𝑗1\displaystyle=\ \max\_{i}\sum\_{j=1}^{N}\left|\bm{x}\_{i}^{T}\bm{x}\_{j}\right|\ \leqslant\ \max\_{i}\sum\_{j=1}^{N}{{\left\|\bm{x}\_{i}\right\|}}\_{\infty}\ {{\left\|\bm{x}\_{j}\right\|}}\_{1} |  | (123) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽M∞​∑j=1NM1=N​M∞​M1,absentsubscript𝑀superscriptsubscript𝑗1𝑁subscript𝑀1𝑁subscript𝑀subscript𝑀1\displaystyle\leqslant\ M\_{\infty}\ \sum\_{j=1}^{N}M\_{1}\ =\ N\ M\_{\infty}\ M\_{1}\ , |  |

where the first inequaltiy is from Hölder’s inequality.
We used

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝑿T​𝑿‖1subscriptnormsuperscript𝑿𝑇𝑿1\displaystyle{{\left\|\bm{X}^{T}\bm{X}\right\|}}\_{1}\ | =maxi​∑j=1N|𝒙iT​𝒙j|⩽maxi​∑j=1N‖𝒙i‖​‖𝒙j‖absentsubscript𝑖superscriptsubscript𝑗1𝑁superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗subscript𝑖superscriptsubscript𝑗1𝑁normsubscript𝒙𝑖normsubscript𝒙𝑗\displaystyle=\ \max\_{i}\sum\_{j=1}^{N}\left|\bm{x}\_{i}^{T}\bm{x}\_{j}\right|\ \leqslant\ \max\_{i}\sum\_{j=1}^{N}{{\left\|\bm{x}\_{i}\right\|}}\ {{\left\|\bm{x}\_{j}\right\|}} |  | (124) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽M​∑j=1NM=N​M2,absent𝑀superscriptsubscript𝑗1𝑁𝑀𝑁superscript𝑀2\displaystyle\leqslant\ M\ \sum\_{j=1}^{N}M\ =\ N\ M^{2}\ , |  |

where the first inequality is from Hölder’s inequality (here the same as the Cauchy-Schwarz inequality).
See proof of Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") for the 1-norm bound on Jssubscript𝐽𝑠J\_{s}. Everything else
follows from the fact that the 1-norm is sub-multiplicative as induced matrix norm.

We consider the minimal ‖𝒑‖norm𝒑{{\left\|\bm{p}\right\|}}.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | min𝒑subscript𝒑\displaystyle\min\_{\bm{p}} | ​‖𝒑‖2 superscriptnorm𝒑2\displaystyle{\mbox{\ ~{}}}{{\left\|\bm{p}\right\|}}^{2} |  | (125) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | s.t. | ​∑ipi=1 subscript𝑖subscript𝑝𝑖1\displaystyle{\mbox{\ ~{}}}\sum\_{i}p\_{i}=1 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∀i:pi≥ 0.\displaystyle{\mbox{\ ~{}}}\forall\_{i}:\ \ p\_{i}\ \geq\ 0\ . |  |

The solution to this minimization problem is
𝒑=(1/N)​𝟏𝒑1𝑁1\bm{p}=(1/N)\bm{1}.
Therefore, we have 1/N⩽‖𝒑‖1𝑁norm𝒑1/\sqrt{N}\leqslant{{\left\|\bm{p}\right\|}}
and 1/N⩽‖𝒑‖21𝑁superscriptnorm𝒑21/N\leqslant{{\left\|\bm{p}\right\|}}^{2}
Using Eq. ([119](#A1.E119 "In A.1.5.2 One Stable State: Fixed Point Near the Mean of the Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) we obtain

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1/N1𝑁\displaystyle 1/\sqrt{N}\ | ⩽‖𝒑new‖⩽ 1/N+pmax.absentnormsuperscript𝒑new1𝑁subscript𝑝\displaystyle\leqslant\ {{\left\|\bm{p}^{\mathrm{new}}\right\|}}\ \leqslant\ 1/\sqrt{N}\ +\ \sqrt{p\_{\max}}\ . |  | (126) |

Moreover

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒑new‖2superscriptnormsuperscript𝒑new2\displaystyle{{\left\|\bm{p}^{\mathrm{new}}\right\|}}^{2}\ | =(𝒑new)T​𝒑new= 1/N+(𝒑new)T​Jm​𝒑⩽ 1/N+‖Jm‖2​‖𝒑‖absentsuperscriptsuperscript𝒑new𝑇superscript𝒑new1𝑁superscriptsuperscript𝒑new𝑇superscriptJ𝑚𝒑1𝑁subscriptnormsuperscriptJ𝑚2norm𝒑\displaystyle=\ (\bm{p}^{\mathrm{new}})^{T}\bm{p}^{\mathrm{new}}\ =\ 1/N\ +\ (\bm{p}^{\mathrm{new}})^{T}\mathrm{J}^{m}\ \bm{p}\ \leqslant\ 1/N\ +\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{p}\right\|}} |  | (127) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽ 1/N+‖Jm‖2,absent1𝑁subscriptnormsuperscriptJ𝑚2\displaystyle\leqslant\ 1/N\ +\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ , |  |

since 𝒑new∈Ssuperscript𝒑newS\bm{p}^{\mathrm{new}}\in\mathrm{S} and 𝒑∈S𝒑S\bm{p}\in\mathrm{S}.

For the fixed point, we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒑∗‖2superscriptnormsuperscript𝒑2\displaystyle{{\left\|\bm{p}^{\*}\right\|}}^{2}\ | =(𝒑∗)T​𝒑∗= 1/N+(𝒑∗)T​Jm​𝒑∗⩽ 1/N+‖Jm‖2​‖𝒑∗‖2,absentsuperscriptsuperscript𝒑𝑇superscript𝒑1𝑁superscriptsuperscript𝒑𝑇superscriptJ𝑚superscript𝒑1𝑁subscriptnormsuperscriptJ𝑚2superscriptnormsuperscript𝒑2\displaystyle=\ (\bm{p}^{\*})^{T}\bm{p}^{\*}\ =\ 1/N\ +\ (\bm{p}^{\*})^{T}\mathrm{J}^{m}\ \bm{p}^{\*}\ \leqslant\ 1/N\ +\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{p}^{\*}\right\|}}^{2}\ , |  | (128) |

and hence

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1/N1𝑁\displaystyle 1/N\ | ⩽‖𝒑∗‖2⩽ 1/N​11−‖Jm‖2= 1/N​(1+‖Jm‖21−‖Jm‖2).absentsuperscriptnormsuperscript𝒑21𝑁11subscriptnormsuperscriptJ𝑚21𝑁1subscriptnormsuperscriptJ𝑚21subscriptnormsuperscriptJ𝑚2\displaystyle\leqslant\ {{\left\|\bm{p}^{\*}\right\|}}^{2}\ \leqslant\ 1/N\frac{1}{1\ -\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}}\ =\ 1/N\ (1\ +\ \frac{{{\left\|\mathrm{J}^{m}\right\|}}\_{2}}{1\ -\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}})\ . |  | (129) |

Therefore, for small ‖Jm‖2subscriptnormsuperscriptJ𝑚2{{\left\|\mathrm{J}^{m}\right\|}}\_{2} we have 𝒑∗≈(1/N)​𝟏superscript𝒑1𝑁1\bm{p}^{\*}\approx(1/N)\bm{1}.

##### A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns.

We move on to the next case, where the patterns 𝒙isubscript𝒙𝑖\bm{x}\_{i} are well separated.
In this case the iteration goes to the pattern to which the initial
𝝃𝝃\bm{\xi} is most similar.
If the initial 𝝃𝝃\bm{\xi} is similar to a vector 𝒙isubscript𝒙𝑖\bm{x}\_{i} then it will
converge to 𝒙isubscript𝒙𝑖\bm{x}\_{i} and 𝒑𝒑\bm{p} will be 𝒆isubscript𝒆𝑖\bm{e}\_{i}.
The main ingredients are again Banach’s Theorem and estimates on the Jacobian norm.

•Proof of a fixed point by Banach Fixed Point Theorem.

→→\rightarrow Mapped Vectors Stay in a Compact Environment.
We show that if 𝒙isubscript𝒙𝑖\bm{x}\_{i} is sufficient dissimilar to
other 𝒙jsubscript𝒙𝑗\bm{x}\_{j} then there is an compact environment of 𝒙isubscript𝒙𝑖\bm{x}\_{i}
(a sphere)
where the fixed point iteration maps this environment into
itself.
The idea of the proof is to define a sphere around 𝒙isubscript𝒙𝑖\bm{x}\_{i}
for which points from the sphere are mapped by f𝑓f into the sphere.

We first need following lemma which bounds the distance
‖𝒙i−f​(𝝃)‖normsubscript𝒙𝑖𝑓𝝃{{\left\|\bm{x}\_{i}\ -\ f(\bm{\xi})\right\|}}, where 𝒙isubscript𝒙𝑖\bm{x}\_{i} is the
pattern that is least separated from 𝝃𝝃\bm{\xi} but
separated from other patterns.

###### Lemma A4.

For a query 𝛏𝛏\bm{\xi} and data 𝐗=(𝐱1,…,𝐱N)𝐗subscript𝐱1…subscript𝐱𝑁\bm{X}=(\bm{x}\_{1},\ldots,\bm{x}\_{N}),
there exists a 𝐱isubscript𝐱𝑖\bm{x}\_{i} that is least separated from 𝛏𝛏\bm{\xi} while
being separated from other 𝐱jsubscript𝐱𝑗\bm{x}\_{j} with j≠i𝑗𝑖j\not=i:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | i𝑖\displaystyle i\ | =arg⁡maxk⁡minj,j≠k⁡(𝝃T​𝒙k−𝝃T​𝒙j)=arg⁡maxk⁡(𝝃T​𝒙k−maxj,j≠k⁡𝝃T​𝒙j)absentsubscript𝑘subscript  𝑗𝑗 𝑘superscript𝝃𝑇subscript𝒙𝑘superscript𝝃𝑇subscript𝒙𝑗subscript𝑘superscript𝝃𝑇subscript𝒙𝑘subscript  𝑗𝑗 𝑘superscript𝝃𝑇subscript𝒙𝑗\displaystyle=\ \arg\max\_{k}\min\_{j,j\not=k}\left(\bm{\xi}^{T}\bm{x}\_{k}\ -\ \bm{\xi}^{T}\bm{x}\_{j}\right)\ =\ \arg\max\_{k}\left(\bm{\xi}^{T}\bm{x}\_{k}\ -\ \max\_{j,j\not=k}\bm{\xi}^{T}\bm{x}\_{j}\right) |  | (130) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 00\displaystyle 0\ | ⩽c=maxk⁡minj,j≠k⁡(𝝃T​𝒙k−𝝃T​𝒙j)=maxk⁡(𝝃T​𝒙k−maxj,j≠k⁡𝝃T​𝒙j).absent𝑐subscript𝑘subscript  𝑗𝑗 𝑘superscript𝝃𝑇subscript𝒙𝑘superscript𝝃𝑇subscript𝒙𝑗subscript𝑘superscript𝝃𝑇subscript𝒙𝑘subscript  𝑗𝑗 𝑘superscript𝝃𝑇subscript𝒙𝑗\displaystyle\leqslant\ c\ =\ \max\_{k}\min\_{j,j\not=k}\left(\bm{\xi}^{T}\bm{x}\_{k}\ -\ \bm{\xi}^{T}\bm{x}\_{j}\right)\ =\ \max\_{k}\left(\bm{\xi}^{T}\bm{x}\_{k}\ -\ \max\_{j,j\not=k}\bm{\xi}^{T}\bm{x}\_{j}\right)\ . |  | (131) |

For 𝐱isubscript𝐱𝑖\bm{x}\_{i}, the following holds:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒙i−f​(𝝃)‖normsubscript𝒙𝑖𝑓𝝃\displaystyle{{\left\|\bm{x}\_{i}\ -\ f(\bm{\xi})\right\|}}\ | ⩽ 2​ϵ​M,absent2italic-ϵ𝑀\displaystyle\leqslant\ 2\ \epsilon\ M\ , |  | (132) |

where

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M𝑀\displaystyle M\ | =maxi⁡‖𝒙i‖,absentsubscript𝑖normsubscript𝒙𝑖\displaystyle=\ \max\_{i}{{\left\|\bm{x}\_{i}\right\|}}\ , |  | (133) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵitalic-ϵ\displaystyle\epsilon\ | =(N−1)​exp⁡(−β​c).absent𝑁1𝛽𝑐\displaystyle=\ (N-1)\ \exp(-\ \beta\ c)\ . |  | (134) |

###### Proof.

For the softmax component i𝑖i we have:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | [softmax​(β​𝑿T​𝝃)]isubscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑖\displaystyle[\mathrm{softmax}(\beta\ \bm{X}^{T}\bm{\xi})]\_{i}\ | =11+∑j≠iexp⁡(β​(𝝃T​𝒙j−𝝃T​𝒙i))≥11+∑j≠iexp⁡(−β​c)absent11subscript𝑗𝑖𝛽superscript𝝃𝑇subscript𝒙𝑗superscript𝝃𝑇subscript𝒙𝑖11subscript𝑗𝑖𝛽𝑐\displaystyle=\ \frac{1}{1\ +\ \sum\_{j\not=i}\exp(\beta\ (\bm{\xi}^{T}\bm{x}\_{j}\ -\ \bm{\xi}^{T}\bm{x}\_{i}))}\ \geq\ \frac{1}{1\ +\ \sum\_{j\not=i}\exp(-\ \beta\ c)} |  | (135) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =11+(N−1)​exp⁡(−β​c)= 1−(N−1)​exp⁡(−β​c)1+(N−1)​exp⁡(−β​c)absent11𝑁1𝛽𝑐1𝑁1𝛽𝑐1𝑁1𝛽𝑐\displaystyle=\ \frac{1}{1\ +\ (N-1)\exp(-\ \beta\ c)}\ =\ 1\ -\ \frac{(N-1)\exp(-\ \beta\ c)}{1\ +\ (N-1)\exp(-\ \beta\ c)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥ 1−(N−1)​exp⁡(−β​c)= 1−ϵabsent1𝑁1𝛽𝑐1italic-ϵ\displaystyle\geq\ 1\ -\ (N-1)\exp(-\ \beta\ c)\ =\ 1\ -\ \epsilon\ |  |

For softmax components k≠i𝑘𝑖k\not=i we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | [softmax​(β​𝑿T​𝝃)]ksubscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑘\displaystyle[\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})]\_{k}\ | =exp⁡(β​(𝝃T​𝒙k−𝝃T​𝒙i))1+∑j≠iexp⁡(β​(𝝃T​𝒙j−𝝃T​𝒙i))⩽exp⁡(−β​c)=ϵN−1.absent𝛽superscript𝝃𝑇subscript𝒙𝑘superscript𝝃𝑇subscript𝒙𝑖1subscript𝑗𝑖𝛽superscript𝝃𝑇subscript𝒙𝑗superscript𝝃𝑇subscript𝒙𝑖𝛽𝑐italic-ϵ𝑁1\displaystyle=\ \frac{\exp(\beta\ (\bm{\xi}^{T}\bm{x}\_{k}\ -\ \bm{\xi}^{T}\bm{x}\_{i}))}{1\ +\ \sum\_{j\not=i}\exp(\beta\ (\bm{\xi}^{T}\bm{x}\_{j}\ -\ \bm{\xi}^{T}\bm{x}\_{i}))}\ \leqslant\ \exp(-\ \beta\ c)\ =\ \frac{\epsilon}{N-1}\ . |  | (136) |

The iteration f𝑓f can be written as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(𝝃)𝑓𝝃\displaystyle f(\bm{\xi})\ | =𝑿​softmax​(β​𝑿T​𝝃)=∑j=1N𝒙j​[softmax​(β​𝑿T​𝝃)]j.absent𝑿softmax𝛽superscript𝑿𝑇𝝃superscriptsubscript𝑗1𝑁subscript𝒙𝑗subscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑗\displaystyle=\ \bm{X}\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ =\ \sum\_{j=1}^{N}\bm{x}\_{j}\ [\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})]\_{j}\ . |  | (137) |

We now can bound ‖𝒙i−f​(𝝃)‖normsubscript𝒙𝑖𝑓𝝃{{\left\|\bm{x}\_{i}\ -\ f(\bm{\xi})\right\|}}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒙i−f​(𝝃)‖normsubscript𝒙𝑖𝑓𝝃\displaystyle{{\left\|\bm{x}\_{i}\ -\ f(\bm{\xi})\right\|}}\ | =‖𝒙i−∑j=1N[softmax​(β​𝑿T​𝝃)]j​𝒙j‖absentnormsubscript𝒙𝑖superscriptsubscript𝑗1𝑁subscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑗subscript𝒙𝑗\displaystyle=\ {{\left\|\bm{x}\_{i}\ -\ \sum\_{j=1}^{N}[\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})]\_{j}\ \bm{x}\_{j}\right\|}} |  | (138) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =‖(1−[softmax​(β​𝑿T​𝝃)]i)​𝒙i−∑j=1,j≠iN[softmax​(β​𝑿T​𝝃)]j​𝒙j‖absentnorm1subscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑖subscript𝒙𝑖superscriptsubscriptformulae-sequence𝑗1𝑗𝑖𝑁subscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑗subscript𝒙𝑗\displaystyle=\ {{\left\|(1-[\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})]\_{i})\ \bm{x}\_{i}\ -\ \sum\_{j=1,j\not=i}^{N}[\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})]\_{j}\ \bm{x}\_{j}\right\|}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽ϵ​‖𝒙i‖+ϵN−1​∑j=1,j≠iN‖𝒙j‖absentitalic-ϵnormsubscript𝒙𝑖italic-ϵ𝑁1superscriptsubscriptformulae-sequence𝑗1𝑗𝑖𝑁normsubscript𝒙𝑗\displaystyle\leqslant\ \epsilon\ {{\left\|\bm{x}\_{i}\right\|}}\ +\ \frac{\epsilon}{N-1}\ \sum\_{j=1,j\not=i}^{N}{{\left\|\bm{x}\_{j}\right\|}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽ϵ​M+ϵN−1​∑j=1,j≠iNM= 2​ϵ​M.absentitalic-ϵ𝑀italic-ϵ𝑁1superscriptsubscriptformulae-sequence𝑗1𝑗𝑖𝑁𝑀2italic-ϵ𝑀\displaystyle\leqslant\ \epsilon\ M\ +\ \frac{\epsilon}{N-1}\ \sum\_{j=1,j\not=i}^{N}M\ =\ 2\ \epsilon\ M\ . |  |

∎

We define ΔisubscriptΔ𝑖\Delta\_{i}, i.e. the separation of pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} from data
𝑿=(𝒙1,…,𝒙N)𝑿subscript𝒙1…subscript𝒙𝑁\bm{X}=(\bm{x}\_{1},\ldots,\bm{x}\_{N}) as:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | =minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)=𝒙iT​𝒙i−maxj,j≠i⁡𝒙iT​𝒙j.absentsubscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖subscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right)\ =\ \bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \max\_{j,j\not=i}\bm{x}\_{i}^{T}\bm{x}\_{j}\ . |  | (139) |

The pattern is separated from the other data if 0<Δi0subscriptΔ𝑖0<\Delta\_{i}.
Using the parallelogram identity, ΔisubscriptΔ𝑖\Delta\_{i} can also be expressed as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | =minj,j≠i⁡12​(‖𝒙i‖2−‖𝒙j‖2+‖𝒙i−𝒙j‖2)absentsubscript  𝑗𝑗 𝑖12superscriptnormsubscript𝒙𝑖2superscriptnormsubscript𝒙𝑗2superscriptnormsubscript𝒙𝑖subscript𝒙𝑗2\displaystyle=\ \min\_{j,j\not=i}\frac{1}{2}\ \left({{\left\|\bm{x}\_{i}\right\|}}^{2}\ -\ {{\left\|\bm{x}\_{j}\right\|}}^{2}\ +\ {{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}^{2}\right) |  | (140) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =12​‖𝒙i‖2−12​maxj,j≠i⁡(‖𝒙j‖2−‖𝒙i−𝒙j‖2).absent12superscriptnormsubscript𝒙𝑖212subscript  𝑗𝑗 𝑖superscriptnormsubscript𝒙𝑗2superscriptnormsubscript𝒙𝑖subscript𝒙𝑗2\displaystyle=\ \frac{1}{2}{{\left\|\bm{x}\_{i}\right\|}}^{2}\ -\ \frac{1}{2}\ \max\_{j,j\not=i}\left({{\left\|\bm{x}\_{j}\right\|}}^{2}\ -\ {{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}^{2}\right)\ . |  |

For ‖𝒙i‖=‖𝒙j‖normsubscript𝒙𝑖normsubscript𝒙𝑗{{\left\|\bm{x}\_{i}\right\|}}={{\left\|\bm{x}\_{j}\right\|}} we have Δi=1/2​minj,j≠i⁡‖𝒙i−𝒙j‖2subscriptΔ𝑖12subscript

𝑗𝑗
𝑖superscriptnormsubscript𝒙𝑖subscript𝒙𝑗2\Delta\_{i}=1/2\min\_{j,j\not=i}{{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}^{2}.

Next we define the sphere where we want to apply
Banach fixed point theorem.

###### Definition 3 (Sphere SisubscriptS𝑖\mathrm{S}\_{i}).

The sphere SisubscriptS𝑖\mathrm{S}\_{i} is defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | SisubscriptS𝑖\displaystyle\mathrm{S}\_{i}\ | :={𝝃∣‖𝝃−𝒙i‖⩽1β​N​M}.assignabsentconditional-set𝝃norm𝝃subscript𝒙𝑖1𝛽𝑁𝑀\displaystyle:=\ \left\{\bm{\xi}\mid{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ \leqslant\ \frac{1}{\beta\ N\ M}\right\}\ . |  | (141) |

###### Lemma A5.

With 𝛏𝛏\bm{\xi} given, if the assumptions

1. A1:

   𝝃𝝃\bm{\xi} is inside sphere: 𝝃∈Si𝝃subscriptS𝑖\bm{\xi}\in\mathrm{S}\_{i},
2. A2:

   data point 𝒙isubscript𝒙𝑖\bm{x}\_{i} is well separated from the other data:

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | ≥2β​N+1β​ln⁡(2​(N−1)​N​β​M2)absent2𝛽𝑁1𝛽2𝑁1𝑁𝛽superscript𝑀2\displaystyle\geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right) |  | (142) |

hold, then f​(𝛏)𝑓𝛏f(\bm{\xi}) is inside the sphere: f​(𝛏)∈Si𝑓𝛏subscriptS𝑖f(\bm{\xi})\in\mathrm{S}\_{i}.
Therefore, with assumption (A2),
f𝑓f is a mapping from SisubscriptS𝑖\mathrm{S}\_{i} into SisubscriptS𝑖\mathrm{S}\_{i}.

###### Proof.

We need the separation Δ~isubscript~Δ𝑖\tilde{\Delta}\_{i} of
𝝃𝝃\bm{\xi} from the data.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~isubscript~Δ𝑖\displaystyle\tilde{\Delta}\_{i}\ | =minj,j≠i⁡(𝝃T​𝒙i−𝝃T​𝒙j).absentsubscript  𝑗𝑗 𝑖superscript𝝃𝑇subscript𝒙𝑖superscript𝝃𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,j\not=i}\left(\bm{\xi}^{T}\bm{x}\_{i}\ -\ \bm{\xi}^{T}\bm{x}\_{j}\right)\ . |  | (143) |

Using the Cauchy-Schwarz inequality, we obtain for 1⩽j⩽N1𝑗𝑁1\leqslant j\leqslant N:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |𝝃T​𝒙j−𝒙iT​𝒙j|superscript𝝃𝑇subscript𝒙𝑗superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗\displaystyle\left|\bm{\xi}^{T}\bm{x}\_{j}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right| | ⩽‖𝝃−𝒙i‖​‖𝒙j‖⩽‖𝝃−𝒙i‖​M.absentnorm𝝃subscript𝒙𝑖normsubscript𝒙𝑗norm𝝃subscript𝒙𝑖𝑀\displaystyle\leqslant\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ {{\left\|\bm{x}\_{j}\right\|}}\ \leqslant\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ M\ . |  | (144) |

We have the lower bound

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~isubscript~Δ𝑖\displaystyle\tilde{\Delta}\_{i}\ | ≥minj,j≠i⁡((𝒙iT​𝒙i−‖𝝃−𝒙i‖​M)−(𝒙iT​𝒙j+‖𝝃−𝒙i‖​M))absentsubscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖norm𝝃subscript𝒙𝑖𝑀superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗norm𝝃subscript𝒙𝑖𝑀\displaystyle\geq\ \min\_{j,j\not=i}\left(\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ M\right)\ -\ \left(\bm{x}\_{i}^{T}\bm{x}\_{j}\ +\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ M\right)\right) |  | (145) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =− 2​‖𝝃−𝒙i‖​M+minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)=Δi− 2​‖𝝃−𝒙i‖​Mabsent2norm𝝃subscript𝒙𝑖𝑀subscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗subscriptΔ𝑖2norm𝝃subscript𝒙𝑖𝑀\displaystyle=\ -\ 2\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ M\ +\ \min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right)\ =\ \Delta\_{i}\ -\ 2\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ M |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥Δi−2β​N,absentsubscriptΔ𝑖2𝛽𝑁\displaystyle\geq\ \Delta\_{i}\ -\ \frac{2}{\beta\ N}\ , |  |

where we used the assumption (A1) of the lemma.

From the proof in Lemma [A4](#ThmlemmaA4 "Lemma A4. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | pmaxsubscript𝑝\displaystyle p\_{\max}\ | =[softmax​(β​𝑿T​𝝃)]i≥ 1−(N−1)​exp⁡(−β​Δ~i)= 1−ϵ~.absentsubscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑖1𝑁1𝛽subscript~Δ𝑖1~italic-ϵ\displaystyle=\ [\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})]\_{i}\ \geq\ 1\ -\ (N-1)\ \exp(-\ \beta\ \tilde{\Delta}\_{i})\ =\ 1\ -\ \tilde{\epsilon}\ . |  | (146) |

Lemma [A4](#ThmlemmaA4 "Lemma A4. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") states that

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒙i−f​(𝝃)‖normsubscript𝒙𝑖𝑓𝝃\displaystyle{{\left\|\bm{x}\_{i}\ -\ f(\bm{\xi})\right\|}}\ | ⩽ 2​ϵ~​M= 2​(N−1)​exp⁡(−β​Δ~i)​Mabsent2~italic-ϵ𝑀2𝑁1𝛽subscript~Δ𝑖𝑀\displaystyle\leqslant\ 2\ \tilde{\epsilon}\ M\ =\ 2\ (N-1)\ \exp(-\ \beta\ \tilde{\Delta}\_{i})\ M |  | (147) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽ 2​(N−1)​exp⁡(−β​(Δi−2β​N))​M.absent2𝑁1𝛽subscriptΔ𝑖2𝛽𝑁𝑀\displaystyle\leqslant\ 2\ (N-1)\ \exp(-\ \beta\ (\Delta\_{i}\ -\ \frac{2}{\beta\ N}))\ M\ . |  |

We have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝒙i−f​(𝝃)‖normsubscript𝒙𝑖𝑓𝝃\displaystyle{{\left\|\bm{x}\_{i}\ -\ f(\bm{\xi})\right\|}} |  | (148) |
|  |  |  |
| --- | --- | --- |
|  | ⩽ 2​(N−1)​exp⁡(−β​(2β​N+1β​ln⁡(2​(N−1)​N​β​M2)−2β​N))​Mabsent2𝑁1𝛽2𝛽𝑁1𝛽2𝑁1𝑁𝛽superscript𝑀22𝛽𝑁𝑀\displaystyle\leqslant\ 2\ (N-1)\ \exp(-\ \beta\ (\frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right)\ -\ \frac{2}{\beta\ N}))\ M |  |
|  |  |  |
| --- | --- | --- |
|  | = 2​(N−1)​exp⁡(−ln⁡(2​(N−1)​N​β​M2))​Mabsent2𝑁12𝑁1𝑁𝛽superscript𝑀2𝑀\displaystyle=\ 2\ (N-1)\ \exp(-\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right))\ M |  |
|  |  |  |
| --- | --- | --- |
|  | =1N​β​M,absent1𝑁𝛽𝑀\displaystyle=\ \frac{1}{N\ \beta\ M}\ , |  |

where we used assumption (A2) of the lemma.
Therefore, f​(𝝃)𝑓𝝃f(\bm{\xi}) is a mapping from the
sphere SisubscriptS𝑖\mathrm{S}\_{i} into the sphere SisubscriptS𝑖\mathrm{S}\_{i}:
If 𝝃∈Si𝝃subscriptS𝑖\bm{\xi}\in\mathrm{S}\_{i} then f​(𝝃)∈Si𝑓𝝃subscriptS𝑖f(\bm{\xi})\in\mathrm{S}\_{i}.
∎

•Contraction mapping.

For applying Banach fixed point theorem we need to show that
f𝑓f is contraction in the compact environment SisubscriptS𝑖\mathrm{S}\_{i}.

###### Lemma A6.

Assume that

1. A1:

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | ≥2β​N+1β​ln⁡(2​(N−1)​N​β​M2),absent2𝛽𝑁1𝛽2𝑁1𝑁𝛽superscript𝑀2\displaystyle\geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right)\ , |  | (149) |

then f𝑓f is a contraction mapping in SisubscriptS𝑖\mathrm{S}\_{i}.

###### Proof.

The version of the mean value theorem Lemma [A32](#ThmlemmaA32 "Lemma A32 (Mean Value Theorem). ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") states for
Jm=∫01J​(λ​𝝃+(1−λ)​𝒙i)​dλsuperscriptJ𝑚superscriptsubscript01J𝜆𝝃1𝜆subscript𝒙𝑖differential-d𝜆\mathrm{J}^{m}=\int\_{0}^{1}\mathrm{J}(\lambda\bm{\xi}+(1-\lambda)\bm{x}\_{i})\ \mathrm{d}\lambda:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(𝝃)𝑓𝝃\displaystyle f(\bm{\xi})\ | =f​(𝒙i)+Jm​(𝝃−𝒙i).absent𝑓subscript𝒙𝑖superscriptJ𝑚𝝃subscript𝒙𝑖\displaystyle=\ f(\bm{x}\_{i})\ +\ \mathrm{J}^{m}\ (\bm{\xi}\ -\ \bm{x}\_{i})\ . |  | (150) |

Therefore

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖f​(𝝃)−f​(𝒙i)‖norm𝑓𝝃𝑓subscript𝒙𝑖\displaystyle{{\left\|f(\bm{\xi})\ -\ f(\bm{x}\_{i})\right\|}}\ | ⩽‖Jm‖2​‖𝝃−𝒙i‖.absentsubscriptnormsuperscriptJ𝑚2norm𝝃subscript𝒙𝑖\displaystyle\leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ . |  | (151) |

We define 𝝃~=λ​𝝃+(1−λ)​𝒙i~𝝃𝜆𝝃1𝜆subscript𝒙𝑖\tilde{\bm{\xi}}=\lambda\bm{\xi}+(1-\lambda)\bm{x}\_{i} for some λ∈[0,1]𝜆01\lambda\in[0,1].
From the proof in Lemma [A4](#ThmlemmaA4 "Lemma A4. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | pmax​(𝝃~)subscript𝑝~𝝃\displaystyle p\_{\max}(\tilde{\bm{\xi}})\ | =[softmax​(β​𝑿T​𝝃~)]i≥ 1−(N−1)​exp⁡(−β​Δ~i)= 1−ϵ~,absentsubscriptdelimited-[]softmax𝛽superscript𝑿𝑇~𝝃𝑖1𝑁1𝛽subscript~Δ𝑖1~italic-ϵ\displaystyle=\ [\mathrm{softmax}(\beta\ \bm{X}^{T}\ \tilde{\bm{\xi}})]\_{i}\ \geq\ 1\ -\ (N-1)\ \exp(-\ \beta\ \tilde{\Delta}\_{i})\ =\ 1\ -\ \tilde{\epsilon}\ , |  | (152) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵ~~italic-ϵ\displaystyle\tilde{\epsilon}\ | =(N−1)​exp⁡(−β​Δ~i),absent𝑁1𝛽subscript~Δ𝑖\displaystyle=\ (N-1)\ \exp(-\ \beta\ \tilde{\Delta}\_{i})\ , |  | (153) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~isubscript~Δ𝑖\displaystyle\tilde{\Delta}\_{i}\ | =minj,j≠i⁡(𝝃~T​𝒙i−𝝃~T​𝒙j).absentsubscript  𝑗𝑗 𝑖superscript~𝝃𝑇subscript𝒙𝑖superscript~𝝃𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,j\not=i}\left(\tilde{\bm{\xi}}^{T}\bm{x}\_{i}\ -\ \tilde{\bm{\xi}}^{T}\bm{x}\_{j}\right)\ . |  | (154) |

First we compute an upper bound on ϵ~~italic-ϵ\tilde{\epsilon}.
We need the separation Δ~isubscript~Δ𝑖\tilde{\Delta}\_{i} of
𝝃𝝃\bm{\xi} from the data.
Using the Cauchy-Schwarz inequality, we obtain for 1⩽j⩽N1𝑗𝑁1\leqslant j\leqslant N:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |𝝃~T​𝒙j−𝒙iT​𝒙j|superscript~𝝃𝑇subscript𝒙𝑗superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗\displaystyle\left|\tilde{\bm{\xi}}^{T}\bm{x}\_{j}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right| | ⩽‖𝝃~−𝒙i‖​‖𝒙j‖⩽‖𝝃~−𝒙i‖​M.absentnorm~𝝃subscript𝒙𝑖normsubscript𝒙𝑗norm~𝝃subscript𝒙𝑖𝑀\displaystyle\leqslant\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{x}\_{i}\right\|}}\ {{\left\|\bm{x}\_{j}\right\|}}\ \leqslant\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{x}\_{i}\right\|}}\ M\ . |  | (155) |

We have the lower bound on Δ~isubscript~Δ𝑖\tilde{\Delta}\_{i}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~isubscript~Δ𝑖\displaystyle\tilde{\Delta}\_{i}\ | ≥minj,j≠i⁡((𝒙iT​𝒙i−‖𝝃~−𝒙i‖​M)−(𝒙iT​𝒙j+‖𝝃~−𝒙i‖​M))absentsubscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖norm~𝝃subscript𝒙𝑖𝑀superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗norm~𝝃subscript𝒙𝑖𝑀\displaystyle\geq\ \min\_{j,j\not=i}\left(\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{x}\_{i}\right\|}}\ M\right)\ -\ \left(\bm{x}\_{i}^{T}\bm{x}\_{j}\ +\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{x}\_{i}\right\|}}\ M\right)\right) |  | (156) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =− 2​‖𝝃~−𝒙i‖​M+minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)=Δi− 2​‖𝝃~−𝒙i‖​Mabsent2norm~𝝃subscript𝒙𝑖𝑀subscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗subscriptΔ𝑖2norm~𝝃subscript𝒙𝑖𝑀\displaystyle=\ -\ 2\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{x}\_{i}\right\|}}\ M\ +\ \min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right)\ =\ \Delta\_{i}\ -\ 2\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{x}\_{i}\right\|}}\ M |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥Δi− 2​‖𝝃−𝒙i‖​M,absentsubscriptΔ𝑖2norm𝝃subscript𝒙𝑖𝑀\displaystyle\geq\ \Delta\_{i}\ -\ 2\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ M\ , |  |

where we used ‖𝝃~−𝒙i‖=λ​‖𝝃−𝒙i‖⩽‖𝝃−𝒙i‖norm~𝝃subscript𝒙𝑖𝜆norm𝝃subscript𝒙𝑖norm𝝃subscript𝒙𝑖{{\left\|\tilde{\bm{\xi}}-\bm{x}\_{i}\right\|}}=\lambda{{\left\|\bm{\xi}-\bm{x}\_{i}\right\|}}\leqslant{{\left\|\bm{\xi}-\bm{x}\_{i}\right\|}}.
From the definition of ϵ~~italic-ϵ\tilde{\epsilon} in Eq. ([152](#A1.E152 "In Proof. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵ~~italic-ϵ\displaystyle\tilde{\epsilon}\ | =(N−1)​exp⁡(−β​Δ~i)absent𝑁1𝛽subscript~Δ𝑖\displaystyle=\ (N-1)\ \exp(-\ \beta\ \tilde{\Delta}\_{i}) |  | (157) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽(N−1)​exp⁡(−β​(Δi− 2​‖𝝃−𝒙i‖​M))absent𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖𝑀\displaystyle\leqslant\ (N-1)\ \exp\left(-\ \beta\ \left(\Delta\_{i}\ -\ 2\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ M\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽(N−1)​exp⁡(−β​(Δi−2β​N)),absent𝑁1𝛽subscriptΔ𝑖2𝛽𝑁\displaystyle\leqslant\ (N-1)\ \exp\left(-\ \beta\ \left(\Delta\_{i}\ -\ \frac{2}{\beta\ N}\right)\right)\ , |  |

where we used 𝝃∈Si𝝃subscriptS𝑖\bm{\xi}\in\mathrm{S}\_{i}, therefore ‖𝝃−𝒙i‖⩽1β​N​Mnorm𝝃subscript𝒙𝑖1𝛽𝑁𝑀{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ \leqslant\ \frac{1}{\beta\ N\ M}.

Next we compute an lower bound on ϵ~~italic-ϵ\tilde{\epsilon}.
We start with an upper on Δ~isubscript~Δ𝑖\tilde{\Delta}\_{i}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~isubscript~Δ𝑖\displaystyle\tilde{\Delta}\_{i}\ | ⩽minj,j≠i⁡((𝒙iT​𝒙i+‖𝝃~−𝒙i‖​M)−(𝒙iT​𝒙j−‖𝝃~−𝒙i‖​M))absentsubscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖norm~𝝃subscript𝒙𝑖𝑀superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗norm~𝝃subscript𝒙𝑖𝑀\displaystyle\leqslant\ \min\_{j,j\not=i}\left(\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ +\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{x}\_{i}\right\|}}\ M\right)\ -\ \left(\bm{x}\_{i}^{T}\bm{x}\_{j}\ -\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{x}\_{i}\right\|}}\ M\right)\right) |  | (158) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | = 2​‖𝝃~−𝒙i‖​M+minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)=Δi+ 2​‖𝝃~−𝒙i‖​Mabsent2norm~𝝃subscript𝒙𝑖𝑀subscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗subscriptΔ𝑖2norm~𝝃subscript𝒙𝑖𝑀\displaystyle=\ 2\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{x}\_{i}\right\|}}\ M\ +\ \min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right)\ =\ \Delta\_{i}\ +\ 2\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{x}\_{i}\right\|}}\ M |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽Δi+ 2​‖𝝃−𝒙i‖​M,absentsubscriptΔ𝑖2norm𝝃subscript𝒙𝑖𝑀\displaystyle\leqslant\ \Delta\_{i}\ +\ 2\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ M\ , |  |

where we used ‖𝝃~−𝒙i‖=λ​‖𝝃−𝒙i‖⩽‖𝝃−𝒙i‖norm~𝝃subscript𝒙𝑖𝜆norm𝝃subscript𝒙𝑖norm𝝃subscript𝒙𝑖{{\left\|\tilde{\bm{\xi}}-\bm{x}\_{i}\right\|}}=\lambda{{\left\|\bm{\xi}-\bm{x}\_{i}\right\|}}\leqslant{{\left\|\bm{\xi}-\bm{x}\_{i}\right\|}}.
From the definition of ϵ~~italic-ϵ\tilde{\epsilon} in Eq. ([152](#A1.E152 "In Proof. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵ~~italic-ϵ\displaystyle\tilde{\epsilon}\ | =(N−1)​exp⁡(−β​Δ~i)absent𝑁1𝛽subscript~Δ𝑖\displaystyle=\ (N-1)\ \exp(-\ \beta\ \tilde{\Delta}\_{i}) |  | (159) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥(N−1)​exp⁡(−β​(Δi+ 2​‖𝝃−𝒙i‖​M))absent𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖𝑀\displaystyle\geq\ (N-1)\ \exp\left(-\ \beta\ \left(\Delta\_{i}\ +\ 2\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ M\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥(N−1)​exp⁡(−β​(Δi+2β​N)),absent𝑁1𝛽subscriptΔ𝑖2𝛽𝑁\displaystyle\geq\ (N-1)\ \exp\left(-\ \beta\ \left(\Delta\_{i}\ +\ \frac{2}{\beta\ N}\right)\right)\ , |  |

where we used 𝝃∈Si𝝃subscriptS𝑖\bm{\xi}\in\mathrm{S}\_{i}, therefore ‖𝝃−𝒙i‖⩽1β​N​Mnorm𝝃subscript𝒙𝑖1𝛽𝑁𝑀{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ \leqslant\ \frac{1}{\beta\ N\ M}.

Now we bound the Jacobian.
We can assume ϵ~⩽0.5~italic-ϵ0.5\tilde{\epsilon}\leqslant 0.5 otherwise
(1−ϵ~)⩽0.51~italic-ϵ0.5(1-\tilde{\epsilon})\leqslant 0.5 in the following.
From the proof of
Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") we know
for pmax​(𝝃~)≥1−ϵ~subscript𝑝~𝝃1~italic-ϵp\_{\max}(\tilde{\bm{\xi}})\geq 1-\tilde{\epsilon},
then pi​(𝝃~)⩽ϵ~subscript𝑝𝑖~𝝃~italic-ϵp\_{i}(\tilde{\bm{\xi}})\leqslant\tilde{\epsilon}
for pi​(𝝃~)≠pmax​(𝝃~)subscript𝑝𝑖~𝝃subscript𝑝~𝝃p\_{i}(\tilde{\bm{\xi}})\not=p\_{\max}(\tilde{\bm{\xi}}).
Therefore, pi​(𝝃~)​(1−pi​(𝝃~))⩽m⩽ϵ~​(1−ϵ~)subscript𝑝𝑖~𝝃1subscript𝑝𝑖~𝝃𝑚~italic-ϵ1~italic-ϵp\_{i}(\tilde{\bm{\xi}})(1-p\_{i}(\tilde{\bm{\xi}}))\leqslant m\leqslant\tilde{\epsilon}(1-\tilde{\epsilon}) for all i𝑖i.
Next we use the derived upper and lower bound on ϵ~~italic-ϵ\tilde{\epsilon}
in previous Eq. ([61](#A1.E61 "In Lemma A2. ‣ A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) in Lemma [A2](#ThmlemmaA2 "Lemma A2. ‣ A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J​(𝝃~)‖2subscriptnormJ~𝝃2\displaystyle{{\left\|\mathrm{J}(\tilde{\bm{\xi}})\right\|}}\_{2}\ | ⩽ 2​β​N​M2​ϵ~− 2​ϵ~2​β​N​M2absent2𝛽𝑁superscript𝑀2~italic-ϵ2superscript~italic-ϵ2𝛽𝑁superscript𝑀2\displaystyle\leqslant\ 2\ \beta\ N\ M^{2}\ \tilde{\epsilon}\ -\ 2\ \tilde{\epsilon}^{2}\ \beta\ N\ M^{2} |  | (160) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽ 2​β​N​M2​(N−1)​exp⁡(−β​(Δi−2β​N))−absentlimit-from2𝛽𝑁superscript𝑀2𝑁1𝛽subscriptΔ𝑖2𝛽𝑁\displaystyle\leqslant\ 2\ \beta\ N\ M^{2}\ (N-1)\ \exp\left(-\ \beta\ \left(\Delta\_{i}\ -\frac{2}{\beta\ N}\right)\right)\ - |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​(N−1)2​exp⁡(− 2​β​(Δi+2β​N))​β​N​M2.2superscript𝑁122𝛽subscriptΔ𝑖2𝛽𝑁𝛽𝑁superscript𝑀2\displaystyle 2\ (N-1)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{i}\ +\ \frac{2}{\beta\ N}\right)\right)\ \beta\ N\ M^{2}\ . |  |

The bound Eq. ([160](#A1.E160 "In Proof. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) holds for the mean JmsuperscriptJ𝑚\mathrm{J}^{m}, too,
since it averages over J​(𝝃~)J~𝝃\mathrm{J}(\tilde{\bm{\xi}}):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jm‖2subscriptnormsuperscriptJ𝑚2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ | ⩽ 2​β​N​M2​(N−1)​exp⁡(−β​(Δi−2β​N))−absentlimit-from2𝛽𝑁superscript𝑀2𝑁1𝛽subscriptΔ𝑖2𝛽𝑁\displaystyle\leqslant\ 2\ \beta\ N\ M^{2}\ (N-1)\ \exp\left(-\ \beta\ \left(\Delta\_{i}\ -\ \frac{2}{\beta\ N}\right)\right)\ - |  | (161) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​(N−1)2​exp⁡(− 2​β​(Δi+2β​N))​β​N​M2.2superscript𝑁122𝛽subscriptΔ𝑖2𝛽𝑁𝛽𝑁superscript𝑀2\displaystyle 2\ (N-1)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{i}\ +\ \frac{2}{\beta\ N}\right)\right)\ \beta\ N\ M^{2}\ . |  |

The assumption of the lemma is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | ≥2β​N+1β​ln⁡(2​(N−1)​N​β​M2),absent2𝛽𝑁1𝛽2𝑁1𝑁𝛽superscript𝑀2\displaystyle\geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right)\ , |  | (162) |

This is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δi−2β​NsubscriptΔ𝑖2𝛽𝑁\displaystyle\Delta\_{i}\ -\ \frac{2}{\beta\ N}\ | ≥1β​ln⁡(2​(N−1)​N​β​M2),absent1𝛽2𝑁1𝑁𝛽superscript𝑀2\displaystyle\geq\ \frac{1}{\beta}\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right)\ , |  | (163) |

Therefore, the spectral norm ‖J‖2subscriptnormJ2{{\left\|\mathrm{J}\right\|}}\_{2} can be bounded by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ‖Jm‖2⩽ 2​β​(N−1)​exp⁡(−β​1β​ln⁡(2​(N−1)​N​β​M2))​N​M2−subscriptnormsuperscriptJ𝑚2limit-from2𝛽𝑁1𝛽1𝛽2𝑁1𝑁𝛽superscript𝑀2𝑁superscript𝑀2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ \leqslant\ 2\ \beta\ (N-1)\ \exp\left(-\ \beta\ \frac{1}{\beta}\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right)\right)\ N\ M^{2}\ - |  | (164) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​(N−1)2​exp⁡(− 2​β​(Δi+2β​N))​β​N​M22superscript𝑁122𝛽subscriptΔ𝑖2𝛽𝑁𝛽𝑁superscript𝑀2\displaystyle 2\ (N-1)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{i}\ +\ \frac{2}{\beta\ N}\right)\right)\ \beta\ N\ M^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | = 2​β​(N−1)​12​(N−1)​N​β​M2​N​M2−absentlimit-from2𝛽𝑁112𝑁1𝑁𝛽superscript𝑀2𝑁superscript𝑀2\displaystyle=\ \ 2\ \beta\ (N-1)\ \frac{1}{2\ (N-1)\ N\ \beta\ M^{2}}\ N\ M^{2}\ - |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​(N−1)2​exp⁡(− 2​β​(Δi+2β​N))​β​N​M22superscript𝑁122𝛽subscriptΔ𝑖2𝛽𝑁𝛽𝑁superscript𝑀2\displaystyle 2\ (N-1)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{i}\ +\ \frac{2}{\beta\ N}\right)\right)\ \beta\ N\ M^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | = 1−2​(N−1)2​exp⁡(− 2​β​(Δi+2β​N))​β​N​M2<1.absent12superscript𝑁122𝛽subscriptΔ𝑖2𝛽𝑁𝛽𝑁superscript𝑀21\displaystyle=\ 1\ -2\ (N-1)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{i}\ +\ \frac{2}{\beta\ N}\right)\right)\ \beta\ N\ M^{2}\ <1\ . |  |

Therefore, f𝑓f is a contraction mapping in SisubscriptS𝑖\mathrm{S}\_{i}.
∎

•Banach Fixed Point Theorem.
Now we have all ingredients to apply Banach fixed point theorem.

###### Lemma A7.

Assume that

1. A1:

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | ≥2β​N+1β​ln⁡(2​(N−1)​N​β​M2),absent2𝛽𝑁1𝛽2𝑁1𝑁𝛽superscript𝑀2\displaystyle\geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right)\ , |  | (165) |

then f𝑓f has a fixed point in SisubscriptS𝑖\mathrm{S}\_{i}.

###### Proof.

We use Banach fixed point theorem:
Lemma [A5](#ThmlemmaA5 "Lemma A5. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") says that f𝑓f maps from SisubscriptS𝑖\mathrm{S}\_{i} into SisubscriptS𝑖\mathrm{S}\_{i}.
Lemma [A6](#ThmlemmaA6 "Lemma A6. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") says that f𝑓f is a contraction mapping in
SisubscriptS𝑖\mathrm{S}\_{i}.
∎

•Contraction mapping with a fixed point.

We have shown that a fixed point exists. We want to know
how fast the iteration converges to the fixed point.
Let 𝒙i∗superscriptsubscript𝒙𝑖\bm{x}\_{i}^{\*} be the fixed point of the iteration f𝑓f in the sphere SisubscriptS𝑖\mathrm{S}\_{i}.
Using the mean value theorem Lemma [A32](#ThmlemmaA32 "Lemma A32 (Mean Value Theorem). ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we have with
Jm=∫01J​(λ​𝝃+(1−λ)​𝒙i∗)​dλsuperscriptJ𝑚superscriptsubscript01J𝜆𝝃1𝜆superscriptsubscript𝒙𝑖differential-d𝜆\mathrm{J}^{m}=\int\_{0}^{1}\mathrm{J}(\lambda\bm{\xi}+(1-\lambda)\bm{x}\_{i}^{\*})\ \mathrm{d}\lambda:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i∗‖norm𝑓𝝃superscriptsubscript𝒙𝑖\displaystyle{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}\ | =‖f​(𝝃)−f​(𝒙i∗)‖⩽‖Jm‖2​‖𝝃−𝒙i∗‖absentnorm𝑓𝝃𝑓superscriptsubscript𝒙𝑖subscriptnormsuperscriptJ𝑚2norm𝝃superscriptsubscript𝒙𝑖\displaystyle=\ {{\left\|f(\bm{\xi})\ -\ f(\bm{x}\_{i}^{\*})\right\|}}\ \leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}^{\*}\right\|}} |  | (166) |

According to Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"),
if pmax=maxi⁡pi≥1−ϵsubscript𝑝subscript𝑖subscript𝑝𝑖1italic-ϵp\_{\max}=\max\_{i}p\_{i}\geq 1-\epsilon for all
𝒙~=λ​𝝃+(1−λ)​𝒙i∗~𝒙𝜆𝝃1𝜆superscriptsubscript𝒙𝑖\tilde{\bm{x}}=\lambda\bm{\xi}+(1-\lambda)\bm{x}\_{i}^{\*},
then the spectral norm of
the Jacobian is bounded by

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js​(𝒙~)‖2subscriptnormsubscriptJ𝑠~𝒙2\displaystyle{{\left\|\mathrm{J}\_{s}(\tilde{\bm{x}})\right\|}}\_{2}\ | < 2​ϵ​β.absent2italic-ϵ𝛽\displaystyle<\ 2\ \epsilon\ \beta\ . |  | (167) |

The norm of Jacobian at 𝒙~~𝒙\tilde{\bm{x}} is bounded

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J​(𝒙~)‖2subscriptnormJ~𝒙2\displaystyle{{\left\|\mathrm{J}(\tilde{\bm{x}})\right\|}}\_{2}\ | ⩽ 2​β​‖𝑿‖22​ϵ⩽ 2​β​N​M2​ϵ.absent2𝛽superscriptsubscriptnorm𝑿22italic-ϵ2𝛽𝑁superscript𝑀2italic-ϵ\displaystyle\leqslant\ 2\ \beta\ {{\left\|\bm{X}\right\|}}\_{2}^{2}\ \epsilon\ \leqslant\ 2\ \beta\ NM^{2}\ \epsilon\ . |  | (168) |

We used that
the spectral norm ∥.∥2{{\left\|.\right\|}}\_{2} is bounded by the Frobenius norm
∥.∥F{{\left\|.\right\|}}\_{F} which can be expressed by the norm squared of its
column vectors:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝑿‖2subscriptnorm𝑿2\displaystyle{{\left\|\bm{X}\right\|}}\_{2}\ | ⩽‖𝑿‖F=∑i‖𝒙i‖2.absentsubscriptnorm𝑿𝐹subscript𝑖superscriptnormsubscript𝒙𝑖2\displaystyle\leqslant\ {{\left\|\bm{X}\right\|}}\_{F}\ =\ \sqrt{\sum\_{i}{{\left\|\bm{x}\_{i}\right\|}}^{2}}\ . |  | (169) |

Therefore

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝑿‖22superscriptsubscriptnorm𝑿22\displaystyle{{\left\|\bm{X}\right\|}}\_{2}^{2}\ | ⩽N​M2.absent𝑁superscript𝑀2\displaystyle\leqslant\ N\ M^{2}\ . |  | (170) |

The norm of Jacobian of the fixed point iteration is bounded

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jm‖2subscriptnormsuperscriptJ𝑚2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ | ⩽ 2​β​‖𝑿‖22​ϵ⩽ 2​β​N​M2​ϵ.absent2𝛽superscriptsubscriptnorm𝑿22italic-ϵ2𝛽𝑁superscript𝑀2italic-ϵ\displaystyle\leqslant\ 2\ \beta\ {{\left\|\bm{X}\right\|}}\_{2}^{2}\ \epsilon\ \leqslant\ 2\ \beta\ NM^{2}\ \epsilon\ . |  | (171) |

The separation of pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} from data 𝑿=(𝒙1,…,𝒙N)𝑿subscript𝒙1…subscript𝒙𝑁\bm{X}=(\bm{x}\_{1},\ldots,\bm{x}\_{N})
is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | =minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)=𝒙iT​𝒙i−maxj,j≠i⁡𝒙iT​𝒙j.absentsubscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖subscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right)\ =\ \bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \max\_{j,j\not=i}\bm{x}\_{i}^{T}\bm{x}\_{j}\ . |  | (172) |

We need the separation Δ~isubscript~Δ𝑖\tilde{\Delta}\_{i} of
𝒙~=λ​𝝃+(1−λ)​𝒙i∗~𝒙𝜆𝝃1𝜆superscriptsubscript𝒙𝑖\tilde{\bm{x}}=\lambda\bm{\xi}+(1-\lambda)\bm{x}\_{i}^{\*} from the data:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~isubscript~Δ𝑖\displaystyle\tilde{\Delta}\_{i}\ | =minj,j≠i⁡(𝒙~T​𝒙i−𝒙~T​𝒙j).absentsubscript  𝑗𝑗 𝑖superscript~𝒙𝑇subscript𝒙𝑖superscript~𝒙𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,j\not=i}\left(\tilde{\bm{x}}^{T}\bm{x}\_{i}\ -\ \tilde{\bm{x}}^{T}\bm{x}\_{j}\right)\ . |  | (173) |

We compute a lower bound on Δ~isubscript~Δ𝑖\tilde{\Delta}\_{i}.
Using the Cauchy-Schwarz inequality, we obtain for 1⩽j⩽N1𝑗𝑁1\leqslant j\leqslant N:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |𝒙~T​𝒙j−𝒙iT​𝒙j|superscript~𝒙𝑇subscript𝒙𝑗superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗\displaystyle\left|\tilde{\bm{x}}^{T}\bm{x}\_{j}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right| | ⩽‖𝒙~−𝒙i‖​‖𝒙j‖⩽‖𝒙~−𝒙i‖​M.absentnorm~𝒙subscript𝒙𝑖normsubscript𝒙𝑗norm~𝒙subscript𝒙𝑖𝑀\displaystyle\leqslant\ {{\left\|\tilde{\bm{x}}\ -\ \bm{x}\_{i}\right\|}}\ {{\left\|\bm{x}\_{j}\right\|}}\ \leqslant\ {{\left\|\tilde{\bm{x}}\ -\ \bm{x}\_{i}\right\|}}\ M\ . |  | (174) |

We have the lower bound

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~isubscript~Δ𝑖\displaystyle\tilde{\Delta}\_{i}\ | ≥minj,j≠i⁡((𝒙iT​𝒙i−‖𝒙~−𝒙i‖​M)−(𝒙iT​𝒙j+‖𝒙~−𝒙i‖​M))absentsubscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖norm~𝒙subscript𝒙𝑖𝑀superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗norm~𝒙subscript𝒙𝑖𝑀\displaystyle\geq\ \min\_{j,j\not=i}\left(\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ {{\left\|\tilde{\bm{x}}\ -\ \bm{x}\_{i}\right\|}}\ M\right)\ -\ \left(\bm{x}\_{i}^{T}\bm{x}\_{j}\ +\ {{\left\|\tilde{\bm{x}}\ -\ \bm{x}\_{i}\right\|}}\ M\right)\right) |  | (175) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =− 2​‖𝒙~−𝒙i‖​M+minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)=Δi− 2​‖𝒙~−𝒙i‖​M.absent2norm~𝒙subscript𝒙𝑖𝑀subscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗subscriptΔ𝑖2norm~𝒙subscript𝒙𝑖𝑀\displaystyle=\ -\ 2\ {{\left\|\tilde{\bm{x}}\ -\ \bm{x}\_{i}\right\|}}\ M\ +\ \min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right)\ =\ \Delta\_{i}\ -\ 2\ {{\left\|\tilde{\bm{x}}\ -\ \bm{x}\_{i}\right\|}}\ M\ . |  |

Since

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒙~−𝒙i‖norm~𝒙subscript𝒙𝑖\displaystyle{{\left\|\tilde{\bm{x}}\ -\ \bm{x}\_{i}\right\|}}\ | =‖λ​𝝃+(1−λ)​𝒙i∗−𝒙i‖absentnorm𝜆𝝃1𝜆superscriptsubscript𝒙𝑖subscript𝒙𝑖\displaystyle=\ {{\left\|\lambda\bm{\xi}+(1-\lambda)\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}} |  | (176) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽λ​‖𝝃−𝒙i‖+(1−λ)​‖𝒙i∗−𝒙i‖absent𝜆norm𝝃subscript𝒙𝑖1𝜆normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖\displaystyle\leqslant\ \lambda\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ +\ (1-\lambda)\ {{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖},absentnorm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖\displaystyle\leqslant\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ , |  |

we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~isubscript~Δ𝑖\displaystyle\tilde{\Delta}\_{i}\ | ≥Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M.absentsubscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle\geq\ \Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M\ . |  | (177) |

For the softmax component i𝑖i we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | [softmax​(β​𝑿T​𝝃~)]i=11+∑j≠iexp⁡(β​(𝝃~T​𝒙j−𝝃~T​𝒙i))subscriptdelimited-[]softmax𝛽superscript𝑿𝑇~𝝃𝑖11subscript𝑗𝑖𝛽superscript~𝝃𝑇subscript𝒙𝑗superscript~𝝃𝑇subscript𝒙𝑖\displaystyle[\mathrm{softmax}(\beta\ \bm{X}^{T}\tilde{\bm{\xi}})]\_{i}\ =\ \frac{1}{1\ +\ \sum\_{j\not=i}\exp(\beta\ (\tilde{\bm{\xi}}^{T}\bm{x}\_{j}\ -\ \tilde{\bm{\xi}}^{T}\bm{x}\_{i}))} |  | (178) |
|  |  |  |
| --- | --- | --- |
|  | ≥11+∑j≠iexp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M))absent11subscript𝑗𝑖𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle\geq\ \frac{1}{1\ +\ \sum\_{j\not=i}\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))} |  |
|  |  |  |
| --- | --- | --- |
|  | =11+(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M))absent11𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle=\ \frac{1}{1\ +\ (N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))} |  |
|  |  |  |
| --- | --- | --- |
|  | = 1−(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M))1+(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M))absent1𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀1𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle=\ 1\ -\ \frac{(N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))}{1\ +\ (N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))} |  |
|  |  |  |
| --- | --- | --- |
|  | ≥ 1−(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M))absent1𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle\geq\ 1\ -\ (N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M)) |  |
|  |  |  |
| --- | --- | --- |
|  | = 1−ϵ.absent1italic-ϵ\displaystyle=\ 1\ -\ \epsilon\ . |  |

Therefore

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵitalic-ϵ\displaystyle\epsilon\ | =(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M)).absent𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle=\ (N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ . |  | (179) |

We can bound the spectral norm of
the Jacobian, which upper bounds the Lipschitz constant:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jm‖2subscriptnormsuperscriptJ𝑚2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ | ⩽ 2​β​N​M2​(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M)).absent2𝛽𝑁superscript𝑀2𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle\leqslant\ 2\ \beta\ N\ M^{2}\ (N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ . |  | (180) |

For a contraction mapping we require

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Jm‖2< 1,subscriptnormsuperscriptJ𝑚21\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ <\ 1\ , |  | (181) |

which can be ensured by

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​β​N​M2​(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M))< 1.2𝛽𝑁superscript𝑀2𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀1\displaystyle 2\ \beta\ NM^{2}\ (N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ <\ 1\ . |  | (182) |

Solving this inequality for ΔisubscriptΔ𝑖\Delta\_{i} gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | > 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M+1β​ln⁡(2​(N−1)​N​β​M2).absent2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀1𝛽2𝑁1𝑁𝛽superscript𝑀2\displaystyle>\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M\ +\ \frac{1}{\beta}\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right)\ . |  | (183) |

In an environment around 𝒙i∗superscriptsubscript𝒙𝑖\bm{x}\_{i}^{\*} in
which Eq. ([183](#A1.E183 "In A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) holds, f𝑓f is a contraction mapping and every
point converges under the iteration f𝑓f to 𝒙i∗superscriptsubscript𝒙𝑖\bm{x}\_{i}^{\*} when the iteration stays in
the environment.
After every iteration the mapped point f​(𝝃)𝑓𝝃f(\bm{\xi}) is closer to the fixed point 𝒙i∗superscriptsubscript𝒙𝑖\bm{x}\_{i}^{\*}
than the original point 𝒙isubscript𝒙𝑖\bm{x}\_{i}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i∗‖norm𝑓𝝃superscriptsubscript𝒙𝑖\displaystyle{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}\ | ⩽‖Jm‖2​‖𝝃−𝒙i∗‖<‖𝝃−𝒙i∗‖.absentsubscriptnormsuperscriptJ𝑚2norm𝝃superscriptsubscript𝒙𝑖norm𝝃superscriptsubscript𝒙𝑖\displaystyle\leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}^{\*}\right\|}}\ <\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}^{\*}\right\|}}\ . |  | (184) |

Using

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i∗‖⩽‖Jm‖2​‖𝝃−𝒙i∗‖⩽‖Jm‖2​‖𝝃−f​(𝝃)‖+‖Jm‖2​‖f​(𝝃)−𝒙i∗‖,norm𝑓𝝃superscriptsubscript𝒙𝑖subscriptnormsuperscriptJ𝑚2norm𝝃superscriptsubscript𝒙𝑖subscriptnormsuperscriptJ𝑚2norm𝝃𝑓𝝃subscriptnormsuperscriptJ𝑚2norm𝑓𝝃superscriptsubscript𝒙𝑖\displaystyle{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}\ \leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}^{\*}\right\|}}\ \leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ f(\bm{\xi})\right\|}}\ +\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}\ , |  | (185) |

we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i∗‖⩽‖Jm‖21−‖Jm‖2​‖𝝃−f​(𝝃)‖.norm𝑓𝝃superscriptsubscript𝒙𝑖subscriptnormsuperscriptJ𝑚21subscriptnormsuperscriptJ𝑚2norm𝝃𝑓𝝃\displaystyle{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}\ \leqslant\ \frac{{{\left\|\mathrm{J}^{m}\right\|}}\_{2}}{1\ -\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}}\ {{\left\|\bm{\xi}\ -\ f(\bm{\xi})\right\|}}\ . |  | (186) |

For large ΔisubscriptΔ𝑖\Delta\_{i} the iteration is close to the fixed point even after
one update.
This has been confirmed in several experiments.

##### A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns.

The proof concept is the same as for a single pattern
but now for the arithmetic mean of similar patterns.

•Bound on the Jacobian.

The Jacobian of the fixed point iteration is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | JJ\displaystyle\mathrm{J}\ | =β​𝑿​(diag​(𝒑)−𝒑​𝒑T)​𝑿T=𝑿​Js​𝑿T.absent𝛽𝑿diag𝒑𝒑superscript𝒑𝑇superscript𝑿𝑇𝑿subscriptJ𝑠superscript𝑿𝑇\displaystyle=\ \beta\ \bm{X}\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\bm{X}^{T}\ =\ \bm{X}\mathrm{J}\_{s}\bm{X}^{T}\ . |  | (187) |

If we consider pisubscript𝑝𝑖p\_{i} as the probability of selecting
the vector 𝒙isubscript𝒙𝑖\bm{x}\_{i}, then we can define
expectations as E𝒑​[f​(𝒙)]=∑i=1Npi​f​(𝒙i)subscriptE𝒑delimited-[]𝑓𝒙superscriptsubscript𝑖1𝑁subscript𝑝𝑖𝑓subscript𝒙𝑖\mathbf{\mathrm{E}}\_{\bm{p}}[f(\bm{x})]=\sum\_{i=1}^{N}p\_{i}f(\bm{x}\_{i}).
In this setting the matrix

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑿​(diag​(𝒑)−𝒑​𝒑T)​𝑿T𝑿diag𝒑𝒑superscript𝒑𝑇superscript𝑿𝑇\displaystyle\bm{X}\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\bm{X}^{T} |  | (188) |

is the covariance matrix of data 𝑿𝑿\bm{X} when its
vectors are selected according
to the probability 𝒑𝒑\bm{p}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑿​(diag​(𝒑)−𝒑​𝒑T)​𝑿T=𝑿​diag​(𝒑)​𝑿T−𝑿​𝒑​𝒑T​𝑿T𝑿diag𝒑𝒑superscript𝒑𝑇superscript𝑿𝑇𝑿diag𝒑superscript𝑿𝑇𝑿𝒑superscript𝒑𝑇superscript𝑿𝑇\displaystyle\bm{X}\left(\mathrm{diag}(\bm{p})\ -\ \bm{p}\bm{p}^{T}\right)\bm{X}^{T}\ =\ \bm{X}\mathrm{diag}(\bm{p})\bm{X}^{T}\ -\ \bm{X}\bm{p}\bm{p}^{T}\bm{X}^{T} |  | (189) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =∑i=1Npi​𝒙i​𝒙iT−(∑i=1Npi​𝒙i)​(∑i=1Npi​𝒙i)Tabsentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i}\right)\left(\sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i}\right)^{T} |  | (190) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =E𝒑​[𝒙​𝒙T]−E𝒑​[𝒙]​E𝒑​[𝒙]T=Var𝒑​[𝒙],absentsubscriptE𝒑delimited-[]𝒙superscript𝒙𝑇subscriptE𝒑delimited-[]𝒙subscriptE𝒑superscriptdelimited-[]𝒙𝑇subscriptVar𝒑delimited-[]𝒙\displaystyle=\ \mathbf{\mathrm{E}}\_{\bm{p}}[\bm{x}\ \bm{x}^{T}]\ -\ \mathbf{\mathrm{E}}\_{\bm{p}}[\bm{x}]\ \mathbf{\mathrm{E}}\_{\bm{p}}[\bm{x}]^{T}\ =\ \mathbf{\mathrm{Var}}\_{\bm{p}}[\bm{x}]\ , |  | (191) |

therefore we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | JJ\displaystyle\mathrm{J}\ | =β​Var𝒑​[𝒙].absent𝛽subscriptVar𝒑delimited-[]𝒙\displaystyle=\ \beta\ \mathbf{\mathrm{Var}}\_{\bm{p}}[\bm{x}]\ . |  | (192) |

We now elaborate more on this interpretation as variance.
Specifically the singular values of JJ\mathrm{J} (or in other words: the
covariance) should be reasonably small.
The singular values are the key to ensure convergence of the iteration
Eq. ([57](#A1.E57 "In A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
Next we present some thoughts.

1. 1.

   It’s clear that the largest eigenvalue of the covariance matrix
   (equal to the largest singular value)
   is the variance in the direction of the eigenvector
   associated with the largest eigenvalue.
2. 2.

   Furthermore the variance goes to zero as one pisubscript𝑝𝑖p\_{i} goes to
   one, since only one pattern is chosen and there is
   no variance.
3. 3.

   The variance is reasonable small if all patterns
   are chosen with equal probability.
4. 4.

   The variance is small if few similar patterns are
   chosen with high probability.
   If the patterns are sufficient similar, then
   the spectral norm of the covariance matrix is
   smaller than one.

The first three issues have already been adressed.
Now we focus on the last one in greater detail.
We assume that the first l𝑙l patterns
are much more probable (and similar
to one another) than the other patterns.
Therefore, we define:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M𝑀\displaystyle M\ | :=maxi⁡‖𝒙i‖,assignabsentsubscript𝑖normsubscript𝒙𝑖\displaystyle:=\ \max\_{i}{{\left\|\bm{x}\_{i}\right\|}}\ , |  | (193) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | γ𝛾\displaystyle\gamma\ | =∑i=l+1Npi⩽ϵ,absentsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖italic-ϵ\displaystyle=\ \sum\_{i=l+1}^{N}p\_{i}\ \leqslant\ \epsilon\ , |  | (194) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1−γ1𝛾\displaystyle 1-\gamma\ | =∑i=1lpi≥ 1−ϵ,absentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖1italic-ϵ\displaystyle=\ \sum\_{i=1}^{l}p\_{i}\ \geq\ 1\ -\ \epsilon\ , |  | (195) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | p~isubscript~𝑝𝑖\displaystyle\tilde{p}\_{i}\ | :=pi1−γ⩽pi/(1−ϵ),assignabsentsubscript𝑝𝑖1𝛾subscript𝑝𝑖1italic-ϵ\displaystyle:=\ \frac{p\_{i}}{1-\gamma}\ \leqslant\ p\_{i}/(1-\epsilon)\ , |  | (196) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i=1lp~isuperscriptsubscript𝑖1𝑙subscript~𝑝𝑖\displaystyle\sum\_{i=1}^{l}\tilde{p}\_{i}\ | = 1,absent1\displaystyle=\ 1\ , |  | (197) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒎𝒙subscript𝒎𝒙\displaystyle\bm{m}\_{\bm{x}}\ | =1l​∑i=1l𝒙i,absent1𝑙superscriptsubscript𝑖1𝑙subscript𝒙𝑖\displaystyle=\ \frac{1}{l}\ \sum\_{i=1}^{l}\ \bm{x}\_{i}\ , |  | (198) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | mmaxsubscript𝑚\displaystyle m\_{\max}\ | =max1⩽i⩽l⁡‖𝒙i−𝒎𝒙‖.absentsubscript1𝑖𝑙normsubscript𝒙𝑖subscript𝒎𝒙\displaystyle=\ \max\_{1\leqslant i\leqslant l}{{\left\|\bm{x}\_{i}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ . |  | (199) |

M𝑀M is an upper bound on the Euclidean norm of the patterns, which are vectors.
ϵitalic-ϵ\epsilon is an upper bound on the probability γ𝛾\gamma of not choosing one of the first l𝑙l patterns, while
1−ϵ1italic-ϵ1-\epsilon is a lower bound the probability (1−γ)1𝛾(1-\gamma)
of choosing one of the first l𝑙l patterns.
𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} is the arithmetic mean (the center) of the first l𝑙l
patterns.
mmaxsubscript𝑚m\_{\max} is the maximal distance of the patterns to the center 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} .
𝒑~~𝒑\tilde{\bm{p}} is the probability 𝒑𝒑\bm{p}
normalized for the first l𝑙l patterns.

The variance of the first l𝑙l patterns is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Varp~​[𝒙1:l]subscriptVar~𝑝delimited-[]subscript𝒙:1𝑙\displaystyle\mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}]\ | =∑i=1lp~i​𝒙i​𝒙iT−(∑i=1lp~i​𝒙i)​(∑i=1lp~i​𝒙i)Tabsentsuperscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{l}\tilde{p}\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=1}^{l}\tilde{p}\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}\tilde{p}\_{i}\ \bm{x}\_{i}\right)^{T} |  | (200) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1lp~i​(𝒙i−∑i=1lp~i​𝒙i)​(𝒙i−∑i=1lp~i​𝒙i)T.absentsuperscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖superscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖superscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{l}\tilde{p}\_{i}\ \left(\bm{x}\_{i}\ -\ \sum\_{i=1}^{l}\tilde{p}\_{i}\bm{x}\_{i}\right)\ \left(\bm{x}\_{i}\ -\ \sum\_{i=1}^{l}\tilde{p}\_{i}\bm{x}\_{i}\right)^{T}\ . |  |

###### Lemma A8.

With the definitions in Eq. ([193](#A1.E193 "In A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
to Eq. ([200](#A1.E200 "In A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")),
the following bounds on the norm ‖J‖2subscriptnormJ2{{\left\|\mathrm{J}\right\|}}\_{2} of the
Jacobian of the fixed point iteration hold.
The γ𝛾\gamma-bound for ‖J‖2subscriptnormJ2{{\left\|\mathrm{J}\right\|}}\_{2} is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽β​((1−γ)​mmax2+γ​ 2​(2−γ)​M2)absent𝛽1𝛾superscriptsubscript𝑚2𝛾22𝛾superscript𝑀2\displaystyle\leqslant\ \beta\left((1-\gamma)\ m\_{\max}^{2}\ +\ \gamma\ 2\ (2\ -\ \gamma)\ M^{2}\right) |  | (201) |

and the ϵitalic-ϵ\epsilon-bound for ‖J‖2subscriptnormJ2{{\left\|\mathrm{J}\right\|}}\_{2} is:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽β​(mmax2+ϵ​ 2​(2−ϵ)​M2).absent𝛽superscriptsubscript𝑚2italic-ϵ22italic-ϵsuperscript𝑀2\displaystyle\leqslant\ \beta\left(\ m\_{\max}^{2}\ +\ \epsilon\ 2\ (2\ -\ \epsilon)\ M^{2}\right)\ . |  | (202) |

###### Proof.

The variance Varp~​[𝒙1:l]subscriptVar~𝑝delimited-[]subscript𝒙:1𝑙\mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}] can be expressed as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | (1−γ)​Varp~​[𝒙1:l]=∑i=1lpi​(𝒙i−11−γ​∑i=1lpi​𝒙i)​(𝒙i−11−γ​∑i=1lpi​𝒙i)T1𝛾subscriptVar~𝑝delimited-[]subscript𝒙:1𝑙superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖11𝛾superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖11𝛾superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle(1-\gamma)\ \mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}]\ =\ \sum\_{i=1}^{l}p\_{i}\ \left(\bm{x}\_{i}\ -\ \frac{1}{1-\gamma}\ \sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\bm{x}\_{i}\ -\ \frac{1}{1-\gamma}\ \sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T} |  | (203) |
|  |  |  |
| --- | --- | --- |
|  | =∑i=1lpi​𝒙i​𝒙iT−(∑i=1lpi​𝒙i)​11−γ​(∑i=1lpi​𝒙i)T−11−γ​(∑i=1lpi​𝒙i)​(∑i=1lpi​𝒙i)Tabsentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖11𝛾superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇11𝛾superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \frac{1}{1-\gamma}\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\ -\ \frac{1}{1-\gamma}\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T} |  |
|  |  |  |
| --- | --- | --- |
|  | +∑i=1lpi(1−γ)2​(∑i=1lpi​𝒙i)​(∑i=1lpi​𝒙i)T=∑i=1lpi​𝒙i​𝒙iT−11−γ​(∑i=1lpi​𝒙i)​(∑i=1lpi​𝒙i)Tsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖superscript1𝛾2superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇11𝛾superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle+\ \frac{\sum\_{i=1}^{l}p\_{i}}{(1-\gamma)^{2}}\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\ =\ \sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \frac{1}{1-\gamma}\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T} |  |
|  |  |  |
| --- | --- | --- |
|  | =∑i=1lpi​𝒙i​𝒙iT−(∑i=1lpi​𝒙i)​(∑i=1lpi​𝒙i)T+(1−11−γ)​(∑i=1lpi​𝒙i)​(∑i=1lpi​𝒙i)Tabsentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇111𝛾superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\ +\ \left(1\ -\ \frac{1}{1-\gamma}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T} |  |
|  |  |  |
| --- | --- | --- |
|  | =∑i=1lpi​𝒙i​𝒙iT−(∑i=1lpi​𝒙i)​(∑i=1lpi​𝒙i)T−γ1−γ​(∑i=1lpi​𝒙i)​(∑i=1lpi​𝒙i)T.absentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇𝛾1𝛾superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\ -\ \frac{\gamma}{1-\gamma}\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\ . |  |

Therefore, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i=1lpi​𝒙i​𝒙iT−(∑i=1lpi​𝒙i)​(∑i=1lpi​𝒙i)Tsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T} |  | (204) |
|  |  |  |
| --- | --- | --- |
|  | =(1−γ)​Varp~​[𝒙1:l]+γ1−γ​(∑i=1lpi​𝒙i)​(∑i=1lpi​𝒙i)T.absent1𝛾subscriptVar~𝑝delimited-[]subscript𝒙:1𝑙𝛾1𝛾superscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ (1-\gamma)\ \mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}]\ +\ \frac{\gamma}{1-\gamma}\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\ . |  |

We now can reformulate the Jacobian JJ\mathrm{J}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | JJ\displaystyle\mathrm{J}\ | =β(∑i=1lpi𝒙i𝒙iT+∑i=l+1Npi𝒙i𝒙iT\displaystyle=\ \beta\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ +\ \sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\right. |  | (205) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −(∑i=1lpi𝒙i+∑i=l+1Npi𝒙i)(∑i=1lpi𝒙i+∑i=l+1Npi𝒙i)T)\displaystyle-\ \left.\left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\ +\ \sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)\left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\ +\ \sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)^{T}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β(∑i=1lpi𝒙i𝒙iT−(∑i=1lpi𝒙i)(∑i=1lpi𝒙i)T\displaystyle=\ \beta\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑i=l+1Npi​𝒙i​𝒙iT−(∑i=l+1Npi​𝒙i)​(∑i=l+1Npi​𝒙i)Tsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle+\ \left.\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)^{T}\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −(∑i=1lpi𝒙i)(∑i=l+1Npi𝒙i)T−(∑i=l+1Npi𝒙i)(∑i=1lpi𝒙i)T)\displaystyle-\ \left.\left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)^{T}\ -\ \left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)\left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β((1−γ)Varp~[𝒙1:l]+γ1−γ(∑i=1lpi𝒙i)(∑i=1lpi𝒙i)T\displaystyle=\ \beta\ \left((1-\gamma)\ \mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}]\ +\ \frac{\gamma}{1-\gamma}\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑i=l+1Npi​𝒙i​𝒙iT−(∑i=l+1Npi​𝒙i)​(∑i=l+1Npi​𝒙i)Tsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖𝑇\displaystyle+\ \left.\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)^{T}\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −(∑i=1lpi𝒙i)(∑i=l+1Npi𝒙i)T−(∑i=l+1Npi𝒙i)(∑i=1lpi𝒙i)T).\displaystyle-\ \left.\left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)^{T}\ -\ \left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)\left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\right)\ . |  |

The spectral norm of an outer product of two vectors
is the product of the Euclidean norms of the vectors:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒂​𝒃T‖2subscriptnorm𝒂superscript𝒃𝑇2\displaystyle{{\left\|\bm{a}\bm{b}^{T}\right\|}}\_{2}\ | =λmax​(𝒃​𝒂T​𝒂​𝒃T)=‖𝒂‖​λmax​(𝒃​𝒃T)=‖𝒂‖​‖𝒃‖,absentsubscript𝜆𝒃superscript𝒂𝑇𝒂superscript𝒃𝑇norm𝒂subscript𝜆𝒃superscript𝒃𝑇norm𝒂norm𝒃\displaystyle=\ \sqrt{\lambda\_{\max}(\bm{b}\bm{a}^{T}\bm{a}\bm{b}^{T})}\ =\ {{\left\|\bm{a}\right\|}}\ \sqrt{\lambda\_{\max}(\bm{b}\bm{b}^{T})}\ =\ {{\left\|\bm{a}\right\|}}\ {{\left\|\bm{b}\right\|}}\ , |  | (206) |

since 𝒃​𝒃T𝒃superscript𝒃𝑇\bm{b}\bm{b}^{T} has eigenvector 𝒃/‖𝒃‖𝒃norm𝒃\bm{b}/{{\left\|\bm{b}\right\|}} with
eigenvalue ‖𝒃‖2superscriptnorm𝒃2{{\left\|\bm{b}\right\|}}^{2} and otherwise zero eigenvalues.

We now bound the norms of some matrices and vectors:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖∑i=1lpi​𝒙i‖normsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖\displaystyle{{\left\|\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right\|}}\ | ⩽∑i=1lpi​‖𝒙i‖⩽(1−γ)​M,absentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖normsubscript𝒙𝑖1𝛾𝑀\displaystyle\leqslant\ \sum\_{i=1}^{l}p\_{i}\ {{\left\|\bm{x}\_{i}\right\|}}\ \leqslant\ (1-\gamma)\ M\ , |  | (207) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖∑i=l+1Npi​𝒙i‖normsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖\displaystyle{{\left\|\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right\|}}\ | ⩽∑i=l+1Npi​‖𝒙i‖⩽γ​M,absentsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖normsubscript𝒙𝑖𝛾𝑀\displaystyle\leqslant\ \sum\_{i=l+1}^{N}p\_{i}\ {{\left\|\bm{x}\_{i}\right\|}}\ \leqslant\ \gamma\ M\ , |  | (208) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖∑i=l+1Npi​𝒙i​𝒙iT‖2subscriptnormsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇2\displaystyle{{\left\|\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\right\|}}\_{2}\ | ⩽∑i=l+1Npi​‖𝒙i​𝒙iT‖2=∑i=l+1Npi​‖𝒙i‖2⩽∑i=l+1Npi​M2=γ​M2.absentsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscriptnormsubscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇2superscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖superscriptnormsubscript𝒙𝑖2superscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖superscript𝑀2𝛾superscript𝑀2\displaystyle\leqslant\ \sum\_{i=l+1}^{N}p\_{i}\ {{\left\|\bm{x}\_{i}\ \bm{x}\_{i}^{T}\right\|}}\_{2}\ =\ \sum\_{i=l+1}^{N}p\_{i}\ {{\left\|\bm{x}\_{i}\right\|}}^{2}\ \leqslant\ \sum\_{i=l+1}^{N}p\_{i}\ M^{2}\ =\ \gamma\ M^{2}\ . |  | (209) |

In order to bound the variance of the
first l𝑙l patterns,
we compute the vector 𝒂𝒂\bm{a} that minimizes

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(𝒂)𝑓𝒂\displaystyle f(\bm{a})\ | =∑i=1lpi​‖𝒙i−𝒂‖2=∑i=1lpi​(𝒙i−𝒂)T​(𝒙i−𝒂).absentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖superscriptnormsubscript𝒙𝑖𝒂2superscriptsubscript𝑖1𝑙subscript𝑝𝑖superscriptsubscript𝒙𝑖𝒂𝑇subscript𝒙𝑖𝒂\displaystyle=\ \sum\_{i=1}^{l}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bm{a}\right\|}}^{2}\ =\ \sum\_{i=1}^{l}p\_{i}(\bm{x}\_{i}\ -\ \bm{a})^{T}(\bm{x}\_{i}\ -\ \bm{a})\ . |  | (210) |

The solution to

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂f​(𝒂)∂𝒂𝑓𝒂𝒂\displaystyle\frac{\partial f(\bm{a})}{\partial\bm{a}}\ | = 2​∑i=1Npi​(𝒂−𝒙i)= 0absent2superscriptsubscript𝑖1𝑁subscript𝑝𝑖𝒂subscript𝒙𝑖 0\displaystyle=\ 2\ \sum\_{i=1}^{N}p\_{i}(\bm{a}\ -\ \bm{x}\_{i})\ =\ 0 |  | (211) |

is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒂𝒂\displaystyle\bm{a}\ | =∑i=1Npi​𝒙i.absentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖\displaystyle=\ \sum\_{i=1}^{N}p\_{i}\bm{x}\_{i}\ . |  | (212) |

The Hessian of f𝑓f is positive definite since

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂2f​(𝒂)∂𝒂2superscript2𝑓𝒂superscript𝒂2\displaystyle\frac{\partial^{2}f(\bm{a})}{\partial\bm{a}^{2}}\ | = 2​∑i=1Npi​𝑰= 2​𝑰absent2superscriptsubscript𝑖1𝑁subscript𝑝𝑖𝑰2𝑰\displaystyle=\ 2\ \sum\_{i=1}^{N}p\_{i}\ \bm{I}\ =\ 2\ \bm{I} |  | (213) |

and f𝑓f is a convex function.
Hence, the mean

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒙¯¯𝒙\displaystyle\bar{\bm{x}}\ | :=∑i=1Npi​𝒙iassignabsentsuperscriptsubscript𝑖1𝑁subscript𝑝𝑖subscript𝒙𝑖\displaystyle:=\ \sum\_{i=1}^{N}p\_{i}\ \bm{x}\_{i} |  | (214) |

minimizes
∑i=1Npi​‖𝒙i−𝒂‖2superscriptsubscript𝑖1𝑁subscript𝑝𝑖superscriptnormsubscript𝒙𝑖𝒂2\sum\_{i=1}^{N}p\_{i}{{\left\|\bm{x}\_{i}-\bm{a}\right\|}}^{2}.
Therefore, we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i=1lpi​‖𝒙i−𝒙¯‖2superscriptsubscript𝑖1𝑙subscript𝑝𝑖superscriptnormsubscript𝒙𝑖¯𝒙2\displaystyle\sum\_{i=1}^{l}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bar{\bm{x}}\right\|}}^{2}\ | ⩽∑i=1lpi​‖𝒙i−𝒎𝒙‖2⩽(1−γ)​mmax2.absentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖superscriptnormsubscript𝒙𝑖subscript𝒎𝒙21𝛾superscriptsubscript𝑚2\displaystyle\leqslant\ \sum\_{i=1}^{l}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bm{m}\_{\bm{x}}\right\|}}^{2}\ \leqslant\ (1\ -\ \gamma)\ m\_{\max}^{2}\ . |  | (215) |

We now bound the variance on the first l𝑙l patterns:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (1−γ)​‖Varp~​[𝒙1:l]‖21𝛾subscriptnormsubscriptVar~𝑝delimited-[]subscript𝒙:1𝑙2\displaystyle(1-\gamma)\ {{\left\|\mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}]\right\|}}\_{2}\ | ⩽∑i=1lpi​‖(𝒙i−𝒙¯)​(𝒙i−𝒙¯)T‖2absentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscriptnormsubscript𝒙𝑖¯𝒙superscriptsubscript𝒙𝑖¯𝒙𝑇2\displaystyle\leqslant\ \sum\_{i=1}^{l}p\_{i}{{\left\|\left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)\ \left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)^{T}\right\|}}\_{2} |  | (216) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1lpi​‖𝒙i−𝒙¯‖2⩽∑i=1lpi​‖𝒙i−𝒎𝒙‖2⩽(1−γ)​mmax2.absentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖superscriptnormsubscript𝒙𝑖¯𝒙2superscriptsubscript𝑖1𝑙subscript𝑝𝑖superscriptnormsubscript𝒙𝑖subscript𝒎𝒙21𝛾superscriptsubscript𝑚2\displaystyle=\ \sum\_{i=1}^{l}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bar{\bm{x}}\right\|}}^{2}\ \leqslant\ \sum\_{i=1}^{l}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bm{m}\_{\bm{x}}\right\|}}^{2}\ \leqslant\ (1\ -\ \gamma)\ m\_{\max}^{2}\ . |  |

We obtain for the spectral norm of JJ\mathrm{J}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽β((1−γ)∥Varp~[𝒙1:l]∥2\displaystyle\leqslant\ \beta\left((1-\gamma)\ {{\left\|\mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}]\right\|}}\_{2}\right. |  | (217) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +γ1−γ​‖(∑i=1lpi​𝒙i)​(∑i=1lpi​𝒙i)T‖2𝛾1𝛾subscriptnormsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖𝑇2\displaystyle+\ \left.\frac{\gamma}{1-\gamma}\ {{\left\|\left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\right\|}}\_{2}\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +‖∑i=l+1Npi​𝒙i​𝒙iT‖2+‖(∑i=l+1Npi​𝒙i)​(∑i=l+1Npi​𝒙i)T‖2subscriptnormsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇2subscriptnormsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖subscript𝒙𝑖𝑇2\displaystyle+\ \left.{{\left\|\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\right\|}}\_{2}\ +\ {{\left\|\left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)^{T}\right\|}}\_{2}\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∥(∑i=1lpi𝒙i)(∑i=l+1Npi𝒙i)T∥2+∥(∑i=l+1Npi𝒙i)(∑i=1lpi𝒙i)T∥2)\displaystyle+\ \left.{{\left\|\left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)^{T}\right\|}}\_{2}\ +\ {{\left\|\left(\sum\_{i=l+1}^{N}p\_{i}\ \bm{x}\_{i}\right)\left(\sum\_{i=1}^{l}p\_{i}\ \bm{x}\_{i}\right)^{T}\right\|}}\_{2}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽β((1−γ)∥Varp~[𝒙1:l]∥2+γ(1−γ)M2+γM2+γ2M2+\displaystyle\leqslant\ \beta\left((1-\gamma)\ {{\left\|\mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}]\right\|}}\_{2}\ +\ \gamma\ (1-\gamma)\ M^{2}\ +\ \gamma\ M^{2}\ +\ \gamma^{2}\ M^{2}\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | γ(1−γ)M2+γ(1−γ)M2)\displaystyle\left.\gamma\ (1-\gamma)\ M^{2}\ +\ \gamma\ (1-\gamma)\ M^{2}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β​((1−γ)‖Varp~​[𝒙1:l]‖2+γ​ 2​(2−γ)​M2).absent𝛽  1𝛾subscriptnormsubscriptVar~𝑝delimited-[]subscript𝒙:1𝑙2𝛾22𝛾superscript𝑀2\displaystyle=\ \beta\left((1-\gamma)\ \ {{\left\|\mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}]\right\|}}\_{2}\ +\ \gamma\ 2\ (2\ -\ \gamma)\ M^{2}\right)\ . |  |

Combining the previous two estimates immediately
leads to Eq. ([201](#A1.E201 "In Lemma A8. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).

The function h​(x)=x​2​(2−x)ℎ𝑥𝑥22𝑥h(x)=x2(2-x) has the derivative h′​(x)=4​(1−x)superscriptℎ′𝑥41𝑥h^{\prime}(x)=4(1-x).
Therefore, h​(x)ℎ𝑥h(x) is monotone increasing for x<1𝑥1x<1.
For 0⩽γ⩽ϵ<10𝛾italic-ϵ10\leqslant\gamma\leqslant\epsilon<1,
we can immediately deduce that
γ​2​(2−γ)⩽ϵ​2​(2−ϵ)𝛾22𝛾italic-ϵ22italic-ϵ\gamma 2(2-\gamma)\leqslant\epsilon 2(2-\epsilon).
Since ϵitalic-ϵ\epsilon is larger than γ𝛾\gamma,
we obtain the following ϵitalic-ϵ\epsilon-bound for ‖J‖2subscriptnormJ2{{\left\|\mathrm{J}\right\|}}\_{2}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽β​(mmax2+ϵ​ 2​(2−ϵ)​M2).absent𝛽superscriptsubscript𝑚2italic-ϵ22italic-ϵsuperscript𝑀2\displaystyle\leqslant\ \beta\left(\ m\_{\max}^{2}\ +\ \epsilon\ 2\ (2\ -\ \epsilon)\ M^{2}\right)\ . |  | (218) |

∎

We revisit the bound on (1−γ)​Varp~​[𝒙1:l]1𝛾subscriptVar~𝑝delimited-[]subscript𝒙:1𝑙(1-\gamma)\ \mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}].
The trace ∑k=1deksuperscriptsubscript𝑘1𝑑subscript𝑒𝑘\sum\_{k=1}^{d}e\_{k} is the sum of the eigenvalues
eksubscript𝑒𝑘e\_{k}. The spectral norm is equal to the
largest eigenvalue e1subscript𝑒1e\_{1}, that is, the largest singular value.
We obtain:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Varp~​[𝒙1:l]‖2subscriptnormsubscriptVar~𝑝delimited-[]subscript𝒙:1𝑙2\displaystyle{{\left\|\mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}]\right\|}}\_{2}\ | =Tr​(∑i=1lpi​(𝒙i−𝒙¯)​(𝒙i−𝒙¯)T)−∑k=2dekabsentTrsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖subscript𝒙𝑖¯𝒙superscriptsubscript𝒙𝑖¯𝒙𝑇superscriptsubscript𝑘2𝑑subscript𝑒𝑘\displaystyle=\ \mathbf{\mathrm{Tr}}\left(\sum\_{i=1}^{l}p\_{i}\left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)\ \left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)^{T}\right)\ -\ \sum\_{k=2}^{d}e\_{k} |  | (219) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1lpi​Tr​((𝒙i−𝒙¯)​(𝒙i−𝒙¯)T)−∑k=2dekabsentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖Trsubscript𝒙𝑖¯𝒙superscriptsubscript𝒙𝑖¯𝒙𝑇superscriptsubscript𝑘2𝑑subscript𝑒𝑘\displaystyle=\ \sum\_{i=1}^{l}p\_{i}\mathbf{\mathrm{Tr}}\left(\left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)\ \left(\bm{x}\_{i}\ -\ \bar{\bm{x}}\right)^{T}\right)\ -\ \sum\_{k=2}^{d}e\_{k} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1lpi​‖𝒙i−𝒙¯‖2−∑k=2dek.absentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖superscriptnormsubscript𝒙𝑖¯𝒙2superscriptsubscript𝑘2𝑑subscript𝑒𝑘\displaystyle=\ \sum\_{i=1}^{l}p\_{i}{{\left\|\bm{x}\_{i}\ -\ \bar{\bm{x}}\right\|}}^{2}\ -\ \sum\_{k=2}^{d}e\_{k}\ . |  |

Therefore, the tightness of the bound depends on eigenvalues
which are not the largest. That is variations which are not
along the strongest variation weaken the bound.

•Proof of a fixed point by Banach Fixed Point Theorem.

Without restricting the generality,
we assume that the first l𝑙l patterns are
much more probable (and similar
to one another) than the other patterns.
Therefore, we define:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M𝑀\displaystyle M\ | :=maxi⁡‖𝒙i‖,assignabsentsubscript𝑖normsubscript𝒙𝑖\displaystyle:=\ \max\_{i}{{\left\|\bm{x}\_{i}\right\|}}\ , |  | (220) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | γ𝛾\displaystyle\gamma\ | =∑i=l+1Npi⩽ϵ,absentsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖italic-ϵ\displaystyle=\ \sum\_{i=l+1}^{N}p\_{i}\ \leqslant\ \epsilon\ , |  | (221) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1−γ1𝛾\displaystyle 1-\gamma\ | =∑i=1lpi≥ 1−ϵ,absentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖1italic-ϵ\displaystyle=\ \sum\_{i=1}^{l}p\_{i}\ \geq\ 1\ -\ \epsilon\ , |  | (222) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | p~isubscript~𝑝𝑖\displaystyle\tilde{p}\_{i}\ | :=pi1−γ⩽pi/(1−ϵ),assignabsentsubscript𝑝𝑖1𝛾subscript𝑝𝑖1italic-ϵ\displaystyle:=\ \frac{p\_{i}}{1-\gamma}\ \leqslant\ p\_{i}/(1-\epsilon)\ , |  | (223) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i=1lp~isuperscriptsubscript𝑖1𝑙subscript~𝑝𝑖\displaystyle\sum\_{i=1}^{l}\tilde{p}\_{i}\ | = 1,absent1\displaystyle=\ 1\ , |  | (224) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒎𝒙subscript𝒎𝒙\displaystyle\bm{m}\_{\bm{x}}\ | =1l​∑i=1l𝒙i,absent1𝑙superscriptsubscript𝑖1𝑙subscript𝒙𝑖\displaystyle=\ \frac{1}{l}\ \sum\_{i=1}^{l}\ \bm{x}\_{i}\ , |  | (225) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | mmaxsubscript𝑚\displaystyle m\_{\max}\ | =max1⩽i⩽l⁡‖𝒙i−𝒎𝒙‖.absentsubscript1𝑖𝑙normsubscript𝒙𝑖subscript𝒎𝒙\displaystyle=\ \max\_{1\leqslant i\leqslant l}{{\left\|\bm{x}\_{i}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ . |  | (226) |

M𝑀M is an upper bound on the Euclidean norm of the patterns, which are vectors.
ϵitalic-ϵ\epsilon is an upper bound on the probability γ𝛾\gamma of not choosing one of the first l𝑙l patterns, while
1−ϵ1italic-ϵ1-\epsilon is a lower bound the probability (1−γ)1𝛾(1-\gamma)
of choosing one of the first l𝑙l patterns.
𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} is the arithmetic mean (the center) of the first l𝑙l
patterns.
mmaxsubscript𝑚m\_{\max} is the maximal distance of the patterns to the center 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} .
𝒑~~𝒑\tilde{\bm{p}} is the probability 𝒑𝒑\bm{p}
normalized for the first l𝑙l patterns.

•Mapped vectors stay in a compact environment.
We show that if 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} is sufficient dissimilar to
other 𝒙jsubscript𝒙𝑗\bm{x}\_{j} with l<j𝑙𝑗l<j
then there is an compact environment of 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}}
(a sphere)
where the fixed point iteration maps this environment into
itself.
The idea of the proof is to define a sphere around 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}}
for which the points from the sphere are mapped by f𝑓f into the sphere.

We first need following lemma which bounds the distance
‖𝒎𝒙−f​(𝝃)‖normsubscript𝒎𝒙𝑓𝝃{{\left\|\bm{m}\_{\bm{x}}\ -\ f(\bm{\xi})\right\|}} of a 𝝃𝝃\bm{\xi} which is close
to 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}}.

###### Lemma A9.

For a query 𝛏𝛏\bm{\xi} and data 𝐗=(𝐱1,…,𝐱N)𝐗subscript𝐱1…subscript𝐱𝑁\bm{X}=(\bm{x}\_{1},\ldots,\bm{x}\_{N}),
we define

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 00\displaystyle 0\ | ⩽c=minj,l<j⁡(𝝃T​𝒎𝒙−𝝃T​𝒙j)=𝝃T​𝒎𝒙−maxj,l<j⁡𝝃T​𝒙j.absent𝑐subscript  𝑗𝑙 𝑗superscript𝝃𝑇subscript𝒎𝒙superscript𝝃𝑇subscript𝒙𝑗superscript𝝃𝑇subscript𝒎𝒙subscript  𝑗𝑙 𝑗superscript𝝃𝑇subscript𝒙𝑗\displaystyle\leqslant\ c\ =\ \min\_{j,l<j}\left(\bm{\xi}^{T}\bm{m}\_{\bm{x}}\ -\ \bm{\xi}^{T}\bm{x}\_{j}\right)\ =\ \bm{\xi}^{T}\bm{m}\_{\bm{x}}\ -\ \max\_{j,l<j}\bm{\xi}^{T}\bm{x}\_{j}\ . |  | (227) |

The following holds:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒎𝒙−f​(𝝃)‖normsubscript𝒎𝒙𝑓𝝃\displaystyle{{\left\|\bm{m}\_{\bm{x}}\ -\ f(\bm{\xi})\right\|}}\ | ⩽mmax+ 2​γ​M⩽mmax+ 2​ϵ​M,absentsubscript𝑚2𝛾𝑀subscript𝑚2italic-ϵ𝑀\displaystyle\leqslant\ m\_{\max}\ +\ 2\ \gamma\ M\ \leqslant\ m\_{\max}\ +\ 2\ \epsilon\ M\ , |  | (228) |

where

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M𝑀\displaystyle M\ | =maxi⁡‖𝒙i‖,absentsubscript𝑖normsubscript𝒙𝑖\displaystyle=\ \max\_{i}{{\left\|\bm{x}\_{i}\right\|}}\ , |  | (229) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵitalic-ϵ\displaystyle\epsilon\ | =(N−l)​exp⁡(−β​c).absent𝑁𝑙𝛽𝑐\displaystyle=\ (N-l)\ \exp(-\ \beta\ c)\ . |  | (230) |

###### Proof.

Let s=arg⁡maxj,j⩽l⁡𝝃T​𝒙j𝑠subscript

𝑗𝑗
𝑙superscript𝝃𝑇subscript𝒙𝑗s=\arg\max\_{j,j\leqslant l}\bm{\xi}^{T}\bm{x}\_{j},
therefore 𝝃T​𝒎𝒙=1l​∑i=1l𝝃T​𝒙i⩽1l​∑i=1l𝝃T​𝒙s=𝝃T​𝒙ssuperscript𝝃𝑇subscript𝒎𝒙1𝑙superscriptsubscript𝑖1𝑙superscript𝝃𝑇subscript𝒙𝑖1𝑙superscriptsubscript𝑖1𝑙superscript𝝃𝑇subscript𝒙𝑠superscript𝝃𝑇subscript𝒙𝑠\bm{\xi}^{T}\bm{m}\_{\bm{x}}=\frac{1}{l}\ \sum\_{i=1}^{l}\ \bm{\xi}^{T}\bm{x}\_{i}\leqslant\frac{1}{l}\ \sum\_{i=1}^{l}\ \bm{\xi}^{T}\bm{x}\_{s}=\bm{\xi}^{T}\bm{x}\_{s}.
For softmax components j𝑗j with l<j𝑙𝑗l<j we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | [softmax​(β​𝑿T​𝝃)]jsubscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑗\displaystyle[\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})]\_{j}\ | =exp⁡(β​(𝝃T​𝒙j−𝝃T​𝒙s))1+∑k,k≠sexp⁡(β​(𝝃T​𝒙k−𝝃T​𝒙s))⩽exp⁡(−β​c)=ϵN−l,absent𝛽superscript𝝃𝑇subscript𝒙𝑗superscript𝝃𝑇subscript𝒙𝑠1subscript  𝑘𝑘 𝑠𝛽superscript𝝃𝑇subscript𝒙𝑘superscript𝝃𝑇subscript𝒙𝑠𝛽𝑐italic-ϵ𝑁𝑙\displaystyle=\ \frac{\exp(\beta\ (\bm{\xi}^{T}\bm{x}\_{j}\ -\ \bm{\xi}^{T}\bm{x}\_{s}))}{1\ +\ \sum\_{k,k\not=s}\exp(\beta\ (\bm{\xi}^{T}\bm{x}\_{k}\ -\ \bm{\xi}^{T}\bm{x}\_{s}))}\ \leqslant\ \exp(-\ \beta\ c)\ =\ \frac{\epsilon}{N-l}\ , |  | (231) |

since 𝝃T​𝒙s−𝝃T​𝒙j≥𝝃T​𝒎𝒙−𝝃T​𝒙jsuperscript𝝃𝑇subscript𝒙𝑠superscript𝝃𝑇subscript𝒙𝑗superscript𝝃𝑇subscript𝒎𝒙superscript𝝃𝑇subscript𝒙𝑗\bm{\xi}^{T}\bm{x}\_{s}-\bm{\xi}^{T}\bm{x}\_{j}\geq\bm{\xi}^{T}\bm{m}\_{\bm{x}}-\bm{\xi}^{T}\bm{x}\_{j} for each j𝑗j with l<j𝑙𝑗l<j,
therefore 𝝃T​𝒙s−𝝃T​𝒙j≥csuperscript𝝃𝑇subscript𝒙𝑠superscript𝝃𝑇subscript𝒙𝑗𝑐\bm{\xi}^{T}\bm{x}\_{s}-\bm{\xi}^{T}\bm{x}\_{j}\geq c

The iteration f𝑓f can be written as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(𝝃)𝑓𝝃\displaystyle f(\bm{\xi})\ | =𝑿​softmax​(β​𝑿T​𝝃)=∑j=1N𝒙j​[softmax​(β​𝑿T​𝝃)]j.absent𝑿softmax𝛽superscript𝑿𝑇𝝃superscriptsubscript𝑗1𝑁subscript𝒙𝑗subscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑗\displaystyle=\ \bm{X}\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ =\ \sum\_{j=1}^{N}\bm{x}\_{j}\ [\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})]\_{j}\ . |  | (232) |

We set pi=[softmax​(β​𝑿T​𝝃)]isubscript𝑝𝑖subscriptdelimited-[]softmax𝛽superscript𝑿𝑇𝝃𝑖p\_{i}=[\mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})]\_{i},
therefore ∑i=1lpi=1−γ≥1−ϵsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖1𝛾1italic-ϵ\sum\_{i=1}^{l}p\_{i}=1-\gamma\geq 1-\epsilon
and ∑i=l+1Npi=γ⩽ϵsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖𝛾italic-ϵ\sum\_{i=l+1}^{N}p\_{i}=\gamma\leqslant\epsilon.
Therefore

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ‖𝒎𝒙−∑j=1lpj1−γ​𝒙j‖2=‖∑j=1lpj1−γ​(𝒎𝒙−𝒙j)‖2superscriptnormsubscript𝒎𝒙superscriptsubscript𝑗1𝑙subscript𝑝𝑗1𝛾subscript𝒙𝑗2superscriptnormsuperscriptsubscript𝑗1𝑙subscript𝑝𝑗1𝛾subscript𝒎𝒙subscript𝒙𝑗2\displaystyle{{\left\|\bm{m}\_{\bm{x}}\ -\ \sum\_{j=1}^{l}\frac{p\_{j}}{1-\gamma}\ \bm{x}\_{j}\right\|}}^{2}\ =\ {{\left\|\sum\_{j=1}^{l}\frac{p\_{j}}{1-\gamma}\left(\bm{m}\_{\bm{x}}\ -\ \bm{x}\_{j}\right)\right\|}}^{2} |  | (233) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j=1,k=1lpj1−γ​pk1−γ​(𝒎𝒙−𝒙j)T​(𝒎𝒙−𝒙k)absentsuperscriptsubscriptformulae-sequence𝑗1𝑘1𝑙subscript𝑝𝑗1𝛾subscript𝑝𝑘1𝛾superscriptsubscript𝒎𝒙subscript𝒙𝑗𝑇subscript𝒎𝒙subscript𝒙𝑘\displaystyle=\ \sum\_{j=1,k=1}^{l}\frac{p\_{j}}{1-\gamma}\frac{p\_{k}}{1-\gamma}\left(\bm{m}\_{\bm{x}}\ -\ \bm{x}\_{j}\right)^{T}\left(\bm{m}\_{\bm{x}}\ -\ \bm{x}\_{k}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =12​∑j=1,k=1lpj1−γ​pk1−γ​(‖𝒎𝒙−𝒙j‖2+‖𝒎𝒙−𝒙k‖2−‖𝒙j−𝒙k‖2)absent12superscriptsubscriptformulae-sequence𝑗1𝑘1𝑙subscript𝑝𝑗1𝛾subscript𝑝𝑘1𝛾superscriptnormsubscript𝒎𝒙subscript𝒙𝑗2superscriptnormsubscript𝒎𝒙subscript𝒙𝑘2superscriptnormsubscript𝒙𝑗subscript𝒙𝑘2\displaystyle=\ \frac{1}{2}\ \sum\_{j=1,k=1}^{l}\frac{p\_{j}}{1-\gamma}\frac{p\_{k}}{1-\gamma}\left({{\left\|\bm{m}\_{\bm{x}}\ -\ \bm{x}\_{j}\right\|}}^{2}\ +\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \bm{x}\_{k}\right\|}}^{2}\ -\ {{\left\|\bm{x}\_{j}\ -\ \bm{x}\_{k}\right\|}}^{2}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j=1lpj1−γ​‖𝒎𝒙−𝒙j‖2−12​∑j=1,k=1lpj1−γ​pk1−γ​‖𝒙j−𝒙k‖2absentsuperscriptsubscript𝑗1𝑙subscript𝑝𝑗1𝛾superscriptnormsubscript𝒎𝒙subscript𝒙𝑗212superscriptsubscriptformulae-sequence𝑗1𝑘1𝑙subscript𝑝𝑗1𝛾subscript𝑝𝑘1𝛾superscriptnormsubscript𝒙𝑗subscript𝒙𝑘2\displaystyle=\ \sum\_{j=1}^{l}\frac{p\_{j}}{1-\gamma}\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \bm{x}\_{j}\right\|}}^{2}\ -\ \frac{1}{2}\ \sum\_{j=1,k=1}^{l}\frac{p\_{j}}{1-\gamma}\frac{p\_{k}}{1-\gamma}{{\left\|\bm{x}\_{j}\ -\ \bm{x}\_{k}\right\|}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽∑j=1lpj1−γ​‖𝒎𝒙−𝒙j‖2⩽mmax2.absentsuperscriptsubscript𝑗1𝑙subscript𝑝𝑗1𝛾superscriptnormsubscript𝒎𝒙subscript𝒙𝑗2superscriptsubscript𝑚2\displaystyle\leqslant\ \sum\_{j=1}^{l}\frac{p\_{j}}{1-\gamma}\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \bm{x}\_{j}\right\|}}^{2}\ \leqslant\ m\_{\max}^{2}\ . |  |

It follows that

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒎𝒙−∑j=1lpj1−γ​𝒙j‖normsubscript𝒎𝒙superscriptsubscript𝑗1𝑙subscript𝑝𝑗1𝛾subscript𝒙𝑗\displaystyle{{\left\|\bm{m}\_{\bm{x}}\ -\ \sum\_{j=1}^{l}\frac{p\_{j}}{1-\gamma}\ \bm{x}\_{j}\right\|}}\ | ⩽mmaxabsentsubscript𝑚\displaystyle\leqslant\ m\_{\max} |  | (234) |

We now can bound ‖𝒎𝒙−f​(𝝃)‖normsubscript𝒎𝒙𝑓𝝃{{\left\|\bm{m}\_{\bm{x}}\ -\ f(\bm{\xi})\right\|}}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒎𝒙−f​(𝝃)‖normsubscript𝒎𝒙𝑓𝝃\displaystyle{{\left\|\bm{m}\_{\bm{x}}\ -\ f(\bm{\xi})\right\|}}\ | =‖𝒎𝒙−∑j=1Npj​𝒙j‖absentnormsubscript𝒎𝒙superscriptsubscript𝑗1𝑁subscript𝑝𝑗subscript𝒙𝑗\displaystyle=\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \sum\_{j=1}^{N}p\_{j}\ \bm{x}\_{j}\right\|}} |  | (235) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =‖𝒎𝒙−∑j=1lpj​𝒙j−∑j=l+1Npj​𝒙j‖absentnormsubscript𝒎𝒙superscriptsubscript𝑗1𝑙subscript𝑝𝑗subscript𝒙𝑗superscriptsubscript𝑗𝑙1𝑁subscript𝑝𝑗subscript𝒙𝑗\displaystyle=\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \sum\_{j=1}^{l}p\_{j}\ \bm{x}\_{j}\ -\ \sum\_{j=l+1}^{N}p\_{j}\ \bm{x}\_{j}\right\|}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =‖𝒎𝒙−∑j=1lpj1−γ​𝒙j+γ1−γ​∑j=1lpj​𝒙j−∑j=l+1Npj​𝒙j‖absentnormsubscript𝒎𝒙superscriptsubscript𝑗1𝑙subscript𝑝𝑗1𝛾subscript𝒙𝑗𝛾1𝛾superscriptsubscript𝑗1𝑙subscript𝑝𝑗subscript𝒙𝑗superscriptsubscript𝑗𝑙1𝑁subscript𝑝𝑗subscript𝒙𝑗\displaystyle=\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \sum\_{j=1}^{l}\frac{p\_{j}}{1-\gamma}\ \bm{x}\_{j}\ +\ \frac{\gamma}{1-\gamma}\ \sum\_{j=1}^{l}p\_{j}\ \bm{x}\_{j}\ -\ \sum\_{j=l+1}^{N}p\_{j}\ \bm{x}\_{j}\right\|}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽‖𝒎𝒙−∑j=1lpj1−γ​𝒙j‖+γ1−γ​‖∑j=1lpj​𝒙j‖+‖∑j=l+1Npj​𝒙j‖absentnormsubscript𝒎𝒙superscriptsubscript𝑗1𝑙subscript𝑝𝑗1𝛾subscript𝒙𝑗𝛾1𝛾normsuperscriptsubscript𝑗1𝑙subscript𝑝𝑗subscript𝒙𝑗normsuperscriptsubscript𝑗𝑙1𝑁subscript𝑝𝑗subscript𝒙𝑗\displaystyle\leqslant\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \sum\_{j=1}^{l}\frac{p\_{j}}{1-\gamma}\ \bm{x}\_{j}\right\|}}\ +\ \frac{\gamma}{1-\gamma}\ {{\left\|\sum\_{j=1}^{l}p\_{j}\ \bm{x}\_{j}\right\|}}\ +\ {{\left\|\sum\_{j=l+1}^{N}p\_{j}\ \bm{x}\_{j}\right\|}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽‖𝒎𝒙−∑j=1lpj1−γ​𝒙j‖+γ1−γ​∑j=1lpj​M+∑j=l+1Npj​Mabsentnormsubscript𝒎𝒙superscriptsubscript𝑗1𝑙subscript𝑝𝑗1𝛾subscript𝒙𝑗𝛾1𝛾superscriptsubscript𝑗1𝑙subscript𝑝𝑗𝑀superscriptsubscript𝑗𝑙1𝑁subscript𝑝𝑗𝑀\displaystyle\leqslant\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \sum\_{j=1}^{l}\frac{p\_{j}}{1-\gamma}\ \bm{x}\_{j}\right\|}}\ +\ \frac{\gamma}{1-\gamma}\ \sum\_{j=1}^{l}p\_{j}\ M\ +\ \sum\_{j=l+1}^{N}p\_{j}\ M |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽‖𝒎𝒙−∑j=1lpj1−γ​𝒙j‖+ 2​γ​Mabsentnormsubscript𝒎𝒙superscriptsubscript𝑗1𝑙subscript𝑝𝑗1𝛾subscript𝒙𝑗2𝛾𝑀\displaystyle\leqslant\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \sum\_{j=1}^{l}\frac{p\_{j}}{1-\gamma}\ \bm{x}\_{j}\right\|}}\ +\ 2\ \gamma\ M |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽mmax+ 2​γ​M⩽mmax+ 2​ϵ​M,absentsubscript𝑚2𝛾𝑀subscript𝑚2italic-ϵ𝑀\displaystyle\leqslant\ m\_{\max}\ +\ 2\ \gamma\ M\ \leqslant\ m\_{\max}\ +\ 2\ \epsilon\ M\ , |  |

where we applied Eq. ([233](#A1.E233 "In Proof. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) in the penultimate inequality.
This is the statement of the lemma.
∎

The separation of the center (the arithmetic mean)
𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} of the first l𝑙l
from data 𝑿=(𝒙l+1,…,𝒙N)𝑿subscript𝒙𝑙1…subscript𝒙𝑁\bm{X}=(\bm{x}\_{l+1},\ldots,\bm{x}\_{N})
is ΔmsubscriptΔ𝑚\Delta\_{m}, defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔmsubscriptΔ𝑚\displaystyle\Delta\_{m}\ | =minj,l<j⁡(𝒎𝒙T​𝒎𝒙−𝒎𝒙T​𝒙j)=𝒎𝒙T​𝒎𝒙−maxj,l<j⁡𝒎𝒙T​𝒙j.absentsubscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙subscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,l<j}\left(\bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ \bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\right)\ =\ \bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ \max\_{j,l<j}\bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\ . |  | (236) |

The center is separated from the other data 𝒙jsubscript𝒙𝑗\bm{x}\_{j}
with l<j𝑙𝑗l<j if 0<Δm0subscriptΔ𝑚0<\Delta\_{m}.
By the same arguments as in Eq. ([140](#A1.E140 "In A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), ΔmsubscriptΔ𝑚\Delta\_{m} can also be expressed as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔmsubscriptΔ𝑚\displaystyle\Delta\_{m}\ | =minj,l<j⁡12​(‖𝒎𝒙‖2−‖𝒙j‖2+‖𝒎𝒙−𝒙j‖2)absentsubscript  𝑗𝑙 𝑗12superscriptnormsubscript𝒎𝒙2superscriptnormsubscript𝒙𝑗2superscriptnormsubscript𝒎𝒙subscript𝒙𝑗2\displaystyle=\ \min\_{j,l<j}\frac{1}{2}\ \left({{\left\|\bm{m}\_{\bm{x}}\right\|}}^{2}\ -\ {{\left\|\bm{x}\_{j}\right\|}}^{2}\ +\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \bm{x}\_{j}\right\|}}^{2}\right) |  | (237) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =12​‖𝒎𝒙‖2−12​maxj,l<j⁡(‖𝒙j‖2−‖𝒎𝒙−𝒙j‖2).absent12superscriptnormsubscript𝒎𝒙212subscript  𝑗𝑙 𝑗superscriptnormsubscript𝒙𝑗2superscriptnormsubscript𝒎𝒙subscript𝒙𝑗2\displaystyle=\ \frac{1}{2}{{\left\|\bm{m}\_{\bm{x}}\right\|}}^{2}\ -\ \frac{1}{2}\ \max\_{j,l<j}\left({{\left\|\bm{x}\_{j}\right\|}}^{2}\ -\ {{\left\|\bm{m}\_{\bm{x}}\ -\ \bm{x}\_{j}\right\|}}^{2}\right)\ . |  |

For ‖𝒎𝒙‖=‖𝒙j‖normsubscript𝒎𝒙normsubscript𝒙𝑗{{\left\|\bm{m}\_{\bm{x}}\right\|}}={{\left\|\bm{x}\_{j}\right\|}} we have Δm=1/2​minj,l<j⁡‖𝒎𝒙−𝒙j‖2subscriptΔ𝑚12subscript

𝑗𝑙
𝑗superscriptnormsubscript𝒎𝒙subscript𝒙𝑗2\Delta\_{m}=1/2\min\_{j,l<j}{{\left\|\bm{m}\_{\bm{x}}\ -\ \bm{x}\_{j}\right\|}}^{2}.

Next we define the sphere where we want to apply
Banach fixed point theorem.

###### Definition 4 (Sphere SmsubscriptS𝑚\mathrm{S}\_{m}).

The sphere SmsubscriptS𝑚\mathrm{S}\_{m} is defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | SmsubscriptS𝑚\displaystyle\mathrm{S}\_{m}\ | :={𝝃∣‖𝝃−𝒎𝒙‖⩽1β​mmax}.assignabsentconditional-set𝝃norm𝝃subscript𝒎𝒙1𝛽subscript𝑚\displaystyle:=\ \left\{\bm{\xi}\mid{{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ \leqslant\ \frac{1}{\beta\ m\_{\max}}\right\}\ . |  | (238) |

###### Lemma A10.

With 𝛏𝛏\bm{\xi} given, if the assumptions

1. A1:

   𝝃𝝃\bm{\xi} is inside sphere: 𝝃∈Sm𝝃subscriptS𝑚\bm{\xi}\in\mathrm{S}\_{m},
2. A2:

   the center 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} is well separated from other data 𝒙jsubscript𝒙𝑗\bm{x}\_{j}
   with l<j𝑙𝑗l<j:

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | ΔmsubscriptΔ𝑚\displaystyle\Delta\_{m}\ | ≥2​Mβ​mmax−1β​ln⁡(1−β​mmax22​β​(N−l)​M​max⁡{mmax, 2​M}),absent2𝑀𝛽subscript𝑚1𝛽1𝛽superscriptsubscript𝑚22𝛽𝑁𝑙𝑀subscript𝑚2𝑀\displaystyle\geq\ \frac{2\ M}{\beta\ m\_{\max}}\ -\ \frac{1}{\beta}\ \ln\left(\frac{1\ -\ \beta\ m\_{\max}^{2}}{2\ \beta\ (N-l)\ M\ \max\{m\_{\max}\ ,\ 2\ M\}}\right)\ , |  | (239) |
3. A3:

   the distance mmaxsubscript𝑚m\_{\max} of similar patterns to the center is sufficient small:

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | β​mmax2𝛽superscriptsubscript𝑚2\displaystyle\beta\ m\_{\max}^{2}\ | ⩽ 1absent1\displaystyle\leqslant\ 1 |  | (240) |

hold, then f​(𝛏)∈Sm𝑓𝛏subscriptS𝑚f(\bm{\xi})\in\mathrm{S}\_{m}.
Therefore, under conditions (A2) and (A3), f𝑓f is a mapping from SmsubscriptS𝑚\mathrm{S}\_{m} into SmsubscriptS𝑚\mathrm{S}\_{m}.

###### Proof.

We need the separation Δ~msubscript~Δ𝑚\tilde{\Delta}\_{m} of
𝝃𝝃\bm{\xi} from the rest of the data, which is the last N−l𝑁𝑙N-l
data points 𝑿=(𝒙l+1,…,𝒙N)𝑿subscript𝒙𝑙1…subscript𝒙𝑁\bm{X}=(\bm{x}\_{l+1},\ldots,\bm{x}\_{N}).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~msubscript~Δ𝑚\displaystyle\tilde{\Delta}\_{m}\ | =minj,l<j⁡(𝝃T​𝒎𝒙−𝝃T​𝒙j).absentsubscript  𝑗𝑙 𝑗superscript𝝃𝑇subscript𝒎𝒙superscript𝝃𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,l<j}\left(\bm{\xi}^{T}\bm{m}\_{\bm{x}}\ -\ \bm{\xi}^{T}\bm{x}\_{j}\right)\ . |  | (241) |

Using the Cauchy-Schwarz inequality, we obtain for l+1⩽j⩽N𝑙1𝑗𝑁l+1\leqslant j\leqslant N:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |𝝃T​𝒙j−𝒎𝒙T​𝒙j|superscript𝝃𝑇subscript𝒙𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗\displaystyle\left|\bm{\xi}^{T}\bm{x}\_{j}\ -\ \bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\right| | ⩽‖𝝃−𝒎𝒙‖​‖𝒙j‖⩽‖𝝃−𝒎𝒙‖​M.absentnorm𝝃subscript𝒎𝒙normsubscript𝒙𝑗norm𝝃subscript𝒎𝒙𝑀\displaystyle\leqslant\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ {{\left\|\bm{x}\_{j}\right\|}}\ \leqslant\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ . |  | (242) |

We have the lower bound

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~msubscript~Δ𝑚\displaystyle\tilde{\Delta}\_{m}\ | ≥minj,l<j⁡((𝒎𝒙T​𝒎𝒙−‖𝝃−𝒎𝒙‖​M)−(𝒎𝒙T​𝒙j+‖𝝃−𝒎𝒙‖​M))absentsubscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙norm𝝃subscript𝒎𝒙𝑀superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗norm𝝃subscript𝒎𝒙𝑀\displaystyle\geq\ \min\_{j,l<j}\left(\left(\bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\right)\ -\ \left(\bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\ +\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\right)\right) |  | (243) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =− 2​‖𝝃−𝒎𝒙‖​M+minj,l<j⁡(𝒎𝒙T​𝒎𝒙−𝒎𝒙T​𝒙j)=Δm− 2​‖𝝃−𝒎𝒙‖​Mabsent2norm𝝃subscript𝒎𝒙𝑀subscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗subscriptΔ𝑚2norm𝝃subscript𝒎𝒙𝑀\displaystyle=\ -\ 2\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ +\ \min\_{j,l<j}\left(\bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ \bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\right)\ =\ \Delta\_{m}\ -\ 2\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥Δm− 2​Mβ​mmax,absentsubscriptΔ𝑚2𝑀𝛽subscript𝑚\displaystyle\geq\ \Delta\_{m}\ -\ 2\ \frac{M}{\beta\ m\_{\max}}\ , |  |

where we used the assumption (A1) of the lemma.

From the proof in Lemma [A9](#ThmlemmaA9 "Lemma A9. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i=1lpisuperscriptsubscript𝑖1𝑙subscript𝑝𝑖\displaystyle\sum\_{i=1}^{l}p\_{i}\ | ≥ 1−(N−l)​exp⁡(−β​Δ~m)= 1−ϵ~,absent1𝑁𝑙𝛽subscript~Δ𝑚1~italic-ϵ\displaystyle\geq\ 1\ -\ (N-l)\ \exp(-\ \beta\ \tilde{\Delta}\_{m})\ =\ 1\ -\ \tilde{\epsilon}\ , |  | (244) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i=l+1Npisuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖\displaystyle\sum\_{i=l+1}^{N}p\_{i}\ | ⩽(N−l)​exp⁡(−β​Δ~m)=ϵ~.absent𝑁𝑙𝛽subscript~Δ𝑚~italic-ϵ\displaystyle\leqslant\ (N-l)\ \exp(-\ \beta\ \tilde{\Delta}\_{m})\ =\ \tilde{\epsilon}\ . |  | (245) |

Lemma [A9](#ThmlemmaA9 "Lemma A9. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") states that

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒎𝒙−f​(𝝃)‖normsubscript𝒎𝒙𝑓𝝃\displaystyle{{\left\|\bm{m}\_{\bm{x}}\ -\ f(\bm{\xi})\right\|}}\ | ⩽mmax+ 2​ϵ~​Mabsentsubscript𝑚2~italic-ϵ𝑀\displaystyle\leqslant\ m\_{\max}\ +\ 2\ \tilde{\epsilon}\ M |  | (246) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽mmax+ 2​(N−l)​exp⁡(−β​Δ~m)​M.absentsubscript𝑚2𝑁𝑙𝛽subscript~Δ𝑚𝑀\displaystyle\leqslant\ m\_{\max}\ +\ 2\ (N-l)\ \exp(-\ \beta\ \tilde{\Delta}\_{m})\ M\ . |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽mmax+ 2​(N−l)​exp⁡(−β​(Δm− 2​Mβ​mmax))​M.absentsubscript𝑚2𝑁𝑙𝛽subscriptΔ𝑚2𝑀𝛽subscript𝑚𝑀\displaystyle\leqslant\ m\_{\max}\ +\ 2\ (N-l)\ \exp(-\ \beta\ (\Delta\_{m}\ -\ 2\ \frac{M}{\beta\ m\_{\max}}))\ M\ . |  |

Therefore, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝒎𝒙−f​(𝝃)‖⩽mmax+ 2​(N−l)​exp⁡(−β​(Δm− 2​Mβ​mmax))​Mnormsubscript𝒎𝒙𝑓𝝃subscript𝑚2𝑁𝑙𝛽subscriptΔ𝑚2𝑀𝛽subscript𝑚𝑀\displaystyle{{\left\|\bm{m}\_{\bm{x}}\ -\ f(\bm{\xi})\right\|}}\ \leqslant\ m\_{\max}\ +\ 2\ (N-l)\ \exp\left(-\ \beta\ (\Delta\_{m}\ -\ 2\ \frac{M}{\beta\ m\_{\max}})\right)\ M |  | (247) |
|  |  |  |
| --- | --- | --- |
|  | ⩽mmax+ 2(N−l)exp(−β(2​Mβ​mmax−\displaystyle\leqslant\ m\_{\max}\ +\ 2\ (N-l)\ \exp\left(-\ \beta\ \left(\frac{2\ M}{\beta\ m\_{\max}}\ -\right.\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 1βln(1−β​mmax22​β​(N−l)​M​max⁡{mmax, 2​M})− 2Mβ​mmax))M\displaystyle\left.\left.\frac{1}{\beta}\ \ln\left(\frac{1\ -\ \beta\ m\_{\max}^{2}}{2\ \beta\ (N-l)\ M\ \max\{m\_{\max}\ ,\ 2\ M\}}\right)\ -\ 2\ \frac{M}{\beta\ m\_{\max}}\right)\right)\ M |  |
|  |  |  |
| --- | --- | --- |
|  | =mmax+ 2​(N−l)​1−β​mmax22​β​(N−l)​M​max⁡{mmax, 2​M}​Mabsentsubscript𝑚2𝑁𝑙1𝛽superscriptsubscript𝑚22𝛽𝑁𝑙𝑀subscript𝑚2𝑀𝑀\displaystyle=\ m\_{\max}\ +\ 2\ (N-l)\ \frac{1\ -\ \beta\ m\_{\max}^{2}}{2\ \beta\ (N-l)\ M\ \max\{m\_{\max}\ ,\ 2\ M\}}\ M |  |
|  |  |  |
| --- | --- | --- |
|  | ⩽mmax+1−β​mmax2β​mmax=1β​mmax,absentsubscript𝑚1𝛽superscriptsubscript𝑚2𝛽subscript𝑚1𝛽subscript𝑚\displaystyle\leqslant\ m\_{\max}\ +\ \frac{1\ -\ \beta\ m\_{\max}^{2}}{\beta\ m\_{\max}}\ =\ \frac{1}{\beta\ m\_{\max}}\ , |  |

where we used assumption (A2) of the lemma.
Therefore, f​(𝝃)𝑓𝝃f(\bm{\xi}) is a mapping from the
sphere SmsubscriptS𝑚\mathrm{S}\_{m} into the sphere SmsubscriptS𝑚\mathrm{S}\_{m}.

|  |  |  |  |
| --- | --- | --- | --- |
|  | mmax=max1⩽i⩽l⁡‖𝒙i−𝒎𝒙‖subscript𝑚subscript1𝑖𝑙normsubscript𝒙𝑖subscript𝒎𝒙\displaystyle m\_{\max}=\max\_{1\leqslant i\leqslant l}{{\left\|\bm{x}\_{i}-\bm{m}\_{\bm{x}}\right\|}} |  | (248) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =max1⩽i⩽l⁡‖𝒙i−1/l​∑j=1l𝒙j‖absentsubscript1𝑖𝑙normsubscript𝒙𝑖1𝑙superscriptsubscript𝑗1𝑙subscript𝒙𝑗\displaystyle=\max\_{1\leqslant i\leqslant l}{{\left\|\bm{x}\_{i}-1/l\sum\_{j=1}^{l}\bm{x}\_{j}\right\|}} |  | (249) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =max1⩽i⩽l⁡‖1/l​∑j=1l(𝒙i−𝒙j)‖absentsubscript1𝑖𝑙norm1𝑙superscriptsubscript𝑗1𝑙subscript𝒙𝑖subscript𝒙𝑗\displaystyle=\max\_{1\leqslant i\leqslant l}{{\left\|1/l\sum\_{j=1}^{l}(\bm{x}\_{i}-\bm{x}\_{j})\right\|}} |  | (250) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⩽max1⩽i,j⩽l⁡‖𝒙i−𝒙j‖absentsubscriptformulae-sequence1𝑖𝑗𝑙normsubscript𝒙𝑖subscript𝒙𝑗\displaystyle\leqslant\max\_{1\leqslant i,j\leqslant l}{{\left\|\bm{x}\_{i}-\bm{x}\_{j}\right\|}} |  | (251) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⩽max1⩽i⩽l⁡‖𝒙i‖+max1⩽j⩽l⁡‖𝒙i‖absentsubscript1𝑖𝑙normsubscript𝒙𝑖subscript1𝑗𝑙normsubscript𝒙𝑖\displaystyle\leqslant\max\_{1\leqslant i\leqslant l}{{\left\|\bm{x}\_{i}\right\|}}+\max\_{1\leqslant j\leqslant l}{{\left\|\bm{x}\_{i}\right\|}} |  | (252) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⩽2​Mabsent2𝑀\displaystyle\leqslant 2M |  | (253) |

∎

•Contraction mapping.

For applying Banach fixed point theorem we need to show that
f𝑓f is contraction in the compact environment SmsubscriptS𝑚\mathrm{S}\_{m}.

###### Lemma A11.

Assume that

1. A1:

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | ΔmsubscriptΔ𝑚\displaystyle\Delta\_{m}\ | ≥2​Mβ​mmax−1β​ln⁡(1−β​mmax22​β​(N−l)​M​max⁡{mmax, 2​M}),absent2𝑀𝛽subscript𝑚1𝛽1𝛽superscriptsubscript𝑚22𝛽𝑁𝑙𝑀subscript𝑚2𝑀\displaystyle\geq\ \frac{2\ M}{\beta\ m\_{\max}}\ -\ \frac{1}{\beta}\ \ln\left(\frac{1\ -\ \beta\ m\_{\max}^{2}}{2\ \beta\ (N-l)\ M\ \max\{m\_{\max}\ ,\ 2\ M\}}\right)\ , |  | (254) |

   and
2. A2:

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | β​mmax2𝛽superscriptsubscript𝑚2\displaystyle\beta\ m\_{\max}^{2}\ | ⩽ 1,absent1\displaystyle\leqslant\ 1\ , |  | (255) |

then f𝑓f is a contraction mapping in SmsubscriptS𝑚\mathrm{S}\_{m}.

###### Proof.

The version of the mean value theorem Lemma [A32](#ThmlemmaA32 "Lemma A32 (Mean Value Theorem). ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") states for the symmetric
Jm=∫01J​(λ​𝝃+(1−λ)​𝒎𝒙)​dλsuperscriptJ𝑚superscriptsubscript01J𝜆𝝃1𝜆subscript𝒎𝒙differential-d𝜆\mathrm{J}^{m}=\int\_{0}^{1}\mathrm{J}(\lambda\bm{\xi}+(1-\lambda)\bm{m}\_{\bm{x}})\ \mathrm{d}\lambda:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(𝝃)𝑓𝝃\displaystyle f(\bm{\xi})\ | =f​(𝒎𝒙)+Jm​(𝝃−𝒎𝒙).absent𝑓subscript𝒎𝒙superscriptJ𝑚𝝃subscript𝒎𝒙\displaystyle=\ f(\bm{m}\_{\bm{x}})\ +\ \mathrm{J}^{m}\ (\bm{\xi}\ -\ \bm{m}\_{\bm{x}})\ . |  | (256) |

In complete analogy to Lemma [A6](#ThmlemmaA6 "Lemma A6. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we get:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖f​(𝝃)−f​(𝒎𝒙)‖norm𝑓𝝃𝑓subscript𝒎𝒙\displaystyle{{\left\|f(\bm{\xi})\ -\ f(\bm{m}\_{\bm{x}})\right\|}}\ | ⩽‖Jm‖2​‖𝝃−𝒎𝒙‖.absentsubscriptnormsuperscriptJ𝑚2norm𝝃subscript𝒎𝒙\displaystyle\leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ . |  | (257) |

We define 𝝃~=λ​𝝃+(1−λ)​𝒎𝒙~𝝃𝜆𝝃1𝜆subscript𝒎𝒙\tilde{\bm{\xi}}=\lambda\bm{\xi}+(1-\lambda)\bm{m}\_{\bm{x}}
for some λ∈[0,1]𝜆01\lambda\in[0,1].
We need the separation Δ~msubscript~Δ𝑚\tilde{\Delta}\_{m} of
𝝃~~𝝃\tilde{\bm{\xi}} from the rest of the data, which is the last N−l𝑁𝑙N-l
data points 𝑿=(𝒙l+1,…,𝒙N)𝑿subscript𝒙𝑙1…subscript𝒙𝑁\bm{X}=(\bm{x}\_{l+1},\ldots,\bm{x}\_{N}).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~msubscript~Δ𝑚\displaystyle\tilde{\Delta}\_{m}\ | =minj,l<j⁡(𝝃~T​𝒎𝒙−𝝃~T​𝒙j).absentsubscript  𝑗𝑙 𝑗superscript~𝝃𝑇subscript𝒎𝒙superscript~𝝃𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,l<j}\left(\tilde{\bm{\xi}}^{T}\bm{m}\_{\bm{x}}\ -\ \tilde{\bm{\xi}}^{T}\bm{x}\_{j}\right)\ . |  | (258) |

From the proof in Lemma [A9](#ThmlemmaA9 "Lemma A9. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵ~~italic-ϵ\displaystyle\tilde{\epsilon}\ | =(N−l)​exp⁡(−β​Δ~m),absent𝑁𝑙𝛽subscript~Δ𝑚\displaystyle=\ (N-l)\ \exp(-\ \beta\ \tilde{\Delta}\_{m})\ , |  | (259) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i=1lpi​(𝝃~)superscriptsubscript𝑖1𝑙subscript𝑝𝑖~𝝃\displaystyle\sum\_{i=1}^{l}p\_{i}(\tilde{\bm{\xi}})\ | ≥ 1−(N−l)​exp⁡(−β​Δ~m)= 1−ϵ~,absent1𝑁𝑙𝛽subscript~Δ𝑚1~italic-ϵ\displaystyle\geq\ 1\ -\ (N-l)\ \exp(-\ \beta\ \tilde{\Delta}\_{m})\ =\ 1\ -\ \tilde{\epsilon}\ , |  | (260) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i=l+1Npi​(𝝃~)superscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖~𝝃\displaystyle\sum\_{i=l+1}^{N}p\_{i}(\tilde{\bm{\xi}})\ | ⩽(N−l)​exp⁡(−β​Δ~m)=ϵ~.absent𝑁𝑙𝛽subscript~Δ𝑚~italic-ϵ\displaystyle\leqslant\ (N-l)\ \exp(-\ \beta\ \tilde{\Delta}\_{m})\ =\ \tilde{\epsilon}\ . |  | (261) |

We first compute an upper bound on ϵ~~italic-ϵ\tilde{\epsilon}.
Using the Cauchy-Schwarz inequality, we obtain for l+1⩽j⩽N𝑙1𝑗𝑁l+1\leqslant j\leqslant N:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |𝝃~T​𝒙j−𝒎𝒙T​𝒙j|superscript~𝝃𝑇subscript𝒙𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗\displaystyle\left|\tilde{\bm{\xi}}^{T}\bm{x}\_{j}\ -\ \bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\right| | ⩽‖𝝃~−𝒎𝒙‖​‖𝒙j‖⩽‖𝝃~−𝒎𝒙‖​M.absentnorm~𝝃subscript𝒎𝒙normsubscript𝒙𝑗norm~𝝃subscript𝒎𝒙𝑀\displaystyle\leqslant\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ {{\left\|\bm{x}\_{j}\right\|}}\ \leqslant\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ . |  | (262) |

We have the lower bound on Δ~msubscript~Δ𝑚\tilde{\Delta}\_{m}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~msubscript~Δ𝑚\displaystyle\tilde{\Delta}\_{m}\ | ≥minj,l<j⁡((𝒎𝒙T​𝒎𝒙−‖𝝃~−𝒎𝒙‖​M)−(𝒎𝒙T​𝒙j+‖𝝃~−𝒎𝒙‖​M))absentsubscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙norm~𝝃subscript𝒎𝒙𝑀superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗norm~𝝃subscript𝒎𝒙𝑀\displaystyle\geq\ \min\_{j,l<j}\left(\left(\bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\right)\ -\ \left(\bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\ +\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\right)\right) |  | (263) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =− 2​‖𝝃~−𝒎𝒙‖​M+minj,l<j⁡(𝒎𝒙T​𝒎𝒙−𝒎𝒙T​𝒙j)=Δm− 2​‖𝝃~−𝒎𝒙‖​Mabsent2norm~𝝃subscript𝒎𝒙𝑀subscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗subscriptΔ𝑚2norm~𝝃subscript𝒎𝒙𝑀\displaystyle=\ -\ 2\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ +\ \min\_{j,l<j}\left(\bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ \bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\right)\ =\ \Delta\_{m}\ -\ 2\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥Δm− 2​‖𝝃−𝒎𝒙‖​M.absentsubscriptΔ𝑚2norm𝝃subscript𝒎𝒙𝑀\displaystyle\geq\ \Delta\_{m}\ -\ 2\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ . |  |

where we used ‖𝝃~−𝒎𝒙‖=λ​‖𝝃−𝒎𝒙‖⩽‖𝝃−𝒎𝒙‖norm~𝝃subscript𝒎𝒙𝜆norm𝝃subscript𝒎𝒙norm𝝃subscript𝒎𝒙{{\left\|\tilde{\bm{\xi}}-\bm{m}\_{\bm{x}}\right\|}}=\lambda{{\left\|\bm{\xi}-\bm{m}\_{\bm{x}}\right\|}}\leqslant{{\left\|\bm{\xi}-\bm{m}\_{\bm{x}}\right\|}}.
We obtain the upper bound on ϵ~~italic-ϵ\tilde{\epsilon}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵ~~italic-ϵ\displaystyle\tilde{\epsilon}\ | ⩽(N−l)​exp⁡(−β​(Δm− 2​‖𝝃−𝒎𝒙‖​M))absent𝑁𝑙𝛽subscriptΔ𝑚2norm𝝃subscript𝒎𝒙𝑀\displaystyle\leqslant\ (N-l)\ \exp\left(-\ \beta\ \left(\Delta\_{m}\ -\ 2\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\right)\right) |  | (264) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽(N−l)​exp⁡(−β​(Δm−2​Mβ​mmax)).absent𝑁𝑙𝛽subscriptΔ𝑚2𝑀𝛽subscript𝑚\displaystyle\leqslant\ (N-l)\ \exp\left(-\ \beta\ \left(\Delta\_{m}\ -\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)\ . |  |

where we used that in the sphere SisubscriptS𝑖\mathrm{S}\_{i} holds:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝝃−𝒎𝒙‖⩽1β​mmax,norm𝝃subscript𝒎𝒙1𝛽subscript𝑚\displaystyle{{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ \leqslant\ \frac{1}{\beta\ m\_{\max}}\ , |  | (265) |

therefore

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​‖𝝃−𝒎𝒙‖​M⩽2​Mβ​mmax.2norm𝝃subscript𝒎𝒙𝑀2𝑀𝛽subscript𝑚\displaystyle 2\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ \leqslant\ \frac{2\ M}{\beta\ m\_{\max}}\ . |  | (266) |

Next we compute a lower bound on ϵ~~italic-ϵ\tilde{\epsilon}
and to this end start with the upper bound on Δ~msubscript~Δ𝑚\tilde{\Delta}\_{m}
using the same arguments as in Eq. ([158](#A1.E158 "In Proof. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) in combination with Eq. ([266](#A1.E266 "In Proof. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~msubscript~Δ𝑚\displaystyle\tilde{\Delta}\_{m}\ | ≥minj,l<j⁡((𝒎𝒙T​𝒎𝒙+‖𝝃~−𝒎𝒙‖​M)−(𝒎𝒙T​𝒙j−‖𝝃~−𝒎𝒙‖​M))absentsubscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙norm~𝝃subscript𝒎𝒙𝑀superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗norm~𝝃subscript𝒎𝒙𝑀\displaystyle\geq\ \min\_{j,l<j}\left(\left(\bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ +\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\right)\ -\ \left(\bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\ -\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\right)\right) |  | (267) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | = 2​‖𝝃~−𝒎𝒙‖​M+minj,l<j⁡(𝒎𝒙T​𝒎𝒙−𝒎𝒙T​𝒙j)=Δm+ 2​‖𝝃~−𝒎𝒙‖​Mabsent2norm~𝝃subscript𝒎𝒙𝑀subscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗subscriptΔ𝑚2norm~𝝃subscript𝒎𝒙𝑀\displaystyle=\ 2\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ +\ \min\_{j,l<j}\left(\bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ \bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\right)\ =\ \Delta\_{m}\ +\ 2\ {{\left\|\tilde{\bm{\xi}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥Δm+ 2​‖𝝃−𝒎𝒙‖​M.absentsubscriptΔ𝑚2norm𝝃subscript𝒎𝒙𝑀\displaystyle\geq\ \Delta\_{m}\ +\ 2\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ . |  |

where we used ‖𝝃~−𝒎𝒙‖=λ​‖𝝃−𝒎𝒙‖⩽‖𝝃−𝒎𝒙‖norm~𝝃subscript𝒎𝒙𝜆norm𝝃subscript𝒎𝒙norm𝝃subscript𝒎𝒙{{\left\|\tilde{\bm{\xi}}-\bm{m}\_{\bm{x}}\right\|}}=\lambda{{\left\|\bm{\xi}-\bm{m}\_{\bm{x}}\right\|}}\leqslant{{\left\|\bm{\xi}-\bm{m}\_{\bm{x}}\right\|}}.
We obtain the lower bound on ϵ~~italic-ϵ\tilde{\epsilon}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵ~~italic-ϵ\displaystyle\tilde{\epsilon}\ | ≥(N−l)​exp⁡(−β​(Δm+2​Mβ​mmax)),absent𝑁𝑙𝛽subscriptΔ𝑚2𝑀𝛽subscript𝑚\displaystyle\geq\ (N-l)\ \exp\left(-\ \beta\ \left(\Delta\_{m}\ +\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)\ , |  | (268) |

where we used that in the sphere SisubscriptS𝑖\mathrm{S}\_{i} holds:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝝃−𝒎𝒙‖⩽1β​mmax,norm𝝃subscript𝒎𝒙1𝛽subscript𝑚\displaystyle{{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ \leqslant\ \frac{1}{\beta\ m\_{\max}}\ , |  | (269) |

therefore

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​‖𝝃−𝒎𝒙‖​M⩽2​Mβ​mmax.2norm𝝃subscript𝒎𝒙𝑀2𝑀𝛽subscript𝑚\displaystyle 2\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ \leqslant\ \frac{2\ M}{\beta\ m\_{\max}}\ . |  | (270) |

From Lemma [A8](#ThmlemmaA8 "Lemma A8. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J​(𝝃~)‖2subscriptnormJ~𝝃2\displaystyle{{\left\|\mathrm{J}(\tilde{\bm{\xi}})\right\|}}\_{2}\ | ⩽β​(mmax2+ϵ~​ 2​(2−ϵ~)​M2)absent𝛽superscriptsubscript𝑚2~italic-ϵ22~italic-ϵsuperscript𝑀2\displaystyle\leqslant\ \beta\left(\ m\_{\max}^{2}\ +\ \tilde{\epsilon}\ 2\ (2\ -\ \tilde{\epsilon})\ M^{2}\right) |  | (271) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β​(mmax2+ϵ~​4​M2− 2​ϵ~2​M2)absent𝛽superscriptsubscript𝑚2~italic-ϵ4superscript𝑀22superscript~italic-ϵ2superscript𝑀2\displaystyle=\ \beta\left(m\_{\max}^{2}\ +\ \tilde{\epsilon}4\ M^{2}\ -\ 2\ \tilde{\epsilon}^{2}\ M^{2}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽β(mmax2+(N−l)exp(−β(Δm−2​Mβ​mmax))4M2−\displaystyle\leqslant\ \beta\left(m\_{\max}^{2}\ +\ (N-l)\ \exp\left(-\ \beta\ \left(\Delta\_{m}\ -\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)4\ M^{2}\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2(N−l)2exp(− 2β(Δm+2​Mβ​mmax))M2).\displaystyle\left.2\ (N-l)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{m}\ +\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)\ M^{2}\right)\ . |  |

The bound Eq. ([271](#A1.E271 "In Proof. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) holds for the mean JmsuperscriptJ𝑚\mathrm{J}^{m}, too,
since it averages over J​(𝝃~)J~𝝃\mathrm{J}(\tilde{\bm{\xi}}):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jm‖2subscriptnormsuperscriptJ𝑚2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ | ⩽β(mmax2+(N−l)exp(−β(Δm−2​Mβ​mmax))4M2−\displaystyle\leqslant\ \beta\left(m\_{\max}^{2}\ +\ (N-l)\ \exp\left(-\ \beta\ \left(\Delta\_{m}\ -\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)4\ M^{2}\ -\right. |  | (272) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2(N−l)2exp(− 2β(Δm+2​Mβ​mmax))M2).\displaystyle\left.2\ (N-l)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{m}\ +\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)\ M^{2}\right)\ . |  |

The assumption of the lemma is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔmsubscriptΔ𝑚\displaystyle\Delta\_{m}\ | ≥2​Mβ​mmax−1β​ln⁡(1−β​mmax22​β​(N−l)​M​max⁡{mmax, 2​M}),absent2𝑀𝛽subscript𝑚1𝛽1𝛽superscriptsubscript𝑚22𝛽𝑁𝑙𝑀subscript𝑚2𝑀\displaystyle\geq\ \frac{2\ M}{\beta\ m\_{\max}}\ -\ \frac{1}{\beta}\ \ln\left(\frac{1\ -\ \beta\ m\_{\max}^{2}}{2\ \beta\ (N-l)\ M\ \max\{m\_{\max}\ ,\ 2\ M\}}\right)\ , |  | (273) |

Therefore, we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δm−2​Mβ​mmaxsubscriptΔ𝑚2𝑀𝛽subscript𝑚\displaystyle\Delta\_{m}\ -\ \frac{2\ M}{\beta\ m\_{\max}}\ | ≥−1β​ln⁡(1−β​mmax22​β​(N−l)​M​max⁡{mmax, 2​M}).absent1𝛽1𝛽superscriptsubscript𝑚22𝛽𝑁𝑙𝑀subscript𝑚2𝑀\displaystyle\geq\ -\ \frac{1}{\beta}\ \ln\left(\frac{1\ -\ \beta\ m\_{\max}^{2}}{2\ \beta\ (N-l)\ M\ \max\{m\_{\max}\ ,\ 2\ M\}}\right)\ . |  | (274) |

Therefore, the spectral norm ‖Jm‖2subscriptnormsuperscriptJ𝑚2{{\left\|\mathrm{J}^{m}\right\|}}\_{2} can be bounded by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Jm‖2⩽subscriptnormsuperscriptJ𝑚2absent\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ \leqslant |  | (275) |
|  |  |  |
| --- | --- | --- |
|  | β(mmax2+(N−l)exp(−β(−1βln(1−β​mmax22​β​(N−l)​M​max⁡{mmax, 2​M})))\displaystyle\beta\left(m\_{\max}^{2}\ +\ (N-l)\ \exp\left(-\ \beta\ \left(-\ \frac{1}{\beta}\ \ln\left(\frac{1\ -\ \beta\ m\_{\max}^{2}}{2\ \beta\ (N-l)\ M\ \max\{m\_{\max}\ ,\ 2\ M\}}\right)\right)\right)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 4M2− 2(N−l)2exp(− 2β(Δm+2​Mβ​mmax))M2)\displaystyle\left.4\ M^{2}\ -\ 2\ (N-l)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{m}\ +\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)\ M^{2}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | =β(mmax2+(N−l)exp(ln(1−β​mmax22​β​(N−l)​M​max⁡{mmax, 2​M}))\displaystyle=\ \beta\left(m\_{\max}^{2}\ +\ (N-l)\ \exp\left(\ln\left(\frac{1\ -\ \beta\ m\_{\max}^{2}}{2\ \beta\ (N-l)\ M\ \max\{m\_{\max}\ ,\ 2\ M\}}\right)\right)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 4M2− 2(N−l)2exp(− 2β(Δm+2​Mβ​mmax))M2)\displaystyle\left.4\ M^{2}\ -\ 2\ (N-l)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{m}\ +\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)\ M^{2}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | =β(mmax2+(N−l)1−β​mmax22​β​(N−l)​M​max⁡{mmax, 2​M} 4M2−\displaystyle=\ \beta\left(m\_{\max}^{2}\ +\ (N-l)\ \frac{1\ -\ \beta\ m\_{\max}^{2}}{2\ \beta\ (N-l)\ M\ \max\{m\_{\max}\ ,\ 2\ M\}}\ 4\ M^{2}\ -\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2(N−l)2exp(− 2β(Δm+2​Mβ​mmax))M2)\displaystyle\left.2\ (N-l)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{m}\ +\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)\ M^{2}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | =β​mmax2+1−β​mmax2max⁡{mmax, 2​M}​ 2​M−absent𝛽superscriptsubscript𝑚2limit-from1𝛽superscriptsubscript𝑚2subscript𝑚2𝑀2𝑀\displaystyle=\ \beta m\_{\max}^{2}\ +\ \frac{1\ -\ \beta\ m\_{\max}^{2}}{\ \max\{m\_{\max}\ ,\ 2\ M\}}\ 2\ M\ - |  |
|  |  |  |
| --- | --- | --- |
|  | β​ 2​(N−l)2​exp⁡(− 2​β​(Δm+2​Mβ​mmax))​M2𝛽2superscript𝑁𝑙22𝛽subscriptΔ𝑚2𝑀𝛽subscript𝑚superscript𝑀2\displaystyle\beta\ 2\ (N-l)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{m}\ +\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)\ M^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | ⩽β​mmax2+ 1−β​mmax2−β​ 2​(N−l)2​exp⁡(− 2​β​(Δm+2​Mβ​mmax))​M2absent𝛽superscriptsubscript𝑚21𝛽superscriptsubscript𝑚2𝛽2superscript𝑁𝑙22𝛽subscriptΔ𝑚2𝑀𝛽subscript𝑚superscript𝑀2\displaystyle\leqslant\ \beta m\_{\max}^{2}\ +\ 1\ -\ \beta\ m\_{\max}^{2}\ -\ \beta\ 2\ (N-l)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{m}\ +\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)\ M^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | = 1−β​ 2​(N−l)2​exp⁡(− 2​β​(Δm+2​Mβ​mmax))​M2<1.absent1𝛽2superscript𝑁𝑙22𝛽subscriptΔ𝑚2𝑀𝛽subscript𝑚superscript𝑀21\displaystyle=\ 1\ -\ \beta\ 2\ (N-l)^{2}\ \exp\left(-\ 2\ \beta\ \left(\Delta\_{m}\ +\ \frac{2\ M}{\beta\ m\_{\max}}\right)\right)\ M^{2}\ <1\ . |  |

For the last but one inequality we used
2​M⩽max⁡{mmax,2​M}2𝑀subscript𝑚2𝑀2M\leqslant\max\{m\_{\max},2M\}.

Therefore, f𝑓f is a contraction mapping in SmsubscriptS𝑚\mathrm{S}\_{m}.
∎

•Banach Fixed Point Theorem.
Now we have all ingredients to apply Banach fixed point theorem.

###### Lemma A12.

Assume that

1. A1:

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | ΔmsubscriptΔ𝑚\displaystyle\Delta\_{m}\ | ≥2​Mβ​mmax−1β​ln⁡(1−β​mmax22​β​(N−l)​M​max⁡{mmax, 2​M}),absent2𝑀𝛽subscript𝑚1𝛽1𝛽superscriptsubscript𝑚22𝛽𝑁𝑙𝑀subscript𝑚2𝑀\displaystyle\geq\ \frac{2\ M}{\beta\ m\_{\max}}\ -\ \frac{1}{\beta}\ \ln\left(\frac{1\ -\ \beta\ m\_{\max}^{2}}{2\ \beta\ (N-l)\ M\ \max\{m\_{\max}\ ,\ 2\ M\}}\right)\ , |  | (276) |

   and
2. A2:

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | β​mmax2𝛽superscriptsubscript𝑚2\displaystyle\beta\ m\_{\max}^{2}\ | ⩽ 1,absent1\displaystyle\leqslant\ 1\ , |  | (277) |

then f𝑓f has a fixed point in SmsubscriptS𝑚\mathrm{S}\_{m}.

###### Proof.

We use Banach fixed point theorem:
Lemma [A10](#ThmlemmaA10 "Lemma A10. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") says that f𝑓f maps from the compact set SmsubscriptS𝑚\mathrm{S}\_{m} into
the same compact set SmsubscriptS𝑚\mathrm{S}\_{m}.
Lemma [A11](#ThmlemmaA11 "Lemma A11. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") says that f𝑓f is a contraction mapping in
SmsubscriptS𝑚\mathrm{S}\_{m}.
∎

•Contraction mapping with a fixed point.

We assume that the first l𝑙l patterns
are much more probable (and similar
to one another) than the other patterns.
Therefore, we define:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M𝑀\displaystyle M\ | :=maxi⁡‖𝒙i‖,assignabsentsubscript𝑖normsubscript𝒙𝑖\displaystyle:=\ \max\_{i}{{\left\|\bm{x}\_{i}\right\|}}\ , |  | (278) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | γ𝛾\displaystyle\gamma\ | =∑i=l+1Npi⩽ϵ,absentsuperscriptsubscript𝑖𝑙1𝑁subscript𝑝𝑖italic-ϵ\displaystyle=\ \sum\_{i=l+1}^{N}p\_{i}\ \leqslant\ \epsilon\ , |  | (279) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1−γ1𝛾\displaystyle 1-\gamma\ | =∑i=1lpi≥ 1−ϵ,absentsuperscriptsubscript𝑖1𝑙subscript𝑝𝑖1italic-ϵ\displaystyle=\ \sum\_{i=1}^{l}p\_{i}\ \geq\ 1\ -\ \epsilon\ , |  | (280) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | p~isubscript~𝑝𝑖\displaystyle\tilde{p}\_{i}\ | :=pi1−γ⩽pi/(1−ϵ),assignabsentsubscript𝑝𝑖1𝛾subscript𝑝𝑖1italic-ϵ\displaystyle:=\ \frac{p\_{i}}{1-\gamma}\ \leqslant\ p\_{i}/(1-\epsilon)\ , |  | (281) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i=1lp~isuperscriptsubscript𝑖1𝑙subscript~𝑝𝑖\displaystyle\sum\_{i=1}^{l}\tilde{p}\_{i}\ | = 1,absent1\displaystyle=\ 1\ , |  | (282) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒎𝒙subscript𝒎𝒙\displaystyle\bm{m}\_{\bm{x}}\ | =1l​∑i=1l𝒙i,absent1𝑙superscriptsubscript𝑖1𝑙subscript𝒙𝑖\displaystyle=\ \frac{1}{l}\ \sum\_{i=1}^{l}\ \bm{x}\_{i}\ , |  | (283) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | mmaxsubscript𝑚\displaystyle m\_{\max}\ | =max1⩽i⩽l⁡‖𝒙i−𝒎𝒙‖.absentsubscript1𝑖𝑙normsubscript𝒙𝑖subscript𝒎𝒙\displaystyle=\ \max\_{1\leqslant i\leqslant l}{{\left\|\bm{x}\_{i}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ . |  | (284) |

M𝑀M is an upper bound on the Euclidean norm of the patterns, which are vectors.
ϵitalic-ϵ\epsilon is an upper bound on the probability γ𝛾\gamma of not choosing one of the first l𝑙l patterns, while
1−ϵ1italic-ϵ1-\epsilon is a lower bound the probability (1−γ)1𝛾(1-\gamma)
of choosing one of the first l𝑙l patterns.
𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} is the arithmetic mean (the center) of the first l𝑙l
patterns.
mmaxsubscript𝑚m\_{\max} is the maximal distance of the patterns to the center 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} .
𝒑~~𝒑\tilde{\bm{p}} is the probability 𝒑𝒑\bm{p}
normalized for the first l𝑙l patterns.

The variance of the first l𝑙l patterns is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Varp~​[𝒙1:l]subscriptVar~𝑝delimited-[]subscript𝒙:1𝑙\displaystyle\mathbf{\mathrm{Var}}\_{\tilde{p}}[\bm{x}\_{1:l}]\ | =∑i=1lp~i​𝒙i​𝒙iT−(∑i=1lp~i​𝒙i)​(∑i=1lp~i​𝒙i)Tabsentsuperscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇superscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖superscriptsuperscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{l}\tilde{p}\_{i}\ \bm{x}\_{i}\ \bm{x}\_{i}^{T}\ -\ \left(\sum\_{i=1}^{l}\tilde{p}\_{i}\ \bm{x}\_{i}\right)\ \left(\sum\_{i=1}^{l}\tilde{p}\_{i}\ \bm{x}\_{i}\right)^{T} |  | (285) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1lp~i​(𝒙i−∑i=1lp~i​𝒙i)​(𝒙i−∑i=1lp~i​𝒙i)T.absentsuperscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖superscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖superscriptsubscript𝒙𝑖superscriptsubscript𝑖1𝑙subscript~𝑝𝑖subscript𝒙𝑖𝑇\displaystyle=\ \sum\_{i=1}^{l}\tilde{p}\_{i}\ \left(\bm{x}\_{i}\ -\ \sum\_{i=1}^{l}\tilde{p}\_{i}\bm{x}\_{i}\right)\ \left(\bm{x}\_{i}\ -\ \sum\_{i=1}^{l}\tilde{p}\_{i}\bm{x}\_{i}\right)^{T}\ . |  |

We have shown that a fixed point exists. We want to know
how fast the iteration converges to the fixed point.
Let 𝒎𝒙∗superscriptsubscript𝒎𝒙\bm{m}\_{\bm{x}}^{\*} be the fixed point of the iteration f𝑓f in the sphere SmsubscriptS𝑚\mathrm{S}\_{m}.
Using the mean value theorem Lemma [A32](#ThmlemmaA32 "Lemma A32 (Mean Value Theorem). ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), we have with
Jm=∫01J​(λ​𝝃+(1−λ)​𝒎𝒙∗)​dλsuperscriptJ𝑚superscriptsubscript01J𝜆𝝃1𝜆superscriptsubscript𝒎𝒙differential-d𝜆\mathrm{J}^{m}=\int\_{0}^{1}\mathrm{J}(\lambda\bm{\xi}+(1-\lambda)\bm{m}\_{\bm{x}}^{\*})\ \mathrm{d}\lambda:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒎𝒙∗‖norm𝑓𝝃superscriptsubscript𝒎𝒙\displaystyle{{\left\|f(\bm{\xi})\ -\ \bm{m}\_{\bm{x}}^{\*}\right\|}}\ | =‖f​(𝝃)−f​(𝒎𝒙∗)‖⩽‖Jm‖2​‖𝝃−𝒎𝒙∗‖absentnorm𝑓𝝃𝑓superscriptsubscript𝒎𝒙subscriptnormsuperscriptJ𝑚2norm𝝃superscriptsubscript𝒎𝒙\displaystyle=\ {{\left\|f(\bm{\xi})\ -\ f(\bm{m}\_{\bm{x}}^{\*})\right\|}}\ \leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}^{\*}\right\|}} |  | (286) |

According to Lemma [A8](#ThmlemmaA8 "Lemma A8. ‣ A.1.5.4 Metastable States: Fixed Points Near Mean of Similar Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
the following bounds on the norm ‖J‖2subscriptnormJ2{{\left\|\mathrm{J}\right\|}}\_{2} of the
Jacobian of the fixed point iteration hold.
The γ𝛾\gamma-bound for ‖J‖2subscriptnormJ2{{\left\|\mathrm{J}\right\|}}\_{2} is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽β​((1−γ)​mmax2+γ​ 2​(2−γ)​M2),absent𝛽1𝛾superscriptsubscript𝑚2𝛾22𝛾superscript𝑀2\displaystyle\leqslant\ \beta\left((1-\gamma)\ m\_{\max}^{2}\ +\ \gamma\ 2\ (2\ -\ \gamma)\ M^{2}\right)\ , |  | (287) |

while the ϵitalic-ϵ\epsilon-bound for ‖J‖2subscriptnormJ2{{\left\|\mathrm{J}\right\|}}\_{2} is:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖J‖2subscriptnormJ2\displaystyle{{\left\|\mathrm{J}\right\|}}\_{2}\ | ⩽β​(mmax2+ϵ​ 2​(2−ϵ)​M2).absent𝛽superscriptsubscript𝑚2italic-ϵ22italic-ϵsuperscript𝑀2\displaystyle\leqslant\ \beta\left(\ m\_{\max}^{2}\ +\ \epsilon\ 2\ (2\ -\ \epsilon)\ M^{2}\right)\ . |  | (288) |

From the last condition we require
for a contraction mapping:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | βmmax2  𝛽superscriptsubscript𝑚2\displaystyle\beta\ \ m\_{\max}^{2}\ | < 1.absent1\displaystyle<\ 1\ . |  | (289) |

We want to see how large ϵitalic-ϵ\epsilon is.
The separation of center
𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} from data 𝑿=(𝒙l+1,…,𝒙N)𝑿subscript𝒙𝑙1…subscript𝒙𝑁\bm{X}=(\bm{x}\_{l+1},\ldots,\bm{x}\_{N})
is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔmsubscriptΔ𝑚\displaystyle\Delta\_{m}\ | =minj,l<j⁡(𝒎𝒙T​𝒎𝒙−𝒎𝒙T​𝒙j)=𝒎𝒙T​𝒎𝒙−maxj,l<j⁡𝒎𝒙T​𝒙j.absentsubscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙subscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,l<j}\left(\bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ \bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\right)\ =\ \bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ \max\_{j,l<j}\bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\ . |  | (290) |

We need the separation Δ~msubscript~Δ𝑚\tilde{\Delta}\_{m} of
𝒙~=λ​𝝃+(1−λ)​𝒎𝒙∗~𝒙𝜆𝝃1𝜆superscriptsubscript𝒎𝒙\tilde{\bm{x}}=\lambda\bm{\xi}+(1-\lambda)\bm{m}\_{\bm{x}}^{\*} from the data.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~msubscript~Δ𝑚\displaystyle\tilde{\Delta}\_{m}\ | =minj,l<j⁡(𝒙~T​𝒎𝒙−𝒙~T​𝒙j).absentsubscript  𝑗𝑙 𝑗superscript~𝒙𝑇subscript𝒎𝒙superscript~𝒙𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,l<j}\left(\tilde{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ \tilde{\bm{x}}^{T}\bm{x}\_{j}\right)\ . |  | (291) |

We compute a lower bound on Δ~msubscript~Δ𝑚\tilde{\Delta}\_{m}.
Using the Cauchy-Schwarz inequality, we obtain for 1⩽j⩽N1𝑗𝑁1\leqslant j\leqslant N:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |𝒙~T​𝒙j−𝒎𝒙T​𝒙j|superscript~𝒙𝑇subscript𝒙𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗\displaystyle\left|\tilde{\bm{x}}^{T}\bm{x}\_{j}\ -\ \bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\right| | ⩽‖𝒙~−𝒎𝒙‖​‖𝒙j‖⩽‖𝒙~−𝒎𝒙‖​M.absentnorm~𝒙subscript𝒎𝒙normsubscript𝒙𝑗norm~𝒙subscript𝒎𝒙𝑀\displaystyle\leqslant\ {{\left\|\tilde{\bm{x}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ {{\left\|\bm{x}\_{j}\right\|}}\ \leqslant\ {{\left\|\tilde{\bm{x}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ . |  | (292) |

We have the lower bound

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~msubscript~Δ𝑚\displaystyle\tilde{\Delta}\_{m}\ | ≥minj,l<j⁡((𝒎𝒙T​𝒎𝒙−‖𝒙~−𝒎𝒙‖​M)−(𝒎𝒙T​𝒙j+‖𝒙~−𝒎𝒙‖​M))absentsubscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙norm~𝒙subscript𝒎𝒙𝑀superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗norm~𝒙subscript𝒎𝒙𝑀\displaystyle\geq\ \min\_{j,l<j}\left(\left(\bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ {{\left\|\tilde{\bm{x}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\right)\ -\ \left(\bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\ +\ {{\left\|\tilde{\bm{x}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\right)\right) |  | (293) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =− 2​‖𝒙~−𝒎𝒙‖​M+minj,l<j⁡(𝒎𝒙T​𝒎𝒙−𝒎𝒙T​𝒙j)=Δm− 2​‖𝒙~−𝒎𝒙‖​M.absent2norm~𝒙subscript𝒎𝒙𝑀subscript  𝑗𝑙 𝑗superscriptsubscript𝒎𝒙𝑇subscript𝒎𝒙superscriptsubscript𝒎𝒙𝑇subscript𝒙𝑗subscriptΔ𝑚2norm~𝒙subscript𝒎𝒙𝑀\displaystyle=\ -\ 2\ {{\left\|\tilde{\bm{x}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ +\ \min\_{j,l<j}\left(\bm{m}\_{\bm{x}}^{T}\bm{m}\_{\bm{x}}\ -\ \bm{m}\_{\bm{x}}^{T}\bm{x}\_{j}\right)\ =\ \Delta\_{m}\ -\ 2\ {{\left\|\tilde{\bm{x}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ M\ . |  |

Since

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒙~−𝒎𝒙‖norm~𝒙subscript𝒎𝒙\displaystyle{{\left\|\tilde{\bm{x}}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ | =‖λ​𝝃+(1−λ)​𝒎𝒙∗−𝒎𝒙‖absentnorm𝜆𝝃1𝜆superscriptsubscript𝒎𝒙subscript𝒎𝒙\displaystyle=\ {{\left\|\lambda\bm{\xi}+(1-\lambda)\bm{m}\_{\bm{x}}^{\*}\ -\ \bm{m}\_{\bm{x}}\right\|}} |  | (294) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽λ​‖𝝃−𝒎𝒙‖+(1−λ)​‖𝒎𝒙∗−𝒎𝒙‖absent𝜆norm𝝃subscript𝒎𝒙1𝜆normsuperscriptsubscript𝒎𝒙subscript𝒎𝒙\displaystyle\leqslant\ \lambda\ {{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}}\ +\ (1-\lambda)\ {{\left\|\bm{m}\_{\bm{x}}^{\*}\ -\ \bm{m}\_{\bm{x}}\right\|}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⩽max⁡{‖𝝃−𝒎𝒙‖,‖𝒎𝒙∗−𝒎𝒙‖},absentnorm𝝃subscript𝒎𝒙normsuperscriptsubscript𝒎𝒙subscript𝒎𝒙\displaystyle\leqslant\ \max\{{{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}},{{\left\|\bm{m}\_{\bm{x}}^{\*}\ -\ \bm{m}\_{\bm{x}}\right\|}}\}\ , |  |

we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ~msubscript~Δ𝑚\displaystyle\tilde{\Delta}\_{m}\ | ≥Δm− 2​max⁡{‖𝝃−𝒎𝒙‖,‖𝒎𝒙∗−𝒎𝒙‖}​M.absentsubscriptΔ𝑚2norm𝝃subscript𝒎𝒙normsuperscriptsubscript𝒎𝒙subscript𝒎𝒙𝑀\displaystyle\geq\ \Delta\_{m}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}},{{\left\|\bm{m}\_{\bm{x}}^{\*}\ -\ \bm{m}\_{\bm{x}}\right\|}}\}\ M\ . |  | (295) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵitalic-ϵ\displaystyle\epsilon\ | =(N−l)​exp⁡(−β​(Δm− 2​max⁡{‖𝝃−𝒎𝒙‖,‖𝒎𝒙∗−𝒎𝒙‖}​M)).absent𝑁𝑙𝛽subscriptΔ𝑚2norm𝝃subscript𝒎𝒙normsuperscriptsubscript𝒎𝒙subscript𝒎𝒙𝑀\displaystyle=\ (N-l)\exp(-\ \beta\ (\Delta\_{m}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{m}\_{\bm{x}}\right\|}},{{\left\|\bm{m}\_{\bm{x}}^{\*}\ -\ \bm{m}\_{\bm{x}}\right\|}}\}\ M))\ . |  | (296) |

#### A.1.6 Properties of Fixed Points Near Stored Pattern

In Subsection [A.1.5.3](#A1.SS1.SSS5.P3 "A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
many stable states that are fixed points near the
stored patterns are considered.
We now consider this case.
In the fist subsection
we investigate the storage capacity if all patterns are sufficiently
separated so that metastable states do not appear.
In the next subsection
we look into the updates required and error when
retrieving the stored patterns.
For metastable states we can do the same analyses if each
metastable state is treated as one state like one pattern.

We see a trade-off that is known from classical Hopfield networks
and for modern Hopfield networks.
Small separation ΔisubscriptΔ𝑖\Delta\_{i} of the pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} from the other
patterns gives high storage capacity. However the convergence
speed is lower and the retrieval error higher.
In contrast, large separation ΔisubscriptΔ𝑖\Delta\_{i} of the pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} from the other
pattern allows the retrieval of patterns with one update step
and exponentially low error.

##### A.1.6.1 Exponentially Many Patterns can be Stored.

From Subsection [A.1.5.3](#A1.SS1.SSS5.P3 "A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") need some definitions.
We assume to have N𝑁N patterns,
the separation of pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} from the other patterns {𝒙1,…,𝒙i−1,𝒙i+1,…,𝒙N}subscript𝒙1…subscript𝒙𝑖1subscript𝒙𝑖1…subscript𝒙𝑁{{\left\{\bm{x}\_{1},\ldots,\bm{x}\_{i-1},\bm{x}\_{i+1},\ldots,\bm{x}\_{N}\right\}}}
is ΔisubscriptΔ𝑖\Delta\_{i}, defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | =minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)=𝒙iT​𝒙i−maxj,j≠i⁡𝒙iT​𝒙j.absentsubscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖subscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right)\ =\ \bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \max\_{j,j\not=i}\bm{x}\_{i}^{T}\bm{x}\_{j}\ . |  | (297) |

The pattern is separated from the other data if 0<Δi0subscriptΔ𝑖0<\Delta\_{i}.
The separation ΔisubscriptΔ𝑖\Delta\_{i} can also be expressed as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | =minj,j≠i⁡12​(‖𝒙i‖2−‖𝒙j‖2+‖𝒙i−𝒙j‖2)absentsubscript  𝑗𝑗 𝑖12superscriptnormsubscript𝒙𝑖2superscriptnormsubscript𝒙𝑗2superscriptnormsubscript𝒙𝑖subscript𝒙𝑗2\displaystyle=\ \min\_{j,j\not=i}\frac{1}{2}\ \left({{\left\|\bm{x}\_{i}\right\|}}^{2}\ -\ {{\left\|\bm{x}\_{j}\right\|}}^{2}\ +\ {{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}^{2}\right) |  | (298) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =12​‖𝒙i‖2−12​maxj,j≠i⁡(‖𝒙j‖2−‖𝒙i−𝒙j‖2).absent12superscriptnormsubscript𝒙𝑖212subscript  𝑗𝑗 𝑖superscriptnormsubscript𝒙𝑗2superscriptnormsubscript𝒙𝑖subscript𝒙𝑗2\displaystyle=\ \frac{1}{2}{{\left\|\bm{x}\_{i}\right\|}}^{2}\ -\ \frac{1}{2}\ \max\_{j,j\not=i}\left({{\left\|\bm{x}\_{j}\right\|}}^{2}\ -\ {{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}^{2}\right)\ . |  |

For ‖𝒙i‖=‖𝒙j‖normsubscript𝒙𝑖normsubscript𝒙𝑗{{\left\|\bm{x}\_{i}\right\|}}={{\left\|\bm{x}\_{j}\right\|}} we have Δi=1/2​minj,j≠i⁡‖𝒙i−𝒙j‖2subscriptΔ𝑖12subscript

𝑗𝑗
𝑖superscriptnormsubscript𝒙𝑖subscript𝒙𝑗2\Delta\_{i}=1/2\min\_{j,j\not=i}{{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}^{2}.
The sphere SisubscriptS𝑖\mathrm{S}\_{i} with center 𝒙isubscript𝒙𝑖\bm{x}\_{i} is defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | SisubscriptS𝑖\displaystyle\mathrm{S}\_{i}\ | ={𝝃∣‖𝝃−𝒙i‖⩽1β​N​M}.absentconditional-set𝝃norm𝝃subscript𝒙𝑖1𝛽𝑁𝑀\displaystyle=\ \left\{\bm{\xi}\mid{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}}\ \leqslant\ \frac{1}{\beta\ N\ M}\right\}\ . |  | (299) |

The maximal length of a pattern is M=maxi⁡‖𝒙i‖𝑀subscript𝑖normsubscript𝒙𝑖M=\max\_{i}{{\left\|\bm{x}\_{i}\right\|}}.

We next define what we mean with storing and retrieving a pattern.

###### Definition 5 (Pattern Stored and Retrieved).

We assume that around every pattern 𝐱isubscript𝐱𝑖\bm{x}\_{i} a sphere SisubscriptS𝑖\mathrm{S}\_{i} is given.
We say 𝐱isubscript𝐱𝑖\bm{x}\_{i} is stored if there is a single fixed point 𝐱i∗∈Sisuperscriptsubscript𝐱𝑖subscriptS𝑖\bm{x}\_{i}^{\*}\in\mathrm{S}\_{i} to
which all points 𝛏∈Si𝛏subscriptS𝑖\bm{\xi}\in\mathrm{S}\_{i} converge,
and Si∩Sj=∅subscriptS𝑖subscriptS𝑗\mathrm{S}\_{i}\cap\mathrm{S}\_{j}=\emptyset for i≠j𝑖𝑗i\not=j.
We say 𝐱isubscript𝐱𝑖\bm{x}\_{i} is retrieved for a given ϵitalic-ϵ\epsilon if
iteration (update rule) Eq. ([92](#A1.E92 "In A.1.5.2 One Stable State: Fixed Point Near the Mean of the Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) gives
a point 𝐱~isubscript~𝐱𝑖\tilde{\bm{x}}\_{i} that is at least
ϵitalic-ϵ\epsilon-close to the single fixed point 𝐱i∗∈Sisuperscriptsubscript𝐱𝑖subscriptS𝑖\bm{x}\_{i}^{\*}\in\mathrm{S}\_{i}.
The retrieval error is ‖𝐱~i−𝐱i‖normsubscript~𝐱𝑖subscript𝐱𝑖{{\left\|\tilde{\bm{x}}\_{i}-\bm{x}\_{i}\right\|}}.

The sphere SisubscriptS𝑖\mathrm{S}\_{i} around pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} can be any a sphere
and do not have the specific sphere defined in Def. [3](#Thmdefinition3 "Definition 3 (Sphere S_𝑖). ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").

For a query 𝝃∈Si𝝃subscriptS𝑖\bm{\xi}\in\mathrm{S}\_{i} to converge to a fixed point
𝒙i∗∈Sisuperscriptsubscript𝒙𝑖subscriptS𝑖\bm{x}\_{i}^{\*}\in\mathrm{S}\_{i} we required for the application
of Banach fixed point theorem
and for ensuring a contraction mapping
the following inequality:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | ≥2β​N+1β​ln⁡(2​(N−1)​N​β​M2).absent2𝛽𝑁1𝛽2𝑁1𝑁𝛽superscript𝑀2\displaystyle\geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right)\ . |  | (300) |

This is the assumption in Lemma [A7](#ThmlemmaA7 "Lemma A7. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
to ensure a fixed point in sphere SisubscriptS𝑖\mathrm{S}\_{i}.
Since replacing (N−1)​N𝑁1𝑁(N-1)N by N2superscript𝑁2N^{2} gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2β​N+1β​ln⁡(2​N2​β​M2)>2β​N+1β​ln⁡(2​(N−1)​N​β​M2),2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀22𝛽𝑁1𝛽2𝑁1𝑁𝛽superscript𝑀2\displaystyle\frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\ >\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ (N-1)\ N\ \beta\ M^{2}\right)\ , |  | (301) |

the inequality follows from following master inequality

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | ≥2β​N+1β​ln⁡(2​N2​β​M2),absent2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2\displaystyle\geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\ , |  | (302) |

If we assume that Si∩Sj≠∅subscriptS𝑖subscriptS𝑗\mathrm{S}\_{i}\cap\mathrm{S}\_{j}\neq\emptyset with i≠j𝑖𝑗i\neq j, then
the triangle inequality with a point from the intersection gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒙i−𝒙j‖normsubscript𝒙𝑖subscript𝒙𝑗\displaystyle{{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}\ | ⩽2β​N​M.absent2𝛽𝑁𝑀\displaystyle\leqslant\ \frac{2}{\beta\ N\ M}\ . |  | (303) |

Therefore, we have using the Cauchy-Schwarz inequality:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | ⩽𝒙iT​(𝒙i−𝒙j)⩽‖𝒙i‖​‖𝒙i−𝒙j‖⩽M​2β​N​M=2β​N.absentsuperscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖subscript𝒙𝑗normsubscript𝒙𝑖normsubscript𝒙𝑖subscript𝒙𝑗𝑀2𝛽𝑁𝑀2𝛽𝑁\displaystyle\leqslant\ \bm{x}\_{i}^{T}\left(\bm{x}\_{i}\ -\ \bm{x}\_{j}\right)\ \leqslant\ {{\left\|\bm{x}\_{i}\right\|}}\ {{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{j}\right\|}}\ \leqslant M\ \frac{2}{\beta\ N\ M}\ =\ \frac{2}{\beta\ N}\ . |  | (304) |

The last inequality is a contraction to Eq. ([302](#A1.E302 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
if we assume that

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 11\displaystyle 1\ | < 2​(N−1)​N​β​M2.absent2𝑁1𝑁𝛽superscript𝑀2\displaystyle<\ 2\ (N-1)\ N\ \beta\ M^{2}\ . |  | (305) |

With this assumption, the spheres SisubscriptS𝑖\mathrm{S}\_{i} and SjsubscriptS𝑗\mathrm{S}\_{j} do
not intersect. Therefore, each 𝒙isubscript𝒙𝑖\bm{x}\_{i} has its separate fixed point in
SisubscriptS𝑖\mathrm{S}\_{i}.
We define

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔminsubscriptΔ\displaystyle\Delta\_{\min}\ | =min1⩽i⩽N⁡Δiabsentsubscript1𝑖𝑁subscriptΔ𝑖\displaystyle=\ \min\_{1\leqslant i\leqslant N}\Delta\_{i} |  | (306) |

to obtain the master inequality

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔminsubscriptΔ\displaystyle\Delta\_{\min}\ | ≥2β​N+1β​ln⁡(2​N2​β​M2).absent2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2\displaystyle\geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\ . |  | (307) |

•Patterns on a sphere.

For simplicity and in accordance with the results of the classical Hopfield
network, we assume all patterns being on a sphere with radius M𝑀M:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∀i:‖𝒙i‖:subscriptfor-all𝑖normsubscript𝒙𝑖\displaystyle\forall\_{i}:\ {{\left\|\bm{x}\_{i}\right\|}}\ | =M.absent𝑀\displaystyle=\ M\ . |  | (308) |

Under assumption Eq. ([305](#A1.E305 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) we have only
to show that the master inequality Eq. ([307](#A1.E307 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) is fulfilled
for each 𝒙isubscript𝒙𝑖\bm{x}\_{i} to have a separate fixed point near each 𝒙isubscript𝒙𝑖\bm{x}\_{i}.

We defined αi​jsubscript𝛼𝑖𝑗\alpha\_{ij} as the angle between 𝒙isubscript𝒙𝑖\bm{x}\_{i} and 𝒙jsubscript𝒙𝑗\bm{x}\_{j}.
The minimal angle αminsubscript𝛼\alpha\_{\min} between two
data points is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | αminsubscript𝛼\displaystyle\alpha\_{\min}\ | =min1⩽i<j⩽N⁡αi​j.absentsubscript1𝑖𝑗𝑁subscript𝛼𝑖𝑗\displaystyle=\ \min\_{1\leqslant i<j\leqslant N}\alpha\_{ij}\ . |  | (309) |

On the sphere with radius M𝑀M we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔminsubscriptΔ\displaystyle\Delta\_{\min}\ | =min1⩽i<j⩽N⁡M2​(1−cos⁡(αi​j))=M2​(1−cos⁡(αmin)),absentsubscript1𝑖𝑗𝑁superscript𝑀21subscript𝛼𝑖𝑗superscript𝑀21subscript𝛼\displaystyle=\ \min\_{1\leqslant i<j\leqslant N}M^{2}(1\ -\ \cos(\alpha\_{ij}))\ =\ M^{2}(1\ -\ \cos(\alpha\_{\min}))\ , |  | (310) |

therefore it is sufficient to show the master inequality on the sphere:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M2​(1−cos⁡(αmin))superscript𝑀21subscript𝛼\displaystyle M^{2}(1\ -\ \cos(\alpha\_{\min}))\ | ≥2β​N+1β​ln⁡(2​N2​β​M2).absent2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2\displaystyle\geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\ . |  | (311) |

Under assumption Eq. ([305](#A1.E305 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) we have only
to show that the master inequality Eq. ([307](#A1.E307 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) is fulfilled
for ΔminsubscriptΔ\Delta\_{\min}. We consider patterns on the sphere, therefore the master
inequality Eq. ([307](#A1.E307 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) becomes Eq. ([311](#A1.E311 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
First we show results when pattern positions
on the sphere are constructed and ΔminsubscriptΔ\Delta\_{\min} is ensured.
Then we move on to
random patterns on a sphere, where ΔminsubscriptΔ\Delta\_{\min} becomes a random variable.

•Storage capacity for patterns placed on the sphere.

Next theorem says how many patterns we can stored
(fixed point with attraction basin near pattern)
if we are allowed to place them on the sphere.

###### Theorem A3 (Storage Capacity (M=2): Placed Patterns).

We assume β=1𝛽1\beta=1 and patterns on the sphere with radius M𝑀M.
If M=2​d−1𝑀2𝑑1M=2\sqrt{d-1} and the dimension d𝑑d
of the space is d≥4𝑑4d\geq 4 or if
M=1.7​d−1𝑀1.7𝑑1M=1.7\sqrt{d-1} and the dimension d𝑑d
of the space is d≥50𝑑50d\geq 50,
then
the number of patterns N𝑁N that can be stored
(fixed point with attraction basin near pattern) is
at least

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | = 22​(d−1).absentsuperscript22𝑑1\displaystyle=\ 2^{2(d-1)}\ . |  | (312) |

###### Proof.

For random patterns on the sphere,
we have to show that the master inequality Eq. ([311](#A1.E311 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
holds:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M2​(1−cos⁡(αmin))superscript𝑀21subscript𝛼\displaystyle M^{2}(1\ -\ \cos(\alpha\_{\min}))\ | ≥2β​N+1β​ln⁡(2​N2​β​M2).absent2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2\displaystyle\geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\ . |  | (313) |

We now place the patterns equidistant on the sphere where the pattern
are separated by an angle αminsubscript𝛼\alpha\_{\min}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∀i:minj,j≠i⁡αi​j=αmin,:subscriptfor-all𝑖subscript  𝑗𝑗 𝑖subscript𝛼𝑖𝑗subscript𝛼\displaystyle\forall\_{i}:\ \min\_{j,j\not=i}\alpha\_{ij}\ =\ \alpha\_{\min}\ , |  | (314) |

In a d𝑑d-dimensional space we can place

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | =(2​παmin)d−1absentsuperscript2𝜋subscript𝛼𝑑1\displaystyle=\ \left(\frac{2\pi}{\alpha\_{\min}}\right)^{d-1} |  | (315) |

points on the sphere.
In a spherical coordinate system a pattern differs from its most
closest patterns by an angle αminsubscript𝛼\alpha\_{\min} and there are d−1𝑑1d-1 angles.
Solving for αminsubscript𝛼\alpha\_{\min} gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | αminsubscript𝛼\displaystyle\alpha\_{\min}\ | =2​πN1/(d−1).absent2𝜋superscript𝑁1𝑑1\displaystyle=\ \frac{2\pi}{N^{1/(d-1)}}\ . |  | (316) |

The number of patterns that can be stored is determined
by the largest N𝑁N that fulfils

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | M2​(1−cos⁡(2​πN1/(d−1)))≥2β​N+1β​ln⁡(2​N2​β​M2).superscript𝑀212𝜋superscript𝑁1𝑑12𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2\displaystyle M^{2}\left(1\ -\ \cos\left(\frac{2\pi}{N^{1/(d-1)}}\right)\right)\ \geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\ . |  | (317) |

We set N=22​(d−1)𝑁superscript22𝑑1N=2^{2(d-1)} and obtain for Eq. ([317](#A1.E317 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | M2​(1−cos⁡(π2))≥2β​ 23​(d−1)+1β​ln⁡(2​β​M2)+1β​ 4​(d−1)​ln⁡2.superscript𝑀21𝜋22𝛽superscript23𝑑11𝛽2𝛽superscript𝑀21𝛽4𝑑12\displaystyle M^{2}\left(1\ -\ \cos\left(\frac{\pi}{2}\right)\right)\ \geq\ \frac{2}{\beta\ 2^{3(d-1)}}\ +\ \frac{1}{\beta}\ \ln\left(2\ \beta\ M^{2}\right)\ +\ \frac{1}{\beta}\ 4\ (d-1)\ln 2\ . |  | (318) |

This inequality is equivalent to

|  |  |  |  |
| --- | --- | --- | --- |
|  | β​M2≥122​(d−1)−1+ln⁡(2​β​M2)+ 4​(d−1)​ln⁡2.𝛽superscript𝑀21superscript22𝑑112𝛽superscript𝑀24𝑑12\displaystyle\beta\ M^{2}\ \geq\ \frac{1}{2^{2(d-1)-1}}\ +\ \ln\left(2\ \beta\ M^{2}\right)\ +\ 4\ (d-1)\ln 2\ . |  | (319) |

The last inequality can be fulfilled with M=K​d−1𝑀𝐾𝑑1M=K\sqrt{d-1} and proper K𝐾K.
For β=1𝛽1\beta=1, d=4𝑑4d=4 and K=2𝐾2K=2 the inequality is fulfilled.
The left hand side minus the right hand side is
4​(d−1)−1/22​(d−1)−1−ln⁡(8​(d−1))−4​(d−1)​ln⁡24𝑑11superscript22𝑑118𝑑14𝑑124(d-1)-1/2^{2(d-1)-1}-\ln(8(d-1))-4(d-1)\ln 2.
Its derivative with respect to d𝑑d is strict positive.
Therefore, the inequality holds for
d≥4𝑑4d\geq 4.

For β=1𝛽1\beta=1, d=50𝑑50d=50 and K=1.7𝐾1.7K=1.7 the inequality is fulfilled.
The left hand side minus the right hand side is
2.89​(d−1)−1/22​(d−1)−1−ln⁡(5.78​(d−1))−4​(d−1)​ln⁡22.89𝑑11superscript22𝑑115.78𝑑14𝑑122.89(d-1)-1/2^{2(d-1)-1}-\ln(5.78(d-1))-4(d-1)\ln 2.
Its derivative with respect to d𝑑d is strict positive.
Therefore, the inequality holds for
d≥50𝑑50d\geq 50.

∎

If we want to store considerably more patterns, then
we have to increase the length of the vectors or the dimension
of the space where the vectors live.
The next theorem shows results for the number of patterns N𝑁N with N=23​(d−1)𝑁superscript23𝑑1N=2^{3(d-1)}.

###### Theorem A4 (Storage Capacity (M=5): Placed Patterns).

We assume β=1𝛽1\beta=1 and patterns on the sphere with radius M𝑀M.
If M=5​d−1𝑀5𝑑1M=5\sqrt{d-1} and the dimension d𝑑d
of the space is d≥3𝑑3d\geq 3 or if
M=4​d−1𝑀4𝑑1M=4\sqrt{d-1} and the dimension d𝑑d
of the space is d≥13𝑑13d\geq 13,
then
the number of patterns N𝑁N that can be stored
(fixed point with attraction basin near pattern) is
at least

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | = 23​(d−1).absentsuperscript23𝑑1\displaystyle=\ 2^{3(d-1)}\ . |  | (320) |

###### Proof.

We set N=23​(d−1)𝑁superscript23𝑑1N=2^{3(d-1)} and obtain for Eq. ([317](#A1.E317 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | M2​(1−cos⁡(π4))≥2β​ 23​(d−1)+1β​ln⁡(2​β​M2)+1β​ 6​(d−1)​ln⁡2.superscript𝑀21𝜋42𝛽superscript23𝑑11𝛽2𝛽superscript𝑀21𝛽6𝑑12\displaystyle M^{2}\left(1\ -\ \cos\left(\frac{\pi}{4}\right)\right)\ \geq\ \frac{2}{\beta\ 2^{3(d-1)}}\ +\ \frac{1}{\beta}\ \ln\left(2\ \beta\ M^{2}\right)\ +\ \frac{1}{\beta}\ 6\ (d-1)\ln 2\ . |  | (321) |

This inequality is equivalent to

|  |  |  |  |
| --- | --- | --- | --- |
|  | β​M2​(1−22)≥123​(d−1)−1+ln⁡(2​β​M2)+ 6​(d−1)​ln⁡2.𝛽superscript𝑀21221superscript23𝑑112𝛽superscript𝑀26𝑑12\displaystyle\beta\ M^{2}\left(1\ -\ \frac{\sqrt{2}}{2}\right)\ \geq\ \frac{1}{2^{3(d-1)-1}}\ +\ \ln\left(2\ \beta\ M^{2}\right)\ +\ 6\ (d-1)\ln 2\ . |  | (322) |

The last inequality can be fulfilled with M=K​d−1𝑀𝐾𝑑1M=K\sqrt{d-1} and proper K𝐾K.
For β=1𝛽1\beta=1, d=13𝑑13d=13 and K=4𝐾4K=4 the inequality is fulfilled.
The left hand side minus the right hand side is
4.686292​(d−1)−1/23​(d−1)−1−ln⁡(32​(d−1))−6​(d−1)​ln⁡24.686292𝑑11superscript23𝑑1132𝑑16𝑑124.686292(d-1)-1/2^{3(d-1)-1}-\ln(32(d-1))-6(d-1)\ln 2.
Its derivative with respect to d𝑑d is strict positive.
Therefore, the inequality holds for
d≥13𝑑13d\geq 13.

For β=1𝛽1\beta=1, d=3𝑑3d=3 and K=5𝐾5K=5 the inequality is fulfilled.
The left hand side minus the right hand side is
7.32233​(d−1)−1/23​(d−1)−1−ln⁡(50​(d−1))−6​(d−1)​ln⁡27.32233𝑑11superscript23𝑑1150𝑑16𝑑127.32233(d-1)-1/2^{3(d-1)-1}-\ln(50(d-1))-6(d-1)\ln 2.
Its derivative with respect to d𝑑d is strict positive.
Therefore, the inequality holds for
d≥3𝑑3d\geq 3.

∎

•Storage capacity for random patterns on the sphere.

Next we investigate random points on the sphere.
Under assumption Eq. ([305](#A1.E305 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) we have
to show that the master inequality Eq. ([311](#A1.E311 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) is fulfilled
for αminsubscript𝛼\alpha\_{\min}, where now αminsubscript𝛼\alpha\_{\min} is now a random variable.
We use results on the
distribution of the minimal angles between random patterns on a sphere
according to Cai et al. ([2013](#bib.bib16)) and Brauchart et al. ([2018](#bib.bib13)).
Theorem 2 in Cai et al. ([2013](#bib.bib16)) gives the distribution of the
minimal angle for random patterns on the unit sphere.
Proposition 3.5 in Brauchart et al. ([2018](#bib.bib13)) gives a lower bound
on the probability of the minimal angle being larger than a given constant.
We require this proposition to derive the probability of pattern having a minimal
angle αminsubscript𝛼\alpha\_{\min}.
Proposition 3.6 in Brauchart et al. ([2018](#bib.bib13)) gives the expectation
of the minimal angle.

We will prove high probability bounds
for the expected storage capacity.
We need the following tail-bound on αminsubscript𝛼\alpha\_{\min} (the minimal angle of
random patterns on a sphere):

###### Lemma A13 ((Brauchart et al., [2018](#bib.bib13))).

Let d𝑑d be the dimension of the pattern space,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | κdsubscript𝜅𝑑\displaystyle\kappa\_{d}\ | :=1d​π​Γ​((d+1)/2)Γ​(d/2).assignabsent1𝑑𝜋Γ𝑑12Γ𝑑2\displaystyle:=\ \frac{1}{d\ \sqrt{\pi}}\ \frac{\Gamma((d+1)/2)}{\Gamma(d/2)}\ . |  | (323) |

and δ>0𝛿0\delta>0 such that κd−12​δ(d−1)⩽1subscript𝜅𝑑12superscript𝛿𝑑11\frac{\kappa\_{d-1}}{2}\delta^{(d-1)}\leqslant 1.
Then

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Pr​(N2d−1​αmin≥δ)Prsuperscript𝑁2𝑑1subscript𝛼𝛿\displaystyle\mathbf{\mathrm{Pr}}(N^{\frac{2}{d-1}}\alpha\_{\min}\ \geq\ \delta)\ | ≥ 1−κd−12​δd−1.absent1subscript𝜅𝑑12superscript𝛿𝑑1\displaystyle\geq\ 1\ -\ \frac{\kappa\_{d-1}}{2}\ \delta^{d-1}\ . |  | (324) |

###### Proof.

The statement of the lemma is Eq. (3-6) from Proposition 3.5 in Brauchart et al. ([2018](#bib.bib13)).
∎

Next we derive upper and lower bounds on the constant κdsubscript𝜅𝑑\kappa\_{d} since we require them
later for proving storage capacity bounds.

###### Lemma A14.

For κdsubscript𝜅𝑑\kappa\_{d} defined in Eq. ([323](#A1.E323 "In Lemma A13 ((Brauchart et al., 2018)). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
we have the following bounds for every d≥1𝑑1d\geq 1:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1exp⁡(1/6)​e​π​d116𝑒𝜋𝑑\displaystyle\frac{1}{\exp(1/6)\ \sqrt{e\ \pi\ d}}\ | ⩽κd⩽exp⁡(1/12)2​π​d< 1.absentsubscript𝜅𝑑1122𝜋𝑑1\displaystyle\leqslant\ \kappa\_{d}\ \leqslant\ \frac{\exp(1/12)}{\sqrt{2\ \pi\ d}}\ <\ 1\ . |  | (325) |

###### Proof.

We use for x>0𝑥0x>0 the following bound related to Stirling’s approximation formula
for the gamma function, c.f. (Olver et al., [2010](#bib.bib72), [(5.6.1)](http://dlmf.nist.gov/5.6.1)):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 11\displaystyle 1\ | <Γ​(x)​(2​π)−12​x12−x​exp⁡(x)<exp⁡(112​x).absentΓ𝑥superscript2𝜋12superscript𝑥12𝑥𝑥112𝑥\displaystyle<\ \Gamma(x)\ (2\ \pi)^{-\ \frac{1}{2}}x^{\frac{1}{2}\ -\ x}\exp(x)\ <\ \exp\left(\frac{1}{12\ x}\right)\ . |  | (326) |

Using Stirling’s formula Eq. ([326](#A1.E326 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), we upper bound κdsubscript𝜅𝑑\kappa\_{d}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | κd=1d​π​Γ​((d+1)/2)Γ​(d/2)<1d​π​exp⁡(16​(d+1))​exp⁡(−d+12)​(d+12)d2exp⁡(−d2)​(d2)d2−12subscript𝜅𝑑1𝑑𝜋Γ𝑑12Γ𝑑21𝑑𝜋16𝑑1𝑑12superscript𝑑12𝑑2𝑑2superscript𝑑2𝑑212\displaystyle\kappa\_{d}\ =\ \frac{1}{d\ \sqrt{\pi}}\ \frac{\Gamma((d+1)/2)}{\Gamma(d/2)}\ <\ \frac{1}{d\ \sqrt{\pi}}\ \frac{\exp\left(\frac{1}{6(d+1)}\right)\ \exp\left(-\ \frac{d+1}{2}\right)\ \left(\frac{d+1}{2}\right)^{\frac{d}{2}}}{\exp\left(-\ \frac{d}{2}\right)\ \left(\frac{d}{2}\right)^{\frac{d}{2}\ -\ \frac{1}{2}}} |  | (327) |
|  |  |  |
| --- | --- | --- |
|  | =1d​π​e​exp⁡(16​(d+1))​(1+1d)d2​d2⩽exp⁡(112)2​π​d.absent1𝑑𝜋𝑒16𝑑1superscript11𝑑𝑑2𝑑21122𝜋𝑑\displaystyle=\ \frac{1}{d\ \sqrt{\pi\ e}}\ \exp\left(\frac{1}{6(d+1)}\right)\ \left(1\ +\ \frac{1}{d}\right)^{\frac{d}{2}}\sqrt{\frac{d}{2}}\ \leqslant\ \frac{\exp\left(\frac{1}{12}\right)}{\sqrt{2\ \pi}\ \sqrt{d}}\ . |  |

For the first inequality, we applied Eq. ([326](#A1.E326 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")),
while for the second we used (1+1d)d<esuperscript11𝑑𝑑𝑒(1+\frac{1}{d})^{d}<e
for d≥1𝑑1d\geq 1.

Next, we lower bound κdsubscript𝜅𝑑\kappa\_{d} by again applying Stirling’s formula Eq. ([326](#A1.E326 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | κd=1d​π​Γ​((d+1)/2)Γ​(d/2)>1d​π​exp⁡(−d+12)​(d+12)d2exp⁡(16​d)​exp⁡(−d2)​(d2)d2−12subscript𝜅𝑑1𝑑𝜋Γ𝑑12Γ𝑑21𝑑𝜋𝑑12superscript𝑑12𝑑216𝑑𝑑2superscript𝑑2𝑑212\displaystyle\kappa\_{d}\ =\ \frac{1}{d\ \sqrt{\pi}}\ \frac{\Gamma((d+1)/2)}{\Gamma(d/2)}\ >\ \frac{1}{d\ \sqrt{\pi}}\ \frac{\exp\left(-\ \frac{d+1}{2}\right)\ \left(\frac{d+1}{2}\right)^{\frac{d}{2}}}{\exp\left(\frac{1}{6\ d}\right)\ \exp\left(-\frac{d}{2}\right)\ \left(\frac{d}{2}\right)^{\frac{d}{2}-\frac{1}{2}}} |  | (328) |
|  |  |  |
| --- | --- | --- |
|  | =1d​π​e​exp⁡(16​d)​(1+1d)d2​d2≥1exp⁡(16)​e​π​d,absent1𝑑𝜋𝑒16𝑑superscript11𝑑𝑑2𝑑2116𝑒𝜋𝑑\displaystyle=\ \frac{1}{d\ \sqrt{\pi\ e}\ \exp\left(\frac{1}{6\ d}\right)}\ \left(1+\frac{1}{d}\right)^{\frac{d}{2}}\sqrt{\frac{d}{2}}\ \geq\ \frac{1}{\exp\left(\frac{1}{6}\right)\ \sqrt{e\ \pi\ d}}\ , |  |

where the last inequality holds because of monotonicity of (1+1d)dsuperscript11𝑑𝑑(1+\frac{1}{d})^{d} and using the fact that for d=1𝑑1d=1 it takes on the value 2.
∎

We require a bound on cos\cos to bound the master inequality
Eq. ([311](#A1.E311 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).

###### Lemma A15.

For 0⩽x⩽π0𝑥𝜋0\leqslant x\leqslant\pi the function cos\cos can be upper bounded by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | cos⁡(x)𝑥\displaystyle\cos(x)\ | ⩽ 1−x25.absent1superscript𝑥25\displaystyle\leqslant\ 1\ -\ \frac{x^{2}}{5}\ . |  | (329) |

###### Proof.

We use the infinite product representation of cos\cos, c.f. (Olver et al., [2010](#bib.bib72), [(4.22.2)](http://dlmf.nist.gov/4.22.2)):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | cos⁡(x)𝑥\displaystyle\cos(x)\ | =∏n=1∞(1−4​x2(2​n−1)2​π2).absentsuperscriptsubscriptproduct𝑛114superscript𝑥2superscript2𝑛12superscript𝜋2\displaystyle=\ \prod\_{n=1}^{\infty}\left(1-\frac{4\ x^{2}}{(2n-1)^{2}\ \pi^{2}}\right)\ . |  | (330) |

Since it holds that

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1−4​x2(2​n−1)2​π2⩽ 114superscript𝑥2superscript2𝑛12superscript𝜋21\displaystyle 1\ -\ \frac{4\ x^{2}}{(2n-1)^{2}\ \pi^{2}}\ \leqslant\ 1 |  | (331) |

for |x|⩽π𝑥𝜋|x|\leqslant\pi and n≥2𝑛2n\geq 2, we can get the following upper bound on
Eq. ([330](#A1.E330 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | cos⁡(x)𝑥\displaystyle\cos(x)\ | ⩽∏n=12(1−4​x2(2​n−1)2​π2)=(1−4​x2π2)​(1−4​x29​π2)absentsuperscriptsubscriptproduct𝑛1214superscript𝑥2superscript2𝑛12superscript𝜋214superscript𝑥2superscript𝜋214superscript𝑥29superscript𝜋2\displaystyle\leqslant\ \prod\_{n=1}^{2}\left(1-\frac{4\ x^{2}}{(2n-1)^{2}\pi^{2}}\right)\ =\ \left(1\ -\ \frac{4\ x^{2}}{\pi^{2}}\right)\ \left(1\ -\ \frac{4\ x^{2}}{9\ \pi^{2}}\right) |  | (332) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | = 1−40​x29​π2+16​x49​π4⩽ 1−40​x29​π2+16​x29​π2absent140superscript𝑥29superscript𝜋216superscript𝑥49superscript𝜋4140superscript𝑥29superscript𝜋216superscript𝑥29superscript𝜋2\displaystyle=\ 1\ -\ \frac{40\ x^{2}}{9\ \pi^{2}}\ +\ \frac{16\ x^{4}}{9\ \pi^{4}}\ \leqslant\ 1\ -\ \frac{40\ x^{2}}{9\ \pi^{2}}\ +\ \frac{16\ x^{2}}{9\ \pi^{2}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | = 1−24​x29​π2⩽ 1−x25.absent124superscript𝑥29superscript𝜋21superscript𝑥25\displaystyle=\ 1\ -\ \frac{24\ x^{2}}{9\ \pi^{2}}\ \leqslant\ 1\ -\ \frac{x^{2}}{5}\ . |  |

The last but one inequality uses x⩽π𝑥𝜋x\leqslant\pi,
which implies x/π⩽1𝑥𝜋1x/\pi\leqslant 1.
Thus Eq. ([329](#A1.E329 "In Lemma A15. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) is proven.

∎

•Exponential storage capacity: the base c𝑐c as a function of
the parameter β𝛽\beta, the radius of the sphere M𝑀M, the probability p𝑝p,
and the dimension d𝑑d of the space.

We express the number N𝑁N of stored patterns by an exponential
function with base c>1𝑐1c>1 and an exponent linear in d𝑑d.
We derive constraints on he base c𝑐c
as a function of β𝛽\beta, the radius of the sphere M𝑀M,
the probability p𝑝p that all patterns can be stored,
and the dimension d𝑑d of the space.
With β>0𝛽0\beta>0, K>0𝐾0K>0, and d≥2𝑑2d\geq 2 (to ensure a sphere),
the following theorem gives our main result.

###### Theorem A5 (Storage Capacity (Main): Random Patterns).

We assume a failure probability 0<p⩽10𝑝10<p\leqslant 1 and randomly chosen patterns
on the sphere with radius M:=K​d−1assign𝑀𝐾𝑑1M:=K\sqrt{d-1}.
We define

|  |  |  |  |
| --- | --- | --- | --- |
|  | a𝑎\displaystyle a\ | :=2d−1​(1+ln⁡(2​β​K2​p​(d−1))),b:=2​K2​β5,formulae-sequenceassignabsent2𝑑112𝛽superscript𝐾2𝑝𝑑1assign𝑏2superscript𝐾2𝛽5\displaystyle:=\ \frac{2}{d-1}\ (1\ +\ \ln(2\ \beta\ K^{2}\ p\ (d-1)))\ ,\quad b\ :=\ \frac{2\ K^{2}\ \beta}{5}\ , |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | c𝑐\displaystyle c\ | :=bW0(exp(a+ln(b)),\displaystyle:=\ \frac{b}{W\_{0}(\exp(a\ +\ \ln(b))}\ , |  | (333) |

where W0subscript𝑊0W\_{0} is the upper branch of the Lambert W𝑊W function (Olver et al., [2010](#bib.bib72), [(4.13)](http://dlmf.nist.gov/4.13))
and ensure

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | c𝑐\displaystyle c\ | ≥(2p)4d−1.absentsuperscript2𝑝4𝑑1\displaystyle\geq\ \left(\frac{2}{\sqrt{p}}\right)^{\frac{4}{d-1}}\ . |  | (334) |

Then with probability 1−p1𝑝1-p, the number of random patterns
that can be stored is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | ≥p​cd−14.absent𝑝superscript𝑐𝑑14\displaystyle\geq\ \sqrt{p}\ c^{\frac{d-1}{4}}\ . |  | (335) |

Therefore it is proven for c≥3.1546𝑐3.1546c\geq 3.1546 with
β=1𝛽1\beta=1, K=3𝐾3K=3, d=20𝑑20d=20 and p=0.001𝑝0.001p=0.001 (a+ln⁡(b)>1.27𝑎𝑏1.27a+\ln(b)>1.27)
and proven for c≥1.3718𝑐1.3718c\geq 1.3718 with β=1𝛽1\beta=1, K=1𝐾1K=1, d=75𝑑75d=75, and p=0.001𝑝0.001p=0.001
(a+ln⁡(b)<−0.94𝑎𝑏0.94a+\ln(b)<-0.94).

###### Proof.

We consider the probability that the master inequality Eq. ([311](#A1.E311 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
is fulfilled:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr(M2(1−cos(αmin)))≥2β​N+1βln(2N2βM2))≥ 1−p.\displaystyle\mathbf{\mathrm{Pr}}\left(M^{2}(1\ -\cos(\alpha\_{\min})))\ \geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\right)\ \geq\ 1\ -\ p\ . |  | (336) |

Using Eq. ([329](#A1.E329 "In Lemma A15. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), we have:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1−cos⁡(αmin)1subscript𝛼\displaystyle 1\ -\ \cos(\alpha\_{\min})\ | ≥15​αmin2.absent15superscriptsubscript𝛼2\displaystyle\geq\ \frac{1}{5}\ \alpha\_{\min}^{2}\ . |  | (337) |

Therefore, with probability 1−p1𝑝1-p the storage capacity is largest N𝑁N that fulfills

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr​(M2​αm​i​n25≥2β​N+1β​ln⁡(2​N2​β​M2))≥ 1−p.Prsuperscript𝑀2superscriptsubscript𝛼𝑚𝑖𝑛252𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀21𝑝\displaystyle\mathbf{\mathrm{Pr}}\left(M^{2}\frac{\alpha\_{min}^{2}}{5}\ \geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\right)\ \geq\ 1\ -\ p\ . |  | (338) |

This inequality is equivalent to

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | Pr​(N2d−1​αm​i​n≥5​N2d−1M​(2β​N+1β​ln⁡(2​N2​β​M2))12)≥ 1−p.Prsuperscript𝑁2𝑑1subscript𝛼𝑚𝑖𝑛5superscript𝑁2𝑑1𝑀superscript2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2121𝑝\displaystyle\mathbf{\mathrm{Pr}}\left(N^{\frac{2}{d-1}}\ \alpha\_{min}\ \geq\ \frac{\sqrt{5}\ N^{\frac{2}{d-1}}}{M}\ \left(\frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\right)^{\frac{1}{2}}\right)\ \geq\ 1\ -\ p\ . |  | (339) |

We use Eq. ([324](#A1.E324 "In Lemma A13 ((Brauchart et al., 2018)). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) to obtain:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr​(N2d−1​αm​i​n≥5​N2d−1M​(2β​N+1β​ln⁡(2​N2​β​M2))12)Prsuperscript𝑁2𝑑1subscript𝛼𝑚𝑖𝑛5superscript𝑁2𝑑1𝑀superscript2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀212\displaystyle\mathbf{\mathrm{Pr}}\left(N^{\frac{2}{d-1}}\ \alpha\_{min}\ \geq\ \frac{\sqrt{5}\ N^{\frac{2}{d-1}}}{M}\ \left(\frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\right)^{\frac{1}{2}}\right) |  | (340) |
|  |  |  |
| --- | --- | --- |
|  | ≥ 1−κd−12​ 5d−12​N2​M−(d−1)​(2β​N+1β​ln⁡(2​N2​β​M2))d−12.absent1subscript𝜅𝑑12superscript5𝑑12superscript𝑁2superscript𝑀𝑑1superscript2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2𝑑12\displaystyle\geq\ 1\ -\ \frac{\kappa\_{d-1}}{2}\ 5^{\frac{d-1}{2}}\ N^{2}\ M^{-(d-1)}\left(\frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\right)^{\frac{d-1}{2}}\ . |  |

For Eq. ([339](#A1.E339 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) to be fulfilled, it is sufficient that

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | κd−12​ 5d−12​N2​M−(d−1)​(2β​N+1β​ln⁡(2​N2​β​M2))d−12−p⩽ 0.subscript𝜅𝑑12superscript5𝑑12superscript𝑁2superscript𝑀𝑑1superscript2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2𝑑12𝑝 0\displaystyle\frac{\kappa\_{d-1}}{2}\ 5^{\frac{d-1}{2}}\ N^{2}\ M^{-(d-1)}\left(\frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta M^{2}\right)\right)^{\frac{d-1}{2}}\ -\ p\ \leqslant\ 0\ . |  | (341) |

If we insert the assumption Eq. ([334](#A1.E334 "In Theorem A5 (Storage Capacity (Main): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) of the theorem
into Eq. ([335](#A1.E335 "In Theorem A5 (Storage Capacity (Main): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), then we obtain N≥2𝑁2N\geq 2.
We now apply the upper bound κd−1/2<κd−1<1subscript𝜅𝑑12subscript𝜅𝑑11\kappa\_{d-1}/2<\kappa\_{d-1}<1 from Eq. ([325](#A1.E325 "In Lemma A14. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
and the upper bound 2β​N⩽1β2𝛽𝑁1𝛽\frac{2}{\beta N}\leqslant\frac{1}{\beta} from N≥2𝑁2N\geq 2 to inequality Eq. ([341](#A1.E341 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
In the resulting inequality we insert N=p​cd−14𝑁𝑝superscript𝑐𝑑14N=\sqrt{p}c^{\frac{d-1}{4}} to check whether it is fulfilled
with this special value of N𝑁N and obtain:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 5d−12​p​cd−12​M−(d−1)​(1β+1β​ln⁡(2​p​cd−12​β​M2))d−12⩽p.superscript5𝑑12𝑝superscript𝑐𝑑12superscript𝑀𝑑1superscript1𝛽1𝛽2𝑝superscript𝑐𝑑12𝛽superscript𝑀2𝑑12𝑝\displaystyle 5^{\frac{d-1}{2}}\ p\ c^{\frac{d-1}{2}}\ M^{-(d-1)}\left(\frac{1}{\beta}\ +\ \frac{1}{\beta}\ \ln\left(2\ p\ c^{\frac{d-1}{2}}\ \beta M^{2}\right)\right)^{\frac{d-1}{2}}\leqslant\ p\ . |  | (342) |

Dividing by p𝑝p, inserting M=K​d−1𝑀𝐾𝑑1M=K\sqrt{d-1},
and exponentiation of the left and right side by 2d−12𝑑1\frac{2}{d-1}
gives:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 5​cK2​(d−1)​(1β+1β​ln⁡(2​β​cd−12​p​K2​(d−1)))− 1⩽ 0.5𝑐superscript𝐾2𝑑11𝛽1𝛽2𝛽superscript𝑐𝑑12𝑝superscript𝐾2𝑑11 0\displaystyle\frac{5\ c}{K^{2}\ (d-1)}\left(\frac{1}{\beta}\ +\ \frac{1}{\beta}\ \ln\left(2\ \beta\ c^{\frac{d-1}{2}}\ p\ K^{2}\ (d-1)\right)\right)\ -\ 1\ \leqslant\ 0\ . |  | (343) |

After some algebraic manipulation, this inequality can be written as

|  |  |  |  |
| --- | --- | --- | --- |
|  | a​c+c​ln⁡(c)−b⩽ 0,𝑎𝑐𝑐𝑐𝑏 0\displaystyle a\ c\ +\ c\ \ln(c)\ -\ b\ \leqslant\ 0\ , |  | (344) |

where we used

|  |  |  |  |
| --- | --- | --- | --- |
|  | a𝑎\displaystyle a\ | :=2d−1​(1+ln⁡(2​β​K2​p​(d−1))),b:=2​K2​β5.formulae-sequenceassignabsent2𝑑112𝛽superscript𝐾2𝑝𝑑1assign𝑏2superscript𝐾2𝛽5\displaystyle:=\ \frac{2}{d-1}\ (1\ +\ \ln(2\ \beta\ K^{2}\ p\ (d-1)))\ ,\quad b\ :=\ \frac{2\ K^{2}\ \beta}{5}\ . |  |

We determine the value c^^𝑐\hat{c} of c𝑐c
which makes the inequality Eq. ([344](#A1.E344 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) equal to zero.
We solve

|  |  |  |  |
| --- | --- | --- | --- |
|  | a​c^+c^​ln⁡(c^)−b= 0𝑎^𝑐^𝑐^𝑐𝑏 0\displaystyle a\ \hat{c}\ +\ \hat{c}\ \ln(\hat{c})\ -\ b\ =\ 0 |  | (345) |

for c^^𝑐\hat{c}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | a​c^+c^​ln⁡(c^)−b= 0𝑎^𝑐^𝑐^𝑐𝑏 0\displaystyle a\ \hat{c}\ +\ \hat{c}\ \ln(\hat{c})\ -\ b\ =\ 0 |  | (346) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | a+ln⁡(c^)=b/c^𝑎^𝑐𝑏^𝑐\displaystyle a\ +\ \ln(\hat{c})\ =\ b/\hat{c} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | a+ln⁡(b)+ln⁡(c^/b)=b/c^𝑎𝑏^𝑐𝑏𝑏^𝑐\displaystyle a\ +\ \ln(b)\ +\ \ln(\hat{c}/b)\ =\ b/\hat{c} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | b/c^+ln⁡(b/c^)=a+ln⁡(b)𝑏^𝑐𝑏^𝑐𝑎𝑏\displaystyle b/\hat{c}\ +\ \ln(b/\hat{c})\ =\ a\ +\ \ln(b) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | b/c^​exp⁡(b/c^)=exp⁡(a+ln⁡(b))𝑏^𝑐𝑏^𝑐𝑎𝑏\displaystyle b/\hat{c}\ \exp(b/\hat{c})\ =\ \exp(a\ +\ \ln(b)) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | b/c^=W0​(exp⁡(a+ln⁡(b)))𝑏^𝑐subscript𝑊0𝑎𝑏\displaystyle b/\hat{c}\ =\ W\_{0}(\exp(a\ +\ \ln(b))) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | c^=bW0(exp(a+ln(b)),\displaystyle\hat{c}\ =\ \frac{b}{W\_{0}(\exp(a\ +\ \ln(b))}\ , |  |

where W0subscript𝑊0W\_{0} is the upper branch of the Lambert W𝑊W function (see Def. [A6](#ThmdefinitionA6 "Definition A6 (Lambert Function). ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
Hence, the solution is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | c^^𝑐\displaystyle\hat{c}\ | =bW0(exp(a+ln(b)).\displaystyle=\ \frac{b}{W\_{0}(\exp(a\ +\ \ln(b))}\ . |  | (347) |

The solution exist, since the Lambert function W0​(x)subscript𝑊0𝑥W\_{0}(x) (Olver et al., [2010](#bib.bib72), [(4.13)](http://dlmf.nist.gov/4.13)) is defined
for −1/e<x1𝑒𝑥-1/e<x and we have 0<exp(a+ln(b)0<\exp(a+\ln(b).

Since c^^𝑐\hat{c} fulfills inequality Eq. ([344](#A1.E344 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
and therefore also Eq. ([342](#A1.E342 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")),
we have a lower bound on the storage capacity N𝑁N:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | ≥p​c^d−14.absent𝑝superscript^𝑐𝑑14\displaystyle\geq\ \sqrt{p}\ \hat{c}^{\frac{d-1}{4}}\ . |  | (348) |

∎

Next we aim at a lower bound on c𝑐c
which does not use the Lambert W𝑊W function (Olver et al., [2010](#bib.bib72), [(4.13)](http://dlmf.nist.gov/4.13)).
Therefore, we upper bound W0(exp(a+ln(b))W\_{0}(\exp(a+\ln(b)) to obtain a lower
bound on c𝑐c, therefore, also a lower bound on the storage
capacity N𝑁N. The lower bound is given in the next corollary.

###### Corollary A1.

We assume a failure probability 0<p⩽10𝑝10<p\leqslant 1 and randomly chosen patterns
on the sphere with radius M=K​d−1𝑀𝐾𝑑1M=K\sqrt{d-1}.
We define

|  |  |  |  |
| --- | --- | --- | --- |
|  | a𝑎\displaystyle a\ | :=2d−1​(1+ln⁡(2​β​K2​p​(d−1))),b:=2​K2​β5.formulae-sequenceassignabsent2𝑑112𝛽superscript𝐾2𝑝𝑑1assign𝑏2superscript𝐾2𝛽5\displaystyle:=\ \frac{2}{d-1}\ (1\ +\ \ln(2\ \beta\ K^{2}\ p\ (d-1)))\ ,\quad b\ :=\ \frac{2\ K^{2}\ \beta}{5}\ . |  |

Using the omega constant Ω≈0.56714329Ω0.56714329\Omega\approx 0.56714329 we set

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | c𝑐\displaystyle c\ | ={bln(Ω​exp⁡(a+ln⁡(b))+ 1Ω​(1+Ω))−1for ​a+ln⁡(b)⩽ 0,b​(a+ln⁡(b))−a+ln⁡(b)a+ln⁡(b)+ 1for ​a+ln⁡(b)> 0\displaystyle=\ \begin{cases}b\ \ln\left(\frac{\Omega\ \exp(a\ +\ \ln(b))\ +\ 1}{\Omega\ (1\ +\ \Omega)}\right)^{-1}&\text{for }\ a\ +\ \ln(b)\ \leqslant\ 0\ ,\\ b\ (a\ +\ \ln(b))^{-\frac{a\ +\ \ln(b)}{a\ +\ \ln(b)\ +\ 1}}&\text{for }\ a\ +\ \ln(b)\ >\ 0\end{cases} |  | (349) |

and ensure

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | c𝑐\displaystyle c\ | ≥(2p)4d−1.absentsuperscript2𝑝4𝑑1\displaystyle\geq\ \left(\frac{2}{\sqrt{p}}\right)^{\frac{4}{d-1}}\ . |  | (350) |

Then with probability 1−p1𝑝1-p, the number of random patterns
that can be stored
is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | ≥p​cd−14.absent𝑝superscript𝑐𝑑14\displaystyle\geq\ \sqrt{p}\ c^{\frac{d-1}{4}}\ . |  | (351) |

Examples are c≥3.1444𝑐3.1444c\geq 3.1444 for
β=1𝛽1\beta=1, K=3𝐾3K=3, d=20𝑑20d=20 and p=0.001𝑝0.001p=0.001 (a+ln⁡(b)>1.27𝑎𝑏1.27a+\ln(b)>1.27)
and c≥1.2585𝑐1.2585c\geq 1.2585 for β=1𝛽1\beta=1 K=1𝐾1K=1, d=75𝑑75d=75, and p=0.001𝑝0.001p=0.001
(a+ln⁡(b)<−0.94𝑎𝑏0.94a+\ln(b)<-0.94).

###### Proof.

We lower bound the c𝑐c defined in Theorem [A5](#ThmtheoremA5 "Theorem A5 (Storage Capacity (Main): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
According to (Hoorfar & Hassani, [2008](#bib.bib46), Theorem 2.3) we have for any real u𝑢u and y>1e𝑦1𝑒y>\frac{1}{e}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W0​(exp⁡(u))subscript𝑊0𝑢\displaystyle W\_{0}(\exp(u))\ | ⩽ln⁡(exp⁡(u)+y1+ln⁡(y)).absent𝑢𝑦1𝑦\displaystyle\leqslant\ \ln\left(\frac{\exp(u)\ +\ y}{1\ +\ \ln(y)}\right)\ . |  | (352) |

To upper bound W0​(x)subscript𝑊0𝑥W\_{0}(x) for x∈[0,1]𝑥01x\in[0,1], we set

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | y𝑦\displaystyle y\ | = 1/W0​(1)= 1/Ω=exp⁡Ω=− 1/ln⁡Ω≈ 1.76322,absent1subscript𝑊011ΩΩ1Ω1.76322\displaystyle=\ 1/W\_{0}(1)\ =\ 1/\Omega\ =\ \exp{\Omega}\ =\ -\ 1/\ln\Omega\ \approx\ 1.76322\ , |  | (353) |

where the Omega constant ΩΩ\Omega is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΩΩ\displaystyle\Omega\ | =(∫−∞∞d​t(et−t)2+π2)−1− 1≈ 0.56714329.absentsuperscriptsuperscriptsubscriptd𝑡superscriptsuperscript𝑒𝑡𝑡2superscript𝜋2110.56714329\displaystyle=\ \left(\int\_{-\infty}^{\infty}\frac{\mathrm{d}t}{\left(e^{t}\ -\ t\right)^{2}\ +\ \pi^{2}}\right)^{-1}\ -\ 1\ \approx\ 0.56714329\ . |  | (354) |

See for these equations the special values of the Lambert W𝑊W function in Lemma [A31](#ThmlemmaA31 "Lemma A31. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
We have the upper bound on W0subscript𝑊0W\_{0}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W0​(exp⁡(u))subscript𝑊0𝑢\displaystyle W\_{0}(\exp(u))\ | ⩽ln⁡(exp⁡(u)+ 1/Ω1+ln⁡(1/Ω))=ln⁡(Ω​exp⁡(u)+ 1Ω​(1+Ω)).absent𝑢1Ω11ΩΩ𝑢1Ω1Ω\displaystyle\leqslant\ \ln\left(\frac{\exp(u)\ +\ 1/\Omega}{1\ +\ \ln(1/\Omega)}\right)\ =\ \ln\left(\frac{\Omega\ \exp(u)\ +\ 1}{\Omega(1\ +\ \Omega)}\right)\ . |  | (355) |

At the right hand side of interval [0,1]01[0,1],
we have u=0𝑢0u=0 and exp⁡(u)=1𝑢1\exp(u)=1 and get:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ln⁡(Ω​ 1+ 1Ω​(1+Ω))=ln⁡(1Ω)=−ln⁡(Ω)=Ω=W0​(1).Ω11Ω1Ω1ΩΩΩsubscript𝑊01\displaystyle\ln\left(\frac{\Omega\ 1\ +\ 1}{\Omega(1\ +\ \Omega)}\right)\ =\ \ln\left(\frac{1}{\Omega}\right)\ =\ -\ \ln\left(\Omega\right)\ =\ \Omega\ =\ W\_{0}(1)\ . |  | (356) |

Therefore, the bound is tight at the right hand side of of interval [0,1]01[0,1],
that is for exp⁡(u)=1𝑢1\exp(u)=1, i.e. u=0𝑢0u=0.
We have derived an bound for W0​(exp⁡(u))subscript𝑊0𝑢W\_{0}(\exp(u)) with exp⁡(u)∈[0,1]𝑢01\exp(u)\in[0,1]
or, equivalently, u∈[−∞,0]𝑢0u\in[-\infty,0].
We obtain from Hoorfar & Hassani ([2008](#bib.bib46), Corollary 2.6) the following bound
on W0​(exp⁡(u))subscript𝑊0𝑢W\_{0}(\exp(u)) for 1<exp⁡(u)1𝑢1<\exp(u), or, equivalently 0<u0𝑢0<u:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W0​(exp⁡(u))subscript𝑊0𝑢\displaystyle W\_{0}(\exp(u))\ | ⩽uu1+u.absentsuperscript𝑢𝑢1𝑢\displaystyle\leqslant\ u^{\frac{u}{1\ +\ u}}\ . |  | (357) |

A lower bound on c^^𝑐\hat{c} is obtained via the upper bounds Eq. ([357](#A1.E357 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
and Eq. ([355](#A1.E355 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) on W0subscript𝑊0W\_{0} as W0>0subscript𝑊00W\_{0}>0.
We set u=a+ln⁡(b)𝑢𝑎𝑏u=a+\ln(b) and obtain

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W0​(exp⁡(a+ln⁡(b)))subscript𝑊0𝑎𝑏\displaystyle W\_{0}(\exp(a\ +\ \ln(b)))\ | ⩽{ln(Ω​exp⁡(a+ln⁡(b))+ 1Ω​(1+Ω))−1for ​a+ln⁡(b)⩽ 0,(a+ln⁡(b))−a+ln⁡(b)a+ln⁡(b)+ 1for ​a+ln⁡(b)> 0\displaystyle\leqslant\ \begin{cases}\ln\left(\frac{\Omega\ \exp(a\ +\ \ln(b))\ +\ 1}{\Omega\ (1\ +\ \Omega)}\right)^{-1}&\text{for }\ a\ +\ \ln(b)\ \leqslant\ 0\ ,\\ (a\ +\ \ln(b))^{-\frac{a\ +\ \ln(b)}{a\ +\ \ln(b)\ +\ 1}}&\text{for }\ a\ +\ \ln(b)\ >\ 0\end{cases} |  | (358) |

We insert this bound into Eq. ([347](#A1.E347 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), the solution for c^^𝑐\hat{c},
to obtain the statement of the theorem.

∎

•Exponential storage capacity: the dimension d𝑑d of the space
as a function of the parameter β𝛽\beta, the radius of the sphere M𝑀M,
and the probability p𝑝p.

We express the number N𝑁N of stored patterns by an exponential
function with base c>1𝑐1c>1 and an exponent linear in d𝑑d.
We derive constraints on the dimension d𝑑d of the space
as a function of β𝛽\beta, the radius of the sphere M𝑀M,
the probability p𝑝p that all patterns can be stored,
and the base of the exponential storage capacity.
The following theorem gives this result.

###### Theorem A6 (Storage Capacity (d computed): Random Patterns).

We assume a failure probability 0<p⩽10𝑝10<p\leqslant 1 and randomly chosen patterns
on the sphere with radius M=K​d−1𝑀𝐾𝑑1M=K\sqrt{d-1}.
We define

|  |  |  |  |
| --- | --- | --- | --- |
|  | a𝑎\displaystyle a\ | :=ln⁡(c)2−K2​β5​c,b:= 1+ln⁡(2​p​β​K2),formulae-sequenceassignabsent𝑐2superscript𝐾2𝛽5𝑐assign𝑏12𝑝𝛽superscript𝐾2\displaystyle:=\ \frac{\ln(c)}{2}\ -\ \frac{K^{2}\ \beta}{5\ c}\ ,\quad b\ :=\ 1\ +\ \ln\left(2\ p\ \beta\ K^{2}\right)\ , |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | d𝑑\displaystyle d\ | ={1+1a​W​(a​exp⁡(−b))for ​a≠0,1+exp⁡(−b)for ​a=0,absentcases11𝑎𝑊𝑎𝑏for 𝑎01𝑏for 𝑎0\displaystyle=\ \begin{cases}1\ +\ \frac{1}{a}\ W(a\ \exp(-b))&\text{for }a\not=0\ ,\\ 1\ +\ \exp(-b)&\text{for }a=0\ ,\end{cases} |  | (359) |

where W𝑊W is the Lambert W𝑊W function (Olver et al., [2010](#bib.bib72), [(4.13)](http://dlmf.nist.gov/4.13)).
For 0<a0𝑎0<a the function W𝑊W is the upper branch W0subscript𝑊0W\_{0} and for a<0𝑎0a<0 we use
the lower branch W−1subscript𝑊1W\_{-1}.
If we ensure that

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | c𝑐\displaystyle c\ | ≥(2p)4d−1,−1e⩽a​exp⁡(−b),formulae-sequenceabsentsuperscript2𝑝4𝑑11𝑒𝑎𝑏\displaystyle\geq\ \left(\frac{2}{\sqrt{p}}\right)^{\frac{4}{d-1}}\ ,\quad\ -\ \frac{1}{e}\ \leqslant\ a\ \exp(-b)\ , |  | (360) |

then with probability 1−p1𝑝1-p, the number of random patterns
that can be stored is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | ≥p​cd−14.absent𝑝superscript𝑐𝑑14\displaystyle\geq\ \sqrt{p}\ c^{\frac{d-1}{4}}\ . |  | (361) |

###### Proof.

We consider the probability that the master inequality Eq. ([311](#A1.E311 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
is fulfilled:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr(M2(1−cos(αmin)))≥2β​N+1βln(2N2βM2))≥ 1−p.\displaystyle\mathbf{\mathrm{Pr}}\left(M^{2}(1\ -\cos(\alpha\_{\min})))\ \geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\right)\ \geq\ 1\ -\ p\ . |  | (362) |

Using Eq. ([329](#A1.E329 "In Lemma A15. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), we have:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1−cos⁡(αmin)1subscript𝛼\displaystyle 1\ -\ \cos(\alpha\_{\min})\ | ≥15​αmin2.absent15superscriptsubscript𝛼2\displaystyle\geq\ \frac{1}{5}\ \alpha\_{\min}^{2}\ . |  | (363) |

Therefore, with probability 1−p1𝑝1-p the storage capacity is largest N𝑁N that fulfills

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr​(M2​αm​i​n25≥2β​N+1β​ln⁡(2​N2​β​M2))≥ 1−p.Prsuperscript𝑀2superscriptsubscript𝛼𝑚𝑖𝑛252𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀21𝑝\displaystyle\mathbf{\mathrm{Pr}}\left(M^{2}\frac{\alpha\_{min}^{2}}{5}\ \geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\right)\ \geq\ 1\ -\ p\ . |  | (364) |

This inequality is equivalent to

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | Pr​(N2d−1​αm​i​n≥5​N2d−1M​(2β​N+1β​ln⁡(2​N2​β​M2))12)≥ 1−p.Prsuperscript𝑁2𝑑1subscript𝛼𝑚𝑖𝑛5superscript𝑁2𝑑1𝑀superscript2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2121𝑝\displaystyle\mathbf{\mathrm{Pr}}\left(N^{\frac{2}{d-1}}\ \alpha\_{min}\ \geq\ \frac{\sqrt{5}\ N^{\frac{2}{d-1}}}{M}\ \left(\frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\right)^{\frac{1}{2}}\right)\ \geq\ 1\ -\ p\ . |  | (365) |

We use Eq. ([324](#A1.E324 "In Lemma A13 ((Brauchart et al., 2018)). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) to obtain:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr​(N2d−1​αm​i​n≥5​N2d−1M​(2β​N+1β​ln⁡(2​N2​β​M2))12)Prsuperscript𝑁2𝑑1subscript𝛼𝑚𝑖𝑛5superscript𝑁2𝑑1𝑀superscript2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀212\displaystyle\mathbf{\mathrm{Pr}}\left(N^{\frac{2}{d-1}}\ \alpha\_{min}\ \geq\ \frac{\sqrt{5}\ N^{\frac{2}{d-1}}}{M}\ \left(\frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\right)^{\frac{1}{2}}\right) |  | (366) |
|  |  |  |
| --- | --- | --- |
|  | ≥ 1−κd−12​ 5d−12​N2​M−(d−1)​(2β​N+1β​ln⁡(2​N2​β​M2))d−12.absent1subscript𝜅𝑑12superscript5𝑑12superscript𝑁2superscript𝑀𝑑1superscript2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2𝑑12\displaystyle\geq\ 1\ -\ \frac{\kappa\_{d-1}}{2}\ 5^{\frac{d-1}{2}}\ N^{2}\ M^{-(d-1)}\left(\frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\right)^{\frac{d-1}{2}}\ . |  |

For Eq. ([365](#A1.E365 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) to be fulfilled, it is sufficient that

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | κd−12​ 5d−12​N2​M−(d−1)​(2β​N+1β​ln⁡(2​N2​β​M2))d−12−p⩽ 0.subscript𝜅𝑑12superscript5𝑑12superscript𝑁2superscript𝑀𝑑1superscript2𝛽𝑁1𝛽2superscript𝑁2𝛽superscript𝑀2𝑑12𝑝 0\displaystyle\frac{\kappa\_{d-1}}{2}\ 5^{\frac{d-1}{2}}\ N^{2}\ M^{-(d-1)}\left(\frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta M^{2}\right)\right)^{\frac{d-1}{2}}\ -\ p\ \leqslant\ 0\ . |  | (367) |

If we insert the assumption Eq. ([360](#A1.E360 "In Theorem A6 (Storage Capacity (d computed): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) of the theorem
into Eq. ([361](#A1.E361 "In Theorem A6 (Storage Capacity (d computed): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), then we obtain N≥2𝑁2N\geq 2.
We now apply the upper bound κd−1/2<κd−1<1subscript𝜅𝑑12subscript𝜅𝑑11\kappa\_{d-1}/2<\kappa\_{d-1}<1 from Eq. ([325](#A1.E325 "In Lemma A14. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
and the upper bound 2β​N⩽1β2𝛽𝑁1𝛽\frac{2}{\beta N}\leqslant\frac{1}{\beta} from N≥2𝑁2N\geq 2 to inequality Eq. ([367](#A1.E367 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
In the resulting inequality we insert N=p​cd−14𝑁𝑝superscript𝑐𝑑14N=\sqrt{p}c^{\frac{d-1}{4}} to check whether it is fulfilled
with this special value of N𝑁N and obtain:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 5d−12​p​cd−12​M−(d−1)​(1β+1β​ln⁡(2​p​cd−12​β​M2))d−12⩽p.superscript5𝑑12𝑝superscript𝑐𝑑12superscript𝑀𝑑1superscript1𝛽1𝛽2𝑝superscript𝑐𝑑12𝛽superscript𝑀2𝑑12𝑝\displaystyle 5^{\frac{d-1}{2}}\ p\ c^{\frac{d-1}{2}}\ M^{-(d-1)}\left(\frac{1}{\beta}\ +\ \frac{1}{\beta}\ \ln\left(2\ p\ c^{\frac{d-1}{2}}\ \beta M^{2}\right)\right)^{\frac{d-1}{2}}\leqslant\ p\ . |  | (368) |

Dividing by p𝑝p, inserting M=K​d−1𝑀𝐾𝑑1M=K\sqrt{d-1},
and exponentiation of the left and right side by 2d−12𝑑1\frac{2}{d-1}
gives:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 5​cK2​(d−1)​(1β+1β​ln⁡(2​β​cd−12​p​K2​(d−1)))− 1⩽ 0.5𝑐superscript𝐾2𝑑11𝛽1𝛽2𝛽superscript𝑐𝑑12𝑝superscript𝐾2𝑑11 0\displaystyle\frac{5\ c}{K^{2}\ (d-1)}\left(\frac{1}{\beta}\ +\ \frac{1}{\beta}\ \ln\left(2\ \beta\ c^{\frac{d-1}{2}}\ p\ K^{2}\ (d-1)\right)\right)\ -\ 1\ \leqslant\ 0\ . |  | (369) |

This inequality Eq. ([369](#A1.E369 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) can be reformulated as:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 1+ln⁡(2​p​β​cd−12​K2​(d−1))−(d−1)​K2​β5​c⩽ 0.12𝑝𝛽superscript𝑐𝑑12superscript𝐾2𝑑1𝑑1superscript𝐾2𝛽5𝑐 0\displaystyle 1\ +\ \ln\left(2\ p\ \beta\ c^{\frac{d-1}{2}}\ K^{2}\ (d-1)\right)\ -\ \frac{(d-1)\ K^{2}\ \beta}{5\ c}\ \leqslant\ 0\ . |  | (370) |

Using

|  |  |  |  |
| --- | --- | --- | --- |
|  | a𝑎\displaystyle a\ | :=ln⁡(c)2−K2​β5​c,b:= 1+ln⁡(2​p​β​K2),formulae-sequenceassignabsent𝑐2superscript𝐾2𝛽5𝑐assign𝑏12𝑝𝛽superscript𝐾2\displaystyle:=\ \frac{\ln(c)}{2}\ -\ \frac{K^{2}\ \beta}{5\ c}\ ,\quad b\ :=\ 1\ +\ \ln\left(2\ p\ \beta\ K^{2}\right)\ , |  |

we write inequality Eq. ([370](#A1.E370 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ln⁡(d−1)+a​(d−1)+b⩽ 0.𝑑1𝑎𝑑1𝑏 0\displaystyle\ln(d-1)\ +\ a\ (d-1)\ +\ b\ \leqslant\ 0\ . |  | (372) |

We determine the value d^^𝑑\hat{d} of d𝑑d
which makes the inequality Eq. ([372](#A1.E372 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) equal to zero.
We solve

|  |  |  |  |
| --- | --- | --- | --- |
|  | ln⁡(d^−1)+a​(d^−1)+b= 0.^𝑑1𝑎^𝑑1𝑏 0\displaystyle\ln(\hat{d}-1)\ +\ a\ (\hat{d}-1)\ +\ b\ =\ 0\ . |  | (373) |

for d^^𝑑\hat{d}

For a≠0𝑎0a\not=0 we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ln⁡(d^−1)+a​(d^−1)+b= 0^𝑑1𝑎^𝑑1𝑏 0\displaystyle\ln(\hat{d}-1)\ +\ a\ (\hat{d}-1)\ +\ b\ =\ 0 |  | (374) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | a​(d^−1)+ln⁡(d^−1)=−b𝑎^𝑑1^𝑑1𝑏\displaystyle a\ (\hat{d}-1)\ +\ \ln(\hat{d}-1)\ =\ -\ b |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | (d^−1)​exp⁡(a​(d^−1))=exp⁡(−b)^𝑑1𝑎^𝑑1𝑏\displaystyle(\hat{d}-1)\exp(a\ (\hat{d}-1))\ =\ \exp(-b) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | a​(d^−1)​exp⁡(a​(d^−1))=a​exp⁡(−b)𝑎^𝑑1𝑎^𝑑1𝑎𝑏\displaystyle a\ (\hat{d}-1)\exp(a\ (\hat{d}-1))\ =\ a\ \exp(-b) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | a​(d^−1)=W​(a​exp⁡(−b))𝑎^𝑑1𝑊𝑎𝑏\displaystyle a\ (\hat{d}-1)\ =\ W(a\ \exp(-b)) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | d^− 1=1a​W​(a​exp⁡(−b))^𝑑11𝑎𝑊𝑎𝑏\displaystyle\hat{d}\ -\ 1\ =\ \frac{1}{a}\ W(a\ \exp(-b)) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔⇔\displaystyle\Leftrightarrow\ \ | d^= 1+1a​W​(a​exp⁡(−b)),^𝑑11𝑎𝑊𝑎𝑏\displaystyle\hat{d}\ =\ 1\ +\ \frac{1}{a}\ W(a\ \exp(-b))\ , |  |

where W𝑊W is the Lambert W𝑊W function (see Def. [A6](#ThmdefinitionA6 "Definition A6 (Lambert Function). ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
For a>0𝑎0a>0 we have to use the upper branch W0subscript𝑊0W\_{0} of the Lambert W𝑊W function
and for a<0𝑎0a<0 we use the lower branch W−1subscript𝑊1W\_{-1} of the Lambert W𝑊W function (Olver et al., [2010](#bib.bib72), [(4.13)](http://dlmf.nist.gov/4.13)).
We have to ensure that
−1/e⩽a​exp⁡(−b)1𝑒𝑎𝑏-1/e\leqslant a\exp(-b) for a solution to exist.
For a=0𝑎0a=0 we have d^=1+exp⁡(−b)^𝑑1𝑏\hat{d}=1+\exp(-b).

Hence, the solution is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | d^^𝑑\displaystyle\hat{d}\ | = 1+1a​W​(a​exp⁡(−b)).absent11𝑎𝑊𝑎𝑏\displaystyle=\ 1\ +\ \frac{1}{a}\ W(a\exp(-b))\ . |  | (375) |

Since d^^𝑑\hat{d} fulfills inequality Eq. ([369](#A1.E369 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
and therefore also Eq. ([368](#A1.E368 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")),
we have a lower bound on the storage capacity N𝑁N:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | ≥p​c^d−14.absent𝑝superscript^𝑐𝑑14\displaystyle\geq\ \sqrt{p}\ \hat{c}^{\frac{d-1}{4}}\ . |  | (376) |

∎

###### Corollary A2.

We assume a failure probability 0<p⩽10𝑝10<p\leqslant 1 and randomly chosen patterns
on the sphere with radius M=K​d−1𝑀𝐾𝑑1M=K\sqrt{d-1}.
We define

|  |  |  |  |
| --- | --- | --- | --- |
|  | a𝑎\displaystyle a\ | :=ln⁡(c)2−K2​β5​c,b:= 1+ln⁡(2​p​β​K2),formulae-sequenceassignabsent𝑐2superscript𝐾2𝛽5𝑐assign𝑏12𝑝𝛽superscript𝐾2\displaystyle:=\ \frac{\ln(c)}{2}\ -\ \frac{K^{2}\ \beta}{5\ c}\ ,\quad b\ :=\ 1\ +\ \ln\left(2\ p\ \beta\ K^{2}\right)\ , |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | d𝑑\displaystyle d\ | = 1+1a​(−ln⁡(−a)+b),absent11𝑎𝑎𝑏\displaystyle=\ 1\ +\ \frac{1}{a}\ \left(-\ \ln(-a)\ +\ b\right)\ , |  | (377) |

and ensure

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | c𝑐\displaystyle c\ | ≥(2p)4d−1,−1e⩽a​exp⁡(−b),a< 0,formulae-sequenceabsentsuperscript2𝑝4𝑑1formulae-sequence1𝑒𝑎𝑏𝑎 0\displaystyle\geq\ \left(\frac{2}{\sqrt{p}}\right)^{\frac{4}{d-1}}\ ,\quad\ -\ \frac{1}{e}\ \leqslant\ a\ \exp(-b)\ ,\quad\ a\ <\ 0\ , |  | (378) |

then with probability 1−p1𝑝1-p, the number of random patterns
that can be stored is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | ≥p​cd−14.absent𝑝superscript𝑐𝑑14\displaystyle\geq\ \sqrt{p}\ c^{\frac{d-1}{4}}\ . |  | (379) |

Setting β=1𝛽1\beta=1, K=3𝐾3K=3, c=2𝑐2c=2 and p=0.001𝑝0.001p=0.001 yields d<24𝑑24d<24.

###### Proof.

For a<0𝑎0a<0 the
Eq. ([359](#A1.E359 "In Theorem A6 (Storage Capacity (d computed): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) from Theorem ([A6](#ThmtheoremA6 "Theorem A6 (Storage Capacity (d computed): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
can be written as

|  |  |  |  |
| --- | --- | --- | --- |
|  | d= 1+W−1​(a​exp⁡(−b))a= 1+W−1​(−exp⁡(−(−ln⁡(−a)+b−1)−1))a𝑑1subscript𝑊1𝑎𝑏𝑎1subscript𝑊1𝑎𝑏11𝑎\displaystyle d\ =\ 1\ +\ \frac{W\_{-1}(a\exp(-b))}{a}\ =\ 1\ +\ \frac{W\_{-1}(-\exp\left(-(-\ln(-a)+b-1)-1\right))}{a} |  | (380) |

From Alzahrani & Salem ([2018](#bib.bib4), Theorem 3.1) we get the following bound on W−1subscript𝑊1W\_{-1}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −ee−1​(u+1)𝑒𝑒1𝑢1\displaystyle-\ \frac{e}{e-1}\ (u+1)\ | <W−1​(−exp⁡(−u−1))<−(u+1).absentsubscript𝑊1𝑢1𝑢1\displaystyle<\ W\_{-1}(-\ \exp(-u-1))\ <\ -\ (u+1)\ . |  | (381) |

for u>0𝑢0u>0. We apply Eq. ([381](#A1.E381 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) to Eq. ([380](#A1.E380 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
with u=−ln⁡(−a)+b−1𝑢𝑎𝑏1u=-\ln(-a)+b-1.

Since a<0𝑎0a<0 we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | d> 1+−ln⁡(−a)+ba.𝑑1𝑎𝑏𝑎\displaystyle d\ >\ 1\ +\ \frac{-\ln(-a)+b}{a}\ . |  | (382) |

∎

•Storage capacity for the expected minimal separation instead of the probability that all patterns
can be stored.
In contrast to the previous paragraph, we want to argue about the storage capacity
for the expected minimal separation.
Therefore, we will use the following bound on the expectation of αminsubscript𝛼\alpha\_{\min} (minimal angle),
which gives also a bound on the expected of ΔminsubscriptΔ\Delta\_{\min} (minimal separation):

###### Lemma A16 (Proposition 3.6 in Brauchart et al. ([2018](#bib.bib13))).

We have the following lower bound on the expectation of αminsubscript𝛼\alpha\_{\min}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E​[N2d−1​αmin]Edelimited-[]superscript𝑁2𝑑1subscript𝛼\displaystyle\mathbf{\mathrm{E}}\left[N^{\frac{2}{d-1}}\ \alpha\_{\min}\right]\ | ≥(Γ​(d2)2​(d−1)​π​Γ​(d−12))−1d−1​Γ​(1+1d−1)​d−1d−1Γ​(2+1d−1):=Cd−1.absentsuperscriptΓ𝑑22𝑑1𝜋Γ𝑑121𝑑1Γ11𝑑1superscript𝑑1𝑑1Γ21𝑑1assignsubscript𝐶𝑑1\displaystyle\geq\ \left(\frac{\Gamma(\frac{d}{2})}{2(d-1)\ \sqrt{\pi}\ \Gamma(\frac{d-1}{2})}\right)^{-\frac{1}{d-1}}\Gamma(1+\frac{1}{d-1})\ \frac{d^{-\frac{1}{d-1}}}{\Gamma(2+\frac{1}{d-1})}\ :=\ C\_{d-1}. |  | (383) |

The bound is valid for all N≥2𝑁2N\geq 2 and d≥2𝑑2d\geq 2.

Let us start with some preliminary estimates.
First of all we need some asymptotics for the constant Cd−1subscript𝐶𝑑1C\_{d-1} in Eq. ([383](#A1.E383 "In Lemma A16 (Proposition 3.6 in Brauchart et al. (2018)). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")):

###### Lemma A17.

The following estimate holds for d≥2𝑑2d\geq 2:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Cdsubscript𝐶𝑑\displaystyle C\_{d}\ | ≥ 1−ln⁡(d+1)d.absent1𝑑1𝑑\displaystyle\geq\ 1\ -\ \frac{\ln(d+1)}{d}\ . |  | (384) |

###### Proof.

The recursion formula for the Gamma function is (Olver et al., [2010](#bib.bib72), [(5.5.1)](http://dlmf.nist.gov/5.5.1)):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Γ​(x+1)Γ𝑥1\displaystyle\Gamma(x+1)\ | =x​Γ​(x).absent𝑥Γ𝑥\displaystyle=\ x\ \Gamma(x)\ . |  | (385) |

We use Eq. ([325](#A1.E325 "In Lemma A14. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) and
the fact that d1d≥1superscript𝑑1𝑑1d^{\frac{1}{d}}\geq 1 for d≥1𝑑1d\geq 1 to obtain:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Cdsubscript𝐶𝑑\displaystyle C\_{d}\ | ≥(2​d)1d​Γ​(1+1d)​(d+1)−1dΓ​(2+1d)=(2​d)1d​(d+1)−1d1−1d>(d+1)1dabsentsuperscript2𝑑1𝑑Γ11𝑑superscript𝑑11𝑑Γ21𝑑superscript2𝑑1𝑑superscript𝑑11𝑑11𝑑superscript𝑑11𝑑\displaystyle\geq\ (2\ \sqrt{d})^{\frac{1}{d}}\Gamma(1+\frac{1}{d})\ \frac{(d+1)^{-\ \frac{1}{d}}}{\Gamma(2+\frac{1}{d})}\ =\ (2\ \sqrt{d})^{\frac{1}{d}}\frac{(d+1)^{-\ \frac{1}{d}}}{1-\frac{1}{d}}\ >\ (d+1)^{\frac{1}{d}} |  | (386) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =exp⁡(−1d​ln⁡(d+1))≥ 1−1d​ln⁡(d+1),absent1𝑑𝑑111𝑑𝑑1\displaystyle=\ \exp(-\frac{1}{d}\ \ln(d+1))\ \geq\ 1\ -\ \frac{1}{d}\ \ln(d+1)\ , |  |

where in the last step we used the elementary inequality exp⁡(x)≥1+x𝑥1𝑥\exp(x)\geq 1+x,
which follows from the mean value theorem.
∎

The next theorem states the number of stored patterns
for the expected minimal separation.

###### Theorem A7 (Storage Capacity (expected separation): Random Patterns).

We assume patterns on the sphere with radius M=K​d−1𝑀𝐾𝑑1M=K\sqrt{d-1}
that are randomly chosen.
Then for all values c≥1𝑐1c\geq 1 for which

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 15​(d−1)​K2​c−1​(1−ln⁡(d−1)(d−1))215𝑑1superscript𝐾2superscript𝑐1superscript1𝑑1𝑑12\displaystyle\frac{1}{5}\ (d-1)\ K^{2}\ c^{-1}(1\ -\ \frac{\ln(d-1)}{(d-1)})^{2}\ | ≥2β​cd−14+1β​ln⁡(2​cd−12​β​(d−1)​K2)absent2𝛽superscript𝑐𝑑141𝛽2superscript𝑐𝑑12𝛽𝑑1superscript𝐾2\displaystyle\geq\ \frac{2}{\beta\ c^{\frac{d-1}{4}}}\ +\ \frac{1}{\beta}\ \ln\left(2\ c^{\frac{d-1}{2}}\ \beta\ (d-1)\ K^{2}\right) |  | (387) |

holds,
the number of stored patterns for the expected minimal separation is at least

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | =cd−14.absentsuperscript𝑐𝑑14\displaystyle=\ c^{\frac{d-1}{4}}\ . |  | (388) |

The inequality Eq. ([387](#A1.E387 "In Theorem A7 (Storage Capacity (expected separation): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) is e.g. fulfilled with β=1𝛽1\beta=1, K=3𝐾3K=3, c=2𝑐2c=2 and d≥17𝑑17d\geq 17.

###### Proof.

Instead of considering the probability that the master inequality Eq. ([311](#A1.E311 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
is fulfilled we now consider whether this inequality is fulfilled for the expected minimal distance.
We consider the expectation of the minimal distance ΔminsubscriptΔ\Delta\_{\min}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E​[Δmin]Edelimited-[]subscriptΔ\displaystyle\mathbf{\mathrm{E}}[\Delta\_{\min}]\ | =E[M2(1−cos(αmin)))]=M2(1−E[cos(αmin))]).\displaystyle=\ \mathbf{\mathrm{E}}[M^{2}(1\ -\ \cos(\alpha\_{\min})))]\ =\ M^{2}(1\ -\ \mathbf{\mathrm{E}}[\cos(\alpha\_{\min}))])\ . |  | (389) |

For this expectation, the master inequality Eq. ([311](#A1.E311 "In A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
becomes

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | M2(1−E[cos(αmin))])≥2β​N+1βln(2N2βM2).\displaystyle M^{2}(1\ -\ \mathbf{\mathrm{E}}[\cos(\alpha\_{\min}))])\ \geq\ \frac{2}{\beta\ N}\ +\ \frac{1}{\beta}\ \ln\left(2\ N^{2}\ \beta\ M^{2}\right)\ . |  | (390) |

We want to find the largest N𝑁N that fulfills this inequality.

We apply Eq. ([329](#A1.E329 "In Lemma A15. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) and Jensen’s inequality to deduce the following lower bound:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1−E​[cos⁡(αmin)]1Edelimited-[]subscript𝛼\displaystyle 1\ -\ \mathbf{\mathrm{E}}[\cos(\alpha\_{\min})]\ | ≥15​E​[αmin2]≥15​E​[αmin]2.absent15Edelimited-[]superscriptsubscript𝛼215Esuperscriptdelimited-[]subscript𝛼2\displaystyle\geq\ \frac{1}{5}\ \mathbf{\mathrm{E}}\left[\alpha\_{\min}^{2}\right]\ \geq\ \frac{1}{5}\ \mathbf{\mathrm{E}}[\alpha\_{\min}]^{2}\ . |  | (391) |

Now we use Eq. ([383](#A1.E383 "In Lemma A16 (Proposition 3.6 in Brauchart et al. (2018)). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) and Eq. ([384](#A1.E384 "In Lemma A17. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) to arrive at

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E​[αmin]2Esuperscriptdelimited-[]subscript𝛼2\displaystyle\mathbf{\mathrm{E}}[\alpha\_{\min}]^{2}\ | ≥N−4d−1​E​[N2d−1​αmin]2≥N−4d−1​Cd−12≥N−4d−1​(1−ln⁡(d−1)(d−1))2,absentsuperscript𝑁4𝑑1Esuperscriptdelimited-[]superscript𝑁2𝑑1subscript𝛼2superscript𝑁4𝑑1superscriptsubscript𝐶𝑑12superscript𝑁4𝑑1superscript1𝑑1𝑑12\displaystyle\geq\ N^{-\frac{4}{d-1}}\ \mathbf{\mathrm{E}}[N^{\frac{2}{d-1}}\ \alpha\_{\min}]^{2}\ \geq\ N^{-\frac{4}{d-1}}\ C\_{d-1}^{2}\ \geq\ N^{-\frac{4}{d-1}}\ (1-\frac{\ln(d-1)}{(d-1)})^{2}\ , |  | (392) |

for sufficiently large d𝑑d.
Thus in order to fulfill Eq. ([390](#A1.E390 "In Proof. ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), it is
enough to find values that satisfy Eq. ([387](#A1.E387 "In Theorem A7 (Storage Capacity (expected separation): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).

∎

##### A.1.6.2 Retrieval of Patterns with One Update and Small Retrieval Error.

Retrieval of a pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} for fixed point 𝒙i∗superscriptsubscript𝒙𝑖\bm{x}\_{i}^{\*} and query 𝝃𝝃\bm{\xi}
is defined via an ϵitalic-ϵ\epsilon by ‖f​(𝝃)−𝒙i∗‖<ϵnorm𝑓𝝃superscriptsubscript𝒙𝑖italic-ϵ{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}<\epsilon,
that is, the update is ϵitalic-ϵ\epsilon-close to the fixed point.
The update rule retrieves a pattern with one update for well separated patterns, that is,
ΔisubscriptΔ𝑖\Delta\_{i} is large.

###### Theorem A8 (Pattern Retrieval with One Update).

With query 𝛏𝛏\bm{\xi}, after one update the distance of the new point f​(𝛏)𝑓𝛏f(\bm{\xi})
to the fixed point 𝐱i∗superscriptsubscript𝐱𝑖\bm{x}\_{i}^{\*} is exponentially small in the separation ΔisubscriptΔ𝑖\Delta\_{i}.
The precise bounds using the Jacobian J=∂f​(𝛏)∂𝛏J𝑓𝛏𝛏\mathrm{J}=\frac{\partial f(\bm{\xi})}{\partial\bm{\xi}} and its value JmsuperscriptJ𝑚\mathrm{J}^{m} in the mean value
theorem are:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i∗‖norm𝑓𝝃superscriptsubscript𝒙𝑖\displaystyle{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}\ | ⩽‖Jm‖2​‖𝝃−𝒙i∗‖,absentsubscriptnormsuperscriptJ𝑚2norm𝝃superscriptsubscript𝒙𝑖\displaystyle\leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}^{\*}\right\|}}\ , |  | (393) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jm‖2subscriptnormsuperscriptJ𝑚2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ | ⩽ 2​β​N​M2​(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M)).absent2𝛽𝑁superscript𝑀2𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle\leqslant\ 2\ \beta\ N\ M^{2}\ (N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ . |  | (394) |

For given ϵitalic-ϵ\epsilon and
sufficient large ΔisubscriptΔ𝑖\Delta\_{i}, we have ‖f​(𝛏)−𝐱i∗‖<ϵnorm𝑓𝛏superscriptsubscript𝐱𝑖italic-ϵ{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}<\epsilon,
that is, retrieval with one update.

###### Proof.

From Eq. ([180](#A1.E180 "In A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jm‖2subscriptnormsuperscriptJ𝑚2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ | ⩽ 2​β​N​M2​(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M)).absent2𝛽𝑁superscript𝑀2𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle\leqslant\ 2\ \beta\ N\ M^{2}\ (N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ . |  | (395) |

After every iteration the mapped point f​(𝝃)𝑓𝝃f(\bm{\xi}) is closer to the fixed point 𝒙i∗superscriptsubscript𝒙𝑖\bm{x}\_{i}^{\*}
than the original point 𝒙isubscript𝒙𝑖\bm{x}\_{i}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i∗‖norm𝑓𝝃superscriptsubscript𝒙𝑖\displaystyle{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}\ | ⩽‖Jm‖2​‖𝝃−𝒙i∗‖.absentsubscriptnormsuperscriptJ𝑚2norm𝝃superscriptsubscript𝒙𝑖\displaystyle\leqslant\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}^{\*}\right\|}}\ . |  | (396) |

For given ϵitalic-ϵ\epsilon and
sufficient large ΔisubscriptΔ𝑖\Delta\_{i}, we have ‖f​(𝝃)−𝒙i∗‖<ϵnorm𝑓𝝃superscriptsubscript𝒙𝑖italic-ϵ{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}<\epsilon,
since ‖Jm‖2subscriptnormsuperscriptJ𝑚2{{\left\|\mathrm{J}^{m}\right\|}}\_{2} foes exponentially fast to zero with increasing ΔisubscriptΔ𝑖\Delta\_{i}.
∎

We want to estimate how large ΔisubscriptΔ𝑖\Delta\_{i} is.
For 𝒙isubscript𝒙𝑖\bm{x}\_{i} we have:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΔisubscriptΔ𝑖\displaystyle\Delta\_{i}\ | =minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)=𝒙iT​𝒙i−maxj,j≠i⁡𝒙iT​𝒙j.absentsubscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖subscript  𝑗𝑗 𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗\displaystyle=\ \min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \bm{x}\_{i}^{T}\bm{x}\_{j}\right)\ =\ \bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \max\_{j,j\not=i}\bm{x}\_{i}^{T}\bm{x}\_{j}\ . |  | (397) |

To estimate how large ΔisubscriptΔ𝑖\Delta\_{i} is,
assume vectors 𝒙∈ℝd𝒙superscriptℝ𝑑\bm{x}\in\mathbb{R}^{d} and 𝒚∈ℝd𝒚superscriptℝ𝑑\bm{y}\in\mathbb{R}^{d} that have as components
standard normally distributed values.
The expected value of the separation of two points
with normally distributed components is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E​[𝒙T​𝒙−𝒙T​𝒚]Edelimited-[]superscript𝒙𝑇𝒙superscript𝒙𝑇𝒚\displaystyle\mathbf{\mathrm{E}}\left[\bm{x}^{T}\bm{x}\ -\ \bm{x}^{T}\bm{y}\right]\ | =∑j=1dE​[xj2]+∑j=1dE​[xj]​∑j=1dE​[yj]=d.absentsuperscriptsubscript𝑗1𝑑Edelimited-[]superscriptsubscript𝑥𝑗2superscriptsubscript𝑗1𝑑Edelimited-[]subscript𝑥𝑗superscriptsubscript𝑗1𝑑Edelimited-[]subscript𝑦𝑗𝑑\displaystyle=\ \sum\_{j=1}^{d}\mathbf{\mathrm{E}}\left[x\_{j}^{2}\right]\ +\ \sum\_{j=1}^{d}\mathbf{\mathrm{E}}\left[x\_{j}\right]\sum\_{j=1}^{d}\mathbf{\mathrm{E}}\left[y\_{j}\right]\ =\ d\ . |  | (398) |

The variance of the separation of two points
with normally distributed components is

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var​[𝒙T​𝒙−𝒙T​𝒚]=E​[(𝒙T​𝒙−𝒙T​𝒚)2]−d2Vardelimited-[]superscript𝒙𝑇𝒙superscript𝒙𝑇𝒚Edelimited-[]superscriptsuperscript𝒙𝑇𝒙superscript𝒙𝑇𝒚2superscript𝑑2\displaystyle\mathbf{\mathrm{Var}}\left[\bm{x}^{T}\bm{x}\ -\ \bm{x}^{T}\bm{y}\right]\ =\ \mathbf{\mathrm{E}}\left[\left(\bm{x}^{T}\bm{x}\ -\ \bm{x}^{T}\bm{y}\right)^{2}\right]\ -\ d^{2} |  | (399) |
|  |  |  |
| --- | --- | --- |
|  | =∑j=1dE​[xj4]+∑j=1,k=1,k≠jdE​[xj2]​E​[xk2]− 2​∑j=1dE​[xj3]​E​[yj]−absent  superscriptsubscript𝑗1𝑑Edelimited-[]superscriptsubscript𝑥𝑗4superscriptsubscriptformulae-sequence𝑗1formulae-sequence𝑘1𝑘𝑗𝑑Edelimited-[]superscriptsubscript𝑥𝑗2Edelimited-[]superscriptsubscript𝑥𝑘2limit-from2superscriptsubscript𝑗1𝑑Edelimited-[]superscriptsubscript𝑥𝑗3Edelimited-[]subscript𝑦𝑗\displaystyle=\ \sum\_{j=1}^{d}\mathbf{\mathrm{E}}\left[x\_{j}^{4}\right]\ +\ \sum\_{j=1,k=1,k\not=j}^{d}\mathbf{\mathrm{E}}\left[x\_{j}^{2}\right]\ \mathbf{\mathrm{E}}\left[x\_{k}^{2}\right]\ \ -\ 2\ \sum\_{j=1}^{d}\mathbf{\mathrm{E}}\left[x\_{j}^{3}\right]\mathbf{\mathrm{E}}\left[y\_{j}\right]\ - |  |
|  |  |  |
| --- | --- | --- |
|  | 2​∑j=1,k=1,k≠jdE​[xj2]​E​[xk]​E​[yk]+∑j=1dE​[xj2]​E​[yj2]+2superscriptsubscriptformulae-sequence𝑗1formulae-sequence𝑘1𝑘𝑗𝑑Edelimited-[]superscriptsubscript𝑥𝑗2Edelimited-[]subscript𝑥𝑘Edelimited-[]subscript𝑦𝑘limit-fromsuperscriptsubscript𝑗1𝑑Edelimited-[]superscriptsubscript𝑥𝑗2Edelimited-[]superscriptsubscript𝑦𝑗2\displaystyle 2\ \sum\_{j=1,k=1,k\not=j}^{d}\mathbf{\mathrm{E}}\left[x\_{j}^{2}\right]\mathbf{\mathrm{E}}\left[x\_{k}\right]\mathbf{\mathrm{E}}\left[y\_{k}\right]\ +\ \sum\_{j=1}^{d}\mathbf{\mathrm{E}}\left[x\_{j}^{2}\right]\ \mathbf{\mathrm{E}}\left[y\_{j}^{2}\right]\ + |  |
|  |  |  |
| --- | --- | --- |
|  | ∑j=1,k=1,k≠jdE​[xj]​E​[yj]​E​[xk]​E​[yk]−d2superscriptsubscriptformulae-sequence𝑗1formulae-sequence𝑘1𝑘𝑗𝑑Edelimited-[]subscript𝑥𝑗Edelimited-[]subscript𝑦𝑗Edelimited-[]subscript𝑥𝑘Edelimited-[]subscript𝑦𝑘superscript𝑑2\displaystyle\sum\_{j=1,k=1,k\not=j}^{d}\mathbf{\mathrm{E}}\left[x\_{j}\right]\mathbf{\mathrm{E}}\left[y\_{j}\right]\mathbf{\mathrm{E}}\left[x\_{k}\right]\mathbf{\mathrm{E}}\left[y\_{k}\right]\ -\ d^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | =3​d+d​(d−1)+d−d2= 3​d.absent3𝑑𝑑𝑑1𝑑superscript𝑑23𝑑\displaystyle=3\ d\ +\ d\ (d-1)\ +\ d\ -\ d^{2}\ =\ 3\ d\ . |  |

The expected value for the separation of two random vectors gives:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Jm‖2subscriptnormsuperscriptJ𝑚2\displaystyle{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ | ⩽ 2​β​N​M2​(N−1)​exp⁡(−β​(d− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M)).absent2𝛽𝑁superscript𝑀2𝑁1𝛽𝑑2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle\leqslant\ 2\ \beta\ N\ M^{2}\ (N-1)\exp(-\ \beta\ (d\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ . |  | (400) |

For the exponential storage we set M=2​d−1𝑀2𝑑1M=2\sqrt{d-1}.
We see the Lipschitz constant ‖Jm‖2subscriptnormsuperscriptJ𝑚2{{\left\|\mathrm{J}^{m}\right\|}}\_{2} decreases exponentially with the dimension.
Therefore, ‖f​(𝝃)−𝒙i∗‖norm𝑓𝝃superscriptsubscript𝒙𝑖{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}} is exponentially small after just one update.
Therefore, the fixed point is well retrieved after one update.

The retrieval error decreases exponentially with the separation ΔisubscriptΔ𝑖\Delta\_{i}.

###### Theorem A9 (Exponentially Small Retrieval Error).

The retrieval error ‖f​(𝛏)−𝐱i‖norm𝑓𝛏subscript𝐱𝑖{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}\right\|}} of pattern 𝐱isubscript𝐱𝑖\bm{x}\_{i}
is bounded by

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i‖norm𝑓𝝃subscript𝒙𝑖\displaystyle{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}\right\|}}\ | ⩽ 2​(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M))​Mabsent2𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀𝑀\displaystyle\leqslant\ 2\ (N-1)\ \exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ M |  | (401) |

and for
‖𝐱i−𝐱i∗‖⩽12​β​Mnormsubscript𝐱𝑖superscriptsubscript𝐱𝑖12𝛽𝑀{{\left\|\bm{x}\_{i}-\bm{x}\_{i}^{\*}\right\|}}\leqslant\frac{1}{2\ \beta\ M}
together with ‖𝐱i−𝛏‖⩽12​β​Mnormsubscript𝐱𝑖𝛏12𝛽𝑀{{\left\|\bm{x}\_{i}-\bm{\xi}\right\|}}\leqslant\frac{1}{2\ \beta\ M}
by

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒙i−𝒙i∗‖normsubscript𝒙𝑖superscriptsubscript𝒙𝑖\displaystyle{{\left\|\bm{x}\_{i}\ -\ \bm{x}\_{i}^{\*}\right\|}}\ | ⩽ 2​e​(N−1)​M​exp⁡(−β​Δi).absent2𝑒𝑁1𝑀𝛽subscriptΔ𝑖\displaystyle\leqslant\ 2\ e\ (N-1)\ M\ \exp(-\ \beta\ \Delta\_{i})\ . |  | (402) |

###### Proof.

We compute the retrieval error which is just
‖f​(𝝃)−𝒙i‖norm𝑓𝝃subscript𝒙𝑖{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}\right\|}}.
From Lemma [A4](#ThmlemmaA4 "Lemma A4. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝒙i−f​(𝝃)‖normsubscript𝒙𝑖𝑓𝝃\displaystyle{{\left\|\bm{x}\_{i}\ -\ f(\bm{\xi})\right\|}}\ | ⩽ 2​ϵ​M,absent2italic-ϵ𝑀\displaystyle\leqslant\ 2\ \epsilon\ M\ , |  | (403) |

From Eq. ([179](#A1.E179 "In A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵitalic-ϵ\displaystyle\epsilon\ | =(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M)).absent𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀\displaystyle=\ (N-1)\exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ . |  | (404) |

For
‖𝒙i−𝒙i∗‖⩽12​β​Mnormsubscript𝒙𝑖superscriptsubscript𝒙𝑖12𝛽𝑀{{\left\|\bm{x}\_{i}-\bm{x}\_{i}^{\*}\right\|}}\leqslant\frac{1}{2\ \beta\ M}
and ‖𝒙i−𝝃‖⩽12​β​Mnormsubscript𝒙𝑖𝝃12𝛽𝑀{{\left\|\bm{x}\_{i}-\bm{\xi}\right\|}}\leqslant\frac{1}{2\ \beta\ M}
Eq. ([404](#A1.E404 "In Proof. ‣ A.1.6.2 Retrieval of Patterns with One Update and Small Retrieval Error. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ϵitalic-ϵ\displaystyle\epsilon\ | ⩽e​(N−1)​M​exp⁡(−β​Δi).absent𝑒𝑁1𝑀𝛽subscriptΔ𝑖\displaystyle\leqslant\ e\ (N-1)\ M\ \exp(-\ \beta\ \Delta\_{i})\ . |  | (405) |

∎

#### A.1.7 Learning Associations

We consider three cases of learning associations,
i.e. three cases of how sets are associated.
(i) Non of the sets is mapped in an associative space.
The raw state pattern 𝒓nsubscript𝒓𝑛\bm{r}\_{n} is the state (query) pattern 𝝃nsubscript𝝃𝑛\bm{\xi}\_{n},
i.e. 𝝃n=𝒓nsubscript𝝃𝑛subscript𝒓𝑛\bm{\xi}\_{n}=\bm{r}\_{n}, and the raw stored pattern 𝒚ssubscript𝒚𝑠\bm{y}\_{s} is
the stored pattern (key), i.e.  𝒙s=𝒚ssubscript𝒙𝑠subscript𝒚𝑠\bm{x}\_{s}=\bm{y}\_{s}.
(ii) Either one of the sets is mapped to the space of the
other set or an association matrix is learned.
(iia) The state patterns are equal to the raw patterns, i.e. 𝝃n=𝒓nsubscript𝝃𝑛subscript𝒓𝑛\bm{\xi}\_{n}=\bm{r}\_{n}, and
raw stored patterns are mapped via 𝑾𝑾\bm{W} to the space of the state patterns,
i.e. 𝒙s=𝑾​𝒚ssubscript𝒙𝑠𝑾subscript𝒚𝑠\bm{x}\_{s}=\bm{W}\bm{y}\_{s}.
(iib) The stored patterns are equal to the raw patterns, i.e. 𝒙s=𝒚ssubscript𝒙𝑠subscript𝒚𝑠\bm{x}\_{s}=\bm{y}\_{s}, and raw
state patterns are mapped via 𝑾𝑾\bm{W} to the space of the stored patterns, i.e. 𝝃n=𝑾T​𝒓nsubscript𝝃𝑛superscript𝑾𝑇subscript𝒓𝑛\bm{\xi}\_{n}=\bm{W}^{T}\bm{r}\_{n}.
(iic) The matrix 𝑾𝑾\bm{W} is an association matrix.
We will compute the derivative of the new state pattern
with respect to 𝑾𝑾\bm{W}, which is valid for all sub-cases (iib)–(iic).
(iii) Both set of patterns are mapped in a common associative space.
A raw state pattern 𝒓nsubscript𝒓𝑛\bm{r}\_{n} is mapped by 𝑾Qsubscript𝑾𝑄\bm{W}\_{Q} to a state pattern (query) 𝝃nsubscript𝝃𝑛\bm{\xi}\_{n},
that is 𝝃n=𝑾Q​𝒓nsubscript𝝃𝑛subscript𝑾𝑄subscript𝒓𝑛\bm{\xi}\_{n}=\bm{W}\_{Q}\bm{r}\_{n}.
A raw stored pattern 𝒚ssubscript𝒚𝑠\bm{y}\_{s} is mapped via
𝑾Ksubscript𝑾𝐾\bm{W}\_{K} to stored pattern (key) 𝒙ssubscript𝒙𝑠\bm{x}\_{s}, that is 𝒙s=𝑾K​𝒚ssubscript𝒙𝑠subscript𝑾𝐾subscript𝒚𝑠\bm{x}\_{s}=\bm{W}\_{K}\bm{y}\_{s}.
We will compute the derivative of the new state pattern with respect to both 𝑾Qsubscript𝑾𝑄\bm{W}\_{Q} and 𝑾Ksubscript𝑾𝐾\bm{W}\_{K}.

##### A.1.7.1 Association of Raw Patterns – No Mapping in an Associative Space.

The sets are associated via their raw patterns,
i.e. the raw state pattern 𝒓nsubscript𝒓𝑛\bm{r}\_{n} is the state (query) pattern 𝝃nsubscript𝝃𝑛\bm{\xi}\_{n},
i.e. 𝝃n=𝒓nsubscript𝝃𝑛subscript𝒓𝑛\bm{\xi}\_{n}=\bm{r}\_{n}, and raw stored pattern 𝒚ssubscript𝒚𝑠\bm{y}\_{s} is
the stored pattern (key), i.e.  𝒙s=𝒚ssubscript𝒙𝑠subscript𝒚𝑠\bm{x}\_{s}=\bm{y}\_{s}.
There is no mapping in an associative space.

The update rule is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =𝑿​𝒑,absent𝑿𝒑\displaystyle=\ \bm{X}\ \bm{p}\ , |  | (406) |

where we used

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝑿T​𝝃).absentsoftmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ \mathrm{softmax}(\beta\ \bm{X}^{T}\bm{\xi})\ . |  | (407) |

The derivative with respect to 𝝃𝝃\bm{\xi} is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝝃new∂𝝃superscript𝝃new𝝃\displaystyle\frac{\partial\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}}\ | =β​𝑿​(diag​(𝒑)−𝒑​𝒑T)​𝑿Tabsent𝛽𝑿diag𝒑𝒑superscript𝒑𝑇superscript𝑿𝑇\displaystyle=\ \beta\ \bm{X}\ \left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ \bm{X}^{T} |  | (408) |

The derivative with respect to 𝑿𝑿\bm{X} is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝑿superscript𝒂𝑇superscript𝝃new𝑿\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{X}}\ | =𝒂​𝒑T+β​𝑿​(diag​(𝒑)−𝒑​𝒑T)​(𝝃T​𝒂).absent𝒂superscript𝒑𝑇𝛽𝑿diag𝒑𝒑superscript𝒑𝑇superscript𝝃𝑇𝒂\displaystyle=\ \bm{a}\ \bm{p}^{T}\ +\ \beta\ \bm{X}\ \left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ (\bm{\xi}^{T}\bm{a})\ . |  | (409) |

These derivatives allow to apply the chain rule if a
Hopfield layer is integrated into a deep neural network.

##### A.1.7.2 Learning an Association Matrix – Only One Set is Mapped in an Associative Space.

Only one of the sets 𝑹𝑹\bm{R} or 𝒀𝒀\bm{Y} is mapped in the space of
the patterns of the other set.
Case (a): the state patterns are equal to the raw patterns 𝝃n=𝒓nsubscript𝝃𝑛subscript𝒓𝑛\bm{\xi}\_{n}=\bm{r}\_{n} and
raw stored patterns are mapped via 𝑾𝑾\bm{W} to the space of the state patterns, i.e. 𝒙s=𝑾​𝒚ssubscript𝒙𝑠𝑾subscript𝒚𝑠\bm{x}\_{s}=\bm{W}\bm{y}\_{s}.
Case (b): the stored patterns are equal to the raw patterns 𝒙s=𝒚ssubscript𝒙𝑠subscript𝒚𝑠\bm{x}\_{s}=\bm{y}\_{s}
and raw state patterns are mapped via 𝑾𝑾\bm{W}
to the space of the stored patterns, i.e.  𝝃n=𝑾T​𝒓nsubscript𝝃𝑛superscript𝑾𝑇subscript𝒓𝑛\bm{\xi}\_{n}=\bm{W}^{T}\bm{r}\_{n}.
Case (c): the matrix 𝑾𝑾\bm{W} associates the sets 𝑹𝑹\bm{R} and 𝒀𝒀\bm{Y}.
This case also includes that 𝑾T=𝑾KT​𝑾Qsuperscript𝑾𝑇superscriptsubscript𝑾𝐾𝑇subscript𝑾𝑄\bm{W}^{T}=\bm{W}\_{K}^{T}\bm{W}\_{Q}, which is treated
in next subsection.
The next subsection focuses on a low rank approximation of 𝑾𝑾\bm{W} by defining
the dimension dksubscript𝑑𝑘d\_{k} of associative space and use the matrices
𝑾KTsuperscriptsubscript𝑾𝐾𝑇\bm{W}\_{K}^{T} and 𝑾Qsubscript𝑾𝑄\bm{W}\_{Q} to define 𝑾𝑾\bm{W}, or equivalently to map 𝑹𝑹\bm{R} and 𝒀𝒀\bm{Y}
into the associative space.

From a mathematical point of view all these case are equal as they
lead to the same update rule.
Therefore, we consider in the following Case (a) with
𝒙s=𝑾​𝒚ssubscript𝒙𝑠𝑾subscript𝒚𝑠\bm{x}\_{s}=\bm{W}\bm{y}\_{s} and 𝝃n=𝒓nsubscript𝝃𝑛subscript𝒓𝑛\bm{\xi}\_{n}=\bm{r}\_{n}.
Still, the following formula are valid for
all three cases (a)–(c).

The update rule is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =𝑾​𝒀​𝒑,absent𝑾𝒀𝒑\displaystyle=\ \bm{W}\ \bm{Y}\ \bm{p}\ , |  | (410) |

where we used

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝒀T​𝑾T​𝝃).absentsoftmax𝛽superscript𝒀𝑇superscript𝑾𝑇𝝃\displaystyle=\ \mathrm{softmax}(\beta\ \bm{Y}^{T}\bm{W}^{T}\bm{\xi})\ . |  | (411) |

We consider the state (query) pattern 𝝃𝝃\bm{\xi} with result 𝝃newsuperscript𝝃new\bm{\xi}^{\mathrm{new}}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =𝑾​𝒀​𝒑=𝑾​𝒀​softmax​(β​𝒀T​𝑾T​𝝃)absent𝑾𝒀𝒑𝑾𝒀softmax𝛽superscript𝒀𝑇superscript𝑾𝑇𝝃\displaystyle=\ \bm{W}\ \bm{Y}\ \bm{p}\ =\ \bm{W}\ \bm{Y}\ \mathrm{softmax}(\beta\ \bm{Y}^{T}\bm{W}^{T}\bm{\xi}) |  | (412) |

For multiple updates this update rule has to be used.
However for a single update, or the last update we consider a
simplified update rule.

Since new state vector 𝝃newsuperscript𝝃new\bm{\xi}^{\mathrm{new}} is projected
by a weight matrix 𝑾Vsubscript𝑾𝑉\bm{W}\_{V}
to another vector, we consider the simplified update rule:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =𝒀​𝒑=𝒀​softmax​(β​𝒀T​𝑾T​𝝃)absent𝒀𝒑𝒀softmax𝛽superscript𝒀𝑇superscript𝑾𝑇𝝃\displaystyle=\ \bm{Y}\ \bm{p}\ =\ \bm{Y}\ \mathrm{softmax}(\beta\ \bm{Y}^{T}\bm{W}^{T}\bm{\xi}) |  | (413) |

The derivative with respect to 𝑾𝑾\bm{W} is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝑾superscript𝒂𝑇superscript𝝃new𝑾\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}}\ | =∂𝝃new∂𝑾​∂𝒂T​𝝃new∂𝝃new=∂𝝃new∂(𝑾T​𝝃)​∂(𝑾T​𝝃)∂𝑾​∂𝒂T​𝝃new∂𝝃new.absentsuperscript𝝃new𝑾superscript𝒂𝑇superscript𝝃newsuperscript𝝃newsuperscript𝝃newsuperscript𝑾𝑇𝝃superscript𝑾𝑇𝝃𝑾superscript𝒂𝑇superscript𝝃newsuperscript𝝃new\displaystyle=\ \frac{\partial\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}}\ \frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ =\ \frac{\partial\bm{\xi}^{\mathrm{new}}}{\partial(\bm{W}^{T}\bm{\xi})}\ \frac{\partial(\bm{W}^{T}\bm{\xi})}{\partial\bm{W}}\ \frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ . |  | (414) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝝃new∂(𝑾T​𝝃)superscript𝝃newsuperscript𝑾𝑇𝝃\displaystyle\frac{\partial\bm{\xi}^{\mathrm{new}}}{\partial(\bm{W}^{T}\bm{\xi})}\ | =β​𝒀​(diag​(𝒑)−𝒑​𝒑T)​𝒀Tabsent𝛽𝒀diag𝒑𝒑superscript𝒑𝑇superscript𝒀𝑇\displaystyle=\ \beta\ \bm{Y}\ \left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ \bm{Y}^{T} |  | (415) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝝃newsuperscript𝒂𝑇superscript𝝃newsuperscript𝝃new\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ | =𝒂.absent𝒂\displaystyle=\ \bm{a}\ . |  | (416) |

We have the product of the 3-dimensional tensor
∂(𝑾T​𝝃)∂𝑾superscript𝑾𝑇𝝃𝑾\frac{\partial(\bm{W}^{T}\bm{\xi})}{\partial\bm{W}} with the
vector 𝒂𝒂\bm{a} which gives a 2-dimensional tensor, i.e. a
matrix:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂(𝑾T​𝝃)∂𝑾​∂𝒂T​𝝃new∂𝝃newsuperscript𝑾𝑇𝝃𝑾superscript𝒂𝑇superscript𝝃newsuperscript𝝃new\displaystyle\frac{\partial(\bm{W}^{T}\bm{\xi})}{\partial\bm{W}}\ \frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ | =∂(𝑾T​𝝃)∂𝑾​𝒂=𝝃T​𝒂​𝑰.absentsuperscript𝑾𝑇𝝃𝑾𝒂superscript𝝃𝑇𝒂𝑰\displaystyle=\ \frac{\partial(\bm{W}^{T}\bm{\xi})}{\partial\bm{W}}\ \bm{a}\ =\ \bm{\xi}^{T}\bm{a}\bm{I}\ . |  | (417) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝑾superscript𝒂𝑇superscript𝝃new𝑾\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}}\ | =β​𝒀​(diag​(𝒑)−𝒑​𝒑T)​𝒀T​(𝝃T​𝒂)=J​(𝝃T​𝒂),absent𝛽𝒀diag𝒑𝒑superscript𝒑𝑇superscript𝒀𝑇superscript𝝃𝑇𝒂Jsuperscript𝝃𝑇𝒂\displaystyle=\ \beta\ \bm{Y}\ \left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ \bm{Y}^{T}(\bm{\xi}^{T}\bm{a})\ =\ \mathrm{J}\ (\bm{\xi}^{T}\bm{a})\ , |  | (418) |

where JJ\mathrm{J} is the Jacobian of the update rule defined in Eq. ([59](#A1.E59 "In A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).

To obtain the derivative of the full update rule Eq. ([412](#A1.E412 "In A.1.7.2 Learning an Association Matrix – Only One Set is Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
we have to add the term

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒂​𝒑T​𝒀T𝒂superscript𝒑𝑇superscript𝒀𝑇\displaystyle\bm{a}\ \bm{p}^{T}\bm{Y}^{T} |  | (419) |

and include the factor 𝑾𝑾\bm{W} to get

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝑾superscript𝒂𝑇superscript𝝃new𝑾\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}}\ | =𝒂​𝒑T​𝒀T+β​𝑾​𝒀​(diag​(𝒑)−𝒑​𝒑T)​𝒀T​(𝝃T​𝒂)absent𝒂superscript𝒑𝑇superscript𝒀𝑇𝛽𝑾𝒀diag𝒑𝒑superscript𝒑𝑇superscript𝒀𝑇superscript𝝃𝑇𝒂\displaystyle=\ \bm{a}\ \bm{p}^{T}\bm{Y}^{T}\ +\ \beta\ \bm{W}\ \bm{Y}\ \left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ \bm{Y}^{T}(\bm{\xi}^{T}\bm{a}) |  | (420) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝒂​𝒑T​𝒀T+𝑾​J​(𝝃T​𝒂).absent𝒂superscript𝒑𝑇superscript𝒀𝑇𝑾Jsuperscript𝝃𝑇𝒂\displaystyle=\ \bm{a}\ \bm{p}^{T}\bm{Y}^{T}\ +\ \bm{W}\ \mathrm{J}\ (\bm{\xi}^{T}\bm{a})\ . |  |

##### A.1.7.3 Learning Two Association Mappings – Both Sets are Mapped in an Associative Space.

Both sets 𝑹𝑹\bm{R} and 𝒀𝒀\bm{Y} are mapped in an associative space.
Every raw state pattern 𝒓nsubscript𝒓𝑛\bm{r}\_{n} is mapped via
𝑾Qsubscript𝑾𝑄\bm{W}\_{Q} to a state pattern (query) 𝝃n=𝑾Q​𝒓nsubscript𝝃𝑛subscript𝑾𝑄subscript𝒓𝑛\bm{\xi}\_{n}=\bm{W}\_{Q}\bm{r}\_{n}.
Every raw stored pattern 𝒚ssubscript𝒚𝑠\bm{y}\_{s} is mapped via
𝑾Ksubscript𝑾𝐾\bm{W}\_{K} to a stored pattern (key) 𝒙s=𝑾K​𝒚ssubscript𝒙𝑠subscript𝑾𝐾subscript𝒚𝑠\bm{x}\_{s}=\bm{W}\_{K}\bm{y}\_{s}.
In the last subsection we considered a single matrix 𝑾𝑾\bm{W}.
For 𝑾T=𝑾KT​𝑾Qsuperscript𝑾𝑇superscriptsubscript𝑾𝐾𝑇subscript𝑾𝑄\bm{W}^{T}=\bm{W}\_{K}^{T}\bm{W}\_{Q} we have the case of the last subsection.
However in this subsection we are looking for a
low rank approximation of 𝑾𝑾\bm{W}.
Toward this end we define the dimension dksubscript𝑑𝑘d\_{k}
of associative space and use the matrices
𝑾KTsuperscriptsubscript𝑾𝐾𝑇\bm{W}\_{K}^{T} and 𝑾Qsubscript𝑾𝑄\bm{W}\_{Q} to map to the associative space.

The update rule is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =𝑿​𝒑,absent𝑿𝒑\displaystyle=\ \bm{X}\ \bm{p}\ , |  | (421) |

where we used

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝑿T​𝝃).absentsoftmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ \mathrm{softmax}(\beta\ \bm{X}^{T}\bm{\xi})\ . |  | (422) |

We consider raw state patterns 𝒓nsubscript𝒓𝑛\bm{r}\_{n} that are mapped to
state patterns 𝝃n=𝑾Q​𝒓nsubscript𝝃𝑛subscript𝑾𝑄subscript𝒓𝑛\bm{\xi}\_{n}=\bm{W}\_{Q}\bm{r}\_{n} with 𝑸T=𝚵=𝑾Q​𝑹superscript𝑸𝑇𝚵subscript𝑾𝑄𝑹\bm{Q}^{T}=\bm{\Xi}=\bm{W}\_{Q}\bm{R}
and raw stored pattern 𝒚ssubscript𝒚𝑠\bm{y}\_{s} that are mapped to
stored patterns 𝒙s=𝑾K​𝒚ssubscript𝒙𝑠subscript𝑾𝐾subscript𝒚𝑠\bm{x}\_{s}=\bm{W}\_{K}\bm{y}\_{s} with 𝑲T=𝑿=𝑾K​𝒀superscript𝑲𝑇𝑿subscript𝑾𝐾𝒀\bm{K}^{T}=\bm{X}=\bm{W}\_{K}\bm{Y}.
The update rule is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =𝑾K​𝒀​𝒑=𝑾K​𝒀​softmax​(β​𝒀T​𝑾KT​𝑾Q​𝒓).absentsubscript𝑾𝐾𝒀𝒑subscript𝑾𝐾𝒀softmax𝛽superscript𝒀𝑇superscriptsubscript𝑾𝐾𝑇subscript𝑾𝑄𝒓\displaystyle=\ \bm{W}\_{K}\ \bm{Y}\ \bm{p}\ =\ \bm{W}\_{K}\ \bm{Y}\ \mathrm{softmax}(\beta\ \bm{Y}^{T}\bm{W}\_{K}^{T}\bm{W}\_{Q}\ \bm{r})\ . |  | (423) |

Since new state vector 𝝃newsuperscript𝝃new\bm{\xi}^{\mathrm{new}} is projected
by a weight matrix 𝑾Vsubscript𝑾𝑉\bm{W}\_{V}
to another vector, we consider the simplified update rule:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =𝒀​𝒑=𝒀​softmax​(β​𝒀T​𝑾KT​𝑾Q​𝒓).absent𝒀𝒑𝒀softmax𝛽superscript𝒀𝑇superscriptsubscript𝑾𝐾𝑇subscript𝑾𝑄𝒓\displaystyle=\ \bm{Y}\ \bm{p}\ =\ \bm{Y}\ \mathrm{softmax}(\beta\ \bm{Y}^{T}\bm{W}\_{K}^{T}\bm{W}\_{Q}\ \bm{r})\ . |  | (424) |

For the simplified update rule,
the vector 𝝃newsuperscript𝝃new\bm{\xi}^{\mathrm{new}} does not live in the associative space but
in the space of raw stored pattern 𝒚𝒚\bm{y}.
However 𝑾Ksubscript𝑾𝐾\bm{W}\_{K} would map it to the associative space.

•Derivative with respect to 𝐖Qsubscript𝐖𝑄\bm{W}\_{Q}.
The derivative with respect to 𝑾Qsubscript𝑾𝑄\bm{W}\_{Q} is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝑾Qsuperscript𝒂𝑇superscript𝝃newsubscript𝑾𝑄\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}\_{Q}}\ | =∂𝝃new∂𝑾Q​∂𝒂T​𝝃new∂𝝃new=∂𝝃new∂(𝑾Q​𝒓)​∂(𝑾Q​𝒓)∂𝑾Q​∂𝒂T​𝝃new∂𝝃new.absentsuperscript𝝃newsubscript𝑾𝑄superscript𝒂𝑇superscript𝝃newsuperscript𝝃newsuperscript𝝃newsubscript𝑾𝑄𝒓subscript𝑾𝑄𝒓subscript𝑾𝑄superscript𝒂𝑇superscript𝝃newsuperscript𝝃new\displaystyle=\ \frac{\partial\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}\_{Q}}\ \frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ =\ \frac{\partial\bm{\xi}^{\mathrm{new}}}{\partial(\bm{W}\_{Q}\ \bm{r})}\ \frac{\partial(\bm{W}\_{Q}\ \bm{r})}{\partial\bm{W}\_{Q}}\ \frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ . |  | (425) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝝃new∂(𝑾Q​𝒓)superscript𝝃newsubscript𝑾𝑄𝒓\displaystyle\frac{\partial\bm{\xi}^{\mathrm{new}}}{\partial(\bm{W}\_{Q}\ \bm{r})}\ | =β​𝒀​(diag​(𝒑)−𝒑​𝒑T)​𝒀T​𝑾KTabsent𝛽𝒀diag𝒑𝒑superscript𝒑𝑇superscript𝒀𝑇superscriptsubscript𝑾𝐾𝑇\displaystyle=\ \beta\ \bm{Y}\ \left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ \bm{Y}^{T}\bm{W}\_{K}^{T} |  | (426) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝝃newsuperscript𝒂𝑇superscript𝝃newsuperscript𝝃new\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ | =𝒂.absent𝒂\displaystyle=\ \bm{a}\ . |  | (427) |

We have the product of the 3-dimensional tensor
∂(𝑾Q​𝒓)∂𝑾Qsubscript𝑾𝑄𝒓subscript𝑾𝑄\frac{\partial(\bm{W}\_{Q}\bm{r})}{\partial\bm{W}\_{Q}} with the
vector 𝒂𝒂\bm{a} which gives a 2-dimensional tensor, i.e. a
matrix:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂(𝑾Q​𝒓)∂𝑾Q​∂𝒂T​𝝃new∂𝝃newsubscript𝑾𝑄𝒓subscript𝑾𝑄superscript𝒂𝑇superscript𝝃newsuperscript𝝃new\displaystyle\frac{\partial(\bm{W}\_{Q}\ \bm{r})}{\partial\bm{W}\_{Q}}\ \frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ | =∂(𝑾Q​𝒓)∂𝑾Q​𝒂=𝒓T​𝒂​𝑰.absentsubscript𝑾𝑄𝒓subscript𝑾𝑄𝒂superscript𝒓𝑇𝒂𝑰\displaystyle=\ \frac{\partial(\bm{W}\_{Q}\ \bm{r})}{\partial\bm{W}\_{Q}}\ \bm{a}\ =\ \bm{r}^{T}\bm{a}\ \bm{I}\ . |  | (428) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝑾Qsuperscript𝒂𝑇superscript𝝃newsubscript𝑾𝑄\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}\_{Q}}\ | =β​𝒀​(diag​(𝒑)−𝒑​𝒑T)​𝒀T​𝑾KT​(𝒓T​𝒂)=J​𝑾KT​(𝒓T​𝒂),absent𝛽𝒀diag𝒑𝒑superscript𝒑𝑇superscript𝒀𝑇superscriptsubscript𝑾𝐾𝑇superscript𝒓𝑇𝒂Jsuperscriptsubscript𝑾𝐾𝑇superscript𝒓𝑇𝒂\displaystyle=\ \beta\ \bm{Y}\ \left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ \bm{Y}^{T}\ \bm{W}\_{K}^{T}(\bm{r}^{T}\bm{a})\ =\ \mathrm{J}\ \bm{W}\_{K}^{T}(\bm{r}^{T}\bm{a})\ , |  | (429) |

where JJ\mathrm{J} is the Jacobian of the update rule defined in Eq. ([59](#A1.E59 "In A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).

To obtain the derivative of the full update rule Eq. ([423](#A1.E423 "In A.1.7.3 Learning Two Association Mappings – Both Sets are Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
we have to include the factor 𝑾Ksubscript𝑾𝐾\bm{W}\_{K}, then get

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝑾Qsuperscript𝒂𝑇superscript𝝃newsubscript𝑾𝑄\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}\_{Q}}\ | =β​𝑾K​𝒀​(diag​(𝒑)−𝒑​𝒑T)​𝒀T​𝑾KT​(𝒓T​𝒂)=𝑾K​J​𝑾KT​(𝒓T​𝒂).absent𝛽subscript𝑾𝐾𝒀diag𝒑𝒑superscript𝒑𝑇superscript𝒀𝑇superscriptsubscript𝑾𝐾𝑇superscript𝒓𝑇𝒂subscript𝑾𝐾Jsuperscriptsubscript𝑾𝐾𝑇superscript𝒓𝑇𝒂\displaystyle=\ \beta\ \bm{W}\_{K}\ \bm{Y}\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ \bm{Y}^{T}\ \bm{W}\_{K}^{T}(\bm{r}^{T}\bm{a})\ =\ \bm{W}\_{K}\ \mathrm{J}\ \bm{W}\_{K}^{T}(\bm{r}^{T}\bm{a})\ . |  | (430) |

•Derivative with respect to 𝐖Ksubscript𝐖𝐾\bm{W}\_{K}.
The derivative with respect to 𝑾Ksubscript𝑾𝐾\bm{W}\_{K} is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝑾Ksuperscript𝒂𝑇superscript𝝃newsubscript𝑾𝐾\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}\_{K}}\ | =∂𝝃new∂𝑾K​∂𝒂T​𝝃new∂𝝃new=∂𝝃new∂(𝑾KT​𝑾Q​𝒓)​∂(𝑾KT​𝑾Q​𝒓)∂𝑾K​∂𝒂T​𝝃new∂𝝃new.absentsuperscript𝝃newsubscript𝑾𝐾superscript𝒂𝑇superscript𝝃newsuperscript𝝃newsuperscript𝝃newsuperscriptsubscript𝑾𝐾𝑇subscript𝑾𝑄𝒓superscriptsubscript𝑾𝐾𝑇subscript𝑾𝑄𝒓subscript𝑾𝐾superscript𝒂𝑇superscript𝝃newsuperscript𝝃new\displaystyle=\ \frac{\partial\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}\_{K}}\ \frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ =\ \frac{\partial\bm{\xi}^{\mathrm{new}}}{\partial(\bm{W}\_{K}^{T}\bm{W}\_{Q}\ \bm{r})}\ \frac{\partial(\bm{W}\_{K}^{T}\bm{W}\_{Q}\ \bm{r})}{\partial\bm{W}\_{K}}\ \frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ . |  | (431) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝝃new∂(𝑾KT​𝑾Q​𝒓)superscript𝝃newsuperscriptsubscript𝑾𝐾𝑇subscript𝑾𝑄𝒓\displaystyle\frac{\partial\bm{\xi}^{\mathrm{new}}}{\partial(\bm{W}\_{K}^{T}\bm{W}\_{Q}\ \bm{r})}\ | =β​𝒀​(diag​(𝒑)−𝒑​𝒑T)​𝒀Tabsent𝛽𝒀diag𝒑𝒑superscript𝒑𝑇superscript𝒀𝑇\displaystyle=\ \beta\ \bm{Y}\ \left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ \bm{Y}^{T} |  | (432) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝝃newsuperscript𝒂𝑇superscript𝝃newsuperscript𝝃new\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ | =𝒂.absent𝒂\displaystyle=\ \bm{a}\ . |  | (433) |

We have the product of the 3-dimensional tensor
∂(𝑾​𝒓)∂𝑾K𝑾𝒓subscript𝑾𝐾\frac{\partial(\bm{W}\bm{r})}{\partial\bm{W}\_{K}} with the
vector 𝒂𝒂\bm{a} which gives a 2-dimensional tensor, i.e. a
matrix:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂(𝑾KT​𝑾Q​𝒓)∂𝑾K​∂𝒂T​𝝃new∂𝝃newsuperscriptsubscript𝑾𝐾𝑇subscript𝑾𝑄𝒓subscript𝑾𝐾superscript𝒂𝑇superscript𝝃newsuperscript𝝃new\displaystyle\frac{\partial(\bm{W}\_{K}^{T}\bm{W}\_{Q}\ \bm{r})}{\partial\bm{W}\_{K}}\ \frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{\xi}^{\mathrm{new}}}\ | =∂(𝑾KT​𝑾Q​𝒓)∂𝑾K​𝒂=𝑾QT​𝒓T​𝒂​𝑰.absentsuperscriptsubscript𝑾𝐾𝑇subscript𝑾𝑄𝒓subscript𝑾𝐾𝒂superscriptsubscript𝑾𝑄𝑇superscript𝒓𝑇𝒂𝑰\displaystyle=\ \frac{\partial(\bm{W}\_{K}^{T}\bm{W}\_{Q}\ \bm{r})}{\partial\bm{W}\_{K}}\ \bm{a}\ =\ \bm{W}\_{Q}^{T}\bm{r}^{T}\bm{a}\ \bm{I}\ . |  | (434) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝑾Ksuperscript𝒂𝑇superscript𝝃newsubscript𝑾𝐾\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}\_{K}}\ | =β​𝒀​(diag​(𝒑)−𝒑​𝒑T)​𝒀T​(𝑾QT​𝒓T​𝒂)=J​(𝑾QT​𝒓T​𝒂),absent𝛽𝒀diag𝒑𝒑superscript𝒑𝑇superscript𝒀𝑇superscriptsubscript𝑾𝑄𝑇superscript𝒓𝑇𝒂Jsuperscriptsubscript𝑾𝑄𝑇superscript𝒓𝑇𝒂\displaystyle=\ \beta\ \bm{Y}\ \left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ \bm{Y}^{T}\ (\bm{W}\_{Q}^{T}\bm{r}^{T}\bm{a})\ =\ \mathrm{J}\ (\bm{W}\_{Q}^{T}\bm{r}^{T}\bm{a})\ , |  | (435) |

where JJ\mathrm{J} is the Jacobian of the update rule defined in Eq. ([59](#A1.E59 "In A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).

To obtain the derivative of the full update rule Eq. ([423](#A1.E423 "In A.1.7.3 Learning Two Association Mappings – Both Sets are Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
we have to add the term

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒂​𝒑T​𝒀T𝒂superscript𝒑𝑇superscript𝒀𝑇\displaystyle\bm{a}\ \bm{p}^{T}\bm{Y}^{T} |  | (436) |

and to include the factor 𝑾Ksubscript𝑾𝐾\bm{W}\_{K}, then get

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒂T​𝝃new∂𝑾Ksuperscript𝒂𝑇superscript𝝃newsubscript𝑾𝐾\displaystyle\frac{\partial\bm{a}^{T}\bm{\xi}^{\mathrm{new}}}{\partial\bm{W}\_{K}}\ | =𝒂​𝒑T​𝒀T+β​𝑾K​𝒀​(diag​(𝒑)−𝒑​𝒑T)​𝒀T​(𝑾QT​𝒓T​𝒂)absent𝒂superscript𝒑𝑇superscript𝒀𝑇𝛽subscript𝑾𝐾𝒀diag𝒑𝒑superscript𝒑𝑇superscript𝒀𝑇superscriptsubscript𝑾𝑄𝑇superscript𝒓𝑇𝒂\displaystyle=\ \bm{a}\ \bm{p}^{T}\bm{Y}^{T}\ +\ \beta\ \bm{W}\_{K}\ \bm{Y}\ \left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ \bm{Y}^{T}(\bm{W}\_{Q}^{T}\bm{r}^{T}\bm{a}) |  | (437) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝒂​𝒑T​𝒀T+𝑾K​J​(𝑾QT​𝒓T​𝒂).absent𝒂superscript𝒑𝑇superscript𝒀𝑇subscript𝑾𝐾Jsuperscriptsubscript𝑾𝑄𝑇superscript𝒓𝑇𝒂\displaystyle=\ \bm{a}\ \bm{p}^{T}\bm{Y}^{T}\ +\ \bm{W}\_{K}\ \mathrm{J}\ (\bm{W}\_{Q}^{T}\bm{r}^{T}\bm{a})\ . |  |

#### A.1.8 Infinite Many Patterns and Forgetting Patterns

In the next subsection we show how
the new Hopfield networks can be used for auto-regressive tasks
by causal masking.
In the following subsection,
we introduce forgetting to the new Hopfield networks
by adding a negative value to the softmax which is larger
if the pattern was observed more in the past.

##### A.1.8.1 Infinite Many Patterns.

The new Hopfield networks can be used for auto-regressive tasks,
that is time series prediction and similar.
Causal masking masks out the future by a large negative
value in the softmax.

We assume to have infinite many stored patterns (keys) 𝒙1,𝒙2,…

subscript𝒙1subscript𝒙2…\bm{x}\_{1},\bm{x}\_{2},\ldots
that are represented by the infinite matrix

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑿𝑿\displaystyle\bm{X}\ | =(𝒙1,𝒙2,…,).\displaystyle=\ \left(\bm{x}\_{1},\bm{x}\_{2},\ldots,\right)\ . |  | (438) |

The pattern index is now a time index, that is,
we observe 𝒙tsubscript𝒙𝑡\bm{x}\_{t} at time t𝑡t.

The pattern matrix at time t𝑡t is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑿tsubscript𝑿𝑡\displaystyle\bm{X}\_{t}\ | =(𝒙1,𝒙2,…,𝒙t).absentsubscript𝒙1subscript𝒙2…subscript𝒙𝑡\displaystyle=\ \left(\bm{x}\_{1},\bm{x}\_{2},\ldots,\bm{x}\_{t}\right)\ . |  | (439) |

The query at time t𝑡t is 𝝃tsubscript𝝃𝑡\bm{\xi}\_{t}.

For Mt=max1⩽i⩽t⁡‖𝒙t‖subscript𝑀𝑡subscript1𝑖𝑡normsubscript𝒙𝑡M\_{t}=\max\_{1\leqslant i\leqslant t}{{\left\|\bm{x}\_{t}\right\|}},
the energy function at time t𝑡t is EtsubscriptE𝑡\mathrm{E}\_{t}

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EtsubscriptE𝑡\displaystyle\mathrm{E}\_{t}\ | =−lse​(β,𝑿tT​𝝃t)+12​𝝃tT​𝝃t+β−1​ln⁡t+12​Mt2absentlse𝛽superscriptsubscript𝑿𝑡𝑇subscript𝝃𝑡12superscriptsubscript𝝃𝑡𝑇subscript𝝃𝑡superscript𝛽1𝑡12superscriptsubscript𝑀𝑡2\displaystyle=\ -\ \mathrm{lse}(\beta,\bm{X}\_{t}^{T}\bm{\xi}\_{t})\ +\ \frac{1}{2}\bm{\xi}\_{t}^{T}\bm{\xi}\_{t}\ +\ \beta^{-1}\ln t\ +\ \frac{1}{2}M\_{t}^{2} |  | (440) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−β−1​ln⁡(∑i=1texp⁡(β​𝒙iT​𝝃t))+12​𝝃tT​𝝃t+β−1​ln⁡t+12​Mt2.absentsuperscript𝛽1superscriptsubscript𝑖1𝑡𝛽superscriptsubscript𝒙𝑖𝑇subscript𝝃𝑡12superscriptsubscript𝝃𝑡𝑇subscript𝝃𝑡superscript𝛽1𝑡12superscriptsubscript𝑀𝑡2\displaystyle=\ -\ \beta^{-1}\ln\left(\sum\_{i=1}^{t}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi}\_{t})\right)\ +\ \frac{1}{2}\bm{\xi}\_{t}^{T}\bm{\xi}\_{t}\ +\ \beta^{-1}\ln t\ +\ \frac{1}{2}M\_{t}^{2}\ . |  | (441) |

The update rule is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃tnewsuperscriptsubscript𝝃𝑡new\displaystyle\bm{\xi}\_{t}^{\mathrm{new}}\ | =𝑿t​𝒑t=𝑿t​softmax​(β​𝑿tT​𝝃t),absentsubscript𝑿𝑡subscript𝒑𝑡subscript𝑿𝑡softmax𝛽superscriptsubscript𝑿𝑡𝑇subscript𝝃𝑡\displaystyle=\ \bm{X}\_{t}\ \bm{p}\_{t}\ =\ \bm{X}\_{t}\ \mathrm{softmax}(\beta\ \bm{X}\_{t}^{T}\bm{\xi}\_{t})\ , |  | (442) |

where we used

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑tsubscript𝒑𝑡\displaystyle\bm{p}\_{t}\ | =softmax​(β​𝑿tT​𝝃t).absentsoftmax𝛽superscriptsubscript𝑿𝑡𝑇subscript𝝃𝑡\displaystyle=\ \mathrm{softmax}(\beta\ \bm{X}\_{t}^{T}\bm{\xi}\_{t})\ . |  | (443) |

We can use an infinite pattern matrix with
an infinite softmax when using causal masking.
The pattern matrix at time t𝑡t is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑿tsubscript𝑿𝑡\displaystyle\bm{X}\_{t}\ | =(𝒙1,𝒙2,…,𝒙t,−α​𝝃t,−α​𝝃t,…),absentsubscript𝒙1subscript𝒙2…subscript𝒙𝑡𝛼subscript𝝃𝑡𝛼subscript𝝃𝑡…\displaystyle=\ \left(\bm{x}\_{1},\bm{x}\_{2},\ldots,\bm{x}\_{t},-\alpha\bm{\xi}\_{t},-\alpha\bm{\xi}\_{t},\ldots\right)\ , |  | (444) |

with the query 𝝃tsubscript𝝃𝑡\bm{\xi}\_{t} and α→∞→𝛼\alpha\to\infty.
The energy function at time t𝑡t is EtsubscriptE𝑡\mathrm{E}\_{t}

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EtsubscriptE𝑡\displaystyle\mathrm{E}\_{t}\ | =−lse​(β,𝑿tT​𝝃t)+12​𝝃tT​𝝃t+β−1​ln⁡t+12​Mt2absentlse𝛽superscriptsubscript𝑿𝑡𝑇subscript𝝃𝑡12superscriptsubscript𝝃𝑡𝑇subscript𝝃𝑡superscript𝛽1𝑡12superscriptsubscript𝑀𝑡2\displaystyle=\ -\ \mathrm{lse}(\beta,\bm{X}\_{t}^{T}\bm{\xi}\_{t})\ +\ \frac{1}{2}\bm{\xi}\_{t}^{T}\bm{\xi}\_{t}\ +\ \beta^{-1}\ln t\ +\ \frac{1}{2}M\_{t}^{2} |  | (445) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−β−1​ln⁡(∑i=1texp⁡(β​𝒙iT​𝝃t)+∑i=t+1⌊α⌋exp⁡(−β​α​‖𝝃t‖2))+12​𝝃tT​𝝃t+absentsuperscript𝛽1superscriptsubscript𝑖1𝑡𝛽superscriptsubscript𝒙𝑖𝑇subscript𝝃𝑡superscriptsubscript𝑖𝑡1𝛼𝛽𝛼superscriptnormsubscript𝝃𝑡2limit-from12superscriptsubscript𝝃𝑡𝑇subscript𝝃𝑡\displaystyle=\ -\ \beta^{-1}\ln\left(\sum\_{i=1}^{t}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi}\_{t})\ +\ \sum\_{i=t+1}^{\lfloor\alpha\rfloor}\exp(-\beta\alpha{{\left\|\bm{\xi}\_{t}\right\|}}^{2})\right)\ +\ \frac{1}{2}\bm{\xi}\_{t}^{T}\bm{\xi}\_{t}\ + |  | (446) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | β−1​ln⁡t+12​Mt2.superscript𝛽1𝑡12superscriptsubscript𝑀𝑡2\displaystyle~{}~{}~{}~{}~{}~{}~{}\beta^{-1}\ln t\ +\ \frac{1}{2}M\_{t}^{2}\ . |  |

For α→∞→𝛼\alpha\to\infty and ‖𝝃t‖>0normsubscript𝝃𝑡0{{\left\|\bm{\xi}\_{t}\right\|}}>0 this becomes

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EtsubscriptE𝑡\displaystyle\mathrm{E}\_{t}\ | =−lse​(β,𝑿tT​𝝃t)+12​𝝃tT​𝝃t+β−1​ln⁡t+12​Mt2absentlse𝛽superscriptsubscript𝑿𝑡𝑇subscript𝝃𝑡12superscriptsubscript𝝃𝑡𝑇subscript𝝃𝑡superscript𝛽1𝑡12superscriptsubscript𝑀𝑡2\displaystyle=\ -\ \mathrm{lse}(\beta,\bm{X}\_{t}^{T}\bm{\xi}\_{t})\ +\ \frac{1}{2}\bm{\xi}\_{t}^{T}\bm{\xi}\_{t}\ +\ \beta^{-1}\ln t\ +\ \frac{1}{2}M\_{t}^{2} |  | (447) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−β−1​ln⁡(∑i=1texp⁡(β​𝒙iT​𝝃t))+12​𝝃tT​𝝃t+β−1​ln⁡t+12​Mt2.absentsuperscript𝛽1superscriptsubscript𝑖1𝑡𝛽superscriptsubscript𝒙𝑖𝑇subscript𝝃𝑡12superscriptsubscript𝝃𝑡𝑇subscript𝝃𝑡superscript𝛽1𝑡12superscriptsubscript𝑀𝑡2\displaystyle=\ -\ \beta^{-1}\ln\left(\sum\_{i=1}^{t}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi}\_{t})\right)\ +\ \frac{1}{2}\bm{\xi}\_{t}^{T}\bm{\xi}\_{t}\ +\ \beta^{-1}\ln t\ +\ \frac{1}{2}M\_{t}^{2}\ . |  | (448) |

##### A.1.8.2 Forgetting Patterns.

We introduce forgetting to the new Hopfield networks
by adding a negative value in the softmax which increases
with patterns that are more in the past.

We assume to have infinite many patterns 𝒙1,𝒙2,…

subscript𝒙1subscript𝒙2…\bm{x}\_{1},\bm{x}\_{2},\ldots
that are represented by the infinite matrix

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑿𝑿\displaystyle\bm{X}\ | =(𝒙1,𝒙2,…,).\displaystyle=\ \left(\bm{x}\_{1},\bm{x}\_{2},\ldots,\right)\ . |  | (449) |

The pattern index is now a time index, that is,
we observe 𝒙tsubscript𝒙𝑡\bm{x}\_{t} at time t𝑡t.

The pattern matrix at time t𝑡t is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑿tsubscript𝑿𝑡\displaystyle\bm{X}\_{t}\ | =(𝒙1,𝒙2,…,𝒙t).absentsubscript𝒙1subscript𝒙2…subscript𝒙𝑡\displaystyle=\ \left(\bm{x}\_{1},\bm{x}\_{2},\ldots,\bm{x}\_{t}\right)\ . |  | (450) |

The query at time t𝑡t is 𝝃tsubscript𝝃𝑡\bm{\xi}\_{t}.

The energy function with forgetting parameter
γ𝛾\gamma at time t𝑡t is EtsubscriptE𝑡\mathrm{E}\_{t}

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EtsubscriptE𝑡\displaystyle\mathrm{E}\_{t}\ | =−lse​(β,𝑿tT​𝝃t−γ​(t−1,t−2,…,0)T)+12​𝝃tT​𝝃t+β−1​ln⁡t+12​Mt2absentlse𝛽superscriptsubscript𝑿𝑡𝑇subscript𝝃𝑡𝛾superscript𝑡1𝑡2…0𝑇12superscriptsubscript𝝃𝑡𝑇subscript𝝃𝑡superscript𝛽1𝑡12superscriptsubscript𝑀𝑡2\displaystyle=\ -\ \mathrm{lse}(\beta,\bm{X}\_{t}^{T}\bm{\xi}\_{t}\ -\ \gamma(t-1,t-2,\ldots,0)^{T})\ +\ \frac{1}{2}\bm{\xi}\_{t}^{T}\bm{\xi}\_{t}\ +\ \beta^{-1}\ln t\ +\ \frac{1}{2}M\_{t}^{2} |  | (451) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−β−1​ln⁡(∑i=1Texp⁡(β​𝒙iT​𝝃t−γ​(t−i)))+12​𝝃tT​𝝃t+β−1​ln⁡t+12​Mt2.absentsuperscript𝛽1superscriptsubscript𝑖1𝑇𝛽superscriptsubscript𝒙𝑖𝑇subscript𝝃𝑡𝛾𝑡𝑖12superscriptsubscript𝝃𝑡𝑇subscript𝝃𝑡superscript𝛽1𝑡12superscriptsubscript𝑀𝑡2\displaystyle=\ -\ \beta^{-1}\ln\left(\sum\_{i=1}^{T}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi}\_{t}\ -\ \gamma(t-i))\right)\ +\ \frac{1}{2}\bm{\xi}\_{t}^{T}\bm{\xi}\_{t}\ +\ \beta^{-1}\ln t\ +\ \frac{1}{2}M\_{t}^{2}\ . |  | (452) |

The update rule is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃tnewsuperscriptsubscript𝝃𝑡new\displaystyle\bm{\xi}\_{t}^{\mathrm{new}}\ | =𝑿t​𝒑t=𝑿t​softmax​(β​𝑿tT​𝝃t),absentsubscript𝑿𝑡subscript𝒑𝑡subscript𝑿𝑡softmax𝛽superscriptsubscript𝑿𝑡𝑇subscript𝝃𝑡\displaystyle=\ \bm{X}\_{t}\ \bm{p}\_{t}\ =\ \bm{X}\_{t}\ \mathrm{softmax}(\beta\bm{X}\_{t}^{T}\bm{\xi}\_{t})\ , |  | (453) |

where we used

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑tsubscript𝒑𝑡\displaystyle\bm{p}\_{t}\ | =softmax​(β​𝑿tT​𝝃t).absentsoftmax𝛽superscriptsubscript𝑿𝑡𝑇subscript𝝃𝑡\displaystyle=\ \mathrm{softmax}(\beta\bm{X}\_{t}^{T}\bm{\xi}\_{t})\ . |  | (454) |

#### A.1.9 Number of Spurious States

The energy EE\mathrm{E} is defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EE\displaystyle\mathrm{E}\ | =−lse​(β,𝑿T​𝝃)+12​𝝃T​𝝃+β−1​ln⁡N+12​M2absentlse𝛽superscript𝑿𝑇𝝃12superscript𝝃𝑇𝝃superscript𝛽1𝑁12superscript𝑀2\displaystyle=\ -\ \mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\ +\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}M^{2} |  | (455) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−β−1​ln⁡(∑i=1Nexp⁡(β​𝒙iT​𝝃))+β−1​ln⁡N+12​𝝃T​𝝃+12​M2.absentsuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃superscript𝛽1𝑁12superscript𝝃𝑇𝝃12superscript𝑀2\displaystyle=\ -\ \beta^{-1}\ln\left(\sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)\ +\ \beta^{-1}\ln N\ +\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}M^{2}\ . |  | (456) |

Since the negative exponential function is strict monotonic decreasing,
exp⁡(−E)E\exp(-\mathrm{E}) has minima, where EE\mathrm{E} has maxima, and has maxima,
where as has minima EE\mathrm{E}.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | exp⁡(−E)E\displaystyle\exp(-\mathrm{E})\ | =exp⁡(lse​(β,𝑿T​𝝃))​exp⁡(−12​𝝃T​𝝃)​Cabsentlse𝛽superscript𝑿𝑇𝝃12superscript𝝃𝑇𝝃𝐶\displaystyle=\ \exp(\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi}))\ \exp(-\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi})\ C |  | (457) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(∑i=1Nexp⁡(β​𝒙iT​𝝃))β−1​exp⁡(−12​𝝃T​𝝃)​Cabsentsuperscriptsuperscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃superscript𝛽112superscript𝝃𝑇𝝃𝐶\displaystyle=\left(\sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)^{\beta^{-1}}\ \exp(-\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi})\ C |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(∑i=1Nexp⁡(β​𝒙iT​𝝃))β−1​(exp⁡(−β​12​𝝃T​𝝃))β−1​Cabsentsuperscriptsuperscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃superscript𝛽1superscript𝛽12superscript𝝃𝑇𝝃superscript𝛽1𝐶\displaystyle=\left(\sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)^{\beta^{-1}}\ \left(\exp(-\ \beta\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi})\right)^{\beta^{-1}}\ C |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(∑i=1Nexp⁡(β​(𝒙iT​𝝃−12​𝝃T​𝝃)))β−1​Cabsentsuperscriptsuperscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃12superscript𝝃𝑇𝝃superscript𝛽1𝐶\displaystyle=\left(\sum\_{i=1}^{N}\exp(\beta\ (\bm{x}\_{i}^{T}\bm{\xi}\ -\ \frac{1}{2}\bm{\xi}^{T}\bm{\xi}))\right)^{\beta^{-1}}\ C |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(∑i=1Nexp⁡(12​β​𝒙iT​𝒙i−12​β​(𝝃−𝒙i)T​(𝝃−𝒙i)))β−1​Cabsentsuperscriptsuperscriptsubscript𝑖1𝑁12𝛽superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖12𝛽superscript𝝃subscript𝒙𝑖𝑇𝝃subscript𝒙𝑖superscript𝛽1𝐶\displaystyle=\left(\sum\_{i=1}^{N}\exp(\frac{1}{2}\ \beta\ \bm{x}\_{i}^{T}\bm{x}\_{i}\ -\ \frac{1}{2}\ \beta\ (\bm{\xi}\ -\ \bm{x}\_{i})^{T}(\bm{\xi}\ -\ \bm{x}\_{i}))\right)^{\beta^{-1}}\ C |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(∑i=1Nλ​(𝒙i,β)​G​(𝝃;𝒙i,β−1​𝑰))β−1​C,absentsuperscriptsuperscriptsubscript𝑖1𝑁𝜆subscript𝒙𝑖𝛽𝐺  𝝃subscript𝒙𝑖superscript𝛽1𝑰superscript𝛽1𝐶\displaystyle=\left(\sum\_{i=1}^{N}\lambda(\bm{x}\_{i},\beta)\ G(\bm{\xi};\bm{x}\_{i},\beta^{-1}\ \bm{I})\right)^{\beta^{-1}}\ C\ , |  |

where C𝐶C is a positive constant,
λ​(𝒙i,β)=exp⁡(12​β​𝒙iT​𝒙i)𝜆subscript𝒙𝑖𝛽12𝛽superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖\lambda(\bm{x}\_{i},\beta)=\exp(\frac{1}{2}\beta\bm{x}\_{i}^{T}\bm{x}\_{i}) and
G​(𝝃;𝒙i,β−1​𝑰)𝐺

𝝃subscript𝒙𝑖superscript𝛽1𝑰G(\bm{\xi};\bm{x}\_{i},\beta^{-1}\bm{I}) is the Gaussian with mean 𝒙isubscript𝒙𝑖\bm{x}\_{i} and covariance matrix β−1​𝑰superscript𝛽1𝑰\beta^{-1}\bm{I}.

Since C𝐶C is a positive constant and xβ−1=exp⁡(β−1​ln⁡x)superscript𝑥superscript𝛽1superscript𝛽1𝑥x^{\beta^{-1}}=\exp(\beta^{-1}\ln x) is
strict monotonic for positive x𝑥x, the minima of EE\mathrm{E} are the maxima of

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i=1Nλ​(𝒙i,β)​G​(𝝃;𝒙i,β−1​𝑰).superscriptsubscript𝑖1𝑁𝜆subscript𝒙𝑖𝛽𝐺  𝝃subscript𝒙𝑖superscript𝛽1𝑰\displaystyle\sum\_{i=1}^{N}\lambda(\bm{x}\_{i},\beta)\ G(\bm{\xi};\bm{x}\_{i},\beta^{-1}\ \bm{I})\ . |  | (458) |

In Carreira-Perpiñán &
Williams ([2003](#bib.bib19)) it was shown
that Eq. ([458](#A1.E458 "In A.1.9 Number of Spurious States ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) can have more than N𝑁N modes, that is, more
than N𝑁N maxima.

### A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function

For β>0𝛽0\beta>0, the softmax is defined as

###### Definition A1 (Softmax).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝒙)absentsoftmax𝛽𝒙\displaystyle=\ \mathrm{softmax}(\beta\bm{x}) |  | (459) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | pisubscript𝑝𝑖\displaystyle p\_{i}\ | =[softmax​(β​𝒙)]i=exp⁡(β​xi)∑kexp⁡(β​xk).absentsubscriptdelimited-[]softmax𝛽𝒙𝑖𝛽subscript𝑥𝑖subscript𝑘𝛽subscript𝑥𝑘\displaystyle=\ [\mathrm{softmax}(\beta\bm{x})]\_{i}\ =\ \frac{\exp(\beta x\_{i})}{\sum\_{k}\exp(\beta x\_{k})}\ . |  | (460) |

We also need the log-sum-exp function (lselse\mathrm{lse}), defined as

###### Definition A2 (Log-Sum-Exp Function).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | lse​(β,𝒙)lse𝛽𝒙\displaystyle\mathrm{lse}(\beta,\bm{x})\ | =β−1​ln⁡(∑i=1Nexp⁡(β​xi)).absentsuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽subscript𝑥𝑖\displaystyle=\ \beta^{-1}\ln\left(\sum\_{i=1}^{N}\exp(\beta x\_{i})\right)\ . |  | (461) |

We can formulate the lselse\mathrm{lse}
in another base:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | βasubscript𝛽𝑎\displaystyle\beta\_{a}\ | =βln⁡a,absent𝛽𝑎\displaystyle=\ \frac{\beta}{\ln a}\ , |  | (462) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | lse​(β,𝒙)lse𝛽𝒙\displaystyle\mathrm{lse}(\beta,\bm{x})\ | =β−1​ln⁡(∑i=1Nexp⁡(β​xi))absentsuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽subscript𝑥𝑖\displaystyle=\ \beta^{-1}\ln\left(\sum\_{i=1}^{N}\exp(\beta\ x\_{i})\right) |  | (463) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(βa​ln⁡a)−1​ln⁡(∑i=1Nexp⁡(βa​ln⁡a​xi))absentsuperscriptsubscript𝛽𝑎𝑎1superscriptsubscript𝑖1𝑁subscript𝛽𝑎𝑎subscript𝑥𝑖\displaystyle=\ \left(\beta\_{a}\ \ln a\right)^{-1}\ln\left(\sum\_{i=1}^{N}\exp(\beta\_{a}\ \ln a\ x\_{i})\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(βa)−1​loga⁡(∑i=1Naβa​xi).absentsuperscriptsubscript𝛽𝑎1subscript𝑎superscriptsubscript𝑖1𝑁superscript𝑎subscript𝛽𝑎subscript𝑥𝑖\displaystyle=\ \left(\beta\_{a}\right)^{-1}\log\_{a}\left(\sum\_{i=1}^{N}a^{\beta\_{a}\ x\_{i}}\right)\ . |  |

In particular, the base a=2𝑎2a=2 can be used to speed up computations.

Next, we give the relation between the softmax and the lselse\mathrm{lse} function.

###### Lemma A18.

The softmax is the gradient of the lselse\mathrm{lse}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | softmax​(β​𝒙)softmax𝛽𝒙\displaystyle\mathrm{softmax}(\beta\bm{x})\ | =∇𝒙lse​(β,𝒙).absentsubscript∇𝒙lse𝛽𝒙\displaystyle=\ \nabla\_{\bm{x}}\mathrm{lse}(\beta,\bm{x})\ . |  | (464) |

In the next lemma we report some important properties of the lselse\mathrm{lse} function.

###### Lemma A19.

We define

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | LL\displaystyle\mathrm{L}\ | :=𝒛T​𝒙−β−1​∑i=1Nzi​ln⁡ziassignabsentsuperscript𝒛𝑇𝒙superscript𝛽1superscriptsubscript𝑖1𝑁subscript𝑧𝑖subscript𝑧𝑖\displaystyle:=\ \bm{z}^{T}\bm{x}\ -\ \beta^{-1}\sum\_{i=1}^{N}z\_{i}\ln z\_{i} |  | (465) |

with L≥𝐳T​𝐱Lsuperscript𝐳𝑇𝐱\mathrm{L}\geq\bm{z}^{T}\bm{x}.
The lselse\mathrm{lse} is the maximum of LL\mathrm{L} on
the N𝑁N-dimensional
simplex D𝐷D with D={𝐳∣∑izi=1,0⩽zi}𝐷conditional-set𝐳formulae-sequencesubscript𝑖subscript𝑧𝑖10subscript𝑧𝑖D=\{\bm{z}\mid\sum\_{i}z\_{i}=1,0\leqslant z\_{i}\}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | lse​(β,𝒙)lse𝛽𝒙\displaystyle\mathrm{lse}(\beta,\bm{x})\ | =max𝒛∈D⁡𝒛T​𝒙−β−1​∑i=1Nzi​ln⁡zi.absentsubscript𝒛𝐷superscript𝒛𝑇𝒙superscript𝛽1superscriptsubscript𝑖1𝑁subscript𝑧𝑖subscript𝑧𝑖\displaystyle=\ \max\_{\bm{z}\in D}\bm{z}^{T}\bm{x}\ -\ \beta^{-1}\sum\_{i=1}^{N}z\_{i}\ln z\_{i}\ . |  | (466) |

The softmax 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x}) is the argument
of the maximum of LL\mathrm{L} on
the N𝑁N-dimensional
simplex D𝐷D with D={𝐳∣∑izi=1,0⩽zi}𝐷conditional-set𝐳formulae-sequencesubscript𝑖subscript𝑧𝑖10subscript𝑧𝑖D=\{\bm{z}\mid\sum\_{i}z\_{i}=1,0\leqslant z\_{i}\}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝒙)=arg⁡max𝒛∈D⁡𝒛T​𝒙−β−1​∑i=1Nzi​ln⁡zi.absentsoftmax𝛽𝒙subscript𝒛𝐷superscript𝒛𝑇𝒙superscript𝛽1superscriptsubscript𝑖1𝑁subscript𝑧𝑖subscript𝑧𝑖\displaystyle=\ \mathrm{softmax}(\beta\bm{x})\ =\ \arg\max\_{\bm{z}\in D}\bm{z}^{T}\bm{x}\ -\ \beta^{-1}\sum\_{i=1}^{N}z\_{i}\ln z\_{i}\ . |  | (467) |

###### Proof.

Eq. ([466](#A1.E466 "In Lemma A19. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) is obtained from Equation (8) in Gao & Pavel ([2017](#bib.bib38)) and
Eq. ([467](#A1.E467 "In Lemma A19. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) from Equation (11) in Gao & Pavel ([2017](#bib.bib38)).
∎

From a physical point of view, the lselse\mathrm{lse} function represents the “free energy”
in statistical thermodynamics (Gao & Pavel, [2017](#bib.bib38)).

Next we consider the Jacobian of the softmax and its properties.

###### Lemma A20.

The Jacobian JssubscriptJ𝑠\mathrm{J}\_{s} of the softmax 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x}) is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | JssubscriptJ𝑠\displaystyle\mathrm{J}\_{s}\ | =∂softmax​(β​𝒙)∂𝒙=β​(diag​(𝒑)−𝒑​𝒑T),absentsoftmax𝛽𝒙𝒙𝛽diag𝒑𝒑superscript𝒑𝑇\displaystyle=\ \frac{\partial\mathrm{softmax}(\beta\bm{x})}{\partial\bm{x}}\ =\ \beta\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\ , |  | (468) |

which gives the elements

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | [Js]i​jsubscriptdelimited-[]subscriptJ𝑠𝑖𝑗\displaystyle[\mathrm{J}\_{s}]\_{ij}\ | ={β​pi​(1−pi)for​i=j−β​pi​pjfor​i≠j.absentcases𝛽subscript𝑝𝑖1subscript𝑝𝑖for𝑖𝑗𝛽subscript𝑝𝑖subscript𝑝𝑗for𝑖𝑗\displaystyle=\ \begin{cases}\beta p\_{i}(1-p\_{i})&\text{for}\ i=j\\ -\beta p\_{i}p\_{j}&\text{for}\ i\not=j\end{cases}\ . |  | (469) |

Next we show that JssubscriptJ𝑠\mathrm{J}\_{s} has eigenvalue 00.

###### Lemma A21.

The Jacobian JssubscriptJ𝑠\mathrm{J}\_{s} of the softmax function 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x})
has a zero eigenvalue with eigenvector 𝟏1\bm{1}.

###### Proof.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | [Js​𝟏]isubscriptdelimited-[]subscriptJ𝑠1𝑖\displaystyle[\mathrm{J}\_{s}\bm{1}]\_{i}\ | =β​(pi​(1−pi)−∑j,j≠ipi​pj)=β​pi​(1−∑jpj)=0.absent𝛽subscript𝑝𝑖1subscript𝑝𝑖subscript  𝑗𝑗 𝑖subscript𝑝𝑖subscript𝑝𝑗𝛽subscript𝑝𝑖1subscript𝑗subscript𝑝𝑗0\displaystyle=\ \beta\left(p\_{i}(1-p\_{i})\ -\ \sum\_{j,j\not=i}p\_{i}p\_{j}\right)\ =\ \beta\ p\_{i}(1\ -\ \sum\_{j}p\_{j})\ =0\ . |  | (470) |

∎

Next we show that 00 is the smallest eigenvalue of JssubscriptJ𝑠\mathrm{J}\_{s}, therefore
JssubscriptJ𝑠\mathrm{J}\_{s} is positive semi-definite but not (strict) positive definite.

###### Lemma A22.

The Jacobian JssubscriptJ𝑠\mathrm{J}\_{s} of the softmax 𝐩=softmax​(β​𝛏)𝐩softmax𝛽𝛏\bm{p}=\mathrm{softmax}(\beta\bm{\xi})
is symmetric and positive semi-definite.

###### Proof.

For an arbitrary 𝒛𝒛\bm{z}, we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒛T​(diag​(𝒑)−𝒑​𝒑T)​𝒛superscript𝒛𝑇diag𝒑𝒑superscript𝒑𝑇𝒛\displaystyle\bm{z}^{T}\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\bm{z}\ | =∑ipi​zi2−(∑ipi​zi)2absentsubscript𝑖subscript𝑝𝑖superscriptsubscript𝑧𝑖2superscriptsubscript𝑖subscript𝑝𝑖subscript𝑧𝑖2\displaystyle=\ \sum\_{i}p\_{i}z\_{i}^{2}-\left(\sum\_{i}p\_{i}z\_{i}\right)^{2} |  | (471) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(∑ipi​zi2)​(∑ipi)−(∑ipi​zi)2≥ 0.absentsubscript𝑖subscript𝑝𝑖superscriptsubscript𝑧𝑖2subscript𝑖subscript𝑝𝑖superscriptsubscript𝑖subscript𝑝𝑖subscript𝑧𝑖2 0\displaystyle=\ \left(\sum\_{i}p\_{i}z\_{i}^{2}\right)\ \left(\sum\_{i}p\_{i}\right)-\left(\sum\_{i}p\_{i}z\_{i}\right)^{2}\ \geq\ 0\ . |  |

The last inequality hold true because the Cauchy-Schwarz inequality
says (𝒂T​𝒂)​(𝒃T​𝒃)≥(𝒂T​𝒃)2superscript𝒂𝑇𝒂superscript𝒃𝑇𝒃superscriptsuperscript𝒂𝑇𝒃2(\bm{a}^{T}\bm{a})(\bm{b}^{T}\bm{b})\geq(\bm{a}^{T}\bm{b})^{2}, which is the last
inequality with ai=zi​pisubscript𝑎𝑖subscript𝑧𝑖subscript𝑝𝑖a\_{i}=z\_{i}\sqrt{p\_{i}} and bi=pisubscript𝑏𝑖subscript𝑝𝑖b\_{i}=\sqrt{p\_{i}}.
Consequently (diag​(𝒑)−𝒑​𝒑T)diag𝒑𝒑superscript𝒑𝑇\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right) is positive semi-definite.

Alternatively ∑ipi​zi2−(∑ipi​zi)2subscript𝑖subscript𝑝𝑖superscriptsubscript𝑧𝑖2superscriptsubscript𝑖subscript𝑝𝑖subscript𝑧𝑖2\sum\_{i}p\_{i}z\_{i}^{2}-\left(\sum\_{i}p\_{i}z\_{i}\right)^{2} can
be viewed as the expected second moment minus the mean squared which
gives the variance that is larger equal to zero.

The Jacobian is 0<β0𝛽0<\beta times a positive semi-definite matrix, which is
a positive semi-definite matrix.
∎

Moreover, the softmax is a monotonic map, as described in the next lemma.

###### Lemma A23.

The softmax softmax​(β​𝐱)softmax𝛽𝐱\mathrm{softmax}(\beta\bm{x}) is monotone for β>0𝛽0\beta>0, that is,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (softmax​(β​𝒙)−softmax​(β​𝒙′))T​(𝒙−𝒙′)superscriptsoftmax𝛽𝒙softmax𝛽superscript𝒙′𝑇𝒙superscript𝒙′\displaystyle\left(\mathrm{softmax}(\beta\bm{x})\ -\ \mathrm{softmax}(\beta\bm{x}^{\prime})\right)^{T}\left(\bm{x}\ -\ \bm{x}^{\prime}\right)\ | ≥ 0.absent 0\displaystyle\geq\ 0\ . |  | (472) |

###### Proof.

We use the version of mean value theorem Lemma [A32](#ThmlemmaA32 "Lemma A32 (Mean Value Theorem). ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") with the symmetric matrix
Jsm=∫01Js​(λ​𝒙+(1−λ)​𝒙′)​dλsuperscriptsubscriptJ𝑠𝑚superscriptsubscript01subscriptJ𝑠𝜆𝒙1𝜆superscript𝒙′differential-d𝜆\mathrm{J}\_{s}^{m}=\int\_{0}^{1}\mathrm{J}\_{s}(\lambda\bm{x}\ +\ (1-\lambda)\bm{x}^{\prime})\ \mathrm{d}\lambda:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | softmax​(𝒙)−softmax​(𝒙′)softmax𝒙softmaxsuperscript𝒙′\displaystyle\mathrm{softmax}(\bm{x})\ -\ \mathrm{softmax}(\bm{x}^{\prime})\ | =Jsm​(𝒙−𝒙′).absentsuperscriptsubscriptJ𝑠𝑚𝒙superscript𝒙′\displaystyle=\ \mathrm{J}\_{s}^{m}\ \left(\bm{x}\ -\ \bm{x}^{\prime}\right)\ . |  | (473) |

Therefore

|  |  |  |  |
| --- | --- | --- | --- |
|  | (softmax​(𝒙)−softmax​(𝒙′))T​(𝒙−𝒙′)=(𝒙−𝒙′)T​Jsm​(𝒙−𝒙′)≥ 0,superscriptsoftmax𝒙softmaxsuperscript𝒙′𝑇𝒙superscript𝒙′superscript𝒙superscript𝒙′𝑇superscriptsubscriptJ𝑠𝑚𝒙superscript𝒙′ 0\displaystyle\left(\mathrm{softmax}(\bm{x})\ -\ \mathrm{softmax}(\bm{x}^{\prime})\right)^{T}\left(\bm{x}\ -\ \bm{x}^{\prime}\right)\ =\ \left(\bm{x}\ -\ \bm{x}^{\prime}\right)^{T}\mathrm{J}\_{s}^{m}\ \left(\bm{x}\ -\ \bm{x}^{\prime}\right)\ \geq\ 0\ , |  | (474) |

since JsmsuperscriptsubscriptJ𝑠𝑚\mathrm{J}\_{s}^{m} is positive semi-definite.
For all λ𝜆\lambda the Jacobians
Js​(λ​𝒙+(1−λ)​𝒙′)subscriptJ𝑠𝜆𝒙1𝜆superscript𝒙′\mathrm{J}\_{s}(\lambda\bm{x}\ +\ (1-\lambda)\bm{x}^{\prime})
are positive semi-definite
according to Lemma [A22](#ThmlemmaA22 "Lemma A22. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
Since

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒙T​Jsm​𝒙superscript𝒙𝑇superscriptsubscriptJ𝑠𝑚𝒙\displaystyle\bm{x}^{T}\mathrm{J}\_{s}^{m}\bm{x}\ | =∫01𝒙T​Js​(λ​𝒙+(1−λ)​𝒙′)​𝒙​dλ≥ 0absentsuperscriptsubscript01superscript𝒙𝑇subscriptJ𝑠𝜆𝒙1𝜆superscript𝒙′𝒙differential-d𝜆 0\displaystyle=\ \int\_{0}^{1}\bm{x}^{T}\mathrm{J}\_{s}(\lambda\bm{x}\ +\ (1-\lambda)\bm{x}^{\prime})\ \bm{x}\ \mathrm{d}\lambda\ \geq\ 0 |  | (475) |

is an integral over positive values for every 𝒙𝒙\bm{x},
JsmsuperscriptsubscriptJ𝑠𝑚\mathrm{J}\_{s}^{m} is positive semi-definite, too.
∎

Next we give upper bounds on the norm of JssubscriptJ𝑠\mathrm{J}\_{s}.

###### Lemma A24.

For a softmax 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x}) with
m=maxi⁡pi​(1−pi)𝑚subscript𝑖subscript𝑝𝑖1subscript𝑝𝑖m=\max\_{i}p\_{i}(1-p\_{i}), the spectral norm of
the Jacobian JssubscriptJ𝑠\mathrm{J}\_{s} of the softmax is bounded:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js‖2subscriptnormsubscriptJ𝑠2\displaystyle{{\left\|\mathrm{J}\_{s}\right\|}}\_{2}\ | ⩽ 2​m​β,absent2𝑚𝛽\displaystyle\leqslant\ 2\ m\ \beta\ , |  | (476) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js‖1subscriptnormsubscriptJ𝑠1\displaystyle{{\left\|\mathrm{J}\_{s}\right\|}}\_{1}\ | ⩽ 2​m​β,absent2𝑚𝛽\displaystyle\leqslant\ 2\ m\ \beta\ , |  | (477) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js‖∞subscriptnormsubscriptJ𝑠\displaystyle{{\left\|\mathrm{J}\_{s}\right\|}}\_{\infty}\ | ⩽ 2​m​β.absent2𝑚𝛽\displaystyle\leqslant\ 2\ m\ \beta\ . |  | (478) |

In particular everywhere holds

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js‖2subscriptnormsubscriptJ𝑠2\displaystyle{{\left\|\mathrm{J}\_{s}\right\|}}\_{2}\ | ⩽12​β.absent12𝛽\displaystyle\leqslant\ \frac{1}{2}\ \beta\ . |  | (479) |

If pmax=maxi⁡pi≥1−ϵ≥0.5subscript𝑝subscript𝑖subscript𝑝𝑖1italic-ϵ0.5p\_{\max}=\max\_{i}p\_{i}\geq 1-\epsilon\geq 0.5, then for the spectral norm of
the Jacobian holds

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js‖2subscriptnormsubscriptJ𝑠2\displaystyle{{\left\|\mathrm{J}\_{s}\right\|}}\_{2}\ | ⩽ 2ϵβ− 2ϵ2β< 2ϵβ.\displaystyle\leqslant\ 2\ \epsilon\ \beta\ \ -\ 2\ \epsilon^{2}\ \beta\ \ <\ 2\ \epsilon\ \beta\ . |  | (480) |

###### Proof.

We consider the maximum absolute column sum norm

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝑨‖1subscriptnorm𝑨1\displaystyle{{\left\|\bm{A}\right\|}}\_{1}\ | =maxj​∑i|ai​j|absentsubscript𝑗subscript𝑖subscript𝑎𝑖𝑗\displaystyle=\ \max\_{j}\sum\_{i}{{\left|a\_{ij}\right|}} |  | (481) |

and the maximum absolute row sum norm

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝑨‖∞subscriptnorm𝑨\displaystyle{{\left\|\bm{A}\right\|}}\_{\infty}\ | =maxi​∑j|ai​j|.absentsubscript𝑖subscript𝑗subscript𝑎𝑖𝑗\displaystyle=\ \max\_{i}\sum\_{j}{{\left|a\_{ij}\right|}}\ . |  | (482) |

We have for 𝑨=Js=β​(diag​(𝒑)−𝒑​𝒑T)𝑨subscriptJ𝑠𝛽diag𝒑𝒑superscript𝒑𝑇\bm{A}=\mathrm{J}\_{s}=\beta\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑j|ai​j|subscript𝑗subscript𝑎𝑖𝑗\displaystyle\sum\_{j}{{\left|a\_{ij}\right|}}\ | =β​(pi​(1−pi)+∑j,j≠ipi​pj)=β​pi​(1− 2​pi+∑jpj)absent𝛽subscript𝑝𝑖1subscript𝑝𝑖subscript  𝑗𝑗 𝑖subscript𝑝𝑖subscript𝑝𝑗𝛽subscript𝑝𝑖12subscript𝑝𝑖subscript𝑗subscript𝑝𝑗\displaystyle=\ \beta\ \left(p\_{i}(1-p\_{i})\ +\ \sum\_{j,j\not=i}p\_{i}p\_{j}\right)\ =\ \beta\ p\_{i}\ (1\ -\ 2p\_{i}\ +\ \sum\_{j}p\_{j}) |  | (483) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | = 2​β​pi​(1−pi)⩽ 2​m​β,absent2𝛽subscript𝑝𝑖1subscript𝑝𝑖2𝑚𝛽\displaystyle=\ 2\ \beta\ p\_{i}\ (1-p\_{i})\ \leqslant\ 2\ m\ \beta\ , |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i|ai​j|subscript𝑖subscript𝑎𝑖𝑗\displaystyle\sum\_{i}{{\left|a\_{ij}\right|}}\ | =β​(pj​(1−pj)+∑i,i≠jpj​pi)=β​pj​(1− 2​pj+∑ipi)absent𝛽subscript𝑝𝑗1subscript𝑝𝑗subscript  𝑖𝑖 𝑗subscript𝑝𝑗subscript𝑝𝑖𝛽subscript𝑝𝑗12subscript𝑝𝑗subscript𝑖subscript𝑝𝑖\displaystyle=\ \beta\ \left(p\_{j}\ (1-p\_{j})\ +\ \sum\_{i,i\not=j}p\_{j}p\_{i}\right)\ =\ \beta\ p\_{j}\ (1\ -\ 2p\_{j}\ +\ \sum\_{i}p\_{i}) |  | (484) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | = 2​β​pj​(1−pj)⩽ 2​m​β.absent2𝛽subscript𝑝𝑗1subscript𝑝𝑗2𝑚𝛽\displaystyle=\ 2\ \beta\ p\_{j}\ (1-p\_{j})\ \leqslant\ 2\ m\ \beta\ . |  |

Therefore, we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js‖1subscriptnormsubscriptJ𝑠1\displaystyle{{\left\|\mathrm{J}\_{s}\right\|}}\_{1}\ | ⩽ 2​m​β,absent2𝑚𝛽\displaystyle\leqslant\ 2\ m\ \beta\ , |  | (485) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js‖∞subscriptnormsubscriptJ𝑠\displaystyle{{\left\|\mathrm{J}\_{s}\right\|}}\_{\infty}\ | ⩽ 2​m​β,absent2𝑚𝛽\displaystyle\leqslant\ 2\ m\ \beta\ , |  | (486) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖Js‖2subscriptnormsubscriptJ𝑠2\displaystyle{{\left\|\mathrm{J}\_{s}\right\|}}\_{2}\ | ⩽‖Js‖1​‖Js‖∞⩽ 2​m​β.absentsubscriptnormsubscriptJ𝑠1subscriptnormsubscriptJ𝑠2𝑚𝛽\displaystyle\leqslant\ \sqrt{{{\left\|\mathrm{J}\_{s}\right\|}}\_{1}{{\left\|\mathrm{J}\_{s}\right\|}}\_{\infty}}\ \leqslant\ 2\ m\ \beta\ . |  | (487) |

The last inequality is a direct consequence of Hölder’s inequality.

For 0⩽pi⩽10subscript𝑝𝑖10\leqslant p\_{i}\leqslant 1, we have pi​(1−pi)⩽0.25subscript𝑝𝑖1subscript𝑝𝑖0.25p\_{i}(1-p\_{i})\leqslant 0.25.
Therefore, m⩽0.25𝑚0.25m\leqslant 0.25 for all values of pisubscript𝑝𝑖p\_{i}.

If pmax≥1−ϵ≥0.5subscript𝑝1italic-ϵ0.5p\_{\max}\geq 1-\epsilon\geq 0.5 (ϵ⩽0.5italic-ϵ0.5\epsilon\leqslant 0.5), then
1−pmax⩽ϵ1subscript𝑝italic-ϵ1-p\_{\max}\leqslant\epsilon and for pi≠pmaxsubscript𝑝𝑖subscript𝑝p\_{i}\not=p\_{\max}
pi⩽ϵsubscript𝑝𝑖italic-ϵp\_{i}\leqslant\epsilon.
The derivative ∂x​(1−x)/∂x=1−2​x>0𝑥1𝑥𝑥12𝑥0\partial x(1-x)/\partial x=1-2x>0 for x<0.5𝑥0.5x<0.5, therefore
x​(1−x)𝑥1𝑥x(1-x) increases with x𝑥x for x<0.5𝑥0.5x<0.5.
Using x=1−pmax𝑥1subscript𝑝x=1-p\_{\max} and for pi≠pmaxsubscript𝑝𝑖subscript𝑝p\_{i}\not=p\_{\max}
x=pi𝑥subscript𝑝𝑖x=p\_{i}, we obtain
pi​(1−pi)⩽ϵ​(1−ϵ)subscript𝑝𝑖1subscript𝑝𝑖italic-ϵ1italic-ϵp\_{i}(1-p\_{i})\leqslant\epsilon(1-\epsilon) for all i𝑖i.
Consequently, we have m⩽ϵ​(1−ϵ)𝑚italic-ϵ1italic-ϵm\leqslant\epsilon(1-\epsilon).
∎

Using the bounds on the norm of the Jacobian,
we give some Lipschitz properties of the softmax function.

###### Lemma A25.

The softmax function 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x})
is (β/2)𝛽2(\beta/2)-Lipschitz.
The softmax function 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x})
is (2​β​m)2𝛽𝑚(2\beta m)-Lipschitz
in a convex environment U𝑈U for which
m=max𝐱∈U⁡maxi⁡pi​(1−pi)𝑚subscript𝐱𝑈subscript𝑖subscript𝑝𝑖1subscript𝑝𝑖m=\max\_{\bm{x}\in U}\max\_{i}p\_{i}(1-p\_{i}).
For pmax=min𝐱∈U⁡maxi⁡pi=1−ϵsubscript𝑝subscript𝐱𝑈subscript𝑖subscript𝑝𝑖1italic-ϵp\_{\max}=\min\_{\bm{x}\in U}\max\_{i}p\_{i}=1-\epsilon,
the softmax function 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x})
is (2​β​ϵ)2𝛽italic-ϵ(2\beta\epsilon)-Lipschitz.
For β<2​m𝛽2𝑚\beta<2m, the softmax 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x})
is contractive in U𝑈U on which m𝑚m is defined.

###### Proof.

The version of mean value theorem Lemma [A32](#ThmlemmaA32 "Lemma A32 (Mean Value Theorem). ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") states for the symmetric matrix
Jsm=∫01J​(λ​𝒙+(1−λ)​𝒙′)​dλsuperscriptsubscriptJ𝑠𝑚superscriptsubscript01J𝜆𝒙1𝜆superscript𝒙′differential-d𝜆\mathrm{J}\_{s}^{m}=\int\_{0}^{1}\mathrm{J}(\lambda\bm{x}+(1-\lambda)\bm{x}^{\prime})\ \mathrm{d}\lambda:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | softmax​(𝒙)−softmax​(𝒙′)softmax𝒙softmaxsuperscript𝒙′\displaystyle\mathrm{softmax}(\bm{x})\ -\ \mathrm{softmax}(\bm{x}^{\prime})\ | =Jsm​(𝒙−𝒙′).absentsuperscriptsubscriptJ𝑠𝑚𝒙superscript𝒙′\displaystyle=\ \mathrm{J}\_{s}^{m}\ \left(\bm{x}\ -\ \bm{x}^{\prime}\right)\ . |  | (488) |

According to Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")
for all 𝒙~=λ𝒙+(1−λ)𝒙′)\tilde{\bm{x}}=\lambda\bm{x}+(1-\lambda)\bm{x}^{\prime})

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Js​(𝒙~)‖2⩽ 2​m~​β,subscriptnormsubscriptJ𝑠~𝒙22~𝑚𝛽\displaystyle{{\left\|\mathrm{J}\_{s}(\tilde{\bm{x}})\right\|}}\_{2}\ \leqslant\ 2\ \tilde{m}\ \beta\ , |  | (489) |

where m~=maxi⁡p~i​(1−p~i)~𝑚subscript𝑖subscript~𝑝𝑖1subscript~𝑝𝑖\tilde{m}=\max\_{i}\tilde{p}\_{i}(1-\tilde{p}\_{i}).
Since 𝒙∈U𝒙𝑈\bm{x}\in U and 𝒙′∈Usuperscript𝒙′𝑈\bm{x}^{\prime}\in U we have 𝒙~∈U~𝒙𝑈\tilde{\bm{x}}\in U,
since U𝑈U is convex.
For m=max𝒙∈U⁡maxi⁡pi​(1−pi)𝑚subscript𝒙𝑈subscript𝑖subscript𝑝𝑖1subscript𝑝𝑖m=\max\_{\bm{x}\in U}\max\_{i}p\_{i}(1-p\_{i}) we have
m~⩽m~𝑚𝑚\tilde{m}\leqslant m for all m~~𝑚\tilde{m}. Therefore, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Js​(𝒙~)‖2⩽ 2​m​βsubscriptnormsubscriptJ𝑠~𝒙22𝑚𝛽\displaystyle{{\left\|\mathrm{J}\_{s}(\tilde{\bm{x}})\right\|}}\_{2}\ \leqslant\ 2\ m\ \beta |  | (490) |

which also holds for the mean:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Jsm‖2⩽ 2​m​β.subscriptnormsuperscriptsubscriptJ𝑠𝑚22𝑚𝛽\displaystyle{{\left\|\mathrm{J}\_{s}^{m}\right\|}}\_{2}\ \leqslant\ 2\ m\ \beta\ . |  | (491) |

Therefore,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖softmax​(𝒙)−softmax​(𝒙′)‖normsoftmax𝒙softmaxsuperscript𝒙′\displaystyle{{\left\|\mathrm{softmax}(\bm{x})\ -\ \mathrm{softmax}(\bm{x}^{\prime})\right\|}}\ | ⩽‖Jsm‖2​‖𝒙−𝒙′‖⩽ 2​m​β​‖𝒙−𝒙′‖.absentsubscriptnormsuperscriptsubscriptJ𝑠𝑚2norm𝒙superscript𝒙′2𝑚𝛽norm𝒙superscript𝒙′\displaystyle\leqslant\ {{\left\|\mathrm{J}\_{s}^{m}\right\|}}\_{2}\ {{\left\|\bm{x}\ -\ \bm{x}^{\prime}\right\|}}\ \leqslant\ 2\ m\ \beta\ {{\left\|\bm{x}\ -\ \bm{x}^{\prime}\right\|}}\ . |  | (492) |

From Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") we know m⩽1/4𝑚14m\leqslant 1/4 globally.
For pmax=min𝒙∈U⁡maxi⁡pi=1−ϵsubscript𝑝subscript𝒙𝑈subscript𝑖subscript𝑝𝑖1italic-ϵp\_{\max}=\min\_{\bm{x}\in U}\max\_{i}p\_{i}=1-\epsilon we have
according to Lemma [A24](#ThmlemmaA24 "Lemma A24. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"): m⩽ϵ𝑚italic-ϵm\leqslant\epsilon.
∎

For completeness we present a result about cocoercivity of the softmax:

###### Lemma A26.

For m=max𝐱∈U⁡maxi⁡pi​(1−pi)𝑚subscript𝐱𝑈subscript𝑖subscript𝑝𝑖1subscript𝑝𝑖m=\max\_{\bm{x}\in U}\max\_{i}p\_{i}(1-p\_{i}),
softmax function 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x})
is 1/(2​m​β)12𝑚𝛽1/(2m\beta)-cocoercive in U𝑈U, that is,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (softmax​(𝒙)−softmax​(𝒙′))T​(𝒙−𝒙′)superscriptsoftmax𝒙softmaxsuperscript𝒙′𝑇𝒙superscript𝒙′\displaystyle\left(\mathrm{softmax}(\bm{x})\ -\ \mathrm{softmax}(\bm{x}^{\prime})\right)^{T}\left(\bm{x}\ -\ \bm{x}^{\prime}\right)\ | ≥12​m​β​‖softmax​(𝒙)−softmax​(𝒙′)‖.absent12𝑚𝛽normsoftmax𝒙softmaxsuperscript𝒙′\displaystyle\geq\ \frac{1}{2\ m\ \beta}{{\left\|\mathrm{softmax}(\bm{x})\ -\ \mathrm{softmax}(\bm{x}^{\prime})\right\|}}. |  | (493) |

In particular
the softmax function 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x})
is (2/β)2𝛽(2/\beta)-cocoercive everywhere.
With pmax=min𝐱∈U⁡maxi⁡pi=1−ϵsubscript𝑝subscript𝐱𝑈subscript𝑖subscript𝑝𝑖1italic-ϵp\_{\max}=\min\_{\bm{x}\in U}\max\_{i}p\_{i}=1-\epsilon,
the softmax function 𝐩=softmax​(β​𝐱)𝐩softmax𝛽𝐱\bm{p}=\mathrm{softmax}(\beta\bm{x})
is 1/(2​β​ϵ)12𝛽italic-ϵ1/(2\beta\epsilon)-cocoercive in U𝑈U.

###### Proof.

We apply the Baillon-Haddad theorem (e.g. Theorem 1 in Gao & Pavel ([2017](#bib.bib38)))
together with Lemma [A25](#ThmlemmaA25 "Lemma A25. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
∎

Finally, we introduce the Legendre transform and use it
to describe further properties of the lselse\mathrm{lse}.
We start with the definition of the convex conjugate.

###### Definition A3 (Convex Conjugate).

The Convex Conjugate (Legendre-Fenchel transform)
of a function f𝑓f from a Hilbert Space X𝑋X to [−∞,∞][-\infty,\infty] is f∗superscript𝑓f^{\*} which is defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f∗​(𝒙∗)superscript𝑓superscript𝒙\displaystyle f^{\*}(\bm{x}^{\*})\ | =sup𝒙∈X(𝒙T​𝒙∗−f​(𝒙)),𝒙∗∈Xformulae-sequenceabsentsubscriptsupremum𝒙𝑋superscript𝒙𝑇superscript𝒙𝑓𝒙superscript𝒙𝑋\displaystyle=\ \sup\_{\bm{x}\in X}(\bm{x}^{T}\bm{x}^{\*}\ -\ f(\bm{x}))\ ,\quad\bm{x}^{\*}\in X |  | (494) |

See page 219 Def. 13.1 in Bauschke & Combettes ([2017](#bib.bib11)) and page 134 in Garling ([2017](#bib.bib39)).
Next we define the Legendre transform, which is a more restrictive version of the convex
conjugate.

###### Definition A4 (Legendre Transform).

The Legendre transform of a convex function f𝑓f from a convex set
X⊂ℝn𝑋superscriptℝ𝑛X\subset\mathbb{R}^{n} to ℝℝ\mathbb{R} (f:X→ℝ:𝑓→𝑋ℝf:X\rightarrow\mathbb{R}) is f∗superscript𝑓f^{\*}, which is
defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f∗​(𝒙∗)superscript𝑓superscript𝒙\displaystyle f^{\*}(\bm{x}^{\*})\ | =sup𝒙∈X(𝒙T​𝒙∗−f​(𝒙)),𝒙∗∈X∗,formulae-sequenceabsentsubscriptsupremum𝒙𝑋superscript𝒙𝑇superscript𝒙𝑓𝒙superscript𝒙superscript𝑋\displaystyle=\ \sup\_{\bm{x}\in X}(\bm{x}^{T}\bm{x}^{\*}\ -\ f(\bm{x}))\ ,\quad\bm{x}^{\*}\in X^{\*}\ , |  | (495) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | X∗superscript𝑋\displaystyle X^{\*}\ | ={𝒙∗∈ℝn∣sup𝒙∈X(𝒙T​𝒙∗−f​(𝒙))<∞}.absentconditional-setsuperscript𝒙superscriptℝ𝑛subscriptsupremum𝒙𝑋superscript𝒙𝑇superscript𝒙𝑓𝒙\displaystyle=\ \left\{\bm{x}^{\*}\in\mathbb{R}^{n}\mid\sup\_{\bm{x}\in X}(\bm{x}^{T}\bm{x}^{\*}\ -\ f(\bm{x}))<\infty\right\}\ . |  | (496) |

See page 91 in Boyd & Vandenberghe ([2009](#bib.bib12)).

###### Definition A5 (Epi-Sum).

Let f𝑓f and g𝑔g be two functions from X𝑋X to (−∞,∞](-\infty,\infty], then the infimal convolution (or epi-sum) of f𝑓f and g𝑔g is

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​□​g:X→[−∞,∞],𝒙↦inf𝒚∈X(f​(𝒚)+g​(𝒙−𝒚)):𝑓□𝑔formulae-sequence→𝑋maps-to𝒙subscriptinfimum𝒚𝑋𝑓𝒚𝑔𝒙𝒚\displaystyle f\Box g:X\rightarrow[-\infty,\infty]\ ,\ \bm{x}\mapsto\inf\_{\bm{y}\in X}\left(f(\bm{y})+g(\bm{x}-\bm{y})\right) |  | (497) |

See Def. 12.1 in Bauschke & Combettes ([2017](#bib.bib11)).

###### Lemma A27.

Let f𝑓f and g𝑔g be functions from X𝑋X to (−∞,∞](-\infty,\infty]. Then
the following hold:

1. 1.

   Convex Conjugate of norm squared

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | (12∥.∥2)∗\displaystyle\left(\frac{1}{2}{{\left\|.\right\|}}^{2}\right)^{\*}\ | =12∥.∥2.\displaystyle=\ \frac{1}{2}{{\left\|.\right\|}}^{2}\ . |  | (498) |
2. 2.

   Convex Conjugate of a function multiplied by scalar 0<α∈ℝ0𝛼ℝ0<\alpha\in\mathbb{R}

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | (α​f)∗superscript𝛼𝑓\displaystyle\left(\alpha\ f\right)^{\*}\ | =αf∗(./α).\displaystyle=\ \alpha\ f^{\*}(./\alpha)\ . |  | (499) |
3. 3.

   Convex Conjugate of the sum of a function and a scalar β∈ℝ𝛽ℝ\beta\in\mathbb{R}

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | (f+β)∗superscript𝑓𝛽\displaystyle\left(f\ +\ \beta\right)^{\*}\ | =f∗−β.absentsuperscript𝑓𝛽\displaystyle=\ f^{\*}\ -\ \beta\ . |  | (500) |
4. 4.

   Convex Conjugate of affine transformation of the arguments. Let 𝑨𝑨\bm{A} be
   a non-singular matrix and 𝒃𝒃\bm{b} a vector

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | (f​(𝑨​𝒙+𝒃))∗superscript𝑓𝑨𝒙𝒃\displaystyle\left(f\left(\bm{A}\bm{x}\ +\ \bm{b}\right)\right)^{\*}\ | =f∗​(𝑨−T​𝒙∗)−𝒃T​𝑨−T​𝒙∗.absentsuperscript𝑓superscript𝑨𝑇superscript𝒙superscript𝒃𝑇superscript𝑨𝑇superscript𝒙\displaystyle=\ f^{\*}\left(\bm{A}^{-T}\bm{x}^{\*}\right)\ -\ \bm{b}^{T}\bm{A}^{-T}\bm{x}^{\*}\ . |  | (501) |
5. 5.

   Convex Conjugate of epi-sums

   |  |  |  |  |  |
   | --- | --- | --- | --- | --- |
   |  | (f​□​g)∗superscript𝑓□𝑔\displaystyle\left(f\Box g\right)^{\*}\ | =f∗+g∗.absentsuperscript𝑓superscript𝑔\displaystyle=\ f^{\*}+g^{\*}\ . |  | (502) |

###### Proof.

1. 1.

   Since h​(t):=t22assignℎ𝑡superscript𝑡22h(t):=\frac{t^{2}}{2} is a non-negative convex function and h​(t)=0⇔t=0iffℎ𝑡0𝑡0h(t)=0\iff t=0 we have because of Proposition 11.3.3 in Garling ([2017](#bib.bib39)) that h​(‖x‖)∗=h∗​(‖x∗‖)ℎsuperscriptnorm𝑥superscriptℎnormsuperscript𝑥h\left({{\left\|x\right\|}}\right)^{\*}=h^{\*}\left({{\left\|x^{\*}\right\|}}\right). Additionally, by example (a) on page 137 we get for 1<p<∞1𝑝1<p<\infty and 1p+1q=11𝑝1𝑞1\frac{1}{p}+\frac{1}{q}=1 that (|t|pp)∗=|t∗|qqsuperscriptsuperscript𝑡𝑝𝑝superscriptsuperscript𝑡𝑞𝑞\left(\frac{|t|^{p}}{p}\right)^{\*}=\frac{|t^{\*}|^{q}}{q}. Putting all together we get the desired result. The same result can also be deduced from page 222 Example 13.6 in Bauschke & Combettes ([2017](#bib.bib11)).
2. 2.

   Follows immediately from the definition since

   |  |  |  |
   | --- | --- | --- |
   |  | α​f∗​(𝒙∗α)=α​sup𝒙∈X(𝒙T​𝒙∗α−f​(𝒙))=sup𝒙∈X(𝒙T​𝒙∗−α​f​(𝒙))=(α​f)∗​(𝒙∗)𝛼superscript𝑓superscript𝒙𝛼𝛼subscriptsupremum𝒙𝑋superscript𝒙𝑇superscript𝒙𝛼𝑓𝒙subscriptsupremum𝒙𝑋superscript𝒙𝑇superscript𝒙𝛼𝑓𝒙superscript𝛼𝑓superscript𝒙\displaystyle\alpha f^{\*}\left(\frac{\bm{x}^{\*}}{\alpha}\right)=\alpha\sup\_{\bm{x}\in X}\left(\bm{x}^{T}\frac{\bm{x}^{\*}}{\alpha}\ -\ f(\bm{x})\right)=\sup\_{\bm{x}\in X}(\bm{x}^{T}\bm{x}^{\*}-\alpha f(\bm{x}))=(\alpha f)^{\*}(\bm{x}^{\*}) |  |
3. 3.

   (f+β)∗:=sup𝒙∈X(𝒙T𝒙∗−f(𝒙)−β)=:f∗−β(f+\beta)^{\*}:=\sup\_{\bm{x}\in X}\left(\bm{x}^{T}\bm{x}^{\*}-f(\bm{x})-\beta\right)=:f^{\*}-\beta
4. 4.

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | (f​(𝑨​𝒙+𝒃))∗​(𝒙∗)superscript𝑓𝑨𝒙𝒃superscript𝒙\displaystyle\left(f\left(\bm{A}\bm{x}+\bm{b}\right)\right)^{\*}(\bm{x}^{\*}) | =sup𝒙∈X(𝒙T​𝒙∗−f​(𝑨​𝒙+𝒃))absentsubscriptsupremum𝒙𝑋superscript𝒙𝑇superscript𝒙𝑓𝑨𝒙𝒃\displaystyle=\sup\_{\bm{x}\in X}\left(\bm{x}^{T}\bm{x}^{\*}-f\left(\bm{A}\bm{x}+\bm{b}\right)\right) |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | =sup𝒙∈X((𝑨​𝒙+𝒃)T​𝑨−T​𝒙∗−f​(𝑨​𝒙+𝒃))−𝒃T​𝑨−T​𝒙∗absentsubscriptsupremum𝒙𝑋superscript𝑨𝒙𝒃𝑇superscript𝑨𝑇superscript𝒙𝑓𝑨𝒙𝒃superscript𝒃𝑇superscript𝑨𝑇superscript𝒙\displaystyle=\sup\_{\bm{x}\in X}\left(\left(\bm{A}\bm{x}+\bm{b}\right)^{T}\bm{A}^{-T}\bm{x}^{\*}-f\left(\bm{A}\bm{x}+\bm{b}\right)\right)-\bm{b}^{T}\bm{A}^{-T}\bm{x}^{\*} |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | =sup𝒚∈X(𝒚T​𝑨−T​𝒙∗−f​(𝒚))−𝒃T​𝑨−T​𝒙∗absentsubscriptsupremum𝒚𝑋superscript𝒚𝑇superscript𝑨𝑇superscript𝒙𝑓𝒚superscript𝒃𝑇superscript𝑨𝑇superscript𝒙\displaystyle=\sup\_{\bm{y}\in X}\left(\bm{y}^{T}\bm{A}^{-T}\bm{x}^{\*}-f\left(\bm{y}\right)\right)-\bm{b}^{T}\bm{A}^{-T}\bm{x}^{\*} |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | =f∗​(𝑨−T​𝒙∗)−𝒃T​𝑨−T​𝒙∗absentsuperscript𝑓superscript𝑨𝑇superscript𝒙superscript𝒃𝑇superscript𝑨𝑇superscript𝒙\displaystyle=f^{\*}\left(\bm{A}^{-T}\bm{x}^{\*}\right)-\bm{b}^{T}\bm{A}^{-T}\bm{x}^{\*} |  |
5. 5.

   From Proposition 13.24 (i) in Bauschke & Combettes ([2017](#bib.bib11)) and Proposition 11.4.2 in Garling ([2017](#bib.bib39)) we get

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | (f​□​g)∗​(𝒙∗)superscript𝑓□𝑔superscript𝒙\displaystyle\left(f\Box g\right)^{\*}(\bm{x}^{\*}) | =sup𝒙∈X(𝒙T​𝒙∗−inf𝒚∈X(f​(𝒚)−g​(𝒙−𝒚)))absentsubscriptsupremum𝒙𝑋superscript𝒙𝑇superscript𝒙subscriptinfimum𝒚𝑋𝑓𝒚𝑔𝒙𝒚\displaystyle=\sup\_{\bm{x}\in X}\left(\bm{x}^{T}\bm{x}^{\*}-\inf\_{\bm{y}\in X}\left(f(\bm{y})-g(\bm{x}-\bm{y})\right)\right) |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | =sup𝒙,𝒚∈X(𝒙T​𝒙∗−f​(𝒚)−g​(𝒙−𝒚))absentsubscriptsupremum 𝒙𝒚𝑋superscript𝒙𝑇superscript𝒙𝑓𝒚𝑔𝒙𝒚\displaystyle=\sup\_{\bm{x},\bm{y}\in X}\left(\bm{x}^{T}\bm{x}^{\*}-f(\bm{y})-g(\bm{x}-\bm{y})\right) |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | =sup𝒙,𝒚∈X((𝒚T​𝒙∗−f​(𝒚))+((𝒙−𝒚)T​𝒙∗−g​(𝒙−𝒚)))absentsubscriptsupremum 𝒙𝒚𝑋superscript𝒚𝑇superscript𝒙𝑓𝒚superscript𝒙𝒚𝑇superscript𝒙𝑔𝒙𝒚\displaystyle=\sup\_{\bm{x},\bm{y}\in X}\left(\left(\bm{y}^{T}\bm{x}^{\*}-f(\bm{y})\right)+\left(\left(\bm{x}-\bm{y}\right)^{T}\bm{x}^{\*}-g(\bm{x}-\bm{y})\right)\right) |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | =f∗​(𝒙∗)+g∗​(𝒙∗)absentsuperscript𝑓superscript𝒙superscript𝑔superscript𝒙\displaystyle=f^{\*}(\bm{x}^{\*})+g^{\*}(\bm{x}^{\*}) |  |

∎

###### Lemma A28.

The Legendre transform of the lselse\mathrm{lse} is the
negative entropy function, restricted to the probability simplex
and vice versa.
For the log-sum exponential

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(𝒙)𝑓𝒙\displaystyle f(\bm{x})\ | =ln⁡(∑i=1nexp⁡(xi)),absentsuperscriptsubscript𝑖1𝑛subscript𝑥𝑖\displaystyle=\ \ln\left(\sum\_{i=1}^{n}\exp(x\_{i})\right)\ , |  | (503) |

the Legendre transform is the
negative entropy function, restricted to the probability simplex:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f∗​(𝒙∗)superscript𝑓superscript𝒙\displaystyle f^{\*}(\bm{x}^{\*})\ | ={∑i=1nxi∗​ln⁡(xi∗) for ​ 0⩽xi∗​ and ​∑i=1nxi∗=1∞ otherwise .absentcasessuperscriptsubscript𝑖1𝑛superscriptsubscript𝑥𝑖subscriptsuperscript𝑥𝑖 for  0subscriptsuperscript𝑥𝑖 and superscriptsubscript𝑖1𝑛subscriptsuperscript𝑥𝑖1 otherwise \displaystyle=\ \begin{cases}\sum\_{i=1}^{n}x\_{i}^{\*}\ln(x^{\*}\_{i})&\text{ for }\ 0\leqslant x^{\*}\_{i}\ \text{ and }\ \sum\_{i=1}^{n}x^{\*}\_{i}=1\\ \infty&\text{ otherwise }\end{cases}\ . |  | (504) |

For the negative entropy function, restricted to the probability simplex:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(𝒙)𝑓𝒙\displaystyle f(\bm{x})\ | ={∑i=1nxi​ln⁡(xi) for ​ 0⩽xi​ and ​∑i=1nxi=1∞ otherwise .absentcasessuperscriptsubscript𝑖1𝑛subscript𝑥𝑖subscript𝑥𝑖 for  0subscript𝑥𝑖 and superscriptsubscript𝑖1𝑛subscript𝑥𝑖1 otherwise \displaystyle=\ \begin{cases}\sum\_{i=1}^{n}x\_{i}\ln(x\_{i})&\text{ for }\ 0\leqslant x\_{i}\ \text{ and }\ \sum\_{i=1}^{n}x\_{i}=1\\ \infty&\text{ otherwise }\end{cases}\ . |  | (505) |

the Legendre transform is the
log-sum exponential

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f∗​(𝒙∗)superscript𝑓superscript𝒙\displaystyle f^{\*}(\bm{x}^{\*})\ | =ln⁡(∑i=1nexp⁡(xi∗)),absentsuperscriptsubscript𝑖1𝑛superscriptsubscript𝑥𝑖\displaystyle=\ \ln\left(\sum\_{i=1}^{n}\exp(x\_{i}^{\*})\right)\ , |  | (506) |

###### Proof.

See page 93 Example 3.25 in Boyd & Vandenberghe ([2009](#bib.bib12))
and (Gao & Pavel, [2017](#bib.bib38)).
If f𝑓f is a regular convex function (lower semi-continuous convex function),
then f∗∗=fsuperscript𝑓absent𝑓f^{\*\*}=f according to
page 135 Exercise 11.2.3 in Garling ([2017](#bib.bib39)).
If f𝑓f is lower semi-continuous and convex, then f∗∗=fsuperscript𝑓absent𝑓f^{\*\*}=f according to
Theorem 13.37 (Fenchel-Moreau) in Bauschke & Combettes ([2017](#bib.bib11)).
The log-sum-exponential is continuous and convex.
∎

###### Lemma A29.

Let 𝐗​𝐗T𝐗superscript𝐗𝑇\bm{X}\bm{X}^{T} be non-singular and X𝑋X a Hilbert space.
We define

|  |  |  |  |
| --- | --- | --- | --- |
|  | X∗={𝒂∣0⩽𝑿T​(𝑿​𝑿T)−1​𝒂, 1T​𝑿T​(𝑿​𝑿T)−1​𝒂= 1}.superscript𝑋conditional-set𝒂formulae-sequence0superscript𝑿𝑇superscript𝑿superscript𝑿𝑇1𝒂superscript1𝑇superscript𝑿𝑇superscript𝑿superscript𝑿𝑇1𝒂1\displaystyle X^{\*}\ =\ \left\{\bm{a}\mid 0\ \leqslant\ \bm{X}^{T}\left(\bm{X}\bm{X}^{T}\right)^{-1}\bm{a}\ ,\ \ \bm{1}^{T}\bm{X}^{T}\left(\bm{X}\bm{X}^{T}\right)^{-1}\bm{a}\ =\ 1\right\}\ . |  | (507) |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | Xv={𝒂∣𝒂=𝑿T​𝝃,𝝃∈X}.superscript𝑋𝑣conditional-set𝒂formulae-sequence𝒂superscript𝑿𝑇𝝃𝝃𝑋\displaystyle X^{v}\ =\ \left\{\bm{a}\mid\bm{a}=\bm{X}^{T}\bm{\xi}\ ,\ \ \bm{\xi}\in X\right\}\ . |  | (508) |

The Legendre transform of lse​(β,𝐗T​𝛏)lse𝛽superscript𝐗𝑇𝛏\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})
with 𝛏∈X𝛏𝑋\bm{\xi}\in X is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (lse​(β,𝑿T​𝝃))∗​(𝝃∗)superscriptlse𝛽superscript𝑿𝑇𝝃superscript𝝃\displaystyle\left(\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\right)^{\*}(\bm{\xi}^{\*})\ | =(lse​(β,𝒗))∗​(𝑿T​(𝑿​𝑿T)−1​𝝃∗),absentsuperscriptlse𝛽𝒗superscript𝑿𝑇superscript𝑿superscript𝑿𝑇1superscript𝝃\displaystyle=\ \left(\mathrm{lse}(\beta,\bm{v})\right)^{\*}\left(\bm{X}^{T}\left(\bm{X}\bm{X}^{T}\right)^{-1}\bm{\xi}^{\*}\right)\ , |  | (509) |

with 𝛏∗∈X∗superscript𝛏superscript𝑋\bm{\xi}^{\*}\in X^{\*} and 𝐯∈Xv𝐯superscript𝑋𝑣\bm{v}\in X^{v}.
The domain of (lse​(β,𝐗T​𝛏))∗superscriptlse𝛽superscript𝐗𝑇𝛏\left(\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\right)^{\*} is X∗superscript𝑋X^{\*}.

Furthermore we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (lse​(β,𝑿T​𝝃))∗∗superscriptlse𝛽superscript𝑿𝑇𝝃absent\displaystyle\left(\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\right)^{\*\*}\ | =lse​(β,𝑿T​𝝃).absentlse𝛽superscript𝑿𝑇𝝃\displaystyle=\ \mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\ . |  | (510) |

###### Proof.

We use the definition of the Legendre transform:

|  |  |  |  |
| --- | --- | --- | --- |
|  | (lse​(β,𝑿T​𝝃))∗​(𝝃∗)=sup𝝃∈X𝝃T​𝝃∗−lse​(β,𝑿T​𝝃)superscriptlse𝛽superscript𝑿𝑇𝝃superscript𝝃subscriptsupremum𝝃𝑋superscript𝝃𝑇superscript𝝃lse𝛽superscript𝑿𝑇𝝃\displaystyle\left(\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\right)^{\*}(\bm{\xi}^{\*})\ =\ \sup\_{\bm{\xi}\in X}\bm{\xi}^{T}\bm{\xi}^{\*}\ -\ \mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi}) |  | (511) |
|  |  |  |
| --- | --- | --- |
|  | =sup𝝃∈X(𝑿T​𝝃)T​𝑿T​(𝑿​𝑿T)−1​𝝃∗−lse​(β,𝑿T​𝝃)absentsubscriptsupremum𝝃𝑋superscriptsuperscript𝑿𝑇𝝃𝑇superscript𝑿𝑇superscript𝑿superscript𝑿𝑇1superscript𝝃lse𝛽superscript𝑿𝑇𝝃\displaystyle=\ \sup\_{\bm{\xi}\in X}\left(\bm{X}^{T}\bm{\xi}\right)^{T}\bm{X}^{T}\left(\bm{X}\bm{X}^{T}\right)^{-1}\bm{\xi}^{\*}\ -\ \mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi}) |  |
|  |  |  |
| --- | --- | --- |
|  | =sup𝒗∈Xv𝒗T​𝑿T​(𝑿​𝑿T)−1​𝝃∗−lse​(β,𝒗)absentsubscriptsupremum𝒗superscript𝑋𝑣superscript𝒗𝑇superscript𝑿𝑇superscript𝑿superscript𝑿𝑇1superscript𝝃lse𝛽𝒗\displaystyle=\ \sup\_{\bm{v}\in X^{v}}\bm{v}^{T}\bm{X}^{T}\left(\bm{X}\bm{X}^{T}\right)^{-1}\bm{\xi}^{\*}\ -\ \mathrm{lse}(\beta,\bm{v}) |  |
|  |  |  |
| --- | --- | --- |
|  | =sup𝒗∈Xv𝒗T​𝒗∗−lse​(β,𝒗)absentsubscriptsupremum𝒗superscript𝑋𝑣superscript𝒗𝑇superscript𝒗lse𝛽𝒗\displaystyle=\ \sup\_{\bm{v}\in X^{v}}\bm{v}^{T}\bm{v}^{\*}\ -\ \mathrm{lse}(\beta,\bm{v}) |  |
|  |  |  |
| --- | --- | --- |
|  | =(lse​(β,𝒗))∗​(𝒗∗)=(lse​(β,𝒗))∗​(𝑿T​(𝑿​𝑿T)−1​𝝃∗),absentsuperscriptlse𝛽𝒗superscript𝒗superscriptlse𝛽𝒗superscript𝑿𝑇superscript𝑿superscript𝑿𝑇1superscript𝝃\displaystyle=\ \left(\mathrm{lse}(\beta,\bm{v})\right)^{\*}(\bm{v}^{\*})\ =\ \left(\mathrm{lse}(\beta,\bm{v})\right)^{\*}\left(\bm{X}^{T}\left(\bm{X}\bm{X}^{T}\right)^{-1}\bm{\xi}^{\*}\right)\ , |  |

where we used
𝒗∗=𝑿T​(𝑿​𝑿T)−1​𝝃∗superscript𝒗superscript𝑿𝑇superscript𝑿superscript𝑿𝑇1superscript𝝃\bm{v}^{\*}=\bm{X}^{T}\left(\bm{X}\bm{X}^{T}\right)^{-1}\bm{\xi}^{\*}.

According to page 93 Example 3.25 in Boyd & Vandenberghe ([2009](#bib.bib12)),
the equations for the maximum max𝒗∈Xv⁡𝒗T​𝒗∗−lse​(β,𝒗)subscript𝒗superscript𝑋𝑣superscript𝒗𝑇superscript𝒗lse𝛽𝒗\max\_{\bm{v}\in X^{v}}\bm{v}^{T}\bm{v}^{\*}\ -\ \mathrm{lse}(\beta,\bm{v})
are solvable if and only if 0<𝒗∗=𝑿T​(𝑿​𝑿T)−1​𝝃∗0superscript𝒗superscript𝑿𝑇superscript𝑿superscript𝑿𝑇1superscript𝝃0<\bm{v}^{\*}=\bm{X}^{T}\left(\bm{X}\bm{X}^{T}\right)^{-1}\bm{\xi}^{\*} and
𝟏T​𝒗∗=𝟏T​𝑿T​(𝑿​𝑿T)−1​𝝃∗=1superscript1𝑇superscript𝒗superscript1𝑇superscript𝑿𝑇superscript𝑿superscript𝑿𝑇1superscript𝝃1\bm{1}^{T}\bm{v}^{\*}=\bm{1}^{T}\bm{X}^{T}\left(\bm{X}\bm{X}^{T}\right)^{-1}\bm{\xi}^{\*}=1.
Therefore, we assumed 𝝃∗∈X∗superscript𝝃superscript𝑋\bm{\xi}^{\*}\in X^{\*}.

The domain of (lse​(β,𝑿T​𝝃))∗superscriptlse𝛽superscript𝑿𝑇𝝃\left(\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\right)^{\*} is X∗superscript𝑋X^{\*}, since
on page 93 Example 3.25 in Boyd & Vandenberghe ([2009](#bib.bib12)) it was shown that outside X∗superscript𝑋X^{\*}
the sup𝒗∈Xv𝒗T​𝒗∗−lse​(β,𝒗)subscriptsupremum𝒗superscript𝑋𝑣superscript𝒗𝑇superscript𝒗lse𝛽𝒗\sup\_{\bm{v}\in X^{v}}\bm{v}^{T}\bm{v}^{\*}\ -\ \mathrm{lse}(\beta,\bm{v}) is not bounded.

Using

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(β​𝑿T​𝝃),absentsoftmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ \mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi})\ , |  | (512) |

the Hessian of lse​(β,𝑿T​𝝃)lse𝛽superscript𝑿𝑇𝝃\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂2lse​(β,𝑿T​𝝃)∂𝝃2superscript2lse𝛽superscript𝑿𝑇𝝃superscript𝝃2\displaystyle\frac{\partial^{2}\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})}{\partial\bm{\xi}^{2}}\ | =β​𝑿​(diag​(𝒑)−𝒑​𝒑T)​𝑿Tabsent𝛽𝑿diag𝒑𝒑superscript𝒑𝑇superscript𝑿𝑇\displaystyle=\ \beta\ \bm{X}\left(\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T}\right)\bm{X}^{T} |  | (513) |

is positive semi-definite since diag​(𝒑)−𝒑​𝒑Tdiag𝒑𝒑superscript𝒑𝑇\mathrm{diag}(\bm{p})-\bm{p}\bm{p}^{T} is positive semi-definite
according to Lemma [A22](#ThmlemmaA22 "Lemma A22. ‣ A.2 Properties of Softmax, Log-Sum-Exponential, Legendre Transform, Lambert W Function ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
Therefore, lse​(β,𝑿T​𝝃)lse𝛽superscript𝑿𝑇𝝃\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi}) is convex and continuous.

If f𝑓f is a regular convex function (lower semi-continuous convex function),
then f∗∗=fsuperscript𝑓absent𝑓f^{\*\*}=f according to
page 135 Exercise 11.2.3 in Garling ([2017](#bib.bib39)).
If f𝑓f is lower semi-continuous and convex, then f∗∗=fsuperscript𝑓absent𝑓f^{\*\*}=f according to
Theorem 13.37 (Fenchel-Moreau) in Bauschke & Combettes ([2017](#bib.bib11)).
Consequently we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (lse​(β,𝑿T​𝝃))∗∗superscriptlse𝛽superscript𝑿𝑇𝝃absent\displaystyle\left(\mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\right)^{\*\*}\ | =lse​(β,𝑿T​𝝃).absentlse𝛽superscript𝑿𝑇𝝃\displaystyle=\ \mathrm{lse}(\beta,\bm{X}^{T}\bm{\xi})\ . |  | (514) |

∎

We introduce the Lambert W𝑊W function and some of its properties, since it is
needed to derive bounds on the storage capacity of our new Hopfield networks.

###### Definition A6 (Lambert Function).

The Lambert W𝑊W function (Olver et al., [2010](#bib.bib72), [(4.13)](http://dlmf.nist.gov/4.13)) is the inverse function of

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(y)𝑓𝑦\displaystyle f(y)\ | =y​ey.absent𝑦superscript𝑒𝑦\displaystyle=\ ye^{y}\ . |  | (515) |

The Lambert W𝑊W function has an upper branch W0subscript𝑊0W\_{0} for −1⩽y1𝑦-1\leqslant y and a lower
branch W−1subscript𝑊1W\_{-1} for y⩽−1𝑦1y\leqslant-1.
We use W𝑊W if a formula holds for both branches.
We have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(x)𝑊𝑥\displaystyle W(x)\ | =y⇒y​ey=x.absent𝑦⇒𝑦superscript𝑒𝑦𝑥\displaystyle=\ y\ \Rightarrow ye^{y}\ =\ x\ . |  | (516) |

We present some identities for the Lambert W𝑊W function (Olver et al., [2010](#bib.bib72), [(4.13)](http://dlmf.nist.gov/4.13)):

###### Lemma A30.

Identities for the Lambert W𝑊W function are

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(x)​eW​(x)𝑊𝑥superscript𝑒𝑊𝑥\displaystyle W(x)\ e^{W(x)}\ | =x,absent𝑥\displaystyle=\ x\ , |  | (517) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(x​ex)𝑊𝑥superscript𝑒𝑥\displaystyle W(xe^{x})\ | =x,absent𝑥\displaystyle=\ x\ , |  | (518) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | eW​(x)superscript𝑒𝑊𝑥\displaystyle e^{W(x)}\ | =xW​(x),absent𝑥𝑊𝑥\displaystyle=\ \frac{x}{W(x)}\ , |  | (519) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | e−W​(x)superscript𝑒𝑊𝑥\displaystyle e^{-W(x)}\ | =W​(x)x,absent𝑊𝑥𝑥\displaystyle=\ \frac{W(x)}{x}\ , |  | (520) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | en​W​(x)superscript𝑒𝑛𝑊𝑥\displaystyle e^{nW(x)}\ | =(xW​(x))n,absentsuperscript𝑥𝑊𝑥𝑛\displaystyle=\ \left(\frac{x}{W(x)}\right)^{n}\ , |  | (521) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W0​(x​ln⁡x)subscript𝑊0𝑥𝑥\displaystyle W\_{0}\left(x\ \ln x\right)\ | =ln⁡xfor ​x≥1e,formulae-sequenceabsent𝑥for 𝑥1𝑒\displaystyle=\ \ln x\quad\text{for }\ x\ \geq\ \frac{1}{e}\ , |  | (522) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W−1​(x​ln⁡x)subscript𝑊1𝑥𝑥\displaystyle W\_{-1}\left(x\ \ln x\right)\ | =ln⁡xfor ​x⩽1e,formulae-sequenceabsent𝑥for 𝑥1𝑒\displaystyle=\ \ln x\quad\text{for }\ x\ \leqslant\ \frac{1}{e}\ , |  | (523) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(x)𝑊𝑥\displaystyle W(x)\ | =ln⁡xW​(x)for ​x≥−1e,formulae-sequenceabsent𝑥𝑊𝑥for 𝑥1𝑒\displaystyle=\ \ln\frac{x}{W(x)}\quad\text{for }\ x\ \geq\ -\ \frac{1}{e}\ , |  | (524) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(n​xnW​(x)n−1)𝑊𝑛superscript𝑥𝑛𝑊superscript𝑥𝑛1\displaystyle W\left(\frac{n\ x^{n}}{W\left(x\right)^{n-1}}\right)\ | =n​W​(x)for ​n,x> 0,formulae-sequenceabsent  𝑛𝑊𝑥for 𝑛𝑥 0\displaystyle=\ n\ W(x)\quad\text{for }\ n,x\ >\ 0\ , |  | (525) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(x)+W​(y)𝑊𝑥𝑊𝑦\displaystyle W(x)\ +\ W(y)\ | =W​(x​y​(1W​(x)+1W​(y)))for ​x,y> 0,formulae-sequenceabsent  𝑊𝑥𝑦1𝑊𝑥1𝑊𝑦for 𝑥𝑦 0\displaystyle=\ W\left(x\ y\ \left(\frac{1}{W(x)}\ +\ \frac{1}{W(y)}\right)\right)\quad\text{for }\ x,y\ >\ 0\ , |  | (526) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W0​(−ln⁡xx)subscript𝑊0𝑥𝑥\displaystyle W\_{0}\left(-\ \frac{\ln x}{x}\right)\ | =−ln⁡xfor ​ 0<x⩽e,formulae-sequenceabsent𝑥for  0𝑥𝑒\displaystyle=\ -\ \ln x\quad\text{for }\ 0\ <\ x\ \leqslant\ e\ , |  | (527) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W−1​(−ln⁡xx)subscript𝑊1𝑥𝑥\displaystyle W\_{-1}\left(-\ \frac{\ln x}{x}\right)\ | =−ln⁡xfor ​x>e,formulae-sequenceabsent𝑥for 𝑥𝑒\displaystyle=\ -\ \ln x\quad\text{for }\ x\ >\ e\ , |  | (528) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | e−W​(−ln⁡x)superscript𝑒𝑊𝑥\displaystyle e^{-\ W(-\ \ln x)}\ | =W​(−ln⁡x)−ln⁡xfor ​x≠ 1.formulae-sequenceabsent𝑊𝑥𝑥for 𝑥1\displaystyle=\ \frac{W(-\ \ln x)}{-\ \ln x}\quad\text{for }\ x\ \neq\ 1\ . |  | (529) |

We also present some special values for the Lambert W𝑊W function (Olver et al., [2010](#bib.bib72), [(4.13)](http://dlmf.nist.gov/4.13)):

###### Lemma A31.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(0)𝑊0\displaystyle W(0)\ | = 0,absent 0\displaystyle=\ 0\ , |  | (530) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(e)𝑊𝑒\displaystyle W(e)\ | = 1,absent1\displaystyle=\ 1\ , |  | (531) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(−1e)𝑊1𝑒\displaystyle W\left(-\frac{1}{e}\right)\ | =−1,absent1\displaystyle=\ -1\ , |  | (532) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(e1+e)𝑊superscript𝑒1𝑒\displaystyle W\left(e^{1+e}\right)\ | =e,absent𝑒\displaystyle=\ e\ , |  | (533) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(2​ln⁡2)𝑊22\displaystyle W\left(2\ln 2\right)\ | =ln⁡2,absent2\displaystyle=\ \ln 2\ , |  | (534) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(1)𝑊1\displaystyle W(1)\ | =Ω,absentΩ\displaystyle=\ \Omega\ , |  | (535) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(1)𝑊1\displaystyle W(1)\ | =e−W​(1)=ln⁡(1W​(1))=−ln⁡W​(1),absentsuperscript𝑒𝑊11𝑊1𝑊1\displaystyle=\ e^{-W(1)}\ =\ \ln\left(\frac{1}{W(1)}\right)\ =\ -\ \ln W(1)\ , |  | (536) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(−π2)𝑊𝜋2\displaystyle W\left(-\frac{\pi}{2}\right)\ | =i​π2,absent𝑖𝜋2\displaystyle=\ \frac{i\pi}{2}\ , |  | (537) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W​(−1)𝑊1\displaystyle W(-1)\ | ≈−0.31813+1.33723​i,absent0.318131.33723𝑖\displaystyle\approx\ -0.31813+1.33723i\ , |  | (538) |

where the Omega constant ΩΩ\Omega is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ΩΩ\displaystyle\Omega\ | =(∫−∞∞d​t(et−t)2+π2)−1− 1≈ 0.56714329.absentsuperscriptsuperscriptsubscriptd𝑡superscriptsuperscript𝑒𝑡𝑡2superscript𝜋2110.56714329\displaystyle=\ \left(\int\_{-\infty}^{\infty}\frac{\mathrm{d}t}{\left(e^{t}\ -\ t\right)^{2}\ +\ \pi^{2}}\right)^{-1}\ -\ 1\ \approx\ 0.56714329\ . |  | (539) |

We need in some proofs a version of the mean value theorem as
given in the next lemma.

###### Lemma A32 (Mean Value Theorem).

Let U⊂ℝn𝑈superscriptℝ𝑛U\subset\mathbb{R}^{n} be open, f:U→ℝm:𝑓→𝑈superscriptℝ𝑚f:U\to\mathbb{R}^{m} continuously differentiable, and
𝐱∈U𝐱𝑈\bm{x}\in U as well as 𝐡∈ℝn𝐡superscriptℝ𝑛\bm{h}\in\mathbb{R}^{n} vectors such that the line segment
𝐱+t​𝐡𝐱𝑡𝐡\bm{x}+t\bm{h} for 0⩽t⩽10𝑡10\leqslant t\leqslant 1 is in U𝑈U. Then the following holds:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(𝒙+𝒉)−f​(𝒙)𝑓𝒙𝒉𝑓𝒙\displaystyle f(\bm{x}\ +\ \bm{h})\ -\ f(\bm{x})\ | =(∫01J​(𝒙+t​𝒉)​dt)​𝒉,absentsuperscriptsubscript01𝐽𝒙𝑡𝒉differential-d𝑡𝒉\displaystyle=\ \left(\int\_{0}^{1}J(\bm{x}\ +\ t\ \bm{h})\ \mathrm{d}t\right)\ \bm{h}\ , |  | (540) |

where J𝐽J is the Jacobian of f𝑓f and the integral of the matrix is component-wise.

###### Proof.

Let f1,…,fm

subscript𝑓1…subscript𝑓𝑚f\_{1},\ldots,f\_{m} denote the components of f𝑓f and define
gi:[0,1]→ℝ:subscript𝑔𝑖→01ℝg\_{i}:[0,1]\to\mathbb{R} by

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | gi​(t)subscript𝑔𝑖𝑡\displaystyle g\_{i}(t)\ | =fi​(𝒙+t​𝒉),absentsubscript𝑓𝑖𝒙𝑡𝒉\displaystyle=\ f\_{i}(\bm{x}\ +\ t\ \bm{h})\ , |  | (541) |

then we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | fi​(𝒙+𝒉)−fi​(𝒙)=gi​(1)−gi​(0)=∫01g′​(t)​dtsubscript𝑓𝑖𝒙𝒉subscript𝑓𝑖𝒙subscript𝑔𝑖1subscript𝑔𝑖0superscriptsubscript01superscript𝑔′𝑡differential-d𝑡\displaystyle f\_{i}(\bm{x}\ +\ \bm{h})\ -\ f\_{i}(\bm{x})\ =\ g\_{i}(1)\ -\ g\_{i}(0)\ =\ \int\_{0}^{1}g^{\prime}(t)\ \mathrm{d}t |  | (542) |
|  |  |  |
| --- | --- | --- |
|  | ∫01(∑j=1n∂fi∂xj​(𝒙+t​𝒉)​hj)​dt=∑j=1n(∫01∂fi∂xj​(𝒙+t​𝒉)​dt)​hj.superscriptsubscript01superscriptsubscript𝑗1𝑛subscript𝑓𝑖subscript𝑥𝑗𝒙𝑡𝒉subscriptℎ𝑗differential-d𝑡superscriptsubscript𝑗1𝑛superscriptsubscript01subscript𝑓𝑖subscript𝑥𝑗𝒙𝑡𝒉differential-d𝑡subscriptℎ𝑗\displaystyle\int\_{0}^{1}\left(\sum\_{j=1}^{n}\frac{\partial f\_{i}}{\partial x\_{j}}(\bm{x}\ +\ t\ \bm{h})\ h\_{j}\right)\ \mathrm{d}t\ =\ \sum\_{j=1}^{n}\left(\int\_{0}^{1}\frac{\partial f\_{i}}{\partial x\_{j}}(\bm{x}\ +\ t\ \bm{h})\ \mathrm{d}t\right)\ h\_{j}\ . |  |

The statement follows since the Jacobian J𝐽J has as entries ∂fi∂xjsubscript𝑓𝑖subscript𝑥𝑗\frac{\partial f\_{i}}{\partial x\_{j}}.
∎

### A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield)

#### A.3.1 Modern Hopfield Networks: Introduction

##### A.3.1.1 Additional Memory and Attention for Neural Networks.

Modern Hopfield networks may serve as additional memory for neural networks.
Different approaches have been suggested to equip neural networks with
an additional memory beyond recurrent connections.
The neural Turing machine (NTM) is a neural network equipped with an
external memory and an attention process (Graves et al., [2014](#bib.bib41)).
The NTM can write to the memory and can read from it.
A memory network (Weston et al., [2014](#bib.bib104)) consists of a memory together
with the components:
(1) input feature map (converts the incoming input to the internal feature representation)
(2) generalization (updates old memories given the new input),
(3) output feature map (produces a new output),
(4) response (converts the output into the response format).
Memory networks are generalized to an end-to-end trained model, where
the arg⁡max\arg\max memory call is
replaced by a differentiable softmaxsoftmax\mathrm{softmax} (Sukhbaatar et al., [2015a](#bib.bib88); [b](#bib.bib89)).
Linear Memory Network use a linear autoencoder for sequences
as a memory (Carta et al., [2020](#bib.bib20)).

To enhance RNNs with additional associative memory like Hopfield networks
have been proposed (Ba et al., [2016a](#bib.bib6); [b](#bib.bib7)).
The associative memory stores hidden states of the RNN, retrieves
stored states if they are similar to actual ones, and has a forgetting parameter.
The forgetting and storing parameters of the RNN associative memory
have been
generalized to learned matrices (Zhang & Zhou, [2017](#bib.bib118)).
LSTMs with associative memory via Holographic Reduced Representations have
been proposed (Danihelka et al., [2016](#bib.bib28)).

Recently most approaches to new memories are based on attention.
The neural Turing machine (NTM) is equipped with an
external memory and an attention process (Graves et al., [2014](#bib.bib41)).
End to end memory networks (EMN) make the attention scheme
of memory networks (Weston et al., [2014](#bib.bib104)) differentiable
by replacing arg⁡max\arg\max through a softmaxsoftmax\mathrm{softmax} (Sukhbaatar et al., [2015a](#bib.bib88); [b](#bib.bib89)).
EMN with dot products became very popular and implement a key-value
attention (Daniluk et al., [2017](#bib.bib29)) for self-attention.
An enhancement of EMN is the transformer (Vaswani et al., [2017a](#bib.bib96); [b](#bib.bib97))
and its extensions (Dehghani et al., [2018](#bib.bib30)).
The transformer had great impact on the natural language processing
(NLP) community as new records in NLP benchmarks have been achieved
(Vaswani et al., [2017a](#bib.bib96); [b](#bib.bib97)).
MEMO uses the transformer
attention mechanism for reasoning over longer distances (Banino et al., [2020](#bib.bib9)).
Current state-of-the-art for language processing is
a transformer architecture called
“the Bidirectional Encoder Representations from Transformers”
(BERT) (Devlin et al., [2018](#bib.bib32); [2019](#bib.bib33)).

##### A.3.1.2 Modern Hopfield networks: Overview.

The storage capacity of classical binary Hopfield networks (Hopfield, [1982](#bib.bib47))
has been shown to be very limited.
In a d𝑑d-dimensional space,
the standard Hopfield model can store d𝑑d uncorrelated patterns
without errors but only
C​d/ln⁡(d)𝐶𝑑𝑑Cd/\ln(d) random patterns with
C<1/2𝐶12C<1/2 for a fixed stable pattern or C<1/4𝐶14C<1/4 if all patterns
are stable (McEliece et al., [1987](#bib.bib70)).
The same bound holds for nonlinear learning rules (Mazza, [1997](#bib.bib69)).
Using tricks-of-trade and allowing
small retrieval errors, the storage capacity
is about 0.138​d0.138𝑑0.138d (Crisanti et al., [1986](#bib.bib27); Hertz et al., [1991](#bib.bib43); Torres et al., [2002](#bib.bib95)).
If the learning rule is not related to the Hebb rule then up to d𝑑d
patterns can be stored (Abu-Mostafa & StJacques, [1985](#bib.bib1)).
Using Hopfield networks with non-zero diagonal matrices,
the storage can be
increased to C​d​ln⁡(d)𝐶𝑑𝑑Cd\ln(d) (Folli et al., [2017](#bib.bib37)).
In contrast to the storage capacity, the number of energy minima
(spurious states, stable states) of Hopfield networks
is exponentially in d𝑑d (Tanaka & Edwards, [1980](#bib.bib91); Bruck & Roychowdhury, [1990](#bib.bib15); Wainrib & Touboul, [2013](#bib.bib100)).

Recent advances
in the field of binary Hopfield networks (Hopfield, [1982](#bib.bib47))
led to new properties of Hopfield networks.
The stability of spurious states or metastable states
was sensibly reduced by a Hamiltonian treatment for
the new relativistic Hopfield model (Barra et al., [2018](#bib.bib10)).
Recently the storage capacity of Hopfield networks
could be increased by new energy functions.
Interaction functions of the form F​(x)=xn𝐹𝑥superscript𝑥𝑛F(x)=x^{n} lead to storage capacity of
αn​dn−1subscript𝛼𝑛superscript𝑑𝑛1\alpha\_{n}d^{n-1}, where αnsubscript𝛼𝑛\alpha\_{n} depends on the allowed error
probability (Krotov & Hopfield, [2016](#bib.bib59); [2018](#bib.bib60); Demircigil et al., [2017](#bib.bib31))
(see (Krotov & Hopfield, [2018](#bib.bib60)) for the non-binary case).
Interaction functions of the form F​(x)=xn𝐹𝑥superscript𝑥𝑛F(x)=x^{n} lead to storage capacity of
αn​dn−1cn​ln⁡dsubscript𝛼𝑛superscript𝑑𝑛1subscript𝑐𝑛𝑑\alpha\_{n}\frac{d^{n-1}}{c\_{n}\ln d} for cn>2​(2​n−3)!!subscript𝑐𝑛2double-factorial2𝑛3c\_{n}>2(2n-3)!! (Demircigil et al., [2017](#bib.bib31)).

Interaction functions of the form F​(x)=exp⁡(x)𝐹𝑥𝑥F(x)=\exp(x)
lead to exponential storage capacity of
2d/2superscript2𝑑22^{d/2} where all stored patterns are fixed points but the radius of
attraction vanishes (Demircigil et al., [2017](#bib.bib31)).
It has been shown that the network converges with high probability
after one update (Demircigil et al., [2017](#bib.bib31)).

#### A.3.2 Energy and Update Rule for Binary Modern Hopfield Networks

We follow (Demircigil et al., [2017](#bib.bib31)) where the goal is to store a set of input data
𝒙1,…,𝒙N

subscript𝒙1…subscript𝒙𝑁\bm{x}\_{1},\ldots,\bm{x}\_{N}
that are represented by the matrix

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑿𝑿\displaystyle\bm{X}\ | =(𝒙1,…,𝒙N).absentsubscript𝒙1…subscript𝒙𝑁\displaystyle=\ \left(\bm{x}\_{1},\ldots,\bm{x}\_{N}\right)\ . |  | (543) |

The
𝒙isubscript𝒙𝑖\bm{x}\_{i} is pattern with binary components
xi​j∈{−1,+1}subscript𝑥𝑖𝑗11x\_{ij}\in\{-1,+1\} for all i𝑖i and j𝑗j.
𝝃𝝃\bm{\xi} is the actual state of the units of the Hopfield model.
Krotov and Hopfield (Krotov & Hopfield, [2016](#bib.bib59)) defined the energy function EE\mathrm{E}
with the interaction function F𝐹F that evaluates
the dot product between patterns 𝒙isubscript𝒙𝑖\bm{x}\_{i}
and the actual state 𝝃𝝃\bm{\xi}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | EE\displaystyle\mathrm{E}\ | =−∑i=1NF​(𝝃T​𝒙i)absentsuperscriptsubscript𝑖1𝑁𝐹superscript𝝃𝑇subscript𝒙𝑖\displaystyle=\ -\ \sum\_{i=1}^{N}F\left(\bm{\xi}^{T}\bm{x}\_{i}\right) |  | (544) |

with F​(a)=an𝐹𝑎superscript𝑎𝑛F(a)=a^{n}, where n=2𝑛2n=2 gives the energy function of the classical
Hopfield network. This allows to store αn​dn−1subscript𝛼𝑛superscript𝑑𝑛1\alpha\_{n}d^{n-1} patterns (Krotov & Hopfield, [2016](#bib.bib59)).
Krotov and Hopfield (Krotov & Hopfield, [2016](#bib.bib59)) suggested for minimizing this energy
an asynchronous updating dynamics T=(Tj)𝑇subscript𝑇𝑗T=(T\_{j}) for component ξjsubscript𝜉𝑗\xi\_{j}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Tj​(𝝃)subscript𝑇𝑗𝝃\displaystyle T\_{j}(\bm{\xi})\ | :=sgn[∑i=1N(F​(xi​j+∑l≠jxi​l​ξl)−F​(−xi​j+∑l≠jxi​l​ξl))]assignabsentsgndelimited-[]superscriptsubscript𝑖1𝑁𝐹subscript𝑥𝑖𝑗subscript𝑙𝑗subscript𝑥𝑖𝑙subscript𝜉𝑙𝐹subscript𝑥𝑖𝑗subscript𝑙𝑗subscript𝑥𝑖𝑙subscript𝜉𝑙\displaystyle:=\ \mathop{\mathrm{sgn}\,}\Bigl{[}\sum\limits\_{i=1}^{N}\bigl{(}F\bigl{(}x\_{ij}\ +\ \sum\limits\_{l\neq j}x\_{il}\ \xi\_{l}\bigr{)}\ -\ F\bigl{(}-\ x\_{ij}\ +\ \sum\limits\_{l\neq j}x\_{il}\ \xi\_{l}\bigr{)}\bigr{)}\Bigr{]} |  | (545) |

While Krotov and Hopfield used F​(a)=an𝐹𝑎superscript𝑎𝑛F(a)=a^{n},
Demircigil et al. (Demircigil et al., [2017](#bib.bib31)) went a step further and analyzed
the model with the energy function F​(a)=exp⁡(a)𝐹𝑎𝑎F(a)=\exp(a), which
leads to an exponential
storage capacity of N=2d/2𝑁superscript2𝑑2N=2^{d/2}.
Furthermore with a single update the final pattern
is recovered with high probability.
These statements are given in next theorem.

###### Theorem A10 (Storage Capacity for Binary Modern Hopfield Nets (Demircigil et al. 2017)).

Consider the generalized Hopfield model with the dynamics
described in Eq. ([545](#A1.E545 "In A.3.2 Energy and Update Rule for Binary Modern Hopfield Networks ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) and
interaction function F𝐹F given by F​(x)=ex𝐹𝑥superscript𝑒𝑥F(x)=e^{x}.
For a fixed 0<α<ln⁡(2)/20𝛼220<\alpha<\ln(2)/2
let N=exp⁡(α​d)+1𝑁𝛼𝑑1N=\exp\left(\alpha d\right)+1 and let 𝐱1,…,𝐱N

subscript𝐱1…subscript𝐱𝑁\bm{x}\_{1},\ldots,\bm{x}\_{N}
be N𝑁N patterns chosen uniformly at random from {−1,+1}dsuperscript11𝑑\{-1,+1\}^{d}.
Moreover fix ϱ∈[0,1/2)italic-ϱ012\varrho\in[0,1/2).
For any i𝑖i and any 𝐱~isubscript~𝐱𝑖\widetilde{\bm{x}}\_{i} taken uniformly at random
from the Hamming sphere with radius ϱ​ditalic-ϱ𝑑\varrho d centered in 𝐱isubscript𝐱𝑖\bm{x}\_{i},
𝒮​(𝐱i,ϱ​d)𝒮subscript𝐱𝑖italic-ϱ𝑑\mathcal{S}(\bm{x}\_{i},\varrho d), where ϱ​ditalic-ϱ𝑑\varrho d is assumed to be an integer,
it holds that

|  |  |  |
| --- | --- | --- |
|  | Pr(∃i∃j:Tj(𝒙~i)≠xi​j)→ 0,\displaystyle\mathbf{\mathrm{Pr}}\left(\exists i\;\exists j:\ T\_{j}\left(\widetilde{\bm{x}}\_{i}\right)\ \neq\ x\_{ij}\right)\ \rightarrow\ 0\ , |  |

if α𝛼\alpha is chosen in dependence of ϱitalic-ϱ\varrho such that

|  |  |  |
| --- | --- | --- |
|  | α<I​(1−2​ϱ)2𝛼𝐼12italic-ϱ2\displaystyle\alpha\ <\ \frac{I(1-2\varrho)}{2} |  |

with

|  |  |  |
| --- | --- | --- |
|  | I:a↦12​((1+a)​ln⁡(1+a)+(1−a)​ln⁡(1−a)).:𝐼maps-to𝑎121𝑎1𝑎1𝑎1𝑎\displaystyle I:\ a\ \mapsto\ \frac{1}{2}\left((1+a)\ln(1+a)\ +\ (1-a)\ln(1-a)\right)\ . |  |

###### Proof.

The proof can be found in Demircigil et al. ([2017](#bib.bib31)).
∎

The number of patterns N=exp⁡(α​d)+1𝑁𝛼𝑑1N=\exp\left(\alpha d\right)+1 is
exponential in the number d𝑑d of components.
The result

|  |  |  |
| --- | --- | --- |
|  | Pr(∃i∃j:Tj(𝒙~i)≠xi​j)→ 0\displaystyle\mathbf{\mathrm{Pr}}\left(\exists i\;\exists j:\ T\_{j}\left(\widetilde{\bm{x}}\_{i}\right)\ \neq\ x\_{ij}\right)\ \rightarrow\ 0 |  |

means that one update for each component
is sufficient to recover the pattern with high probability.
The constraint α<I​(1−2​ϱ)2𝛼𝐼12italic-ϱ2\alpha<\frac{I(1-2\varrho)}{2} on α𝛼\alpha gives the
trade-off between the radius of attraction ϱ​ditalic-ϱ𝑑\varrho d and the number
N=exp⁡(α​d)+1𝑁𝛼𝑑1N=\exp\left(\alpha d\right)+1 of
pattern that can be stored.

Theorem [A10](#ThmtheoremA10 "Theorem A10 (Storage Capacity for Binary Modern Hopfield Nets (Demircigil et al. 2017)). ‣ A.3.2 Energy and Update Rule for Binary Modern Hopfield Networks ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") in particular implies that

|  |  |  |
| --- | --- | --- |
|  | Pr(∃i∃j:Tj(𝒙i)≠xi​j)→ 0\displaystyle\mathbf{\mathrm{Pr}}\left(\exists i\;\exists j:\ T\_{j}\left(\bm{x}\_{i}\right)\ \neq\ x\_{ij}\right)\ \rightarrow\ 0 |  |

as d→∞→𝑑d\rightarrow\infty, i.e. with a probability converging to 111,
all the patterns are fixed points
of the dynamics.
In this case we can have
α→I​(1)2=ln⁡(2)/2→𝛼𝐼1222\alpha\to\frac{I(1)}{2}=\ln(2)/2.

Krotov and Hopfield define the update dynamics Tj​(𝝃)subscript𝑇𝑗𝝃T\_{j}(\bm{\xi})
in Eq. ([545](#A1.E545 "In A.3.2 Energy and Update Rule for Binary Modern Hopfield Networks ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) via
energy differences of the energy in Eq. ([544](#A1.E544 "In A.3.2 Energy and Update Rule for Binary Modern Hopfield Networks ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
First we express the energy in Eq. ([544](#A1.E544 "In A.3.2 Energy and Update Rule for Binary Modern Hopfield Networks ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) with
F​(a)=exp⁡(a)𝐹𝑎𝑎F(a)=\exp(a) (Demircigil et al., [2017](#bib.bib31))
by the lselse\mathrm{lse} function.
Then we use the mean value theorem to express
the update dynamics Tj​(𝝃)subscript𝑇𝑗𝝃T\_{j}(\bm{\xi})
in Eq. ([545](#A1.E545 "In A.3.2 Energy and Update Rule for Binary Modern Hopfield Networks ‣ A.3 Modern Hopfield Networks: Binary States (Krotov and Hopfield) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) by the softmax function.
For simplicity, we set β=1𝛽1\beta=1 in the following.
There exists a v∈[−1,1]𝑣11v\in[-1,1] with

|  |  |  |  |
| --- | --- | --- | --- |
|  | Tj​(𝝃)subscript𝑇𝑗𝝃\displaystyle T\_{j}(\bm{\xi})\ | =sgn[−E​(ξj=1)+E​(ξj=−1)]=sgn[exp⁡(lse​(ξj=1))−exp⁡(lse​(ξj=−1))]absentsgndelimited-[]Esubscript𝜉𝑗1Esubscript𝜉𝑗1sgndelimited-[]lsesubscript𝜉𝑗1lsesubscript𝜉𝑗1\displaystyle=\ \mathop{\mathrm{sgn}\,}\Bigl{[}-\ \mathrm{E}(\xi\_{j}=1)\ +\ \mathrm{E}(\xi\_{j}=-1)\Bigr{]}\ =\ \mathop{\mathrm{sgn}\,}\Bigl{[}\exp(\mathrm{lse}(\xi\_{j}=1))\ -\ \exp(\mathrm{lse}(\xi\_{j}=-1))\Bigr{]} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =sgn[−(2​𝒆j)T​∇𝝃E​(ξj=v)]=sgn[exp⁡(lse​(ξj=v))​(2​𝒆j)T​lse​(ξj=v)∂𝝃]absentsgndelimited-[]superscript2subscript𝒆𝑗𝑇subscript∇𝝃Esubscript𝜉𝑗𝑣sgndelimited-[]lsesubscript𝜉𝑗𝑣superscript2subscript𝒆𝑗𝑇lsesubscript𝜉𝑗𝑣𝝃\displaystyle=\ \mathop{\mathrm{sgn}\,}\Bigl{[}-\ (2\bm{e}\_{j})^{T}\nabla\_{\bm{\xi}}\mathrm{E}(\xi\_{j}=v)\Bigr{]}\ =\ \mathop{\mathrm{sgn}\,}\Bigl{[}\exp(\mathrm{lse}(\xi\_{j}=v))\ (2\bm{e}\_{j})^{T}\frac{\mathrm{lse}(\xi\_{j}=v)}{\partial\bm{\xi}}\Bigr{]} |  | (546) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =sgn[exp⁡(lse​(ξj=1))​(2​𝒆j)T​𝑿​softmax​(𝑿T​𝝃​(ξj=v))]absentsgndelimited-[]lsesubscript𝜉𝑗1superscript2subscript𝒆𝑗𝑇𝑿softmaxsuperscript𝑿𝑇𝝃subscript𝜉𝑗𝑣\displaystyle=\ \mathop{\mathrm{sgn}\,}\Bigl{[}\exp(\mathrm{lse}(\xi\_{j}=1))\ (2\bm{e}\_{j})^{T}\bm{X}\mathrm{softmax}(\bm{X}^{T}\bm{\xi}(\xi\_{j}=v))\Bigr{]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =sgn[[𝑿​softmax​(𝑿T​𝝃​(ξj=v))]j]=sgn[[𝑿​𝒑​(ξj=v)]j],absentsgndelimited-[]subscriptdelimited-[]𝑿softmaxsuperscript𝑿𝑇𝝃subscript𝜉𝑗𝑣𝑗sgndelimited-[]subscriptdelimited-[]𝑿𝒑subscript𝜉𝑗𝑣𝑗\displaystyle=\ \mathop{\mathrm{sgn}\,}\Bigl{[}[\bm{X}\mathrm{softmax}(\bm{X}^{T}\bm{\xi}(\xi\_{j}=v))]\_{j}\Bigr{]}\ =\ \mathop{\mathrm{sgn}\,}\Bigl{[}[\bm{X}\bm{p}(\xi\_{j}=v)]\_{j}\Bigr{]}\ , |  |

where 𝒆jsubscript𝒆𝑗\bm{e}\_{j} is the Cartesian unit vector with a one at position j𝑗j and zeros elsewhere,
[.]j[.]\_{j} is the projection to the j𝑗j-th component, and

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p}\ | =softmax​(𝑿T​𝝃).absentsoftmaxsuperscript𝑿𝑇𝝃\displaystyle=\ \mathrm{softmax}(\bm{X}^{T}\bm{\xi})\ . |  | (547) |

### A.4 Hopfield Update Rule is Attention of The Transformer

The Hopfield network update rule
is the attention mechanism used
in transformer and BERT models
(see Fig. [A.2](#A1.F2 "Figure A.2 ‣ A.4 Hopfield Update Rule is Attention of The Transformer ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
To see this, we assume N𝑁N stored (key) patterns 𝒚isubscript𝒚𝑖\bm{y}\_{i}
and S𝑆S state (query) patterns 𝒓isubscript𝒓𝑖\bm{r}\_{i} that are mapped to the
Hopfield space of dimension dksubscript𝑑𝑘d\_{k}.
We set 𝒙i=𝑾KT​𝒚isubscript𝒙𝑖superscriptsubscript𝑾𝐾𝑇subscript𝒚𝑖\bm{x}\_{i}=\bm{W}\_{K}^{T}\bm{y}\_{i}, 𝝃i=𝑾QT​𝒓isubscript𝝃𝑖superscriptsubscript𝑾𝑄𝑇subscript𝒓𝑖\bm{\xi}\_{i}=\bm{W}\_{Q}^{T}\bm{r}\_{i},
and multiply the result of our update rule with 𝑾Vsubscript𝑾𝑉\bm{W}\_{V}.
The matrices 𝒀=(𝒚1,…,𝒚N)T𝒀superscriptsubscript𝒚1…subscript𝒚𝑁𝑇\bm{Y}=(\bm{y}\_{1},\ldots,\bm{y}\_{N})^{T} and 𝑹=(𝒓1,…,𝒓S)T𝑹superscriptsubscript𝒓1…subscript𝒓𝑆𝑇\bm{R}=(\bm{r}\_{1},\ldots,\bm{r}\_{S})^{T} combine the 𝒚isubscript𝒚𝑖\bm{y}\_{i} and 𝒓isubscript𝒓𝑖\bm{r}\_{i}
as row vectors.
We define the matrices 𝑿T=𝑲=𝒀​𝑾Ksuperscript𝑿𝑇𝑲𝒀subscript𝑾𝐾\bm{X}^{T}=\bm{K}=\bm{Y}\bm{W}\_{K}, 𝚵T=𝑸=𝑹​𝑾Qsuperscript𝚵𝑇𝑸𝑹subscript𝑾𝑄\bm{\Xi}^{T}=\bm{Q}=\bm{R}\bm{W}\_{Q},
and 𝑽=𝒀​𝑾K​𝑾V=𝑿T​𝑾V𝑽𝒀subscript𝑾𝐾subscript𝑾𝑉superscript𝑿𝑇subscript𝑾𝑉\bm{V}=\bm{Y}\bm{W}\_{K}\bm{W}\_{V}=\bm{X}^{T}\bm{W}\_{V}, where
𝑾K∈ℝdy×dk,𝑾Q∈ℝdr×dk,𝑾V∈ℝdk×dvformulae-sequencesubscript𝑾𝐾superscriptℝsubscript𝑑𝑦subscript𝑑𝑘formulae-sequencesubscript𝑾𝑄superscriptℝsubscript𝑑𝑟subscript𝑑𝑘subscript𝑾𝑉superscriptℝsubscript𝑑𝑘subscript𝑑𝑣\bm{W}\_{K}\in\mathbb{R}^{d\_{y}\times d\_{k}},\bm{W}\_{Q}\in\mathbb{R}^{d\_{r}\times d\_{k}},\bm{W}\_{V}\in\mathbb{R}^{d\_{k}\times d\_{v}}.
If β=1/dk𝛽1subscript𝑑𝑘\beta=1/\sqrt{d\_{k}} and softmax∈ℝNsoftmaxsuperscriptℝ𝑁\mathrm{softmax}\in\mathbb{R}^{N} is changed to a row vector, we obtain
for the update rule Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need")) multiplied by 𝑾Vsubscript𝑾𝑉\bm{W}\_{V}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | softmax​(1/dk​𝑸​𝑲T)​𝑽=softmax​(β​𝑹​𝑾𝑸​𝑾𝑲T​𝒀T)​𝒀​𝑾𝑲​𝑾𝑽.softmax1subscript𝑑𝑘𝑸superscript𝑲𝑇𝑽softmax𝛽𝑹subscript𝑾𝑸superscriptsubscript𝑾𝑲𝑇superscript𝒀𝑇𝒀subscript𝑾𝑲subscript𝑾𝑽\displaystyle\mathrm{softmax}\left(1/\sqrt{d\_{k}}\ \bm{Q}\ \bm{K}^{T}\right)\ \bm{V}\ =\mathrm{softmax}\left(\beta\ \bm{R}\bm{W}\_{\bm{Q}}\ \bm{W}\_{\bm{K}}^{T}\bm{Y}^{T}\right)\ \bm{Y}\bm{W}\_{\bm{K}}\bm{W}\_{\bm{V}}\ . |  | (548) |

The left part of Eq. ([548](#A1.E548 "In A.4 Hopfield Update Rule is Attention of The Transformer ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) is the transformer attention.
Besides the attention mechanism,
Hopfield networks allow for other functionalities
in deep network architectures,
which we introduce via specific layers
in the next section. The right part of Eq. ([548](#A1.E548 "In A.4 Hopfield Update Rule is Attention of The Transformer ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) serves as starting point for these specific layers.

!(/html/2008.02217/assets/x7.png)

Figure A.2: We generalized
the energy
of binary modern Hopfield networks for allowing continuous states
while keeping fast convergence and
storage capacity properties.
We defined for the new energy also a new update
rule that minimizes the energy.
The new update rule is the attention mechanism of the transformer.
Formulae are modified to express softmaxsoftmax\mathrm{softmax} as row vector as for transformers.
"=="-sign means "keeps the properties".

### A.5 Experiments

#### A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics

##### A.5.1.1 Analysis of operating modes of the heads of a pre-trained BERT model.

We analyzed pre-trained BERT models from Hugging
Face Inc. (Wolf et al., [2019](#bib.bib107)) according to these operating classes.
In Fig. [A.3](#A1.F3 "Figure A.3 ‣ A.5.1.1 Analysis of operating modes of the heads of a pre-trained BERT model. ‣ A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") in the appendix the distribution of the pre-trained
bert-base-cased
model is depicted (for other models see
appendix Section [A.5.1.4](#A1.SS5.SSS1.P4 "A.5.1.4 Learning Dynamics of Transformer and BERT Models. ‣ A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
Operating classes (II) (large metastable states) and
(IV) (small metastable states)
are often observed in the middle layers.
Operating class (I) (averaging over a very large number of patterns)
is abundant in lower layers.
Similar observations have been reported in other studies (Toneva & Wehbe, [2019a](#bib.bib93); [b](#bib.bib94); Tay et al., [2020](#bib.bib92)).
Operating class (III) (medium metastable states) is predominant in the last layers.

!(/html/2008.02217/assets/x8.png)

Figure A.3: Analysis of operating modes of the heads of a pre-trained BERT model.
For each head in each layer, the distribution of the minimal number
k𝑘k of patterns required to sum up the softmaxsoftmax\mathrm{softmax} values to 0.900.900.90 is displayed as a violin plot in a panel.
k𝑘k indicates the size of a metastable state.
The bold number in the center of each panel gives the median k¯¯𝑘\bar{k} of the distribution.
The heads in each layer are sorted according to k¯¯𝑘\bar{k}.
Attention heads belong to the class they mainly operate in.
Class (IV) in blue:
Small metastable state or fixed point close to a single pattern, which
is abundant in the middle layers (6, 7, and 8).
Class (II) in orange: Large metastable state, which is
prominent in middle layers (3, 4, and 5).
Class (I) in red: Very large metastable state or global fixed point,
which is predominant in the first layer.
These heads can potentially be replaced by averaging operations.
Class (III) in green: Medium metastable state,
which is frequently observed in higher layers.
We hypothesize that these heads are used to collect information required to
perform the respective task.
These heads should be the main target to improve transformer and BERT models.

##### A.5.1.2 Experimental Setup.

Transformer architectures are known for their high computational demands.
To investigate the learning dynamics of such a model and at the same time keeping training time manageable, we adopted the BERT-small setting from ELECTRA (Clark et al., [2020](#bib.bib25)). It has 121212 layers, 444 heads and
a reduced hidden size, the sequence length is shortened from 512512512 to 128128128 tokens and the batch size is reduced from
256256256 to 128128128. Additionally, the hidden dimension is reduced from 768768768 to 256256256 and the embedding dimension is
reduced from 768768768 to 128128128 (Clark et al., [2020](#bib.bib25)). The training of such a BERT-small model for 1.451.451.45 million update steps takes roughly four days on a single NVIDIA V100 GPU.

As the code base we use the transformers repository from Hugging Face, Inc (Wolf et al., [2019](#bib.bib107)).
We aim to reproduce the dataset of Devlin et al. ([2019](#bib.bib33)) as close as possible, which consists of
the English Wikipedia dataset and the Toronto BookCorpus dataset (Zhu et al., [2015](#bib.bib119)). Due to recent copyright claims the later is not publicly available anymore. Therefore, the pre-training experiments use an uncased snapshot of the original BookCorpus dataset.

##### A.5.1.3 Hopfield Operating Classes of Transformer and BERT Models.

To better understand how operation modes in attention heads develop, we tracked the distribution of counts k𝑘k (see main paper) over time in a BERT-small model.
At the end of training we visualized the count distribution, grouped into four classes (see Figure [A.4](#A1.F4 "Figure A.4 ‣ A.5.1.4 Learning Dynamics of Transformer and BERT Models. ‣ A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
The thresholds for the classes were chosen according to the thresholds of Figure 2 in the main paper.
However, they are divided by a factor of 444 to adapt to the shorter sequence length of 128128128 compared to 512512512.
From this plot it is clear, that the attention in heads of Class IV commit very early to the operating class of small metastable states.

##### A.5.1.4 Learning Dynamics of Transformer and BERT Models.

To observe this behavior in the early phase of training, we created a ridge plot of the distributions of counts k𝑘k for the
first 20,000

2000020,000 steps (see Figure [A.5](#A1.F5 "Figure A.5 ‣ A.5.1.4 Learning Dynamics of Transformer and BERT Models. ‣ A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") (a)). This plot shows that the attention in heads of middle layers often
change the operation mode to Class IV around 9,000

90009,000 to 10,000

1000010,000 steps. At the same time the second big drop in the loss occurs.
The question arises whether this is functionally important or whether it is an artefact which could be even
harmful. To check if the attention mechanism is still able to learn after the change in the operation mode
we analyzed the gradient flow through the softmaxsoftmax\mathrm{softmax} function.
For every token we calculate the Frobenius norm of the Jacobian of the softmaxsoftmax\mathrm{softmax} over multiple samples.
Then, for every head we plot the distribution of the norm (see Figure [A.5](#A1.F5 "Figure A.5 ‣ A.5.1.4 Learning Dynamics of Transformer and BERT Models. ‣ A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")(b)).
The gradients with respect to the weights are determined by the Jacobian JJ\mathrm{J} defined in Eq. ([59](#A1.E59 "In A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
as can be seen in Eq. ([418](#A1.E418 "In A.1.7.2 Learning an Association Matrix – Only One Set is Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), Eq. ([429](#A1.E429 "In A.1.7.3 Learning Two Association Mappings – Both Sets are Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), and Eq. ([435](#A1.E435 "In A.1.7.3 Learning Two Association Mappings – Both Sets are Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
We can see that
the attention
in heads of Class IV remain almost unchanged during the rest of the training.

!(/html/2008.02217/assets/x9.png)

Figure A.4: Left: Ridge plots of the distribution of counts k𝑘k over time for BERT-small Right: Violin plot of counts k𝑘k after 1,450000

14500001,450000 steps, divided into the four classes from the main paper. The thresholds were adapted to the shorter sequence length.

!(/html/2008.02217/assets/x10.png)

(a) Densities

!(/html/2008.02217/assets/x11.png)

(b) Norm of Jacobian

Figure A.5: (a): change of count density during training is depicted for the first 20,000

2000020,000 steps.
(b): the corresponding distribution of the Frobenius norm of the Jacobian of the softmaxsoftmax\mathrm{softmax} function
is depicted.
The gradients with respect to the weights are determined by the Jacobian JJ\mathrm{J} defined in Eq. ([59](#A1.E59 "In A.1.5.1 General Bound on the Jacobian of the Iteration. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
as can be seen in Eq. ([418](#A1.E418 "In A.1.7.2 Learning an Association Matrix – Only One Set is Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), Eq. ([429](#A1.E429 "In A.1.7.3 Learning Two Association Mappings – Both Sets are Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), and Eq. ([435](#A1.E435 "In A.1.7.3 Learning Two Association Mappings – Both Sets are Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).

##### A.5.1.5 Attention Heads Replaced by Gaussian Averaging Layers.

The self-attention mechanism proposed in Vaswani et al. ([2017a](#bib.bib96)) utilizes the softmaxsoftmax\mathrm{softmax} function to compute
the coefficients of a convex combination over the embedded tokens, where the softmaxsoftmax\mathrm{softmax} is conditioned on the input.
However, our analysis showed that especially in lower layers many heads perform averaging over a very
large number of patterns. This suggests that at this level neither the dependency on the input
nor a fine grained attention to individual positions is necessary. As an alternative to the original
mechanism we propose Gaussian averaging heads which
are computationally more efficient. Here, the softmaxsoftmax\mathrm{softmax} function is replaced by a discrete Gaussian kernel,
where the location μ𝜇\mu and the scale σ𝜎\sigma are learned. In detail, for a sequence length of
N𝑁N tokens we are given a vector of location parameters 𝝁=(μ1,…,μN)T𝝁superscriptsubscript𝜇1…subscript𝜇𝑁𝑇\bm{\mu}=(\mu\_{1},\ldots,\mu\_{N})^{T} and a vector of
corresponding scale parameters 𝝈=(σ1,…,σN)T𝝈superscriptsubscript𝜎1…subscript𝜎𝑁𝑇\bm{\sigma}=(\sigma\_{1},\ldots,\sigma\_{N})^{T}.
We subdivide the interval [−1,1]11[-1,1] into N𝑁N equidistant supporting points
{sj}j=1Nsuperscriptsubscriptsubscript𝑠𝑗𝑗1𝑁\{s\_{j}\}\_{j=1}^{N}, where

|  |  |  |
| --- | --- | --- |
|  | sj=(j−1)−0.5​(N−1)0.5​(N−1).subscript𝑠𝑗𝑗10.5𝑁10.5𝑁1\displaystyle s\_{j}=\frac{(j-1)-0.5~{}(N-1)}{0.5~{}(N-1)}. |  |

The attention [A]i,jsubscriptdelimited-[]𝐴

𝑖𝑗[A]\_{i,j} from the i𝑖i-th token to the j𝑗j-th position is calculated as

|  |  |  |
| --- | --- | --- |
|  | [A]i,j=1zi​exp⁡{−12​(sj−μiσi)2},subscriptdelimited-[]𝐴  𝑖𝑗1subscript𝑧𝑖12superscriptsubscript𝑠𝑗subscript𝜇𝑖subscript𝜎𝑖2\displaystyle[A]\_{i,j}=\frac{1}{z\_{i}}\exp\left\{-\frac{1}{2}\big{(}\frac{s\_{j}-\mu\_{i}}{\sigma\_{i}}\big{)}^{2}\right\}, |  |

where zisubscript𝑧𝑖z\_{i} normalizes the i𝑖i-th row of the attention matrix A𝐴A to sum up to one:

|  |  |  |
| --- | --- | --- |
|  | zi=∑j=1Nexp⁡{−12​(sj−μiσi)2}.subscript𝑧𝑖superscriptsubscript𝑗1𝑁12superscriptsubscript𝑠𝑗subscript𝜇𝑖subscript𝜎𝑖2\displaystyle z\_{i}=\sum\_{j=1}^{N}\exp\left\{-\frac{1}{2}\big{(}\frac{s\_{j}-\mu\_{i}}{\sigma\_{i}}\big{)}^{2}\right\}. |  |

For initialization we uniformly sample a location vector 𝝁∈[−1,1]N𝝁superscript11𝑁\bm{\mu}\in[-1,1]^{N} and
a scale vector 𝝈∈[0.75,1.25]N𝝈superscript0.751.25𝑁\bm{\sigma}\in[0.75,1.25]^{N} per head. A simple way to consider the individual
position of each token at initialization is to use the supporting points μi=sisubscript𝜇𝑖subscript𝑠𝑖\mu\_{i}=s\_{i}
(see Figure [A.6](#A1.F6 "Figure A.6 ‣ A.5.1.5 Attention Heads Replaced by Gaussian Averaging Layers. ‣ A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")). In practice no difference to the random initialization was observed.

•Number of parameters.
Gaussian averaging heads can reduce the number of parameters significantly.
For an input size of N𝑁N tokens, there are 2⋅N⋅2𝑁2\cdot N parameters per head.
In contrast, a standard self-attention head
with word embedding dimension dysubscript𝑑𝑦d\_{y} and projection dimension dksubscript𝑑𝑘d\_{k} has two weight matrices
WQ,WK∈ℝdk×dy

subscript𝑊𝑄subscript𝑊𝐾
superscriptℝsubscript𝑑𝑘subscript𝑑𝑦W\_{Q},W\_{K}\in\mathbb{R}^{d\_{k}\times d\_{y}}, which together amount to 2⋅dk⋅dy⋅2subscript𝑑𝑘subscript𝑑𝑦2\cdot d\_{k}\cdot d\_{y} parameters.
As a concrete example, the BERT-base model from Devlin et al. ([2019](#bib.bib33)) has an embedding dimension dy=768subscript𝑑𝑦768d\_{y}=768,
a projection dimension dk=64subscript𝑑𝑘64d\_{k}=64 and a sequence length of N=512𝑁512N=512.
Compared to the Gaussian head, in this case (2⋅768⋅64)/(2⋅512)=95.5⋅276864⋅251295.5(2\cdot 768\cdot 64)/(2\cdot 512)~{}=95.5 times more
parameters are trained for the attention mechanism itself.
Only for very long sequences (and given
that the word embedding dimension stays the same) the dependence on N𝑁N may become a disadvantage.
But of course, due to the independence from the input the Gaussian averaging head is less expressive
in comparison to the original attention mechanism.
A recently proposed input independent replacement for self-attention is the so called Random Synthesizer
(Tay et al., [2020](#bib.bib92)). Here the softmaxsoftmax\mathrm{softmax}-attention is directly parametrized with an N×N𝑁𝑁N\times N
matrix. This amounts to 0.5⋅N⋅0.5𝑁0.5\cdot N more parameters than Gaussian averaging.

!(/html/2008.02217/assets/x12.png)

Figure A.6: Attentions of a Gaussian averaging head at initialization for sequence length N=128𝑁128N=128. Every line depicts one Gaussian kernel. Here, the location parameters are initialized with the value of the supporting points μi=sisubscript𝜇𝑖subscript𝑠𝑖\mu\_{i}=s\_{i}.

#### A.5.2 Experiment 2: Multiple Instance Learning Datasets.

##### A.5.2.1 Immune Repertoire Classification.

An architecture called DeepRC,
is based on our modern Hopfield networks, for
immune repertoire classification and compared to
other machine learning approaches.
For DeepRC, we consider immune repertoires as input objects,
which are represented as bags of instances. In a bag, each instance is
an immune receptor sequence and each bag can contain a large number of sequences.
At its core, DeepRC consists of a modern
Hopfield network that
extracts information from each repertoire.
The stored patterns (keys) are representations of the immune amino acid
sequences (instances) that are obtained by an 1D convolutional network
with position encoding.
Each state pattern (query) is static and learned via backpropagation.
For details see Widrich et al. ([2020a](#bib.bib105); [b](#bib.bib106)).

Our new Hopfield network has
been integrated into a deep learning architecture
for immune repertoire classification, a massive
multiple instance learning task (Widrich et al., [2020a](#bib.bib105); [b](#bib.bib106)).
Theorem [3](#Thmtheorem3 "Theorem 3. ‣ New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need") states that modern Hopfield networks possess an exponential
storage capacity which enables to tackle massive multiple instance learning (MIL)
problems (Dietterich et al., [1997](#bib.bib34)).
Immune repertoire classification (Emerson et al., [2017](#bib.bib35))
typically requires
to extract few patterns from a large set of sequences, the repertoire,
that are indicative for the respective immune status.
Most MIL methods fail due the large number of instances.

Data is obtained by
experimentally observed immune receptors as well as simulated sequences
sequence motifs (Akbar et al., [2019](#bib.bib3); Weber et al., [2020](#bib.bib103)) with low yet varying degrees
of frequency are implanted.
Four different categories of
datasets are constructed:
(a) Simulated immunosequencing data with implanted motifs,
(b) immunosequencing data generated by long short-term memory (LSTM) with
implanted motifs, (c) real-world immunosequencing data with implanted motifs, and (d)
real-world immunosequencing data with known immune
status (Emerson et al., [2017](#bib.bib35)). Categories (a), (b), and (d) contain approx.
300,000 instances per immune repertoire. With over 30 billion sequences in total, this
represents one of the largest multiple instance
learning experiments ever conducted (Carbonneau et al., [2018](#bib.bib17)).
Despite the massive number of instances as well as the low frequency of sequences
indicative of the respective immune status, deep learning
architectures with modern Hopfield networks outperform
all competing methods with respect to average area under the ROC curve in all four
categories, (a), (b), (c) and (d) (for details see Widrich et al. ([2020a](#bib.bib105))).

We evaluate and compare the performance of
DeepRC to a set of machine learning methods that
serve as baseline, were suggested, or
can readily be adapted to immune repertoire
classification.
The methods comprise
(i) known motif,
which counts how often the known implanted motifs occur,
(ii) Support Vector Machine (SVM) approach
that uses a fixed mapping from a bag
of sequences to the corresponding k𝑘k-mer counts
and used the MinMax and Jaccard kernel,
(iii) k𝑘k-Nearest Neighbor (KNN) with k𝑘k-mer representation,
transforming MinMax and Jaccard kernel to distances,
(iv) logistic regression on the k𝑘k-mer representation,
(v) burden test that first identifies sequences or k𝑘k-mers and
then computes a burden
score per individual, and
(vi) logistic multiple instance learning (lMIL).
On the real-world dataset DeepRC achieved
an AUC of 0.832±0.022plus-or-minus0.8320.0220.832\pm 0.022, followed by
the SVM with MinMax kernel (AUC 0.825±0.022plus-or-minus0.8250.0220.825\pm 0.022) and the burden
test with an AUC of 0.699±0.041plus-or-minus0.6990.0410.699\pm 0.041.
Overall on all datasets, DeepRC outperformed all competing methods
with respect to average AUC
(see Widrich et al. ([2020a](#bib.bib105); [b](#bib.bib106))).

Table [A.1](#A1.T1 "Table A.1 ‣ A.5.2.1 Immune Repertoire Classification. ‣ A.5.2 Experiment 2: Multiple Instance Learning Datasets. ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") reports the average
performance in the simulated immunosequencing datasets
(last column) and the performance on datasets of
the remaining three categories.
DeepRC outperforms all competing methods
with respect to average AUC.
Across categories, the runner-up methods are either the SVM for
MIL problems with MinMax kernel or the burden test.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Real-world | Real-world data with implanted signals | | | | LSTM-generated data | | | | | Simulated |
|  | CMV | OM 1% | OM 0.1% | MM 1% | MM 0.1% | 10% | 1% | 0.5% | 0.1% | 0.05% | avg. |
| DeepRC | 0.832 ±plus-or-minus\pm 0.022 | 1.00 ±plus-or-minus\pm 0.00 | 0.98±plus-or-minus\pm 0.01 | 1.00±plus-or-minus\pm 0.00 | 0.94±plus-or-minus\pm0.01 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 0.846±plus-or-minus\pm 0.223 |
| SVM (MM) | 0.825 ±plus-or-minus\pm 0.022 | 1.00 ±plus-or-minus\pm 0.00 | 0.58±plus-or-minus\pm 0.02 | 1.00±plus-or-minus\pm 0.00 | 0.53±plus-or-minus\pm0.02 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 0.99±plus-or-minus\pm 0.01 | 0.827±plus-or-minus\pm 0.210 |
| SVM (J) | 0.546 ±plus-or-minus\pm 0.021 | 0.99 ±plus-or-minus\pm 0.00 | 0.53±plus-or-minus\pm 0.02 | 1.00±plus-or-minus\pm 0.00 | 0.57±plus-or-minus\pm0.02 | 0.98±plus-or-minus\pm 0.04 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 0.90±plus-or-minus\pm 0.04 | 0.77±plus-or-minus\pm 0.07 | 0.550±plus-or-minus\pm 0.080 |
| KNN (MM) | 0.679 ±plus-or-minus\pm 0.076 | 0.74 ±plus-or-minus\pm 0.24 | 0.49±plus-or-minus\pm 0.03 | 0.67±plus-or-minus\pm 0.18 | 0.50±plus-or-minus\pm0.02 | 0.70±plus-or-minus\pm 0.27 | 0.72±plus-or-minus\pm 0.26 | 0.73±plus-or-minus\pm 0.26 | 0.54±plus-or-minus\pm 0.16 | 0.52±plus-or-minus\pm 0.15 | 0.634±plus-or-minus\pm 0.129 |
| KNN (J) | 0.534 ±plus-or-minus\pm 0.039 | 0.65 ±plus-or-minus\pm 0.16 | 0.48±plus-or-minus\pm 0.03 | 0.70±plus-or-minus\pm 0.20 | 0.51±plus-or-minus\pm0.03 | 0.70±plus-or-minus\pm 0.29 | 0.61±plus-or-minus\pm 0.24 | 0.52±plus-or-minus\pm 0.16 | 0.55±plus-or-minus\pm 0.19 | 0.54±plus-or-minus\pm 0.19 | 0.501±plus-or-minus\pm 0.007 |
| Log. regr. | 0.607 ±plus-or-minus\pm 0.058 | 1.00 ±plus-or-minus\pm 0.00 | 0.54±plus-or-minus\pm 0.04 | 0.99±plus-or-minus\pm 0.00 | 0.51±plus-or-minus\pm0.04 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 0.93±plus-or-minus\pm 0.15 | 0.60±plus-or-minus\pm 0.19 | 0.43±plus-or-minus\pm 0.16 | 0.826±plus-or-minus\pm 0.211 |
| Burden test | 0.699 ±plus-or-minus\pm 0.041 | 1.00 ±plus-or-minus\pm 0.00 | 0.64±plus-or-minus\pm 0.05 | 1.00±plus-or-minus\pm 0.00 | 0.89±plus-or-minus\pm0.02 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 0.79±plus-or-minus\pm 0.28 | 0.549±plus-or-minus\pm 0.074 |
| Log. MIL (KMER) | 0.582 ±plus-or-minus\pm 0.065 | 0.54 ±plus-or-minus\pm 0.07 | 0.51±plus-or-minus\pm 0.03 | 0.99±plus-or-minus\pm 0.00 | 0.62±plus-or-minus\pm0.15 | 1.00±plus-or-minus\pm 0.00 | 0.72±plus-or-minus\pm 0.11 | 0.64±plus-or-minus\pm 0.14 | 0.57±plus-or-minus\pm 0.15 | 0.53±plus-or-minus\pm 0.13 | 0.665±plus-or-minus\pm 0.224 |
| Log. MIL (TCR\textbeta) | 0.515 ±plus-or-minus\pm 0.073 | 0.50 ±plus-or-minus\pm 0.03 | 0.50±plus-or-minus\pm 0.02 | 0.99±plus-or-minus\pm 0.00 | 0.78±plus-or-minus\pm0.03 | 0.54±plus-or-minus\pm 0.09 | 0.57±plus-or-minus\pm 0.16 | 0.47±plus-or-minus\pm 0.09 | 0.51±plus-or-minus\pm 0.07 | 0.50±plus-or-minus\pm 0.12 | 0.501±plus-or-minus\pm 0.016 |
| Known motif b. | – | 1.00 ±plus-or-minus\pm 0.00 | 0.70±plus-or-minus\pm 0.03 | 0.99±plus-or-minus\pm 0.00 | 0.62±plus-or-minus\pm0.04 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 0.890±plus-or-minus\pm 0.168 |
| Known motif c. | – | 0.92 ±plus-or-minus\pm 0.00 | 0.56±plus-or-minus\pm 0.03 | 0.65±plus-or-minus\pm 0.03 | 0.52±plus-or-minus\pm0.03 | 1.00±plus-or-minus\pm 0.00 | 1.00±plus-or-minus\pm 0.00 | 0.99±plus-or-minus\pm 0.01 | 0.72±plus-or-minus\pm 0.09 | 0.63±plus-or-minus\pm 0.09 | 0.738±plus-or-minus\pm 0.202 |

Table A.1: Results immune repertoire classification across all datasets. Results are given in terms of AUC of the competing methods on all datasets. The reported errors are standard deviations across 555 cross-validation (CV) folds (except for the column “Simulated”).
Real-world CMV: Average performance over
555 CV folds on the *cytomegalovirus (CMV) dataset* Emerson et al. ([2017](#bib.bib35)).
Real-world data with implanted signals: Average performance over
555 CV folds for each
of the four datasets. A signal
was implanted with a frequency (=wittness rate) of
1%percent11\% or 0.1%percent0.10.1\%. Either a single motif (“OM”)
or multiple motifs (“MM”) were implanted.
LSTM-generated data: Average performance over
555 CV folds for each
of the 555 datasets. In each dataset, a signal
was implanted with a frequency of 10%percent1010\%,
1%percent11\%, 0.5%percent0.50.5\%, 0.1%percent0.10.1\%, and 0.05%percent0.050.05\%, respectively.
Simulated: Here we report the mean over 18 simulated datasets with implanted signals and varying difficulties. The error reported is the standard deviation of the AUC values across the 18 datasets.

##### A.5.2.2 Multiple Instance Learning Benchmark Datasets.

Classical benchmarking datasets comprise
UCSB breast cancer classification (Kandemir et al., [2014](#bib.bib52)), and
the Elephant, Fox, Tiger datasets (Andrews et al., [2003](#bib.bib5)).

Elephant, Fox and Tiger are MIL datasets for image annotation which comprise
color images from the Corel dataset that have been preprocessed and
segmented. An image consists
of a set of segments (or blobs),
each characterized by color, texture and shape
descriptors.
The datasets have 100 positive and 100 negative
example images.
The latter have been randomly drawn from a pool of photos of
other animals.
Elephant has 1391 instances and 230 features.
Fox has 1320 instances and 230 features.
Tiger has 1220 instances and 230 features.
Furthermore, we use the
UCSB breast cancer classification (Kandemir et al., [2014](#bib.bib52))
dataset, which consists of 2,002 instances across 58 input objects. An instance
represents a patch of a histopathological image of cancerous or normal tissue.
The layer HopfieldPooling is used, which allows for
computing a per-input-object representation by extracting an
average of instances that are indicative for one of the two classes.
The input to the HopfieldPooling layer
is a set of embedded instances 𝒀𝒀\bm{Y} and
a trainable but fixed state (query) pattern 𝑸𝑸\bm{Q} used for averaging of class-indicative instances.
This averaging enables a compression of variable-sized bags to a
fixed-sized representation to discriminate the bags.
We performed a manual hyperparameter search on a validation set.
In detail, we used the following architecture to perform the given task
on the Elephant, Fox, Tiger and UCSCB breast cancer datasets:
(I) we apply fully connected linear embedding layers
with ReLU activation. (II) The output of this embedding serves
as the input to our HopfieldPooling layer where the above described
pooling operation is performed. (III) Thereafter we use ’ReLU - Linear blocks’
as the final linear output layers that perform the classification.
Among other hyperparameters, different hidden layer widths
(for the fully connected pre- and post-HopfieldPooling layers),
learning rates and batch sizes were tried.
Additionally our focus resided on the hyperparameters of the HopfieldPooling layer. Among those were the number of heads,
the head dimension and the scaling factor \textbeta.

| parameter | values |
| --- | --- |
| learning rates | {10−3\{10^{-3}, 10−5}10^{-5}\} |
| learning rate decay (\textgamma) | {0.98,0.96,0.94}0.980.960.94\{0.98,0.96,0.94\} |
| embedding layers | {1,2,3}123\{1,2,3\} |
| layer widths | {32,64,256,1024,2048}326425610242048\{32,64,256,1024,2048\} |
| number of heads | {8,12,16,32}8121632\{8,12,16,32\} |
| head dimensions | {16,32,64}163264\{16,32,64\} |
| scaling factors | {0.1,1.0,10.0}0.11.010.0\{0.1,1.0,10.0\} |
| hidden dimensions | {32,64,128}3264128\{32,64,128\} |
| bag dropout | {0.0,0.75}0.00.75\{0.0,0.75\} |

Table A.2: Hyperparameter search-space of a manual hyperparameter selection on the respective validation sets of the Elephant, Fox, Tiger and UCSB breast cancer datasets.

All models were trained for 160 epochs using the AdamW optimizer (Loshchilov & Hutter, [2017](#bib.bib66)) with exponential
learning rate decay (see Table [A.2](#A1.T2 "Table A.2 ‣ A.5.2.2 Multiple Instance Learning Benchmark Datasets. ‣ A.5.2 Experiment 2: Multiple Instance Learning Datasets. ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")), and validated by 10-fold nested cross validation repeated five times with different splits on the data sets. The reported ROC AUC scores are the average of these repetitions. As overfitting imposed quite a problem, bag dropout was applied as the regularization technique of choice.

#### A.5.3 Experiment 3: Classification on Small UCI Benchmark Datasets

##### A.5.3.1 Motivation.

Datasets with a small number of samples, like the UCI benchmark datasets, are particularly difficult for neural networks to generalize on.
In contrast to their performance on larger datasets, they are consistently outperformed by methods like e.g. gradient boosting, random forests (RF) and support vector machines (SVMs).
Finding samples or even learning prototypes that are highly indicative for the class of a sample (query) suggest the use of Hopfield networks.
We applied a modern Hopfield network via the layer Hopfield.
The input vector is mapped to 𝑹𝑹\bm{R}
using a self-normalizing net (SNN) and 𝑾Ksubscript𝑾𝐾\bm{W}\_{K} is learned,
where the dimension of 𝑾Ksubscript𝑾𝐾\bm{W}\_{K} (the number of stored fixed pattern)
is a hyperparameter.
The output 𝒁𝒁\bm{Z} of Hopfield enters
the output layer.

##### A.5.3.2 Methods compared.

Modern Hopfield networks via the layer Hopfield
are compared to
17 groups of methods (Fernández-Delgado et al., [2014](#bib.bib36); Klambauer et al., [2017a](#bib.bib55)):

1. 1.

   Support Vector Machines
2. 2.

   Random Forest
3. 3.

   Multivariate adaptive regression splines (MARS)
4. 4.

   Boosting
5. 5.

   Rule-based Methods
6. 6.

   Logistic and Multinomial Regression (LMR)
7. 7.

   Discriminant Analysis (DA)
8. 8.

   Bagging
9. 9.

   Nearest Neighbor
10. 10.

    Decision Trees
11. 11.

    Other Ensembles
12. 12.

    Neural Networks (standard NN, BatchNorm, WeighNorm, MSRAinit, LayerNorm, ResNet, Self-Normalizing Nets)
13. 13.

    Bayesian Methods
14. 14.

    Other Methods
15. 15.

    Generalized linear models (GLM)
16. 16.

    Partial Least Squares and Principal Component Regression (PLSR)
17. 17.

    Stacking (Wolpert)

##### A.5.3.3 Experimental design and implementation details.

As specified in the main paper,
we consider 757575 datasets of the
UC Irvine Machine Learning Repository,
which contain less than 1,000

10001,000 samples per dataset,
following the dataset separation into large and small dataset in Klambauer et al. ([2017a](#bib.bib55)).
On each dataset,
we performed a grid-search to determine the best hyperparameter setting and model per dataset.
The hyperparameter search-space of the grid-search is listed in Table [A.3](#A1.T3 "Table A.3 ‣ A.5.3.3 Experimental design and implementation details. ‣ A.5.3 Experiment 3: Classification on Small UCI Benchmark Datasets ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
All models were trained for 100100100 epochs with a mini-batch size of 444 samples using the cross entropy loss and
the PyTorch SGD module for stochastic gradient descent without momentum and without weight decay or dropout.
After each epoch, the model accuracy was computed on a separated validation set.
Using early stopping,
the model with the best validation set accuracy averaged over 161616 consecutive epochs was selected as final model.
This final model was then evaluated against a separated test set to determine the accuracy,
as reported in Tables [2](#S4.T2 "Table 2 ‣ UCI Benchmark Collection. ‣ 4 Experiments ‣ Hopfield Networks is All You Need") and Table <uci_detailed_results.csv> in the supplemental materials.

As network architecture,
we use {0,1,7}017\{0,1,7\} fully connected embedding layers with SELU
Klambauer et al. ([2017a](#bib.bib55)) activation functions and
{32,128,1024}321281024\{32,128,1024\} hidden units per embedding layer.
These embedding layers are followed by the layer Hopfield.
The number of hidden units is also used as number of dimensions
for the Hopfield association space with a number of {1,32}132\{1,32\} heads.
The layer Hopfield is followed by a mapping to the output vector,
which has as dimension the number of classes.
Finally, the softmax function is applied to obtain the predicted probability for a class.

| parameter | values |
| --- | --- |
| learning rates | {0.05}0.05\{0.05\} |
| embedding layers | {0,1,7}017\{0,1,7\} |
| hidden units | {32,128,1024}321281024\{32,128,1024\} |
| heads | {1,32}132\{1,32\} |
| β𝛽\beta | {1.0,0.1,0.001}1.00.10.001\{1.0,0.1,0.001\} |
| # stored patterns | {1,8}⋅n​\_​c​l​a​s​s​e​s⋅18𝑛\_𝑐𝑙𝑎𝑠𝑠𝑒𝑠\{1,8\}\cdot n\\_classes |

Table A.3: Hyperparameter search-space for grid-search on small UCI benchmark datasets.
All models were trained for 100100100 epochs using stochastic gradient descent
with early stopping based on the validation set accuracy and
a minibatch size of 444 samples.
The number of stored patterns is depending on the number of target classes of the individual tasks.

##### A.5.3.4 Results.

We compared the performance of 25 methods based on their method rank.
For this we computed the rank per method per dataset based on the accuracy on the test set,
which was then averaged over all 75 datasets for each method to obtain the method rank.
For the baseline methods we used the scores summarized by (Klambauer et al., [2017a](#bib.bib55)).

#### A.5.4 Experiment 4: Drug Design Benchmark Datasets

##### A.5.4.1 Experimental design and implementation details.

We test Hopfield layers on 4 classification datasets from MoleculeNet (Wu et al., [2017](#bib.bib110)),
which are challenging for deep learning methods.
The first dataset is HIV, which
was introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen.
The second dataset is BACE, which
has IC50 measurements for
binding affinities of inhibitors (molecules) to the human β𝛽\beta-secretase 1 (BACE-1).
The third dataset is BBBP (blood-brain barrier permeability),
which stems from modeling and predicting
the blood-brain barrier permeability (Martins et al., [2012](#bib.bib68)).
The fourth dataset is SIDER (Side Effect Resource) Kuhn et al. ([2016](#bib.bib63)) and contains 1427 approved drugs.
These datasets represent four areas of modeling tasks in drug discovery,
concretely to develop accurate models for predicting
a) new anti-virals (HIV),
b) new protein inhibitors (BACE),
c) metabolic effects (BBBP), and
d) side effects of a chemical compound (SIDER).

We implemented a Hopfield layer HopfieldLayer,
in which we used the training-input as stored-pattern 𝒀𝒀\bm{Y} or key,
the training-label as pattern-projection 𝒀​𝑾V𝒀subscript𝑾𝑉\bm{Y}\bm{W}\_{V} or value and
the input as state-pattern 𝑹𝑹\bm{R} or query. As described in section [A.6](#A1.SS6 "A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") by concatenation of input 𝒛isubscript𝒛𝑖\bm{z}\_{i} and target 𝒕isubscript𝒕𝑖\bm{t}\_{i} the matrices 𝑾Ksubscript𝑾𝐾\bm{W}\_{K} and 𝑾Vsubscript𝑾𝑉\bm{W}\_{V}
can be designed such that inside the softmax the input 𝒛isubscript𝒛𝑖\bm{z}\_{i} is used and outside the softmax the target 𝒕isubscript𝒕𝑖\bm{t}\_{i}.

All hyperparameters were selected on separate validation sets and
we selected the model with the highest validation AUC
on five different random splits.

| parameter | values |
| --- | --- |
| beta | {0.0001,0.001,0.01,0.1,0.2,0.3}0.00010.0010.010.10.20.3\{0.0001,0.001,0.01,0.1,0.2,0.3\} |
| learning rates | {0.0002}0.0002\{0.0002\} |
| heads | {1,32,128,512}132128512\{1,32,128,512\} |
| dropout | {0.0,0.1,0.2}0.00.10.2\{0.0,0.1,0.2\} |
| state-pattern bias | {0.0,−0.1,−0.125,0.15,−0.2}0.00.10.1250.150.2\{0.0,-0.1,-0.125,0.15,-0.2\} |
| association-activation | {None, LeakyReLU } |
| state- and stored-pattern static | {False, True} |
| normalize state- and stored-pattern | {False, True} |
| normalize association projection | {False, True} |
| learnable stored-pattern | {False, True} |

Table A.4: Hyperparameter search-space for grid-search on HIV, BACE, BBBP and SIDER.
All models were trained if applicable for 444 epochs using
Adam and a batch size of 111 sample.

##### A.5.4.2 Results.

We compared the Hopfield layer Hopfieldlayer
to Support Vector Machines (SVMs) (Cortes & Vapnik, [1995](#bib.bib26); Schölkopf & Smola, [2002](#bib.bib85)),
Extreme Gradient Boosting (XGBoost) (Chen & Guestrin, [2016](#bib.bib21)),
Random Forest (RF) (Breiman, [2001](#bib.bib14)),
Deep Neural Networks (DNNs) (LeCun et al., [2015](#bib.bib64); Schmidhuber, [2015](#bib.bib84)), and to
graph neural networks (GNN) like
Graph Convolutional Networks (GCNs) (Kipf & Welling, [2016](#bib.bib54)),
Graph Attention Networks (GATs) (Velic̆ković et al., [2018](#bib.bib98)),
Message Passing Neural Networks (MPNNs) (Gilmer et al., [2017](#bib.bib40)), and
Attentive FP (Xiong et al., [2020](#bib.bib111)).
Our architecture with HopfieldLayer has reached state-of-the-art
for predicting side
effects on SIDER 0.672±0.019plus-or-minus0.6720.0190.672\pm 0.019
as well as for predicting β𝛽\beta-secretase BACE 0.902±0.023plus-or-minus0.9020.0230.902\pm 0.023.
See Table [A.5](#A1.T5 "Table A.5 ‣ A.5.4.2 Results. ‣ A.5.4 Experiment 4: Drug Design Benchmark Datasets ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") for all results, where the results
of other methods are taken from Jiang et al. ([2020](#bib.bib51)).

Table A.5: Results on drug design benchmark datasets. Predictive performance (ROCAUC) on test set as reported by Jiang et al. ([2020](#bib.bib51)) for 50 random splits

| Model | HIV | BACE | BBBP | SIDER |
| --- | --- | --- | --- | --- |
| SVM | 0.822±0.020plus-or-minus0.8220.0200.822\pm 0.020 | 0.893±0.020plus-or-minus0.8930.0200.893\pm 0.020 | 0.919±0.028plus-or-minus0.9190.0280.919\pm 0.028 | 0.630±0.021plus-or-minus0.6300.0210.630\pm 0.021 |
| XGBoost | 0.816±0.020plus-or-minus0.8160.0200.816\pm 0.020 | 0.889±0.021plus-or-minus0.8890.0210.889\pm 0.021 | 0.926±0.026plus-or-minus0.9260.026\mathbf{0.926\pm 0.026} | 0.642±0.020plus-or-minus0.6420.0200.642\pm 0.020 |
| RF | 0.820±0.016plus-or-minus0.8200.0160.820\pm 0.016 | 0.890±0.022plus-or-minus0.8900.0220.890\pm 0.022 | 0.927±0.025plus-or-minus0.9270.025\mathbf{0.927\pm 0.025} | 0.646±0.022plus-or-minus0.6460.0220.646\pm 0.022 |
| GCN | 0.834±0.025plus-or-minus0.8340.025\mathbf{0.834\pm 0.025} | 0.898±0.019plus-or-minus0.8980.0190.898\pm 0.019 | 0.903±0.027plus-or-minus0.9030.0270.903\pm 0.027 | 0.634±0.026plus-or-minus0.6340.0260.634\pm 0.026 |
| GAT | 0.826±0.030plus-or-minus0.8260.0300.826\pm 0.030 | 0.886±0.023plus-or-minus0.8860.0230.886\pm 0.023 | 0.898±0.033plus-or-minus0.8980.0330.898\pm 0.033 | 0.627±0.024plus-or-minus0.6270.0240.627\pm 0.024 |
| DNN | 0.797±0.018plus-or-minus0.7970.0180.797\pm 0.018 | 0.890±0.024plus-or-minus0.8900.0240.890\pm 0.024 | 0.898±0.033plus-or-minus0.8980.0330.898\pm 0.033 | 0.627±0.024plus-or-minus0.6270.0240.627\pm 0.024 |
| MPNN | 0.811±0.031plus-or-minus0.8110.0310.811\pm 0.031 | 0.838±0.027plus-or-minus0.8380.0270.838\pm 0.027 | 0.879±0.037plus-or-minus0.8790.0370.879\pm 0.037 | 0.598±0.031plus-or-minus0.5980.0310.598\pm 0.031 |
| Attentive FP | 0.822±0.026plus-or-minus0.8220.0260.822\pm 0.026 | 0.876±0.023plus-or-minus0.8760.0230.876\pm 0.023 | 0.887±0.032plus-or-minus0.8870.0320.887\pm 0.032 | 0.623±0.026plus-or-minus0.6230.0260.623\pm 0.026 |
| Hopfield (ours) | 0.815±0.023plus-or-minus0.8150.0230.815\pm 0.023 | 0.902±0.023plus-or-minus0.9020.023\mathbf{0.902\pm 0.023} | 0.910±0.026plus-or-minus0.9100.0260.910\pm 0.026 | 0.672±0.019plus-or-minus0.6720.019\mathbf{0.672\pm 0.019} |

### A.6 PyTorch Implementation of Hopfield Layers

The implementation is available at: <https://github.com/ml-jku/hopfield-layers>

#### A.6.1 Introduction

In this section, we describe the implementation
of Hopfield layers in PyTorch (Paszke et al., [2017](#bib.bib73); [2019](#bib.bib74))
and, additionally, provide a brief usage manual.
Possible applications for a Hopfield layer
in a deep network architecture comprise:

* •

  multiple instance learning (MIL) (Dietterich et al., [1997](#bib.bib34)),
* •

  processing of and learning with point sets (Qi et al., [2017a](#bib.bib75); [b](#bib.bib76); Xu et al., [2018](#bib.bib112)),
* •

  set-based and permutation invariant learning (Guttenberg et al., [2016](#bib.bib42); Ravanbakhsh et al., [2016](#bib.bib79); Zaheer et al., [2017](#bib.bib115); Korshunova et al., [2018](#bib.bib58); Ilse et al., [2018](#bib.bib49); Zhai et al., [2020](#bib.bib117)),
* •

  attention-based learning (Vaswani et al., [2017a](#bib.bib96)),
* •

  associative learning,
* •

  natural language processing,
* •

  sequence analysis and time series prediction, and
* •

  storing and retrieving reference or experienced data, e.g. to store training data and retrieve it by the model
  or to store experiences for reinforcement learning.

The Hopfield layer in a deep neural network architecture can implement:

* •

  a memory (storage) with associative retrieval (Danihelka et al., [2016](#bib.bib28); Ba et al., [2016a](#bib.bib6)),
* •

  conditional pooling and averaging operations (Wang et al., [2018](#bib.bib102); Ilse et al., [2020](#bib.bib50)),
* •

  combining data by associations (Agrawal et al., [1993](#bib.bib2)),
* •

  associative credit assignment (e.g. Rescorla-Wagner model or value estimation) (Sutton & Barto, [2018](#bib.bib90)),
  and
* •

  attention mechanisms (Vaswani et al., [2017a](#bib.bib96); Bahdanau et al., [2014](#bib.bib8)).

In particular, a Hopfield layer can substitute
attention layers in architectures of transformer and BERT models.
The Hopfield layer
is designed to be used as plug-in replacement
for existing layers like

* •

  pooling layers (max-pooling or average pooling),
* •

  permutation equivariant layers (Guttenberg et al., [2016](#bib.bib42); Ravanbakhsh et al., [2016](#bib.bib79)),
* •

  GRU & LSTM layers, and
* •

  attention layers.

In contrast to classical Hopfield networks, the Hopfield layer is based
on the modern Hopfield networks with continuous states that have increased storage capacity,
as discussed in the main paper.
Like classical Hopfield networks,
the dynamics of the single heads of a Hopfield layer
follow a energy minimization dynamics.
The energy minimization empowers
our Hopfield layer with several advantages
over other architectural designs like memory cells, associative memory, or
attention mechanisms.
For example, the Hopfield layer has more functionality than
a transformer self-attention layer (Vaswani et al., [2017a](#bib.bib96)) as
described in Sec. [A.6.2](#A1.SS6.SSS2 "A.6.2 Functionality ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
Possible use cases are given in Sec. [A.6.3](#A1.SS6.SSS3 "A.6.3 Usage ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").
Source code will be provided under github.

#### A.6.2 Functionality

Non-standard functionalities that are added by a Hopfield layer are

* •

  Association of two sets,
* •

  Multiple Updates for precise fixed points,
* •

  Variable Beta that determines the kind of fixed points,
* •

  Dimension of the associative space for controlling the storage capacity,
* •

  Static Patterns for fixed pattern search,
  and
* •

  Pattern Normalization to control the fixed point dynamics by norm of
  the patterns and shift of the patterns.

A functional sketch of our Hopfield layer is shown in Fig. [A.7](#A1.F7 "Figure A.7 ‣ A.6.2 Functionality ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need").

•Association of two sets.
The Hopfield layer makes it possible
to associate two sets of vectors.
This general functionality allows

* •

  for transformer-like self-attention,
* •

  for decoder-encoder attention,
* •

  for time series prediction (maybe with positional encoding),
* •

  for sequence analysis,
* •

  for multiple instance learning,
* •

  for learning with point sets,
* •

  for combining data sources by associations,
* •

  for constructing a memory,
* •

  for averaging and pooling operations, and
* •

  for many more.

The first set of vectors consists of S𝑆S raw state patterns
𝑹=(𝒓1,…,𝒓S)T𝑹superscriptsubscript𝒓1…subscript𝒓𝑆𝑇\bm{R}=(\bm{r}\_{1},\ldots,\bm{r}\_{S})^{T} with 𝒓s∈ℝdrsubscript𝒓𝑠superscriptℝsubscript𝑑𝑟\bm{r}\_{s}\in\mathbb{R}^{d\_{r}}
and the second set of vectors consists of N𝑁N raw stored patterns
𝒀=(𝒚1,…,𝒚N)T𝒀superscriptsubscript𝒚1…subscript𝒚𝑁𝑇\bm{Y}=(\bm{y}\_{1},\ldots,\bm{y}\_{N})^{T} with 𝒚i∈ℝdysubscript𝒚𝑖superscriptℝsubscript𝑑𝑦\bm{y}\_{i}\in\mathbb{R}^{d\_{y}}.
Both the S𝑆S raw state patterns and N𝑁N raw stored patterns
are mapped to
an associative space in ℝdksuperscriptℝsubscript𝑑𝑘\mathbb{R}^{d\_{k}} via the matrices 𝑾Q∈ℝdr×dksubscript𝑾𝑄superscriptℝsubscript𝑑𝑟subscript𝑑𝑘\bm{W}\_{Q}\in\mathbb{R}^{d\_{r}\times d\_{k}}
and 𝑾K∈ℝdy×dksubscript𝑾𝐾superscriptℝsubscript𝑑𝑦subscript𝑑𝑘\bm{W}\_{K}\in\mathbb{R}^{d\_{y}\times d\_{k}}, respectively.
We define a matrix 𝑸𝑸\bm{Q} (𝚵Tsuperscript𝚵𝑇\bm{\Xi}^{T}) of state patterns 𝝃n=𝑾Q​𝒓nsubscript𝝃𝑛subscript𝑾𝑄subscript𝒓𝑛\bm{\xi}\_{n}=\bm{W}\_{Q}\bm{r}\_{n} in
an associative space ℝdksuperscriptℝsubscript𝑑𝑘\mathbb{R}^{d\_{k}} and a matrix 𝑲𝑲\bm{K} (𝑿Tsuperscript𝑿𝑇\bm{X}^{T}) of
stored patterns 𝒙i=𝑾K​𝒚ssubscript𝒙𝑖subscript𝑾𝐾subscript𝒚𝑠\bm{x}\_{i}=\bm{W}\_{K}\bm{y}\_{s} in the associative space ℝdksuperscriptℝsubscript𝑑𝑘\mathbb{R}^{d\_{k}}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑸𝑸\displaystyle\bm{Q}\ | =𝚵T=𝑹​𝑾Q,absentsuperscript𝚵𝑇𝑹subscript𝑾𝑄\displaystyle=\ \bm{\Xi}^{T}\ =\ \bm{R}\ \bm{W}\_{Q}\ , |  | (549) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑲𝑲\displaystyle\bm{K}\ | =𝑿T=𝒀​𝑾K.absentsuperscript𝑿𝑇𝒀subscript𝑾𝐾\displaystyle=\ \bm{X}^{T}\ =\ \bm{Y}\ \bm{W}\_{K}\ . |  | (550) |

In the main paper, Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need"))
defines the novel update rule:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝝃newsuperscript𝝃new\displaystyle\bm{\xi}^{\mathrm{new}}\ | =f​(𝝃)=𝑿​softmax​(β​𝑿T​𝝃),absent𝑓𝝃𝑿softmax𝛽superscript𝑿𝑇𝝃\displaystyle=\ f(\bm{\xi})\ =\ \bm{X}\ \mathrm{softmax}(\beta\ \bm{X}^{T}\bm{\xi})\ , |  | (551) |

For multiple patterns, Eq. ([3](#S2.E3 "In New energy function for continuous state Hopfield networks. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need")) becomes:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝚵newsuperscript𝚵new\displaystyle\bm{\Xi}^{\mathrm{new}}\ | =f​(𝚵)=𝑿​softmax​(β​𝑿T​𝚵),absent𝑓𝚵𝑿softmax𝛽superscript𝑿𝑇𝚵\displaystyle=\ f(\bm{\Xi})\ =\ \bm{X}\ \mathrm{softmax}(\beta\ \bm{X}^{T}\bm{\Xi})\ , |  | (552) |

where 𝚵=(𝝃1,…,𝝃N)𝚵subscript𝝃1…subscript𝝃𝑁\bm{\Xi}=(\bm{\xi}\_{1},\ldots,\bm{\xi}\_{N}) is the matrix
of N𝑁N state (query) patterns, 𝑿𝑿\bm{X}
is the matrix of stored (key) patterns,
and 𝚵newsuperscript𝚵new\bm{\Xi}^{\mathrm{new}}
is the matrix of new state patterns,
which are averages over stored patterns.
A new state pattern can also be very similar to a single stored pattern, in which case
we call the stored pattern to be retrieved.

These matrices allow to rewrite Eq. ([552](#A1.E552 "In A.6.2 Functionality ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) as:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (𝑸new)Tsuperscriptsuperscript𝑸new𝑇\displaystyle\left(\bm{Q}^{\mathrm{new}}\right)^{T}\ | =𝑲T​softmax​(β​𝑲​𝑸T).absentsuperscript𝑲𝑇softmax𝛽𝑲superscript𝑸𝑇\displaystyle=\ \bm{K}^{T}\mathrm{softmax}(\beta\ \bm{K}\ \bm{Q}^{T})\ . |  | (553) |

For β=1/dk𝛽1subscript𝑑𝑘\beta=1/\sqrt{d\_{k}} and changing in Eq. ([553](#A1.E553 "In A.6.2 Functionality ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
softmax∈ℝNsoftmaxsuperscriptℝ𝑁\mathrm{softmax}\in\mathbb{R}^{N} to a row vector (and evaluating a row vector), we obtain:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑸newsuperscript𝑸new\displaystyle\bm{Q}^{\mathrm{new}}\ | =softmax​(1/dk​𝑸​𝑲T)​𝑲,absentsoftmax1subscript𝑑𝑘𝑸superscript𝑲𝑇𝑲\displaystyle=\ \mathrm{softmax}(1/\sqrt{d\_{k}}\ \bm{Q}\ \bm{K}^{T})\ \bm{K}\ , |  | (554) |

where 𝑸newsuperscript𝑸new\bm{Q}^{\mathrm{new}} is again the matrix of new state patterns.
The new state patterns 𝚵newsuperscript𝚵new\bm{\Xi}^{\mathrm{new}} are projected
via 𝑾Vsubscript𝑾𝑉\bm{W}\_{V} to the result patterns 𝒁=𝚵new​𝑾V𝒁superscript𝚵newsubscript𝑾𝑉\bm{Z}=\bm{\Xi}^{\mathrm{new}}\bm{W}\_{V},
where 𝑾V∈ℝdk×dvsubscript𝑾𝑉superscriptℝsubscript𝑑𝑘subscript𝑑𝑣\bm{W}\_{V}\in\mathbb{R}^{d\_{k}\times d\_{v}}.
With the pattern projection 𝑽=𝑲​𝑾V𝑽𝑲subscript𝑾𝑉\bm{V}=\bm{K}\bm{W}\_{V}, we obtain
the update rule Eq. ([10](#S2.E10 "In Hopfield update rule is attention of the transformer. ‣ 2 Modern Hopfield Nets with Continuous States ‣ Hopfield Networks is All You Need")) from the main paper:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒁𝒁\displaystyle\bm{Z}\ | =softmax​(1/dk​𝑸​𝑲T)​𝑽.absentsoftmax1subscript𝑑𝑘𝑸superscript𝑲𝑇𝑽\displaystyle=\ \mathrm{softmax}(1/\sqrt{d\_{k}}\ \bm{Q}\ \bm{K}^{T})\ \bm{V}\ . |  | (555) |

•Multiple Updates.
The update Eq. ([553](#A1.E553 "In A.6.2 Functionality ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
can be iteratively applied to the initial state 𝝃𝝃\bm{\xi}
of every Hopfield layer head.
After the last update, the new states 𝚵newsuperscript𝚵new\bm{\Xi}^{\mathrm{new}} are projected
via 𝑾Vsubscript𝑾𝑉\bm{W}\_{V} to the result patterns 𝒁=𝚵new​𝑾V𝒁superscript𝚵newsubscript𝑾𝑉\bm{Z}=\bm{\Xi}^{\mathrm{new}}\bm{W}\_{V}.
Therefore, the Hopfield layer allows multiple update steps
in the forward pass without changing the number of parameters.
The number of update steps can be given for every Hopfield
head individually.
Furthermore, it is possible to set a threshold
for the number of updates of every Hopfield
head based on ‖𝝃−𝝃new‖2subscriptnorm𝝃superscript𝝃new2{{\left\|\bm{\xi}-\bm{\xi}^{\mathrm{new}}\right\|}}\_{2}.
In the general case of multiple initial states 𝚵𝚵\bm{\Xi}, the maximum over the individual norms is taken.

•Variable β𝛽\beta.
In the main paper,
we have identified β𝛽\beta as a crucial
parameter for the fixed point dynamics of the Hopfield network,
which governs the operating mode of the attention heads.
In appendix, e.g. in Lemma [A7](#ThmlemmaA7 "Lemma A7. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") or
in Eq. ([102](#A1.E102 "In A.1.5.2 One Stable State: Fixed Point Near the Mean of the Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) and Eq. ([103](#A1.E103 "In A.1.5.2 One Stable State: Fixed Point Near the Mean of the Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")),
we showed that the characteristics of the fixed points of the new
modern Hopfield network are
determined by: β𝛽\beta, M𝑀M (maximal pattern norm),
mmaxsubscript𝑚m\_{\max} (spread of the similar patterns), and ‖𝒎𝒙‖normsubscript𝒎𝒙{{\left\|\bm{m}\_{\bm{x}}\right\|}}
(center of the similar patterns).
Low values of β𝛽\beta induce global averaging and higher values of β𝛽\beta
metastable states.
In the transformer attention, the β𝛽\beta parameter
is set to β=1/dk𝛽1subscript𝑑𝑘\beta=1/\sqrt{d\_{k}} as in Eq. ([555](#A1.E555 "In A.6.2 Functionality ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")).
The Hopfield layer, however, allows to freely choose β>0𝛽0\beta>0, since the fixed
point dynamics does not only depend on the dimension of the associative space dksubscript𝑑𝑘d\_{k}.
Additionally, β𝛽\beta heavily influences the gradient flow to the matrices 𝑾Qsubscript𝑾𝑄\bm{W}\_{Q}
and 𝑾Ksubscript𝑾𝐾\bm{W}\_{K}. Thus, finding the right β𝛽\beta for the respective application
can be crucial.

•Variable dimension of the associative space.
Theorem [A5](#ThmtheoremA5 "Theorem A5 (Storage Capacity (Main): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") says that the storage capacity of
the modern Hopfield network grows exponentially with the
dimension of the associative space.
However higher dimension of the associative space
also means less averaging and smaller metastable states.
The dimension of the associative space trades off storage capacity against
the size of metastable states, e.g. over how many pattern is averaged.
In Eq. ([550](#A1.E550 "In A.6.2 Functionality ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) and in Eq. ([549](#A1.E549 "In A.6.2 Functionality ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")),
we assumed N𝑁N raw state patterns 𝑹=(𝒓1,…,𝒓N)T𝑹superscriptsubscript𝒓1…subscript𝒓𝑁𝑇\bm{R}=(\bm{r}\_{1},\ldots,\bm{r}\_{N})^{T} and
S𝑆S raw stored patterns 𝒀=(𝒚1,…,𝒚S)T𝒀superscriptsubscript𝒚1…subscript𝒚𝑆𝑇\bm{Y}=(\bm{y}\_{1},\ldots,\bm{y}\_{S})^{T}
that are mapped to
a dksubscript𝑑𝑘d\_{k}-dimensional associative space
via the matrices 𝑾Q∈ℝdr×dksubscript𝑾𝑄superscriptℝsubscript𝑑𝑟subscript𝑑𝑘\bm{W}\_{Q}\in\mathbb{R}^{d\_{r}\times d\_{k}} and
𝑾K∈ℝdy×dksubscript𝑾𝐾superscriptℝsubscript𝑑𝑦subscript𝑑𝑘\bm{W}\_{K}\in\mathbb{R}^{d\_{y}\times d\_{k}}, respectively.
In the associative space ℝdksuperscriptℝsubscript𝑑𝑘\mathbb{R}^{d\_{k}}, we obtain the state patterns
𝑸=𝚵T=𝑹​𝑾Q𝑸superscript𝚵𝑇𝑹subscript𝑾𝑄\bm{Q}=\bm{\Xi}^{T}=\bm{R}\bm{W}\_{Q} and
the stored patterns 𝑲=𝑿T=𝒀​𝑾K𝑲superscript𝑿𝑇𝒀subscript𝑾𝐾\bm{K}=\bm{X}^{T}=\bm{Y}\ \bm{W}\_{K}.
The Hopfield view relates
the dimension dksubscript𝑑𝑘d\_{k} to the number of input patterns N𝑁N that have to be processed.
The storage capacity depends exponentially on
the dimension dksubscript𝑑𝑘d\_{k} (the dimension of the associative space) and the
size to metastable states is governed by this dimension, too.
Consequently, dksubscript𝑑𝑘d\_{k} should be chosen with respect
to the number N𝑁N of patterns
one wants to store and the desired size of metastable states,
which is the number of patterns one wants to average over.
For example, if the input consists of many low dimensional input patterns,
it makes sense to project the patterns into a higher dimensional space to allow a proper fixed point dynamics.
Intuitively, this coincides with the construction of a
richer feature space for the patterns.

•Static Patterns.
In Eq. ([550](#A1.E550 "In A.6.2 Functionality ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) and Eq. ([549](#A1.E549 "In A.6.2 Functionality ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")),
the N𝑁N raw state patterns 𝑹=(𝒓1,…,𝒓N)T𝑹superscriptsubscript𝒓1…subscript𝒓𝑁𝑇\bm{R}=(\bm{r}\_{1},\ldots,\bm{r}\_{N})^{T}
and S𝑆S raw stored patterns 𝒀=(𝒚1,…,𝒚S)T𝒀superscriptsubscript𝒚1…subscript𝒚𝑆𝑇\bm{Y}=(\bm{y}\_{1},\ldots,\bm{y}\_{S})^{T}
are mapped to
an associative space via the matrices 𝑾Q∈ℝdr×dksubscript𝑾𝑄superscriptℝsubscript𝑑𝑟subscript𝑑𝑘\bm{W}\_{Q}\in\mathbb{R}^{d\_{r}\times d\_{k}}
and 𝑾K∈ℝdy×dksubscript𝑾𝐾superscriptℝsubscript𝑑𝑦subscript𝑑𝑘\bm{W}\_{K}\in\mathbb{R}^{d\_{y}\times d\_{k}}, which gives
the state patterns 𝑸=𝚵T=𝑹​𝑾Q𝑸superscript𝚵𝑇𝑹subscript𝑾𝑄\bm{Q}=\bm{\Xi}^{T}=\bm{R}\bm{W}\_{Q} and
the stored patterns 𝑲=𝑿T=𝒀​𝑾K𝑲superscript𝑿𝑇𝒀subscript𝑾𝐾\bm{K}=\bm{X}^{T}=\bm{Y}\ \bm{W}\_{K}.
We allow for static state and static stored patterns.
Static pattern means that the pattern does not depend on the
network input, i.e. it is determined by the bias weights and
remains constant across different network inputs.
Static state patterns allow to determine whether particular fixed patterns
are among the stored patterns and vice versa.
The static pattern functionality is typically needed if particular patterns
must be identified in the data, e.g. as described
for immune repertoire classification in the main paper,
where a fixed dksubscript𝑑𝑘d\_{k}-dimensional state vector 𝝃𝝃\bm{\xi} is used.

•Pattern Normalization.
In the appendix, e.g. in Lemma [A7](#ThmlemmaA7 "Lemma A7. ‣ A.1.5.3 Many Stable States: Fixed Points Near Stored Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") or
in Eq. ([102](#A1.E102 "In A.1.5.2 One Stable State: Fixed Point Near the Mean of the Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) and Eq. ([103](#A1.E103 "In A.1.5.2 One Stable State: Fixed Point Near the Mean of the Patterns. ‣ A.1.5 Local Convergence of the Update Rule: Fixed Point Iteration ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")),
we showed that the characteristics of the fixed points of the new
modern Hopfield network are
determined by: β𝛽\beta, M𝑀M (maximal pattern norm),
mmaxsubscript𝑚m\_{\max} (spread of the similar patterns), and ‖𝒎𝒙‖normsubscript𝒎𝒙{{\left\|\bm{m}\_{\bm{x}}\right\|}}
(center of the similar patterns).
We already discussed the parameter β𝛽\beta while
the spread of the similar patterns mmaxsubscript𝑚m\_{\max} is given by the data.
The remaining variables M𝑀M and 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}}
that both control the fixed point dynamics are
adjusted pattern normalization.
M𝑀M is the maximal pattern norm and
𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} the center of the similar patterns.
Theorem [A5](#ThmtheoremA5 "Theorem A5 (Storage Capacity (Main): Random Patterns). ‣ A.1.6.1 Exponentially Many Patterns can be Stored. ‣ A.1.6 Properties of Fixed Points Near Stored Pattern ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need") says
that larger M𝑀M allows for more patterns to
be stored. However, the size of metastable
states will decrease with increasing M𝑀M.
The vector 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}} says
how well the (similar) patterns are centered.
If the norm ‖𝒎𝒙‖normsubscript𝒎𝒙{{\left\|\bm{m}\_{\bm{x}}\right\|}} is large,
then this leads to smaller metastable states.
The two parameters M𝑀M and 𝒎𝒙subscript𝒎𝒙\bm{m}\_{\bm{x}}
are controlled by pattern normalization and determine
the size and convergence properties of metastable states.
These two parameters are important for creating large gradients
if heads start with global averaging which has small gradient.
These two parameters can shift a head towards
small metastable states which have
largest gradient as shown in Fig. [A.5](#A1.F5 "Figure A.5 ‣ A.5.1.4 Learning Dynamics of Transformer and BERT Models. ‣ A.5.1 Experiment 1: Attention in Transformers described by Hopfield dynamics ‣ A.5 Experiments ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")(b).
We allow for three different pattern normalizations, where the first is the default setting:

* •

  pattern normalization of the input patterns,
* •

  pattern normalization after mapping into the associative space,
* •

  no pattern normalization.

!(/html/2008.02217/assets/x13.png)

Figure A.7: 
A flowchart of the Hopfield layer.
First, the raw state (query) patterns
𝑹𝑹\bm{R} and the raw stored (key) patterns
𝒀𝒀\bm{Y} are optionally normalized (with layer normalization),
projected and optionally normalized (with layer normalization) again.
The default setting is a layer normalization of the input
patterns, and no layer normalization of the projected patterns.
The raw stored patterns 𝒀𝒀\bm{Y} can in principle be also two different input tensors.
Optionally, multiple updates take place in the
projected space of 𝑸𝑸\bm{Q} and 𝑲𝑲\bm{K}.
This update rule is obtained e.g. from the full update
Eq. ([423](#A1.E423 "In A.1.7.3 Learning Two Association Mappings – Both Sets are Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need")) or the simplified update
Eq. ([424](#A1.E424 "In A.1.7.3 Learning Two Association Mappings – Both Sets are Mapped in an Associative Space. ‣ A.1.7 Learning Associations ‣ A.1 Continuous State Modern Hopfield Networks (A New Concept) ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"))
in the appendix.

#### A.6.3 Usage

As outlined in Sec. [A.6.1](#A1.SS6.SSS1 "A.6.1 Introduction ‣ A.6 PyTorch Implementation of Hopfield Layers ‣ Appendix A Appendix ‣ Hopfield Networks is All You Need"), there
are a variety of possible use cases for the Hopfield layer,
e.g. to build memory networks or transformer models.
The goal of the implementation is therefore
to provide an easy to use Hopfield module
that can be used in a wide range of applications,
be it as part of a larger architecture or as a standalone module.
Consequently, the focus of the Hopfield layer interface is set
on its core parameters: the association of two sets,
the scaling parameter β𝛽\beta,
the maximum number of updates,
the dimension of the associative space,
the possible usage of static patterns,
and the pattern normalization.
The integration into the PyTorch framework is built such
that with all the above functionalities disabled,
the “HopfieldEncoderLayer” and
the “HopfieldDecoderLayer”, both extensions of the
Hopfield module,
can be used as a one-to-one plug-in replacement
for the TransformerEncoderLayer
and the TransformerDecoderLayer, respectively,
of the PyTorch transformer module.

The Hopfield layer can be used to implement or to substitute different layers:

* •

  Pooling layers: We consider the Hopfield layer as a pooling layer if only
  one static state (query) pattern exists. Then, it is de facto
  a pooling over the sequence, which results from the softmax values
  applied on the stored patterns.
  Therefore, our Hopfield layer can act as a pooling layer.
* •

  Permutation equivariant layers: Our Hopfield layer
  can be used as a plug-in replacement for permutation equivariant layers. Since the Hopfield layer is an associative memory it
  assumes no dependency between the input patterns.
* •

  GRU & LSTM layers: Our Hopfield layer can be used as a plug-in replacement for GRU & LSTM layers. Optionally, for substituting
  GRU & LSTM layers,
  positional encoding might be considered.
* •

  Attention layers: Our Hopfield layer can act as an attention layer, where state (query) and stored (key) patterns are different,
  and need to be associated.
* •

  Finally, the extensions of the Hopfield layer
  are able to operate as a self-attention layer
  (HopfieldEncoderLayer) and as cross-attention layer
  (HopfieldDecoderLayer),
  as described in (Vaswani et al., [2017a](#bib.bib96)).
  As such, it can be used as building block of
  transformer-based or general architectures.

## References

* Abu-Mostafa & StJacques (1985)

  Y. Abu-Mostafa and J.-M. StJacques.
  Information capacity of the Hopfield model.
  *IEEE Transactions on Information Theory*, 31, 1985.
  doi: 10.1109/tit.1985.1057069.
* Agrawal et al. (1993)

  R. Agrawal, T. Imieliundefinedski, and A. Swami.
  Mining association rules between sets of items in large databases.
  *SIGMOD Rec.*, 22(2):207–216, 1993.
  doi: 10.1145/170036.170072.
* Akbar et al. (2019)

  R. Akbar, P. A. Robert, M. Pavlović, J. R. Jeliazkov, I. Snapkov,
  A. Slabodkin, C. R. Weber, L. Scheffer, E. Miho, I. H. Haff, et al.
  A compact vocabulary of paratope-epitope interactions enables
  predictability of antibody-antigen binding.
  *bioRxiv*, 2019.
* Alzahrani & Salem (2018)

  F. Alzahrani and A. Salem.
  Sharp bounds for the lambert w𝑤w function.
  *Integral Transforms and Special Functions*, 29(12):971–978, 2018.
* Andrews et al. (2003)

  S. Andrews, I. Tsochantaridis, and T. Hofmann.
  Support vector machines for multiple-instance learning.
  In S. Becker, S. Thrun, and K. Obermayer (eds.), *Advances in
  Neural Information Processing Systems 15*, pp.  577–584. MIT Press, 2003.
* Ba et al. (2016a)

  J. Ba, G. E. Hinton, V. Mnih, J. Z. Leibo, and C. Ionescu.
  Using fast weights to attend to the recent past.
  In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett
  (eds.), *Advances in Neural Information Processing Systems 29*, pp. 4331–4339. Curran Associates, Inc., 2016a.
* Ba et al. (2016b)

  J. Ba, G. E. Hinton, V. Mnih, J. Z. Leibo, and C. Ionescu.
  Using fast weights to attend to the recent past.
  *ArXiv*, 1610.06258, 2016b.
* Bahdanau et al. (2014)

  D. Bahdanau, K. Cho, and Y. Bengio.
  Neural machine translation by jointly learning to align and
  translate.
  *ArXiv*, 1409.0473, 2014.
  appeared in ICRL 2015.
* Banino et al. (2020)

  A. Banino, A. P. Badia, R. Köster, M. J. Chadwick, V. Zambaldi,
  D. Hassabis, C. Barry, M. Botvinick, D. Kumaran, and C. Blundell.
  MEMO: a deep network for flexible combination of episodic memories.
  *ArXiv*, 2001.10913, 2020.
* Barra et al. (2018)

  A. Barra, M. Beccaria, and A. Fachechi.
  A new mechanical approach to handle generalized Hopfield neural
  networks.
  *Neural Networks*, 106:205–222, 2018.
  doi: 10.1016/j.neunet.2018.07.010.
* Bauschke & Combettes (2017)

  H. H. Bauschke and P. L. Combettes.
  *Convex Analysis and Monotone Operator Theory in Hilbert
  Spaces*.
  Cham: Springer International Publishing, 2nd edition, 2017.
  ISBN 978-3-319-48310-8.
  doi: 10.1007/978-3-319-48311-5.
* Boyd & Vandenberghe (2009)

  S. Boyd and L. Vandenberghe.
  *Convex Optimization*.
  Cambridge University Press, 7th edition, 2009.
  ISBN 978-0-521-83378-3.
* Brauchart et al. (2018)

  J. S. Brauchart, A. B. Reznikov, E. B. Saff, I. H. Sloan, Y. G. Wang, and R. S.
  Womersley.
  Random point sets on the sphere - hole radii, covering, and
  separation.
  *Experimental Mathematics*, 27(1):62–81,
  2018.
  doi: 10.1080/10586458.2016.1226209.
* Breiman (2001)

  L. Breiman.
  Random forests.
  *Machine Learning*, 45(1):5–32, 2001.
  doi: 10.1023/A:1010933404324.
* Bruck & Roychowdhury (1990)

  J. Bruck and V. P. Roychowdhury.
  On the number of spurious memories in the Hopfield model.
  *IEEE Transactions on Information Theory*, 36(2):393–397, 1990.
* Cai et al. (2013)

  T. Cai, J. Fan, and T. Jiang.
  Distributions of angles in random packing on spheres.
  *Journal of Machine Learning Research*, 14(21):1837–1864, 2013.
* Carbonneau et al. (2018)

  M.-A. Carbonneau, V. Cheplygina, E. Granger, and G. Gagnon.
  Multiple instance learning: a survey of problem characteristics and
  applications.
  *Pattern Recognition*, 77:329–353, 2018.
* Carbonneau et al. (2016)

  Marc-André Carbonneau, Eric Granger, Alexandre J. Raymond, and Ghyslain
  Gagnon.
  Robust multiple-instance learning ensembles using random subspace
  instance selection.
  *Pattern Recognition*, 58:83 – 99, 2016.
  ISSN 0031-3203.
  doi: https://doi.org/10.1016/j.patcog.2016.03.035.
  URL
  <http://www.sciencedirect.com/science/article/pii/S0031320316300346>.
* Carreira-Perpiñán &
  Williams (2003)

  M. Carreira-Perpiñán and C. K. I. Williams.
  An isotropic Gaussian mixture can have more modes than components.
  Technical Report EDI-INF-RR-0185, The University of Edinburgh, School
  of Informatics, 2003.
* Carta et al. (2020)

  A. Carta, A. Sperduti, and D. Bacciu.
  Encoding-based memory modules for recurrent neural networks.
  *ArXiv*, 2001.11771, 2020.
* Chen & Guestrin (2016)

  T. Chen and C. Guestrin.
  XGBoost: A scalable tree boosting system.
  In *Proceedings of the 22nd ACM SIGKDD International Conference
  on Knowledge Discovery and Data Mining*, pp.  785–794. ACM, 2016.
  doi: 10.1145/2939672.2939785.
* Chen et al. (2006)

  Y. Chen, J. Bi, and J. Z. Wang.
  MILES: Multiple-instance learning via embedded instance selection.
  *IEEE Transactions on Pattern Analysis and Machine
  Intelligence*, 28(12):1931–1947, 2006.
* Cheplygina et al. (2016)

  V Cheplygina, DM Tax, and M Loog.
  Dissimilarity-based ensembles for multiple instance learning.
  *IEEE transactions on neural networks and learning systems*,
  27(6):1379, 2016.
* Cho et al. (2014)

  K. Cho, B. vanMerriënboer, C. Gulcehre, D. Bahdanau, F. Bougares,
  H. Schwenk, and Y. Bengio.
  Learning phrase representations using RNN encoder–decoder for
  statistical machine translation.
  In *Proceedings of the Conference on Empirical Methods in
  Natural Language Processing (EMNLP)*, pp.  1724–1734. Association for
  Computational Linguistics, 2014.
  doi: 10.3115/v1/D14-1179.
* Clark et al. (2020)

  K. Clark, M.-T. Luong, Q. V. Le, and C. D. Manning.
  ELECTRA: Pre-training text encoders as discriminators rather than
  generators.
  *ArXiv*, 2003.10555, 2020.
  appeared in ICLR 2020.
* Cortes & Vapnik (1995)

  C. Cortes and V. Vapnik.
  Support-vector networks.
  *Machine learning*, 20(3):273–297, 1995.
* Crisanti et al. (1986)

  A. Crisanti, D. J. Amit, and H. Gutfreund.
  Saturation level of the Hopfield model for neural network.
  *Europhysics Letters (EPL)*, 2(4):337–341,
  1986.
  doi: 10.1209/0295-5075/2/4/012.
* Danihelka et al. (2016)

  I. Danihelka, G. Wayne, B. Uria, N. Kalchbrenner, and A. Graves.
  Associative long short-term memory.
  In M. F. Balcan and K. Q. Weinberger (eds.), *Proceedings of The
  33rd International Conference on Machine Learning*, volume 48 of
  *Proceedings of Machine Learning Research*, pp.  1986–1994, New York,
  USA, 2016.
* Daniluk et al. (2017)

  M. Daniluk, T. Rocktäschel, J. Welbl, and S. Riedel.
  Frustratingly short attention spans in neural language modeling.
  *ArXiv*, 1702.04521, 2017.
  appeared in ICRL 2017.
* Dehghani et al. (2018)

  M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and L. Kaiser.
  Universal transformers.
  *ArXiv*, 1807.03819, 2018.
  Published at ICLR 2019.
* Demircigil et al. (2017)

  M. Demircigil, J. Heusel, M. Löwe, S. Upgang, and F. Vermet.
  On a model of associative memory with huge storage capacity.
  *Journal of Statistical Physics*, 168(2):288–299, 2017.
* Devlin et al. (2018)

  J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova.
  BERT: pre-training of deep bidirectional transformers for language
  understanding.
  *ArXiv*, 1810.04805, 2018.
* Devlin et al. (2019)

  J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova.
  BERT: pre-training of deep bidirectional transformers for language
  understanding.
  In *Proceedings of the 2019 Conference of the North American
  Chapter of the Association for Computational Linguistics: Human Language
  Technologies, Volume 1 (Long and Short Papers)*, pp.  4171–4186.
  Association for Computational Linguistics, 2019.
* Dietterich et al. (1997)

  T. G. Dietterich, R. H. Lathrop, and T. Lozano-Pérez.
  Solving the multiple instance problem with axis-parallel rectangles.
  *Artificial Intelligence*, 89(1-2):31–71,
  1997.
* Emerson et al. (2017)

  R. O. Emerson, W. S. DeWitt, M. Vignali, J. Gravley, J. K. Hu, E. J. Osborne,
  C. Desmarais, M. Klinger, C. S. Carlson, J. A. Hansen, et al.
  Immunosequencing identifies signatures of cytomegalovirus exposure
  history and HLA-mediated effects on the T cell repertoire.
  *Nature Genetics*, 49(5):659, 2017.
* Fernández-Delgado et al. (2014)

  M. Fernández-Delgado, E. Cernadas, S. Barro, and D. Amorim.
  Do we need hundreds of classifiers to solve real world classification
  problems?
  *The Journal of Machine Learning Research*, 15(1):3133–3181, 2014.
* Folli et al. (2017)

  V. Folli, M. Leonetti, and G. Ruocco.
  On the maximum storage capacity of the Hopfield model.
  *Frontiers in Computational Neuroscience*, 10(144),
  2017.
  doi: 10.3389/fncom.2016.00144.
* Gao & Pavel (2017)

  B. Gao and L. Pavel.
  On the properties of the softmax function with application in game
  theory and reinforcement learning.
  *ArXiv*, 1704.00805, 2017.
* Garling (2017)

  D. J. H. Garling.
  *Analysis on Polish Spaces and an Introduction to Optimal
  Transportation*.
  London Mathematical Society Student Texts. Cambridge University
  Press, 2017.
  ISBN 1108421571.
  doi: 10.1017/9781108377362.
* Gilmer et al. (2017)

  J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl.
  Neural message passing for quantum chemistry.
  In *Proceedings of the 34th International Conference on Machine
  Learning (ICML)*, volume 70, pp.  1263–1272. JMLR.org, 2017.
* Graves et al. (2014)

  A. Graves, G. Wayne, and I. Danihelka.
  Neural turing machines.
  *ArXiv*, 1410.5401, 2014.
* Guttenberg et al. (2016)

  N. Guttenberg, N. Virgo, O. Witkowski, H. Aoki, and R. Kanai.
  Permutation-equivariant neural networks applied to dynamics
  prediction.
  *arXiv*, 1612.04530, 2016.
* Hertz et al. (1991)

  J. Hertz, A. Krogh, and R. G. Palmer.
  *Introduction to the Theory of Neural Computation*.
  Addison-Wesley Longman Publishing Co., Inc., Redwood City, CA, 1991.
  ISBN 0201503956.
* Hochreiter (1991)

  S. Hochreiter.
  Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis,
  Institut für Informatik, Lehrstuhl Prof. Brauer, Technische
  Universität München, 1991.
  Advisor: J. Schmidhuber.
* Hochreiter & Schmidhuber (1997)

  S. Hochreiter and J. Schmidhuber.
  Long short-term memory.
  *Neural Comput.*, 9(8):1735–1780, 1997.
* Hoorfar & Hassani (2008)

  A. Hoorfar and M. Hassani.
  Inequalities on the Lambert w𝑤w function and hyperpower function.
  *Journal of Inequalities in Pure and Applied Mathematics*,
  9(2):1–5, 2008.
* Hopfield (1982)

  J. J. Hopfield.
  Neural networks and physical systems with emergent collective
  computational abilities.
  *Proceedings of the National Academy of Sciences*, 79(8):2554–2558, 1982.
* Hopfield (1984)

  J. J. Hopfield.
  Neurons with graded response have collective computational properties
  like those of two-state neurons.
  *Proceedings of the National Academy of Sciences*, 81(10):3088–3092, 1984.
  doi: 10.1073/pnas.81.10.3088.
* Ilse et al. (2018)

  M. Ilse, J. M. Tomczak, and M. Welling.
  Attention-based deep multiple instance learning.
  *International Conference on Machine Learning (ICML)*, pp. 3376–3391, 2018.
* Ilse et al. (2020)

  M. Ilse, J. M. Tomczak, and M. Welling.
  Deep multiple instance learning for digital histopathology.
  In *Handbook of Medical Image Computing and Computer Assisted
  Intervention*, pp.  521–546. Elsevier, 2020.
* Jiang et al. (2020)

  D. Jiang, Z. Wu, C.-Y. Hsieh, G. Chen, B. Liao, Z. Wang, C. Shen, D. Cao,
  J. Wu, and T. Hou.
  Could graph neural networks learn better molecular representation for
  drug discovery? a comparison study of descriptor-based and graph-based
  models.
  *Journal of Cheminformatics*, 2020.
  doi: 10.21203/rs.3.rs-81439/v1.
* Kandemir et al. (2014)

  M. Kandemir, C. Zhang, and F. A. Hamprecht.
  Empowering multiple instance histopathology cancer diagnosis by cell
  graphs.
  In *International Conference on Medical Image Computing and
  Computer-Assisted Intervention*, pp.  228–235. Springer, 2014.
* Khan et al. (2018)

  M. M. R. Khan, R. B. Arif, M. A. B. Siddique, and M. R. Oishe.
  Study and observation of the variation of accuracies of KNN, SVM,
  LMNN, ENN algorithms on eleven different datasets from UCI machine
  learning repository.
  In *4th International Conference on Electrical Engineering and
  Information & Communication Technology (iCEEiCT)*, pp.  124–129. IEEE,
  2018.
* Kipf & Welling (2016)

  T. N. Kipf and M. Welling.
  Semi-supervised classification with graph convolutional networks.
  *ArXiv*, 1609.02907, 2016.
  in International Conference On Learning Representations (ICLR) 2017.
* Klambauer et al. (2017a)

  G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter.
  Self-normalizing neural networks.
  In *Advances in Neural Information Processing Systems*, pp. 971–980, 2017a.
* Klambauer et al. (2017b)

  G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter.
  Self-normalizing neural networks.
  *ArXiv*, 1706.02515, 2017b.
* Koiran (1994)

  P. Koiran.
  Dynamics of discrete time, continuous state Hopfield networks.
  *Neural Computation*, 6(3):459–468, 1994.
  doi: 10.1162/neco.1994.6.3.459.
* Korshunova et al. (2018)

  I. Korshunova, J. Degrave, F. Huszar, Y. Gal, A. Gretton, and J. Dambre.
  BRUNO: A deep recurrent model for exchangeable data.
  In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi,
  and R. Garnett (eds.), *Advances in Neural Information Processing
  Systems 31*, pp.  7190–7198. Curran Associates, Inc., 2018.
* Krotov & Hopfield (2016)

  D. Krotov and J. J. Hopfield.
  Dense associative memory for pattern recognition.
  In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett
  (eds.), *Advances in Neural Information Processing Systems*, pp. 1172–1180. Curran Associates, Inc., 2016.
* Krotov & Hopfield (2018)

  D. Krotov and J. J. Hopfield.
  Dense associative memory is robust to adversarial inputs.
  *Neural Computation*, 30(12):3151–3167,
  2018.
* Krotov & Hopfield (2020)

  D. Krotov and J. J. Hopfield.
  Large associative memory problem in neurobiology and machine
  learning.
  *ArXiv*, 2008.06996, 2020.
* Küçükaşcı &
  Baydoğan (2018)

  E. Ş. Küçükaşcı and M. G. Baydoğan.
  Bag encoding strategies in multiple instance learning problems.
  *Information Sciences*, 467:559–578, 2018.
* Kuhn et al. (2016)

  M. Kuhn, I. Letunic, L. J. Jensen, and P. Bork.
  The SIDER database of drugs and side effects.
  *Nucleic Acids Research*, 44(D1):D1075–D1079, 2016.
  doi: 10.1093/nar/gkv1075.
* LeCun et al. (2015)

  Y. LeCun, Y. Bengio, and G. Hinton.
  Deep learning.
  *Nature*, 521:436–444, 2015.
* Lipp & Boyd (2016)

  T. Lipp and S. Boyd.
  Variations and extension of the convex–concave procedure.
  *Optimization and Engineering*, 17(2):263–287, 2016.
  doi: 10.1007/s11081-015-9294-x.
* Loshchilov & Hutter (2017)

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  *arXiv preprint arXiv:1711.05101*, 2017.
* Maron & Lozano-Pérez (1998)

  O. Maron and T. Lozano-Pérez.
  A framework for multiple-instance learning.
  In M. I. Jordan, M. J. Kearns, and S. A. Solla (eds.), *Advances
  in Neural Information Processing Systems*, pp.  570–576. MIT Press, 1998.
* Martins et al. (2012)

  I. F. Martins, A. L. Teixeira, L. Pinheiro, and A. O. Falcao.
  A Bayesian approach to in silico blood-brain barrier penetration
  modeling.
  *Journal of Chemical Information and Modeling*, 52(6):1686–1697, 2012.
  doi: 10.1021/ci300124c.
* Mazza (1997)

  C. Mazza.
  On the storage capacity of nonlinear neural networks.
  *Neural Networks*, 10(4):593–597, 1997.
  doi: 10.1016/S0893-6080(97)00017-8.
* McEliece et al. (1987)

  R. J. McEliece, E. C. Posner, E. R. Rodemich, and S. S. Venkatesh.
  The capacity of the Hopfield associative memory.
  *IEEE Trans. Inf. Theor.*, 33(4):461–482,
  1987.
  doi: 10.1109/TIT.1987.1057328.
* Meyer (1976)

  R. R. Meyer.
  Sufficient conditions for the convergence of monotonic mathematical
  programming algorithms.
  *Journal of Computer and System Sciences*, 12(1):108–121, 1976.
  doi: 10.1016/S0022-0000(76)80021-9.
* Olver et al. (2010)

  F. W. J. Olver, D. W. Lozier, R. F. Boisvert, and C. W. Clark.
  *NIST handbook of mathematical functions*.
  Cambridge University Press, 1 pap/cdr edition, 2010.
  ISBN 9780521192255.
* Paszke et al. (2017)

  A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin,
  A. Desmaison, L. Antiga, and A. Lerer.
  Automatic differentiation in PyTorch.
  In *Workshop in Advances in Neural Information Processing
  Systems (NeurIPS)*, 2017.
* Paszke et al. (2019)

  A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen,
  Z. Lin, N. Gimelshein, L. Antiga, et al.
  PyTorch: An imperative style, high-performance deep learning
  library.
  In *Advances in Neural Information Processing Systems*, pp. 8026–8037, 2019.
* Qi et al. (2017a)

  C. R. Qi, H. Su, M. Kaichun, and L. J. Guibas.
  PointNet: Deep learning on point sets for 3d classification and
  segmentation.
  In *IEEE Conference on Computer Vision and Pattern Recognition
  (CVPR)*, pp.  77–85, 2017a.
  doi: 10.1109/CVPR.2017.16.
* Qi et al. (2017b)

  C. R. Qi, L. Yi, H. Su, and L. J. Guibas.
  PointNet++: Deep hierarchical feature learning on point sets in a
  metric space.
  In *31st International Conference on Neural Information
  Processing Systems*, pp.  5105–5114. Curran Associates Inc.,
  2017b.
* Rangarajan et al. (1996)

  A. Rangarajan, S. Gold, and E. Mjolsness.
  A novel optimizing network architecture with applications.
  *Neural Computation*, 8(5):1041–1060, 1996.
  doi: 10.1162/neco.1996.8.5.1041.
* Rangarajan et al. (1999)

  A. Rangarajan, A. Yuille, and Eric E. Mjolsness.
  Convergence properties of the softassign quadratic assignment
  algorithm.
  *Neural Computation*, 11(6):1455–1474,
  1999.
  doi: 10.1162/089976699300016313.
* Ravanbakhsh et al. (2016)

  S. Ravanbakhsh, J. Schneider, and B. Poczos.
  Deep learning with sets and point clouds.
  *arXiv*, 1611.04500, 2016.
* Schlag & Schmidhuber (2018)

  I. Schlag and J. Schmidhuber.
  Learning to reason with third order tensor products.
  In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi,
  and R. Garnett (eds.), *Advances in Neural Information Processing
  Systems 31*, pp.  9981–9993. Curran Associates, Inc., 2018.
* Schlag et al. (2019)

  I. Schlag, P. Smolensky, R. Fernandez, N. Jojic, J. Schmidhuber, and J. Gao.
  Enhancing the transformer with explicit relational encoding for math
  problem solving.
  *arXiv*, 1910.06611, 2019.
* Schlag et al. (2021)

  I. Schlag, K. Irie, and J. Schmidhuber.
  Linear transformers are secretly fast weight memory systems.
  *arXiv*, 2102.11174, 2021.
* Schmidhuber (1992)

  J. Schmidhuber.
  Learning to control fast-weight memories: An alternative to dynamic
  recurrent networks.
  In *Neural Computations, Volume: 4, Issue: 1*, pp.  131 – 139.
  MIT Press, 1992.
* Schmidhuber (2015)

  J. Schmidhuber.
  Deep learning in neural networks: An overview.
  *Neural Networks*, 61:85–117, 2015.
  doi: 10.1016/j.neunet.2014.09.003.
* Schölkopf & Smola (2002)

  B. Schölkopf and A. J. Smola.
  *Learning with Kernels – Support Vector Machines,
  Regularization, Optimization, and Beyond*.
  MIT Press, Cambridge, MA, 2002.
* Sriperumbudur & Lanckriet (2009)

  B. K. Sriperumbudur and G. R. Lanckriet.
  On the convergence of the concave-convex procedure.
  In Y. Bengio, D. Schuurmans, J. D. Lafferty, C. K. I. Williams, and
  A. Culotta (eds.), *Advances in Neural Information Processing Systems
  22*, pp.  1759–1767. Curran Associates, Inc., 2009.
* Subramanian et al. (2016)

  G. Subramanian, B. Ramsundar, V. Pande, and R. A. Denny.
  Computational modeling of β𝛽\beta-Secretase 1 (BACE-1) inhibitors
  using ligand based approaches.
  *Journal of Chemical Information and Modeling*, 56(10):1936–1949, 2016.
  doi: 10.1021/acs.jcim.6b00290.
* Sukhbaatar et al. (2015a)

  S. Sukhbaatar, A. Szlam, J. Weston, and R. Fergus.
  End-to-end memory networks.
  In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett
  (eds.), *Advances in Neural Information Processing Systems 28*, pp. 2440–2448. Curran Associates, Inc., 2015a.
* Sukhbaatar et al. (2015b)

  S. Sukhbaatar, A. Szlam, J. Weston, and R. Fergus.
  End-to-end memory networks.
  *ArXiv*, 1503.08895, 2015b.
* Sutton & Barto (2018)

  R. S. Sutton and A. G. Barto.
  *Reinforcement Learning: An Introduction*.
  MIT Press, Cambridge, MA, 2 edition, 2018.
* Tanaka & Edwards (1980)

  F. Tanaka and S. F. Edwards.
  Analytic theory of the ground state properties of a spin glass. I.
  Ising spin glass.
  *Journal of Physics F: Metal Physics*, 10(12):2769–2778, 1980.
  doi: 10.1088/0305-4608/10/12/017.
* Tay et al. (2020)

  Y. Tay, D. Bahri, D. Metzler, D.-C. Juan, Z. Zhao, and C. Zheng.
  Synthesizer: Rethinking self-attention in transformer models.
  *ArXiv*, 2005.00743, 2020.
* Toneva & Wehbe (2019a)

  M. Toneva and L. Wehbe.
  Interpreting and improving natural-language processing (in machines)
  with natural language-processing (in the brain).
  In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), *Advances in Neural
  Information Processing Systems 32*, pp.  14954–14964. Curran Associates,
  Inc., 2019a.
* Toneva & Wehbe (2019b)

  M. Toneva and L. Wehbe.
  Interpreting and improving natural-language processing (in machines)
  with natural language-processing (in the brain).
  *arXiv*, 1905.11833, 2019b.
* Torres et al. (2002)

  J. J. Torres, L. Pantic, and Hilbert H. J. Kappen.
  Storage capacity of attractor neural networks with depressing
  synapses.
  *Phys. Rev. E*, 66:061910, 2002.
  doi: 10.1103/PhysRevE.66.061910.
* Vaswani et al. (2017a)

  A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
  L. Kaiser, and I. Polosukhin.
  Attention is all you need.
  In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus,
  S. Vishwanathan, and R. Garnett (eds.), *Advances in Neural Information
  Processing Systems 30*, pp.  5998–6008. Curran Associates, Inc.,
  2017a.
* Vaswani et al. (2017b)

  A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
  L. Kaiser, and I. Polosukhin.
  Attention is all you need.
  *ArXiv*, 1706.03762, 2017b.
* Velic̆ković et al. (2018)

  P. Velic̆ković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and
  Y. Bengio.
  Graph attention networks.
  *arXiv*, 1710.10903, 2018.
  in International Conference On Learning Representations (ICLR) 2018.
* Wainberg et al. (2016)

  M. Wainberg, B. Alipanahi, and B. J. Frey.
  Are random forests truly the best classifiers?
  *The Journal of Machine Learning Research*, 17(1):3837–3841, 2016.
* Wainrib & Touboul (2013)

  G. Wainrib and J. Touboul.
  Topological and dynamical complexity of random neural networks.
  *Phys. Rev. Lett.*, 110:118101, 2013.
  doi: 10.1103/PhysRevLett.110.118101.
* Wang (2000)

  J. Wang.
  Solving the multiple-instance problem: A lazy learning approach.
  In *Proceedings of the 17th International Conference on Machine
  Learning (ICML)*, 2000.
* Wang et al. (2018)

  X. Wang, Y. Yan, P. Tang, X. Bai, and W. Liu.
  Revisiting multiple instance neural networks.
  *Pattern Recognition*, 74:15–24, 2018.
* Weber et al. (2020)

  C. R. Weber, R. Akbar, A. Yermanos, M. Pavlović, I. Snapkov, G. K. Sandve,
  S. T. Reddy, and V. Greiff.
  immuneSIM: tunable multi-feature simulation of B- and T-cell
  receptor repertoires for immunoinformatics benchmarking.
  *Bioinformatics*, 36(11):3594–3596, 2020.
  doi: 10.1093/bioinformatics/btaa158.
* Weston et al. (2014)

  J. Weston, S. Chopra, and A. Bordes.
  Memory networks.
  *ArXiv*, 1410.3916, 2014.
* Widrich et al. (2020a)

  M. Widrich, B. Schäfl, M. Pavlović, H. Ramsauer, L. Gruber,
  M. Holzleitner, J. Brandstetter, G. K. Sandve, V. Greiff, S. Hochreiter, and
  G. Klambauer.
  Modern Hopfield networks and attention for immune repertoire
  classification.
  *ArXiv*, 2007.13505, 2020a.
* Widrich et al. (2020b)

  M. Widrich, B. Schäfl, M. Pavlović, H. Ramsauer, L. Gruber,
  M. Holzleitner, J. Brandstetter, G. K. Sandve, V. Greiff, S. Hochreiter, and
  G. Klambauer.
  Modern Hopfield networks and attention for immune repertoire
  classification.
  In *Advances in Neural Information Processing Systems*. Curran
  Associates, Inc., 2020b.
* Wolf et al. (2019)

  T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac,
  T. Rault, R. Louf, M. Funtowicz, and J. Brew.
  HuggingFace’s transformers: State-of-the-art natural language
  processing.
  *ArXiv*, 1910.03771, 2019.
* Wu (1983)

  J. C. F. Wu.
  On the convergence properties of the em algorithm.
  *Ann. Statist.*, 11(1):95–103, 1983.
  doi: 10.1214/aos/1176346060.
* Wu et al. (2018)

  X. Wu, X. Liu, W. Li, and Q. Wu.
  Improved expressivity through dendritic neural networks.
  In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi,
  and R. Garnett (eds.), *Advances in Neural Information Processing
  Systems 31*, pp.  8057–8068. Curran Associates, Inc., 2018.
* Wu et al. (2017)

  Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu,
  K. Leswing, and V. Pande.
  MoleculeNet: A benchmark for molecular machine learning.
  *arXiv*, 1703.00564, 2017.
* Xiong et al. (2020)

  Z. Xiong, D. Wang, X. Liu, F. Zhong, X. Wan, X. Li, Z. Li, X. Luo, K. Chen,
  H. Jiang, and M. Zheng.
  Pushing the boundaries of molecular representation for drug discovery
  with the graph attention mechanism.
  *Journal of Medicinal Chemistry*, 63(16):8749–8760, 2020.
  doi: 10.1021/acs.jmedchem.9b00959.
* Xu et al. (2018)

  Y. Xu, T. Fan, M. Xu, L. Zeng, and Y. Qiao.
  SpiderCNN: Deep learning on point sets with parameterized
  convolutional filters.
  In V. Ferrari, M. Hebert, C. Sminchisescu, and Y. Weiss (eds.),
  *European Conference on Computer Vision (ECCV)*, pp.  90–105. Springer
  International Publishing, 2018.
* Yuille & Rangarajan (2002)

  A. L. Yuille and A. Rangarajan.
  The concave-convex procedure (CCCP).
  In T. G. Dietterich, S. Becker, and Z. Ghahramani (eds.),
  *Advances in Neural Information Processing Systems 14*, pp. 1033–1040. MIT Press, 2002.
* Yuille & Rangarajan (2003)

  A. L. Yuille and A. Rangarajan.
  The concave-convex procedure.
  *Neural Computation*, 15(4):915–936, 2003.
  doi: 10.1162/08997660360581958.
* Zaheer et al. (2017)

  M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. R. Salakhutdinov, and A. J.
  Smola.
  Deep sets.
  In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus,
  S. Vishwanathan, and R. Garnett (eds.), *Advances in Neural Information
  Processing Systems 30*, pp.  3391–3401. Curran Associates, Inc., 2017.
* Zangwill (1969)

  W. I. Zangwill.
  *Nonlinear programming: a unified approach*.
  Prentice-Hall international series in management. Englewood Cliffs,
  N.J., 1969.
  ISBN 9780136235798.
* Zhai et al. (2020)

  S. Zhai, W. Talbott, M. A. Bautista, C. Guestrin, and J. M. Susskind.
  Set distribution networks: a generative model for sets of images.
  *arXiv*, 2006.10705, 2020.
* Zhang & Zhou (2017)

  W. Zhang and B. Zhou.
  Learning to update auto-associative memory in recurrent neural
  networks for improving sequence memorization.
  *ArXiv*, 1709.06493, 2017.
* Zhu et al. (2015)

  Y. Zhu, R. Kiros, R. S. Zemel, R. Salakhutdinov, R. Urtasun, A. Torralba, and
  S. Fidler.
  Aligning books and movies: Towards story-like visual explanations by
  watching movies and reading books.
  *Proceedings of the IEEE international conference on computer
  vision*, pp.  19–27, 2015.
  arXiv 1506.06724.
