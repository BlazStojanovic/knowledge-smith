---
arxiv: '2511.11346'
authors:
- Andreas Grivas
- Lorenzo Loconte
- Emile van Krieken
- Piotr Nawrot
- Yu Zhao
- Euan Wielewski
- Pasquale Minervini
- Edoardo Ponti
- Antonio Vergari
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Fast and Expressive Multi-Token Prediction with Probabilistic Circuits
url: https://arxiv.org/abs/2511.11346
year: 2025
---

[2511.11346] Fast and Expressive Multi-Token Prediction with Probabilistic Circuits














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



# Fast and Expressive Multi-Token Prediction with Probabilistic Circuits

Andreas Grivas11footnotemark: 1 &Lorenzo Loconte11footnotemark: 1 &Emile van Krieken11footnotemark: 1 &Piotr Nawrot11footnotemark: 1 &Yu Zhao11footnotemark: 1 &Euan Wielewski◇ &Pasquale Minervini11footnotemark: 1 &Edoardo Ponti11footnotemark: 1 &Antonio Vergari11footnotemark: 1 &∗School of Informatics, University of Edinburgh  ◇\Diamond NatWest Group

###### Abstract

Multi-token prediction (MTP) is a prominent strategy to significantly speed up generation in large language models (LLMs), including byte-level LLMs, which are tokeniser-free but prohibitively slow. However, existing MTP methods often sacrifice expressiveness by assuming independence between future tokens. In this work, we investigate the trade-off between expressiveness and latency in MTP within the framework of probabilistic circuits (PCs). Our framework, named MtPC, allows one to explore different ways to encode the joint distributions over future tokens by selecting different circuit architectures, generalising classical models such as (hierarchical) mixture models, hidden Markov models and tensor networks. We show the efficacy of MtPC by retrofitting existing byte-level LLMs, such as EvaByte. Our experiments show that, when combined with speculative decoding, MtPC significantly speeds up generation compared to MTP with independence assumptions, while guaranteeing to retain the performance of the original verifier LLM. We also rigorously study the optimal trade-off between expressiveness and latency when exploring the possible parameterisations of MtPC, such as PC architectures and partial layer sharing between the verifier and draft LLMs.

## 1 Introduction

Autoregressive (AR) large language models (LLMs) can only perform single-token prediction (STP) as they generate one token at a time, incurring
significantly high latency, energy demand, and deployment costs.
This affects not only subword models, but even more so the byte-level ones (megabyte; wang2024mambabyte, inter alia).
Among possible alternatives to speed up generation (ankner2024hydra; deepseekai2024; nawrot-etal-2023-efficient; pagnoni2024blt; lancucki2025inference), multi-token prediction (MTP) stands out as it promises to predict a window of multiple tokens all at once, may they be subwords (gloeckle2024better; cai2024medusa) or bytes (gloeckle2024better; evabyte2025).
As such, MTP LLMs can achieve a significantly higher throughput than STP ones, as they decrease the number of forward passes required through the LLM.

Nevertheless, modelling the joint distribution over all future tokens in a window is challenging, as it requires balancing expressiveness, i.e., representing all the dependencies between tokens, and efficiency, i.e., minimising latency.
Existing MTP approaches favour the latter by making an unrealistic assumption: namely, considering all future tokens to be independent  (evabyte2025; cai2024medusa; gloeckle2024better).
This clearly comes at the expense of expressiveness (ankner2024hydra; wertheimer2024accelerating), as the choice of a token for a position within the window cannot influence the probability of the others.

For example, consider the prompt: “*Name a capital of South Africa*”, where *Cape Town* and *Pretoria* are equally likely completions. A byte-level MTP model with independence assumptions over an 8-token window could return
*Cretoria*
as an argmax, because replacing *P* with *C* cannot change the probability of other tokens.
More concerningly, an exponential number of “byte-salad” continuations, such as
*Crptoria*,
*Crpt ria* and
*Crpt roa*,
are then also equally likely, despite having almost zero probability under the STP model.
Recently, basharin2024faster
introduced
dependencies into MTP with a mixture over the future token probabilities.
However, a single mixture can only add limited expressiveness.
Crucially, understanding how to increase expressiveness while optimally trading off efficiency in a systematic way is still an open question.

In this paper, we fill this gap by proposing an MTP framework based on probabilistic circuits (PCs; choi2020pc; vergari2021compositional), which we name MtPC.
MtPC uses PCs to parameterise the joint distribution over future tokens into tractable computational graphs that can encode hierarchical mixture models.
As such, MtPC offers a way to systematically navigate the spectrum of MTP architectural variants, encompassing fully factorised models (evabyte2025; cai2024medusa; gloeckle2024better) and
shallow mixtures (basharin2024faster) but also more expressive parameterisations:
hidden Markov models (HMMs) and binary tree factorisations (BTrees) which are novel for MTP.

Moreover, in contrast to previous work on MTP (evabyte2025; cai2024medusa), MtPC guarantees we match the quality of an AR LLM via speculative decoding  (leviathan2022fast; chen2023accelerating; stern2018blockwise; xia-etal-2024)—exactly for greedy decoding or in expectation for sampling—showing that the throughput sacrificed for the guarantee is not as large as alluded to previously.
We do so by sharing the LLM backbone for the draft and verifier models for different numbers of layers, highlighting how this creates a second dimension to trade-off expressiveness (as hidden representations between draft and verifier can diverge) and latency (as each non-shared layer requires separate forward passes). We illustrate the two trade-offs at the core of MtPC in [Fig. 1](#S1.F1.fig1 "In 1 Introduction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").

In summary, we make the following contributions: C1) we introduce MtPC, a fast MTP framework based on PCs that overcomes the independence assumptions of previous work and generalises tensor decomposition methods (basharin2024faster);
C2)
we rigorously identify trade-offs between acceptance rates in speculative decoding and latency of generation, based on different choices of probabilistic circuit (PC) architectures and partial layer sharing;
C3) we empirically demonstrate the effectiveness of MtPC by repurposing EvaByte (evabyte2025), a byte-level LLM, into our framework.
The choice of this use case is motivated by the fact that existing byte-level LLMs (pagnoni2024blt; wang2024mambabyte) obviate the limitations of sub-word tokenisers—including uneven efficiency (ahia-etal-2023-languages; dagan2024getting), lack of interoperability  (minixhofer2025crosstokenizer), and vulnerabilities  (rumbelow2023; land2024magikarp; geiping2024coercing; salesky-etal-2021-robust)—at the cost of significantly slowing down generation.
We find that MtPC increases the throughput of EvaByte by 5.47×5.47\times with respect to AR generation and 1.22×1.22\times with respect to MTP with independence assumptions.

![Refer to caption](/html/2511.11346/assets/x1.png)

Figure 1: MtPC allows for exploring the trade-off between efficiency (latency) and expressiveness (token acceptance) with different MTP designs in terms of 1) choice of PC architecture ([FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), [HMM](#S3.Ex3 "Equation HMM ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")); 2) choice of layers shared between draft and verifier models in self-speculative decoding. Dotted lines indicate iso-throughput (tokens generated per second) regions, highlighting configurations such as [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") for n=16n=16 tokens and 22 LoRA layers that achieve the best throughput.

## 2 Speeding up Generation with MTP and Speculative Decoding

Given our goal of speeding up LLM generation with MTP while guaranteeing that the STP quality is fully retained through speculative decoding, we introduce these frameworks below.111We adapt notation from the tensor and circuit literature (loconte2025what), see [Appendix A](#A1 "Appendix A Notation ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").

MTP. A classical STP LLM encodes a distribution over sequences of tokens {𝐱t}\{\mathbf{x}\_{t}\} defined over a vocabulary 𝒱\mathcal{V} as ∏tp​(xt+1∣𝐱≤t)\prod\_{t}p(x\_{t+1}\mid\mathbf{x}\_{\leq t}), where 𝐱≤t\mathbf{x}\_{\leq t} is the context, *i.e.* the observed tokens at timestep tt.
MTP (gloeckle2024better) aims to extend an STP LLM that predicts a single token at a time through p​(xt+1∣𝐱≤t)p(x\_{t+1}\mid\mathbf{x}\_{\leq t}), to an MTP model, q𝜽q\_{\bm{\mathcal{\theta}}}, that models the joint probability of a window of nn future tokens and generates them simultaneously, i.e.,

|  |  |  |  |
| --- | --- | --- | --- |
|  | q𝜽​(xt+1,xt+2,…,xt+n∣𝐱≤t).q\_{\bm{\mathcal{\theta}}}(x\_{t+1},x\_{t+2},\ldots,x\_{t+n}\mid\mathbf{x}\_{\leq t}). |  | (1) |

where 𝜽\bm{\mathcal{\theta}} denotes a given parameterisation for the joint.222These parameters depend on tt, we drop the subscript when not needed to avoid clutter.
The first dimension to trade-off expressiveness and efficiency in MTP pertains to compactly representing q𝜽q\_{\bm{\mathcal{\theta}}}.
Unlike for p​(xt+1∣𝐱≤t)p(x\_{t+1}\mid\mathbf{x}\_{\leq t}), we would need to store more than a vector of logits 𝐚∈ℝv\bm{\mathrm{a}}\in\mathbb{R}^{v} of a single univariate categorical distribution for a vocabulary size v=|𝒱|v=|\mathcal{V}| for every timestep tt.
The most expressive, but least efficient way to do so, would be to store an nn-dimensional tensor 𝓐∈ℝv(1)×…×v(n)\bm{\mathcal{A}}\in\mathbb{R}^{v^{(1)}\times\ldots\times v^{(n)}} of logits having vnv^{n} entries, but this scales exponentially in nn.
Next, we review past attempts to avoid storing 𝓐\bm{\mathcal{A}} explicitly.

Fully factorised.
The most commonly used way to boost efficiency is to assume all nn future tokens are independent (evabyte2025; cai2024medusa; gloeckle2024better), that is, q𝜽q\_{\bm{\mathcal{\theta}}} factorizes as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∏i=1nqϕi​(xt+i∣𝐱≤t).\prod\nolimits\_{i=1}^{n}q\_{\phi\_{i}}(x\_{t+i}\mid\mathbf{x}\_{\leq t}). |  | (FF) |

This comes with the benefit that one needs to store only nn vv-dimensional vectors of probabilities ϕi\phi\_{i} to represent the joint distribution in [Eq. 1](#S2.E1 "In 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").
At the same time, as already discussed in the introduction, this severely limits model expressiveness (ankner2024hydra; wertheimer2024accelerating).

Canonical polyadic (CP) factorisation.
Dependencies between future tokens can be recovered by introducing explicit latent variables (Lee2018deterministic).
To this end, basharin2024faster propose to factorise [Eq. 1](#S2.E1 "In 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") via an rr-rank CP decomposition.
A CP decomposition introduces one discrete latent variable, ZZ, that encodes a mixture of rr fully-factorised components, rewriting [Eq. 1](#S2.E1 "In 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑j=1rq​(Z=j∣𝐱≤t)​∏i=1nqϕi,j​(xt+i∣j,𝐱≤t).\sum\nolimits\_{j=1}^{r}q(Z=j\mid\mathbf{x}\_{\leq t})\prod\nolimits\_{i=1}^{n}q\_{\phi\_{i,j}}(x\_{t+i}\mid j,\,\mathbf{x}\_{\leq t}). |  | (CP) |

where q​(j∣𝐱≤t)=ωjq(j\mid\mathbf{x}\_{\leq t})=\omega\_{j} are the mixture coefficients and ϕi,j\phi\_{i,j}
are the parameters of the categorical distribution for mixture component jj at position ii in the MTP window.
333basharin2024faster calls [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") a mixture of experts (MoE), but we note this is incorrect as the weights ωj\omega\_{j} do not depend on future tokens, but only on past ones. As such, they realise a simple conditional mixture.
They argue that training [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") is challenging and requires insights from the MoE literature, while we are able to train them as well as deeper mixture variants easily without MoE-tailored losses (see [Section 4](#S4 "4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).
Before showing how we can generalize both [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") MTP with PCs, we review how to ensure MTP models match the quality of a given STP model.

Speculative decoding  (stern2018blockwise; leviathan2022fast; chen2023accelerating; xia-etal-2024) can be combined with MTP to speed up generation while guaranteeing no loss in quality.
Given a target STP LLM that we wish to accelerate, speculative decoding involves two steps: 1) drafting, where a cheaper MTP draft model generates nn future tokens, and 2) verification, where the target STP model accepts or rejects the generated tokens in parallel according to a pre-defined consistency criterion. The closer the distributions of the draft and verifier are, the more often ‘speculated’ tokens are accepted, speeding up generation.
With speculative decoding we can quantify the trade-off between expressivenss and efficiency in MTP models as their throughput, i.e.

|  |  |  |  |
| --- | --- | --- | --- |
|  | throughput (tok/s)=acceptance rate (toks per eval)/latency (secs per eval)\text{throughput (tok/s)}={\text{acceptance rate {(toks per eval)}}}/{\text{latency {(secs per eval)}}} |  | (2) |

where acceptance rates are a function of the total variation distance between the two distributions (leviathan2022fast; sun2023spectr) and latency measures how computationally expensive an MTP model is during generation.
While previous work, such as basharin2024faster, focused only on measuring acceptance rates, we highlight how both sides of the ratio in [Eq. 2](#S2.E2 "In 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") are important, as they create a spectrum. MtPCs provide a systematic way to navigate such a spectrum (see [Fig. 1](#S1.F1.fig1 "In 1 Introduction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).

## 3 Probabilistic Circuits for Multi-Token Prediction

The idea behind MtPCs is to further decompose the joint distribution in [Eq. 1](#S2.E1 "In 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") into a deep computational graph encoding a hierarchical mixture model, called a probabilistic circuit ([Sections 3.1](#S3.SS1 "3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and [3.2](#S3.SS2 "3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")), and to parameterise it with LLM embeddings ([Section 3.3](#S3.SS3 "3.3 Parameterising PCs with LLMs ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).

### 3.1 Probabilistic Circuits

![Refer to caption](/html/2511.11346/assets/x2.png)

![Refer to caption](/html/2511.11346/assets/x3.png)![Refer to caption](/html/2511.11346/assets/x4.png)

![Refer to caption](/html/2511.11346/assets/x5.png)

Figure 2: PCs allow for modelling a spectrum of dependency structures over sequences of tokens, as shown for the known [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and the novel [HMM](#S3.Ex3 "Equation HMM ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") MTP variants.
Input units are grouped in coloured layers, one for each token, while sum and product layers encoding (hierarchies of) latent
variable distributions
are in grey.
The output unit of each circuit (in blue) computes q𝜽​(xt+1,…,xt+n∣𝐱≤t)q\_{\bm{\mathrm{\theta}}}(x\_{t+1},\ldots,x\_{t+n}\mid\bm{\mathrm{x}}\_{\leq t}).
In the figure
we omit the dependency on the context 𝐱≤t\mathbf{x}\_{\leq t} for readability.

A circuit (darwiche2003differential; choi2020pc; vergari2021compositional), cc, is a parameterised directed acyclic computational graph444In [Fig. 2](#S3.F2 "In 3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), edges directionality is removed for readability, but it is assumed to be from inputs to outputs. over variables 𝐗\bm{\mathrm{X}} encoding a function, c​(𝐗)c(\bm{\mathrm{X}}), and comprises three kinds of computational units: input, product, and sum units.
Each product or sum unit nn receives the outputs of other units as inputs, denoted with the set 𝗂𝗇​(n){\mathsf{in}}(n).
Each unit nn encodes a function, cnc\_{n}, defined as: (i) cn​(𝗌𝖼​(n);ϕ)c\_{n}({\mathsf{sc}}(n);\phi) if nn is an input unit, where cnc\_{n} is a function parameterised by ϕ\phi over variables 𝗌𝖼​(n)⊆𝐗{\mathsf{sc}}(n)\subseteq\bm{\mathrm{X}}, called its scope;
(ii) ∏j∈𝗂𝗇​(n)cj​(𝗌𝖼​(j))\prod\_{j\in{\mathsf{in}}(n)}c\_{j}({\mathsf{sc}}(j)) if nn is a product unit; and
(iii) ∑j∈𝗂𝗇​(n)ωj​cj​(𝗌𝖼​(j))\sum\_{j\in{\mathsf{in}}(n)}\omega\_{j}c\_{j}({\mathsf{sc}}(j)) if nn is a sum unit, with ωj∈ℝ\omega\_{j}\in\mathbb{R} denoting the sum parameters.
The scope of a product or sum unit nn is the union of the scopes of its inputs, i.e., 𝗌𝖼​(n)=⋃j∈𝗂𝗇​(n)𝗌𝖼​(j){\mathsf{sc}}(n)=\bigcup\_{j\in{\mathsf{in}}(n)}{\mathsf{sc}}(j).
[Fig. 2](#S3.F2 "In 3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")
shows examples of circuits, where units of the same scope are grouped into (coloured) layers belonging to a hierarchy that can be easily parallelised on a GPU (mari2023unifying; loconte2025what).

For MtPCs, we use probabilistic circuits (PCs), i.e., circuits modelling a joint distribution over random variables, in our case tokens 𝐗={X1,…,Xn}\bm{\mathrm{X}}=\{X\_{1},\ldots,X\_{n}\}. PCs encode [Eq. 1](#S2.E1 "In 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") as

|  |  |  |  |
| --- | --- | --- | --- |
|  | q𝜽​(xt+1,…,xt+n∣𝐱≤t)=Z𝜽t−1​c​(xt+1,…,xt+n;𝜽t)q\_{\bm{\mathcal{\theta}}}(x\_{t+1},\ldots,x\_{t+n}\mid\bm{\mathrm{x}}\_{\leq t})={Z\_{\bm{\mathcal{\theta}}\_{t}}^{-1}}\>c(x\_{t+1},\ldots,x\_{t+n};\bm{\mathcal{\theta}}\_{t}) |  | (3) |

where 𝜽t={𝝎t,ϕt}\bm{\mathcal{\theta}}\_{t}=\{\bm{\omega}\_{t},\bm{\phi}\_{t}\}
denote the set of circuit parameters, i.e., all sum unit parameters 𝝎t\bm{\omega}\_{t} and input unit parameterisations ϕt\bm{\phi}\_{t} which depend on the context 𝐱≤t\bm{\mathrm{x}}\_{\leq t}; and Z𝜽tZ\_{\bm{\mathcal{\theta}}\_{t}} denotes the partition function of cc, i.e., Z𝜽t=∑xt+1,…,xt+n∈𝒱nc​(xt+1,…,xt+n;𝜽t)Z\_{\bm{\mathcal{\theta}}\_{t}}=\sum\_{x\_{t+1},\ldots,x\_{t+n}\in\mathcal{V}^{n}}c(x\_{t+1},\ldots,x\_{t+n};\bm{\mathcal{\theta}}\_{t}).
Note that the PC architectures we are interested in are already normalised or always allow computing the partition function in a single feedforward step (see choi2020pc and [Section B.1](#A2.SS1 "B.1 Structural properties ‣ Appendix B Background on circuits ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).
At the same time, we can easily sample from PCs in a single feedforward pass, as discussed in [Section B.2](#A2.SS2 "B.2 Sampling a circuit ‣ Appendix B Background on circuits ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").
Crucially, within the framework of PCs,
we can recover the [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") parameterisations for MTP and several other architectures that generalise tensor factorisations (loconte2025what) that can be used as novel MTP models, each offering a different expressiveness-efficiency trade-off.
We do so
while abstracting away from each model’s original formulation and obtain a unified way to parameterise MTP LLMs, as discussed next.

### 3.2 PC Architectures for MTP

MtPC-ff. Representing the commonly used [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") MTP
parameterisation as a PC is simple: we introduce nn input units, each parameterised by ϕi\phi\_{i}, its corresponding token probabilities, and connect them all to a single product unit, as shown in [Fig. 2](#S3.F2 "In 3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") for a distribution over n=4n=4 tokens.

MtPC-cp. Similarly, we can easily encode a [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") factorisation in a shallow PC by i) introducing rr input units for each token (each parameterised by their own probabilities ϕi​j\phi\_{ij}), then ii) multiplying them to retrieve the rr factorised mixture components, which we then iii) aggregate in a sum unit with weights ωj=q​(zj∣𝐱≤t)\omega\_{j}=q(z\_{j}\mid\mathbf{x}\_{\leq t}) (see also Proposition 1 in loconte2025what).
[Fig. 2](#S3.F2 "In 3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") shows this construction for n=4n=4 and r=2r=2.
This basic construction suggests that we can create deeper architectures by interleaving sum and product layers, while overparameterising each layer by increasing the number of units in it (rr).
Furthermore, by implementing [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") as a PC unlocks a faster sampling routine ([Section B.2](#A2.SS2 "B.2 Sampling a circuit ‣ Appendix B Background on circuits ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")) than the one used in basharin2024faster.

MtPC-hmm. As a further example of the expressiveness increase we get by generalising our approach to deeper PCs, we introduce a factorisation that realises a hidden Markov model (HMM), which better captures distant dependencies in the sequence by introducing a sequence of latent variables, in contrast to the single one present in [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").
More precisely, we define an HMM with rr hidden states and truncate its prediction window to nn steps into the future.
We resort to an inhomogeneous HMM, *i.e.*, we do not make the transition matrices time-invariant, as this setup is more expressive and worked better in our experiments, see [Appendix E](#A5 "Appendix E Hidden Markov Models Setup ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").
This simplifies
[Eq. 1](#S2.E1 "In 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") into:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑z1=1r⋯​∑zn=1rq​(z1∣𝐱≤t)​qϕ​(xt+1∣z1,𝐱≤t)​∏i=2nq​(zi∣zi−1,𝐱≤t)​qϕ​(xt+i∣zi,𝐱≤t).\sum\_{z\_{1}=1}^{r}\cdots\sum\_{z\_{n}=1}^{r}q(z\_{1}\mid\mathbf{x}\_{\leq t})q\_{\bm{\phi}}(x\_{t+1}\mid z\_{1},\mathbf{x}\_{\leq t})\prod\_{i=2}^{n}q(z\_{i}\mid z\_{i-1},\mathbf{x}\_{\leq t})q\_{\bm{\phi}}(x\_{t+i}\mid z\_{i},\,\mathbf{x}\_{\leq t}). |  | (HMM) |

[Fig. 2](#S3.F2 "In 3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") illustrates the [HMM](#S3.Ex3 "Equation HMM ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") parameterisation above represented as a circuit, comprising n=4n=4 pairs of sum and product layers stacked, where the parameters 𝝎i\bm{\omega}\_{i} of the former are the transition probabilities q​(zi∣zi−1,𝐱≤t)q(z\_{i}\mid z\_{i-1},\mathbf{x}\_{\leq t}).
Similarly to [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), we can increase rr to overparameterise the circuit with more input units per token and sum units overall, and hence increase expressiveness.

MtPC-btree. One drawback of the [HMM](#S3.Ex3 "Equation HMM ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") parameterisation is the asymmetry of its computational graph, which i) provides fewer latent variables for the early tokens, and ii) increases latency when predicting the last tokens due to its autoregressive token dependencies.
To solve this, we build a PC whose structure resembles that of a binary tree (BTree), effectively encoding a hierarchy of latent variables or a tree tensor factorisation (grasedyck2010hierarchical; cheng2019tree; loconte2025what).
This is done recursively: at each step hh of the hierarchy, given a sequence of nn tokens to split, and a parent latent variable ZlZ\_{l}, we split it into two sub-sequences (xt+1,…,xt+⌊n/2⌋−1)(x\_{t+1},\ldots,x\_{t+\lfloor n/2\rfloor-1}) and (xt+⌊n/2⌋,…,xt+n)(x\_{t+\lfloor n/2\rfloor},\ldots,x\_{t+n}), then factorise [Eq. 1](#S2.E1 "In 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") as a mixture:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑zh=1rq​(zh∣zl,𝐱≤t)​q𝜽​(xt+1,…,xt+⌊n/2⌋−1∣zh,zl,𝐱≤t)​q𝜽​(xt+⌊n/2⌋,…,xt+n∣zh,zl,𝐱≤t)\sum\nolimits\_{z\_{h}=1}^{r}q(z\_{h}\mid z\_{l},\bm{\mathrm{x}}\_{\leq t})q\_{\bm{\mathcal{\theta}}}(x\_{t+1},\ldots,x\_{t+\lfloor n/2\rfloor-1}\mid z\_{h},z\_{l},\bm{\mathrm{x}}\_{\leq t})q\_{\bm{\mathcal{\theta}}}(x\_{t+\lfloor n/2\rfloor},\ldots,x\_{t+n}\mid z\_{h},z\_{l},\bm{\mathrm{x}}\_{\leq t}) |  | (BTree) |

which corresponds to creating a sum unit whose weights are q​(zh∣zl,𝐱≤t)q(z\_{h}\mid z\_{l},\bm{\mathrm{x}}\_{\leq t}) followed by products. We repeat the process while caching intermediate units
until we reach the base case for n=1n=1, for which we create a layer of input units for the corresponding token.
[Fig. 2](#S3.F2 "In 3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") illustrates the [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") circuit built in this way.
Our experiments ([Section 4.2](#S4.SS2 "4.2 Metrics ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")) show that the [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") parameterisation obtains the optimal throughput by lowering the latency of [HMM](#S3.Ex3 "Equation HMM ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), as it samples more latent variables and tokens in parallel, while
achieving similar acceptance rates.

### 3.3 Parameterising PCs with LLMs

Parameterising MtPCs requires two functions: an LLM that maps the context 𝐱≤t∈𝒱t\bm{\mathrm{x}}\_{\leq t}\in\mathcal{V}^{t} into contextual features, and a neural network head that maps the contextual features to the parameters of the circuit 𝜽t\bm{\mathcal{\theta}}\_{t}, realising a neural conditional circuit (shao2020conditional; shao2022conditional).
To extract the contextual features
𝐞t∈ℝd\mathbf{e}\_{t}\in\mathbb{R}^{d}, we use 𝐞t=LLMLoRA​(k)​(𝐱≤t)\bm{\mathrm{e}}\_{t}=\text{LLM}\_{\text{LoRA}\left(k\right)}\left(\bm{\mathrm{x}}\_{\leq t}\right)
where LLMLoRA​(k):𝒱t→ℝd\text{LLM}\_{\text{LoRA}\left(k\right)}\colon\mathcal{V}^{t}\to\mathbb{R}^{d} is the STP backbone with LoRA (hu2022lora) applied to the last k≥0k\geq 0 layers.
As we will discuss in [Section 4.4](#S4.SS4 "4.4 MtPCs with Adapters ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), the number of LoRA layers can impact throughput significantly.
Given 𝐞t\bm{\mathrm{e}}\_{t}, we realise [Eq. 3](#S3.E3 "In 3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") by
computing 𝜽t=gc​(𝐞t)\bm{\mathcal{\theta}}\_{t}=g\_{c}\left(\bm{\mathrm{e}}\_{t}\right), where gcg\_{c} is a neural network head that
outputs both the
input unit parameters, ϕt\bm{\phi}\_{t}, and the sum unit parameters, 𝝎t\bm{\omega}\_{t} ([Section 3.1](#S3.SS1 "3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).
Note that our parameterisation in MtPCs allows us to abstract from the actual structure of the circuit (i.e., [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), [HMM](#S3.Ex3 "Equation HMM ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") or [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")) and just focus on these two sets of tensorised parameters, as we discuss next.

Input unit distributions.
All MtPCs produce joint distributions over token windows by combining categorical distributions over individual tokens ([Fig. 2](#S3.F2 "In 3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).
We follow EvaByte (evabyte2025) and learn nn separate unembedding layers, one per window position.
For models with mixture coefficients, we also learn one unembedding layer per mixture coefficient.555This is efficient even for PCs with high rank due to the small vocabulary size of byte-level LLMs.
As such, instead of a single unembedding matrix mapping ℝd→ℝv\mathbb{R}^{d}\rightarrow\mathbb{R}^{v}, we have an unembedding tensor 𝓦∈ℝn×r×v×d\bm{\mathcal{W}}\in\mathbb{R}^{n\times r\times v\times d}, and compute the input distributions with the usual unembedding operation followed by softmax, i.e., ϕt​i​j=softmax⁡(𝓦i​j​𝐞t)\bm{\phi}\_{tij}=\operatorname{softmax}\left(\bm{\mathcal{W}}\_{ij}\bm{\mathrm{e}}\_{t}\right),
where ii and jj index the position in the MTP window and the rank rr.

Sum unit parameters.
For sum units, instead of mapping embeddings to the vocabulary via 𝓦\bm{\mathcal{W}}, we map to the rank of the sum unit via 𝓡∈ℝz×r×d\bm{\mathcal{R}}\in\mathbb{R}^{z\times r\times d}, where zz is the number of sum units, rr is its rank, and dd the dimensionality of 𝐞t\bm{\mathrm{e}}\_{t}. We compute 𝝎t​i=softmax⁡(𝓡i​𝐞t)\bm{\omega}\_{ti}=\operatorname{softmax}\left(\bm{\mathcal{R}}\_{i}\bm{\mathrm{e}}\_{t}\right), where ii indexes the sum unit.

### 3.4 Speculative Decoding with MtPC

For MtPCs, we design an architecture that is self-drafting (zhang2024; cai2024medusa), i.e. where the draft and verifier models share the same LLM backbone.
We use an MTP head (cai2024medusa; ankner2024hydra) augmented with our circuits to efficiently sample a draft, and an autoregressive STP head as the verifier.
Optionally, we also explore keeping a few final transformer layers separate in the two models by fine-tuning LoRA adaptors for the draft model’s backbone.

Unlike previous self-drafting MTP works (cai2024medusa; ankner2024hydra), we guarantee that the generated tokens are the same as those the autoregressive LLM would generate in expectation by using speculative decoding (leviathan2022fast; chen2023accelerating), *i.e.*, we only generate the subset of drafted tokens accepted by our verifier.
To keep latency low, we make only a single LLM call per speculative decoding cycle by re-using the LLM backbone state computed by the verifier for the draft model, where possible.
We achieve this by modifying the speculative decoding algorithm slightly, as we detail in [Algorithm 2](#alg2 "In Appendix C Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").

Next we report results for sampling, but we also experimented with greedy speculative decoding
(stern2018blockwise) which guarantees argmax consistency.
Both are suitable for MtPCs.

## 4 MtPCs in Action: Retrofitting a Byte-Level LLM

We evaluate MtPC on the challenging tasks of speeding up byte-level LLMs.
MTP is crucial for byte-level LLMs as they require more tokens than sub-word LLMs to generate text with the same length.
Furthermore, byte-level LLMs allow us to explore large window sizes and more mixtures components due to their small vocabulary size.
We implement our MtPCs variants in the cirkit library (cirkit) and provide it in our supplementary materials.

Target model.
We work with EvaByte (evabyte2025) as our byte-level LLM, because it is open source, publicly available, and obtains results that are competitive to subword-level LLMs on benchmarks (evabyte2025).
EvaByte is a 6.5B byte-level model with an embedding size of 40964096, a vocabulary of 320320 byte tokens and a maximum context window of 32k bytes.
EvaByte has been pre-trained as an MTP model with a prediction window of n=8n=8 bytes.
In our experiments, we retrofit the released fine-tuned version of EvaByte, EvaByte-SFT (evabyte2025).
EvaByte-SFT has been fine-tuned on a data mix of Tülu 3 (lambert2024tulu3), OpenCoder (huang2024opencoder) stages one and two, and OpenHermes 2.5.666The information above is from personal communication with the authors.
We note that EvaByte’s solid performance on benchmarks is based on greedy decoding when the model is used for STP, which we name EvaByte-STP.
For this reason, we set EvaByte-STP as the target model for speculative decoding.
We do so to *accelerate generation without sacrificing generation quality*.

Draft models.
We use EvaByte-MTP to refer to EvaByte’s released fully-factorised ([FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")) MTP head.
Speculative decoding results have not been reported in the EvaByte release (evabyte2025), so we include them here as our baseline.
We also further fine-tune EvaByte-MTP to highlight that the model cannot be improved further.
On top of that, we replace the MTP head with our MtPCs heads, including our [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") implementation and novel [HMM](#S3.Ex3 "Equation HMM ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") heads to relax the independence assumptions of the [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") model and increase expressiveness.
We note that EvaByte-MTP-CP with r=1r=1 is equivalent to EvaByte-MTP, as can be seen from [Eq. CP](#S2.Ex2 "In 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").

### 4.1 Training

In order to improve throughput via speculative decoding, we need to make our MTP model’s distribution as similar as possible to EvaByte-STP’s.
We achieve this in the simplest way by instruction fine-tuning our models on a similar data mix to that used for EvaByte-SFT.
As the full details of the data mix are not known and are hard to replicate, we focus on Tülu 3.

Training data.
We fine-tune on the Tülu 3 SFT mix dataset (lambert2024tulu3) which contains 939,344 examples of user/assistant interactions on 18 tasks.
We split the Tulu 3 dataset into training and validation so that we can check throughput on the unseen validation examples.
In order to make sure all tasks are sampled, we shuffle the training data before splitting.
Because we want training to be possible on 2×802\times 80 Gb GPUs, we limit the context length to 81928192 bytes and filter out 34,067 examples which are longer.
We split the remaining 905,277 examples into 99%99\% train and 1%1\% validation.

Initialisation.
We initialise our MTP heads from EvaByte-SFT in a way that guarantees that our EvaByte-MTP-CP is equivalent to EvaByte-MTP.
This guarantees that we leverage previous training: all models start from the same loss and we smoothly move in parameter space from EvaByte-MTP to our more expressive EvaByte-MTP-CP, EvaByte-MTP-HMM and EvaByte-MTP-BTree.

Loss.
We train our MTP models on the packed train split of Tülu 3 with a batch size of 256256 sequences, or ≈2\approx 2m tokens, which is what EvaByte used.
We first train our MTP heads for 1 epoch ([Section 4.3](#S4.SS3 "4.3 MtPCs without Adapters ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).
Then we load the models and continue training for an additional epoch with LoRA ([Section 4.4](#S4.SS4 "4.4 MtPCs with Adapters ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).
We apply EvaByte’s chat template and only train on the assistant’s answers.
We use overlapping prediction windows, as we need to be able to begin speculative decoding from any position during generation.
We minimise the negative log-likelihood of the observed assistant outputs [Eq. 4](#S4.E4 "In 4.1 Training ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), where NN is the number of training sequences and LL is the sequence length for each token in the window.777Our loss over overlapping windows is a composite log-likelihood (varin2011composite).

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ=∑j=1nγj−1​ℒj,ℒj=−∑i=1N∑t=1L(log⁡pθ​(xt+j(i)∣𝐱<t+j(i)))/(N​valid​(i,j))\mathcal{L}=\sum\nolimits\_{j=1}^{n}\gamma^{j-1}\mathcal{L}\_{j},\quad\mathcal{L}\_{j}=-\sum\nolimits\_{i=1}^{N}\sum\nolimits\_{t=1}^{L}({\log p\_{\theta}(x\_{t+j}^{(i)}\mid\mathbf{x}\_{<t+j}^{(i)})})/({N\text{valid}(i,j)}) |  | (4) |

This involves locally normalising the loss by the number of valid tokens for example ii and output jj in the MTP window, valid​(i,j)\text{valid}(i,j).
As in cai2024medusa, we apply exponential discounting for future tokens in the window, but use γ=0.9\gamma=0.9 instead of γ=0.8\gamma=0.8 to account for n>8n>8.
We use the Adam optimiser (kingma2014adam) with a fixed learning rate of 3×10−43\times 10^{-4}.

### 4.2 Metrics

To speed up LLMs generation with speculative decoding, we need to balance speed and expressiveness.
We measure speed using mean latency (μlat\mu\_{\text{lat}}) and expressiveness via the mean acceptance rate (μacc\mu\_{\text{acc}}; li2024eagle), as defined below.
Our goal is to increase throughput.
We obtain a relative throughput speed-up of one method over another by measuring their wall-time speedup ratio (li2024eagle; cai2024medusa).
We assume a batch size of 11 for all evaluations. We report our metrics on two GPUs, the server-grade NVIDIA L40S GPU and the desktop-grade NVIDIA RTX 3090.

Mean Latency
μlat\mu\_{\text{lat}} is the average time taken for each speculative decoding step, *i.e.*, the time needed for the draft model to generate a candidate sequence and the verifier to choose which tokens to accept.
μlat\mu\_{\text{lat}} is higher for less efficient LLMs and MTP heads, and lower for more powerful GPUs, *e.g.* for EvaByte-MTP the L40S ([Tables 1](#S4.T1.fig1 "In 4.3 MtPCs without Adapters ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), [2](#S4.T2.fig1 "Table 2 ‣ 4.3 MtPCs without Adapters ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and [3](#S4.T3.fig1 "Table 3 ‣ 4.4 MtPCs with Adapters ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")) has half the latency of the RTX 3090 ([Tables 4](#A4.T4.fig1 "In Appendix D Additional Results on an RTX 3090 ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), [5](#A4.T5.fig1 "Table 5 ‣ Appendix D Additional Results on an RTX 3090 ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and [6](#A4.T6 "Table 6 ‣ Appendix D Additional Results on an RTX 3090 ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).

Mean acceptance rate
μacc\mu\_{\text{acc}} is the percentage of drafted tokens that are accepted by the target model.
More expressive draft models will have higher accepance rate as they will better approximate the target distribution.
μacc\mu\_{\text{acc}} depends on the size of the MTP window, nn, as we have μacc∈[0,n]\mu\_{\text{acc}}\in[0,n].

Mean throughput μtok/s\mu\_{\text{tok/s}}is measured as in [Eq. 2](#S2.E2 "In 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), i.e., as the ratio μacc/μlat{\mu\_{\text{acc}}}/{\mu\_{\text{lat}}}.

Wall-time speed-up ratio is the relative speed-up of a proposed model compared to a baseline model, measured as the ratio of their throughputs. As baselines, we use autoregressive generation from the STP model, EvaByte-STP, and MTP with independence assumptions, EvaByte-MTP FF.

### 4.3 MtPCs without Adapters

* RQ1:

  Can we increase throughput by increasing the number of mixture components?

We begin with the simplest PC from our framework, MtPC-cp, which relaxes the independence assumption of the widely used MtPC-ff (r=1r=1) by increasing the number of mixture coefficients, rr.
MtPC-cp can increase throughput because it is more expressive yet still very efficient.

[Table 1](#S4.T1.fig1 "In 4.3 MtPCs without Adapters ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") highlights MtPC-cp’s efficiency; the μlat\mu\_{\text{lat}} introduced by MtPC-cp remains relatively unchanged as we increase rr, because the forward pass cost of the output layer is dominated by the expensive LLM calls.
At the same time, MtPC-cp increases the expressiveness of our MTP head by relaxing the unrealistic independence assumptions.
As a result, MtPC-cp with r=128r=128 achieves μacc=5.94\mu\_{\text{acc}}=5.94, an increase of .82.82 tokens over MtPC-ff.
However, the best throughput is obtained for r=32r=32, where MtPC-cp produces 20.820.8 more tok/s than MtPC-ff.
In the last column, we show the maximum attainable throughput (maxtok/s\max\_{\text{tok/s}}), *i.e.*, we disable speculative decoding and accept all tokens. The price paid in throughput for guaranteeing no loss in generation quality is ≈90\approx 90 tok/s for r=32r=32.
While MtPC-cp performs well for n=8n=8, the margin for further improving throughput is small.
This is because for n=8n=8, we can at best achieve μacc=8\mu\_{\text{acc}}=8, and we have already achieved μacc=5.94\mu\_{\text{acc}}=5.94 and have hit diminishing returns.
To obtain substantial boosts in throughput, we need to extend our model to longer window sizes.
Since r=32r=32 worked best, we keep this fixed for the remaining experiments.

| model | rr | μacc\mu\_{\text{acc}} ↑\uparrow | μlat\mu\_{\text{lat}} ↓\downarrow | μtok/s\mu\_{\text{tok/s}} ↑\uparrow | maxtok/s\max\_{\text{tok/s}} |
| --- | --- | --- | --- | --- | --- |
| \rowcolorgray!15 [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 1 | 5.14 ±\pm 0.06 | 0.0290 ±\pm 0.0002 | 180.1 ±\pm 2.8 | 297.50 |
| [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 8 | 5.65 ±\pm 0.02 | 0.0296 ±\pm 0.0001 | 194.5 ±\pm 1.6 | 291.61 |
| 16 | 5.76 ±\pm 0.03 | 0.0299 ±\pm 0.0002 | 196.1 ±\pm 1.9 | 295.94 |
| 32 | 5.84 ±\pm 0.01 | 0.0297 ±\pm 0.0002 | 200.9 ±\pm 1.6 | 292.33 |
| 64 | 5.87 ±\pm 0.09 | 0.0304 ±\pm 0.0001 | 197.2 ±\pm 2.3 | 278.42 |
| 128 | 5.94 ±\pm 0.04 | 0.0320 ±\pm 0.0001 | 188.6 ±\pm 1.1 | 265.51 |

  


Table 1: Increasing the mixture components (rr) increases the throughput (μtok/s\mu\_{\text{tok/s}}) as seen for MtPC-cp (n=8n=8) over our baseline, EvaByte-MTP (FF) (in gray) where
we report the mean ±\pm std over three sets of 250250 prompts.
MtPC-cp increases throughput: it has a larger acceptance rate (μacc\mu\_{\text{acc}}) while latency (μlat\mu\_{\text{lat}}) is almost constant in rr.

* RQ2:

  Do we benefit from more expressive circuit architectures for longer sequences?

We now consider more expressive circuits, such as MtPC-hmm and MtPC-btree, and show that they outperform MtPC-cp for longer MTP windows, highlighting the importance of our extension to general PCs.
We fix r=32r=32 and explore the different PC architectures for both n=8n=8 and the longer window, n=16n=16.
[Table 2](#S4.T2.fig1 "In 4.3 MtPCs without Adapters ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") shows that MtPC-hmm obtains the best μacc\mu\_{\text{acc}} in both cases, however, it strikes an unfavourable balance in the expressiveness–latency trade-off:
Due to being AR, MtPC-hmm has the largest μlat\mu\_{\text{lat}}, and yields poor throughput as a result.
On the other hand, MtPC-btree almost matches the μacc\mu\_{\text{acc}} of MtPC-hmm and has a smaller μlat\mu\_{\text{lat}} footprint. Nevertheless, for n=8n=8, MtPC-cp still obtains the best μtok/s\mu\_{\text{tok/s}}.
However, when we move to n=16n=16, MtPC-btree substantially increases the gap in μacc\mu\_{\text{acc}} from MtPC-cp. This in turn leads to MtPC-btree having the best throughput, with 203.5203.5 tok/s, a speed-up of ×5.08\times 5.08 over EvaByte-STP.
While the gains already obtained by MtPC-btree are solid, fine-tuning the output layer alone can only get us so far.
This is because EvaByte has not been trained to produce representations that are good for predicting 1616 tokens ahead, as we discuss next.

| nn | rr | model | μacc\mu\_{\text{acc}} ↑\uparrow | μlat\mu\_{\text{lat}} ↓\downarrow | μtok/s\mu\_{\text{tok/s}} ↑\uparrow | speed-up ↑\uparrow |
| --- | --- | --- | --- | --- | --- | --- |
| \rowcolorgray!15 1 | 1 | STP | — | 0.0251 | 040.03±\pm 0.0 | 1.00 |
| \rowcolorbrown!15 8 | 1 | [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 5.14 ±\pm 0.06 | 0.0290 ±\pm 0.0002 | 180.1 ±\pm 2.8 | 4.50 |
| 32 | [HMM](#S3.Ex3 "Equation HMM ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 5.95 ±\pm 0.05 | 0.0332 ±\pm 0.0001 | 182.4 ±\pm 0.9 | 4.56 |
| 32 | [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 5.97 ±\pm 0.06 | 0.0310 ±\pm 0.0004 | 196.6 ±\pm 3.8 | 4.91 |
| 32 | [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 5.84 ±\pm 0.01 | 0.0297 ±\pm 0.0002 | 200.9 ±\pm 1.6 | 5.02 |
| \rowcolorbrown!15 16 | 1 | [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 5.38 ±\pm 0.08 | 0.0307 ±\pm 0.0004 | 179.6 ±\pm 3.8 | 4.49 |
| 32 | [HMM](#S3.Ex3 "Equation HMM ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 6.82 ±\pm 0.04 | 0.0397 ±\pm 0.0001 | 174.5 ±\pm 0.7 | 4.36 |
| 32 | [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 6.10 ±\pm 0.05 | 0.0322 ±\pm 0.0001 | 193.4 ±\pm 1.8 | 4.83 |
| 32 | [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 6.71 ±\pm 0.01 | 0.0337 ±\pm 0.0000 | 203.5 ±\pm 0.1 | 5.08 |

  


Table 2: More expressive architectures such as MtPC-btree outperform MtPC-cp in terms of throughput for n=8n=8 and n=16n=16 windows on an L40S GPU. For this experiment, we trained only model heads (no LoRAs layers). The shaded baselines are EvaByte-STP and the EvaByte-MTP (FF) models, trained for the same number of steps as our circuits for a fair comparison.

Takeaway 1:
While increasing the mixture components rr in CP is initially beneficial, it soon hits diminishing returns. Increasing the window size of future tokens nn and adopting more expressive PC architectures unlocks further gains in throughput. Furthermore, while HMM achieves the highest acceptance rates, it incurs high latency. Instead, non-autoregressive variants such as BTREE strike a better balance and hence should be preferred.

### 4.4 MtPCs with Adapters

* RQ3:

  Can we further increase throughput by adapting the draft LLM using LoRA?

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| nn | model | # LoRA | μacc\mu\_{\text{acc}} ↑\uparrow | μlat\mu\_{\text{lat}} ↓\downarrow | μtok/s\mu\_{\text{tok/s}} ↑\uparrow | speed-up ↑\uparrow |
| \rowcolorgray!15 1 | EvaByte-STP | 0 | — | 0.0251 | 40.03 | 1.00 |
| 8 | MtPC-[FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | \cellcolorbrown!150 | \cellcolorbrown!155.15 ±\pm 0.04 | \cellcolorbrown!150.0327 ±\pm 0.0013 | \cellcolorbrown!15163.7 ±\pm 10.5 | \cellcolorbrown!154.09 |
| 1 | 5.16 ±\pm 0.02 | 0.0308 ±\pm 0.0003 | 171.3 ±\pm 1.2 | 4.28 |
| 2 | 5.14 ±\pm 0.06 | 0.0336 ±\pm 0.0036 | 157.2 ±\pm 17.4 | 3.93 |
| 4 | 5.19 ±\pm 0.03 | 0.0330 ±\pm 0.0001 | 160.3 ±\pm 1.4 | 4.01 |
| MtPC-[BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | \cellcolorbrown!150 | \cellcolorbrown!156.04 ±\pm 0.02 | \cellcolorbrown!150.0326 ±\pm 0.0027 | \cellcolorbrown!15190.5 ±\pm 14.8 | \cellcolorbrown!154.76 |
| 1 | 6.15 ±\pm 0.02 | 0.0344 ±\pm 0.0038 | 185.1 ±\pm 20.3 | 4.62 |
| 2 | 6.20 ±\pm 0.05 | 0.0330 ±\pm 0.0000 | 193.0 ±\pm 1.4 | 4.82 |
| 4 | 6.20 ±\pm 0.04 | 0.0348 ±\pm 0.0001 | 183.1 ±\pm 1.0 | 4.57 |
| 16 | MtPC-[FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | \cellcolorbrown!150 | \cellcolorbrown!155.40 ±\pm 0.06 | \cellcolorbrown!150.0305 ±\pm 0.0001 | \cellcolorbrown!15180.3 ±\pm 2.3 | \cellcolorbrown!154.50 |
| 1 | 5.53 ±\pm 0.08 | 0.0311 ±\pm 0.0001 | 182.3 ±\pm 2.5 | 4.55 |
| 2 | 5.63 ±\pm 0.07 | 0.0321 ±\pm 0.0002 | 179.5 ±\pm 2.0 | 4.48 |
| 4 | 5.60 ±\pm 0.03 | 0.0356 ±\pm 0.0034 | 162.2 ±\pm 14.9 | 4.05 |
| MtPC-[BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | \cellcolorbrown!150 | \cellcolorbrown!156.86 ±\pm 0.03 | \cellcolorbrown!150.0340 ±\pm 0.0001 | \cellcolorbrown!15206.1 ±\pm 0.9 | \cellcolorbrown!155.15 |
| 1 | 7.32 ±\pm 0.03 | 0.0346 ±\pm 0.0000 | 218.0 ±\pm 0.6 | 5.45 |
| 2 | 7.53 ±\pm 0.10 | 0.0354 ±\pm 0.0001 | 219.1 ±\pm 3.0 | 5.47 |
| 4 | 7.58 ±\pm 0.14 | 0.0373 ±\pm 0.0003 | 210.2 ±\pm 5.0 | 5.25 |

  


Table 3: Fine-tuning separate layers in the draft model with LoRA adapters can increase the acceptance rate and speed up [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") MtPCs for n=16n=16 and two LoRA layers by 5.47 over STP on an L40s GPU. Nevertheless, the increased acceptance rate comes at increased latency, making further throughput boosts via more LoRA layers unviable for EvaByte. We shade the STP baseline in gray and ablated models trained for the additional epoch without LoRA in brown.

We now consider increasing the expressiveness by adding LoRA layers, as shown in [Table 3](#S4.T3.fig1 "In 4.4 MtPCs with Adapters ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").
We show that while we can improve throughput, we need to be strategic when choosing the number of layers, as very quickly the latency introduced outweighs the expressiveness gained.

The key here is that we need to balance the expressiveness obtained by adding LoRA layers and the latency we introduce because the additional layers are not shared between the draft and the verifier.
For example, if we train adapters for the last 16 (out of 32) layers, we can improve the acceptance rate by 37%, but we introduce a latency of 1.5×1.5\times the cost of a forward pass of the LLM.888We found that training more than 16 layers of EvaByte does not lead to improvements in acceptance rates.
The FF model for n=8n=8 has plateaued, highlighting its limited expressiveness.
We highlight that the improvements of MtPC are consistent across GPUs.
While throughput is ≈×2\approx\times 2 times larger for the server-grade GPU, the relative speed-ups are similar, see [Appendix D](#A4 "Appendix D Additional Results on an RTX 3090 ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").
Interestingly, due to the different balance between the LLM and MtPC latency across GPUs, on the RTX 3090 we hit diminishing returns after adding a single LoRA layer rather than two on the L40s.

Takeaway 2:
Fine-tuning a few layers of the draft model with LoRA increases the acceptance rate but also increases latency. The optimal trade-off is device-specific, but adding LoRAs is always beneficial compared with a fully shared LLM trunk.
Retrofitting models to longer MTP windows yields an even larger increase in throughput when paired with LoRAs.

## 5 Conclusion

Overall, our results show, for the first time, that throughput in MTP LLMs can be increased by 5.47×5.47\times w.r.t. AR and 1.22×1.22\times w.r.t. MTP with independence assumptions, while simultaneously guaranteeing the retention of an AR LLM’s quality. We achieved this goal by identifying key trade-offs between acceptance rates and latency within our framework, MtPC. We enhanced the expressiveness of MTP by getting rid of the independence assumption (gloeckle2024better; evabyte2025), introducing an explicit probabilistic model for inter-token dependencies that facilitates performance guarantees (ankner2024hydra; li2024eagle; deepseekai2024), and generalising mixture-based methods (basharin2024faster) into the PC framework.
Moreover, we decreased latency by modulating the number of layers shared between draft and verifier model branches.
We showcase the throughput gains of MtPC LLMs at scale by retrofitting EvaByte (evabyte2025), a state-of-the-art 6.5B byte-level LLM into our framework.999More in-depth commentary on related work is available in [Appendix H](#A8 "Appendix H Further Related Work ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").
In future work, our framework can be extended by integrating constraints during generation via methods such as Gelato (pmlr-v202-zhang23g) and Ctrl-G (ctrlg).
Unlike those, we would not need to train an auxiliary HMM in MtPCs and we can integrate constraints directly into our PC head.
Moreover, we can further boost expressiveness by leveraging other PCs architectures such as subtractive mixtures (loconte2024subtractive; loconte2025sum)
and continuous latent variable circuits (gala2024pic; gala2024scaling),
while reducing latency through recent advancements in scaling up PCs (liu2024scaling; zhang2025scaling).

#### Author Contributions

AV, AG and LL conceived the initial idea and later discussed it with the other co-authors.
AG was responsible for trading off expressivity and efficiency by limiting the number of LoRA layers and adapting speculative decoding for the self-speculative scenario.
LL was responsible for the speculative decoding implementation and the development of circuits (CP, HMM, BTree).
AG and LL were responsible for designing and coding models and experiments and trained the models with help from EvK, who also helped with training infrastructure.
YZ and PN provided useful feedback and helped with preliminary experiments.
EW provided funding and useful feedback.
PM provided evaluations, guidance and useful feedback.
EP helped design the experimental setup, including the choice of models and data.
AG, AV and EP wrote the paper with help from LL and EvK.
AV and EP supervised all phases of the project.

## Acknowledgements

We would like to thank the members of april lab and Ponti lab for useful feedback during early presentations of this work.

AG was funded by NatWest Group. EvK was funded by the ELIAI project “Gradient-based Learning of Complex Latent Structures”.
EP was funded by the ERC Starting Grant AToM-FM (101222956).
AG and AV were funded by the “UNREAL: Unified Reasoning Layer for Trustworthy ML” project (EP/Y023838/1) selected by the ERC and funded by UKRI EPSRC.

## Reproducibility statement

To ensure reproducibility for our research, we have attached the codebase for implementing all model variants and running their training and evaluation to our submission. In addition, we have provided full details on sampling in circuits in [Appendix B](#A2 "Appendix B Background on circuits ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and on our algorithms for speculative decoding in [Appendix C](#A3 "Appendix C Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").

## Appendix A Notation

We adapt
notation and nomenclature
from the tensor factorisation (kolda2009tensor) and circuit (loconte2025what) literature.

We denote ordered sets of random variables with 𝐗\bm{\mathrm{X}}, 𝐘\bm{\mathrm{Y}} and 𝐙\bm{\mathrm{Z}}, and we use [n][n] to express the set {1,2,…,n}\{1,2,\ldots,n\} with n>0n>0.
The domain of a variable XX is denoted as 𝖽𝗈𝗆​(X)\mathsf{dom}(X), and we denoted as 𝖽𝗈𝗆​(𝐗)=𝖽𝗈𝗆​(X1)×⋯×𝖽𝗈𝗆​(Xn)\mathsf{dom}(\bm{\mathrm{X}})=\mathsf{dom}(X\_{1})\times\cdots\times\mathsf{dom}(X\_{n}) the joint domain of variables 𝐗={Xi}i=1n\bm{\mathrm{X}}=\{X\_{i}\}\_{i=1}^{n}.
We denote scalars with lower-case letters (e.g., a∈ℝa\in\mathbb{R}), vectors with boldface lower-case letters (e.g., 𝐚∈ℝN\bm{\mathrm{a}}\in\mathbb{R}^{N}), matrices with boldface upper-case letters (excluding those used for variables, e.g., 𝐀∈ℝM×N\bm{\mathrm{A}}\in\mathbb{R}^{M\times N}), and tensors with boldface calligraphic letters (e.g., 𝓐∈ℝI1×I2×I3\bm{\mathcal{A}}\in\mathbb{R}^{I\_{1}\times I\_{2}\times I\_{3}}).
Moreover, we use subscripts to denote entries of tensors (e.g., ai​j​ka\_{ijk} is the (i,j,k)(i,j,k)-th entry in 𝓐\bm{\mathcal{A}}).

## Appendix B Background on circuits

Circuits have a long history in theoretical computer science (shpilka2010open) and probabilistic reasoning (darwiche2003differential; darwiche2009modeling).
In their more modern definition and application to machine learning (vergari2019tractable; choi2020pc), circuits are introduced as structured computational graphs, simplified neural networks where one is allowed to use units from a restricted set of neurons (sum, product and input units) and whose connections need to abide certain structural properties to guarantee tractability (choi2020pc; vergari2021compositional), as discussed next.

### B.1 Structural properties

Tractability is to be intended as the ability to exactly compute a given function (operation) over the circuit in time that is polynomial in its size, denoted as |c||c| for a circuit cc, and representing the number of edges between the computational units.
For example, a circuit
cc can exactly integrate *any subset of variables* in time 𝒪​(|c|)\mathcal{O}(|c|) if (i) its input functions can be integrated efficiently and (ii) it is *smooth* and *decomposable* (darwiche2002knowledge; choi2020pc).

###### Definition 1 (Smoothness and decomposability (darwiche2002knowledge; choi2020pc)).

A circuit is *smooth* if for every sum unit nn, all its input units depend on the same variables, i.e., ∀i,j∈𝗂𝗇​(n):𝗌𝖼​(i)=𝗌𝖼​(j)\forall i,j\in{\mathsf{in}}(n)\colon{\mathsf{sc}}(i)={\mathsf{sc}}(j).
A circuit is *decomposable* if the distinct inputs of every product unit nn depend on disjoint sets of variables, i.e., ∀i,j∈𝗂𝗇​(n)​i≠j:𝗌𝖼​(i)∩𝗌𝖼​(j)=∅\forall i,j\in{\mathsf{in}}(n)\>i\neq j\colon{\mathsf{sc}}(i)\cap{\mathsf{sc}}(j)=\varnothing.

Note that all the PC architectures we have discussed in this paper, [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), [HMM](#S3.Ex3 "Equation HMM ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") and [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), are smooth and decomposable circuits. The reader is encouraged to check this by themselves for the architectures in [Fig. 2](#S3.F2 "In 3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits").

Exactly integrating variables out is relevant to compute marginals such as the normalisation constant of the distribution encoded by the circuit ([Eq. 3](#S3.E3 "In 3.1 Probabilistic Circuits ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).
Note that in our implementation, circuits are normalised by design (peharz2015theoretical), as we assume that input distributions are normalised categoricals and all sum units form a convex combination as their weights are parameterised with a softmax function (see [Section 3.3](#S3.SS3 "3.3 Parameterising PCs with LLMs ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).

More importantly for our MtPCs, we can draw samples efficiently from the distribution of a circuit that is both smooth and decomposable, as we discuss in the next sub-section.

### B.2 Sampling a circuit

A smooth and decomposable PC can use ancestral sampling to generate a complete sample for all nn tokens in a window.
In a nutshell, we can iteratively sample each latent variable in the hierarchy encoded by the PC, and then sample the selected input distributions, in the same way one sample one (hierarchical) mixture model by first sampling one component and then drawing a sample from that component.

Operationally, [Algorithm 1](#alg1 "In B.2 Sampling a circuit ‣ Appendix B Background on circuits ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") details the procedure.
We have to sample one input branch for each sum unit we encounter when performing a backward traversal of the circuit computational graph (from the circuit output back to the input distributions).
Such a branch is sampled proportionally to the sum unit weights ωj\omega\_{j}, which encode the mixture components (or equivalently the transition probabilities in an HMM).
Then, when we traverse a product unit, we follow all its input branches.
When we reach an input unit, we sample a token proportionally to the parameters ϕi​j\phi\_{ij} of the categorical distributions encoded in the unit.

Algorithm 1  Sample​(c)\textsc{Sample}(c)

Input: A smooth, decomposable and normalised PC cc encoding a joint distribution qq over the next nn tokens 𝐗={X1,…,Xn}\bm{\mathrm{X}}=\{X\_{1},\ldots,X\_{n}\}
Output: a sample 𝐱∼q​(𝐗)\mathbf{x}\sim q(\bm{\mathrm{X}}).

  

1:𝐱←𝗓𝖾𝗋𝗈𝖾𝗌​(n)\bm{\mathrm{x}}\leftarrow\mathsf{zeroes}(n)
⊳\triangleright init empty sample

2:cn←𝗈𝗎𝗍𝗉𝗎𝗍​(c)c\_{n}\leftarrow\mathsf{output}(c)

3:𝒩←𝗊𝗎𝖾𝗎𝖾​({cn})\mathcal{N}\leftarrow\mathsf{queue}(\{c\_{n}\})
⊳\triangleright traverse the computational graph from outputs to inputs

4:while 𝒩\mathcal{N} not empty do

5:  cn←𝗉𝗈𝗉​(𝒩)c\_{n}\leftarrow\mathsf{pop}(\mathcal{N})

6:  if cn=∑j=1rωj​cjc\_{n}=\sum\_{j=1}^{r}\omega\_{j}c\_{j} then
⊳\triangleright cnc\_{n} is a sum unit

7:   k←𝗌𝖺𝗆𝗉𝗅𝖾𝖢𝖺𝗍𝖾𝗀𝗈𝗋𝗂𝖼𝖺𝗅​(ω1,…,ωr)k\leftarrow\mathsf{sampleCategorical}(\omega\_{1},\ldots,\omega\_{r})
⊳\triangleright sample from a categorical with rr states

8:   𝒩←𝗉𝗎𝗌𝗁​(𝒩,ck)\mathcal{N}\leftarrow\mathsf{push}(\mathcal{N},c\_{k})

9:  else if cn=∏j=1dcjc\_{n}=\prod\_{j=1}^{d}c\_{j} then
⊳\triangleright cnc\_{n} is a product unit with dd inputs

10:   for k=1​…​dk=1\ldots d do

11:     𝒩←𝗉𝗎𝗌𝗁​(𝒩,ck)\mathcal{N}\leftarrow\mathsf{push}(\mathcal{N},c\_{k})
⊳\triangleright visit all inputs of cnc\_{n}

12:  else if cnc\_{n} is an input unit over variable XiX\_{i} and parameters ϕi\phi\_{i} then

13:   xi←𝗌𝖺𝗆𝗉𝗅𝖾𝖢𝖺𝗍𝖾𝗀𝗈𝗋𝗂𝖼𝖺𝗅​(ϕi)x\_{i}\leftarrow\mathsf{sampleCategorical}(\phi\_{i})
⊳\triangleright sample from a categorical with v=|𝒱|v=|\mathcal{V}| states

14:return 𝐱\bm{\mathrm{x}}

If the circuit is smooth and decomposable, by this process we are guaranteed to end up in a set of input units whose scope is the full set of tokens 𝐗\bm{\mathrm{X}} and in which only one input unit is selected per token position ii (line 13 of [Algorithm 1](#alg1 "In B.2 Sampling a circuit ‣ Appendix B Background on circuits ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).
This procedure can be tensorized as to efficiently generate a batch of samples in a single pass over the computational graph of the circuit (vergari2019visualizing; peharz2019ratspn; peharz2020einsum; loconte2025what; liu2024scaling).

Lastly, we remark that this routine is potentially computationally more efficient than the one implemented in basharin2024faster, as the latter is based on autoregressive inverse transform sampling (see loconte2024subtractive for a discussion) and requires sampling one token at a time.

## Appendix C Speculative Decoding

We give pseudocode for our self-speculative decoding algorithm below.
The algorithm accepts between 0 and nn tokens, but always generates between 11 and nn tokens, where nn is the MTP window size.
The algorithm is very similar to vanilla speculative decoding (leviathan2022fast), but our algorithm includes a modification that reduces latency for the self-speculative scenario, and for this it needs to sacrifice the last “free” token typically obtained from the verifier.
The gain in latency is possible because we can evaluate the shared LLM once per draft/verify cycle, while a naive implementation of leviathan2022fast for self-speculative decoding would need two, approximately halving the possible throughput.

In our self-speculative setup, the verifier and draft LLMs share some layers of the backbone. Importantly, the verifier is always computing LLM states ahead of the draft.
As such, we can get away with a single forward pass through the shared LLM, similar to Medusa (cai2024medusa), by re-using the LLM backbone state computed by the verifier for the draft model. For this to work, we cannot accept a “last sample for free” from the verifier (lines 23-30) [Algorithm 3](#alg3 "In Appendix C Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")),
as we would not have the backbone state for this new token and it is not worth paying an extra LLM evaluation for it.
Therefore, in our algorithm we only sample the “free” token from the verifier in the rare case that no tokens are accepted. This is necessary because the model can get caught in successive no-accept states in the sampling case, or get stuck in an infinite loop if we used greedy decoding.
If any tokens were accepted, we use the last state of the shared backbone computed during the verify phase to seed the draft phase. In what follows, if we have no LoRA layers, the algorithm is modified to have a single component: the shared encoder.

Algorithm 2  SharedStateSelfSpeculativeDecoding

Architecture Components:

Three components: Shared Encoder (SS), Verifier (VV), Draft (DD)

Each with their own KV-cache

Given: A prompt of length LL

Initialisation:

Prefill VV to L−1L-1, and DD and SS to LL

Set SS and DD state

Switch to draft/verify cycle:

while true do

Draft stage:

if SS state is not set then

Compute SS by conditioning on the additional token

Use SS state to compute DD state

Parameterize MTPC with DD state

Draft nn tokens

Verify stage:

Compute SS state on n+1n+1 tokens (draft + predecessor)

Compute VV state using SS state

Obtain up to n+1n+1 tokens from speculative decoding

if 0 tokens accepted then

Keep “free” token sampled from last valid logits

Unset SS state (stale)

else

Accept nn tokens (drop “free” token)

Set SS state (hidden state for last accepted token)

Algorithm 3  SelfSpeculativeDecoding​(𝐱≤t,f,h,c,g)\textsc{SelfSpeculativeDecoding}(\bm{\mathrm{x}}\_{\leq t},f,h,c,g)

Input: A prefix 𝐱≤t\bm{\mathrm{x}}\_{\leq t} of length tt, an LLM backbone f:𝒱∗→ℝdf\colon\mathcal{V}^{\*}\to\mathbb{R}^{d}, an LLM head h:ℝd→𝚯h\colon\mathbb{R}^{d}\to\bm{\mathrm{\Theta}} parameterising a PC cc encoding a joint distribution qq over the next nn tokens, and an LLM head g:ℝd→Δvg\colon\mathbb{R}^{d}\to\Delta^{v} computing the next token probabilities.
  
Output: A sentence (𝐱≤t||𝐱t+1:t+s)∈𝒱t+s(\bm{\mathrm{x}}\_{\leq t}\>||\>\bm{\mathrm{x}}\_{t+1:t+s})\in\mathcal{V}^{t+s} where 1≤s≤n+11\leq s\leq n+1.
Moreover, we have that 𝐱t+1:t+s∼p​(xt+1,…,xt+s∣𝐱≤t)\bm{\mathrm{x}}\_{t+1:t+s}\sim p(x\_{t+1},\ldots,x\_{t+s}\mid\bm{\mathrm{x}}\_{\leq t}) as equivalently encoded by the autoregressive single token prediction model consisting of ff and gg only (leviathan2022fast).

  

1:𝐞t←f​(𝐱≤t)\bm{\mathrm{e}}\_{t}\leftarrow f(\bm{\mathrm{x}}\_{\leq t})
⊳\triangleright Compute the last embedding

2:𝜽←h​(𝐞t)\bm{\mathrm{\theta}}\leftarrow h(\bm{\mathrm{e}}\_{t})
⊳\triangleright Compute the circuit parameters

3:

4:Let q​(𝐗t+1:t+n∣𝐱≤t)=1Z𝜽​c​(𝐗t+1:t+n∣𝜽)q(\bm{\mathrm{X}}\_{t+1:t+n}\mid\bm{\mathrm{x}}\_{\leq t})=\frac{1}{Z\_{\bm{\mathrm{\theta}}}}c(\bm{\mathrm{X}}\_{t+1:t+n}\mid\bm{\mathrm{\theta}})

5:𝐱t+1:t+n∼q​(𝐗t+1:t+n∣𝐱≤t)\bm{\mathrm{x}}\_{t+1:t+n}\sim q(\bm{\mathrm{X}}\_{t+1:t+n}\mid\bm{\mathrm{x}}\_{\leq t})
⊳\triangleright Sample nn tokens from cc in time 𝒪​(|c|)\mathcal{O}(|c|)

6:𝐱←𝐱≤t||𝐱t+1:t+n\bm{\mathrm{x}}\leftarrow\bm{\mathrm{x}}\_{\leq t}\>||\>\bm{\mathrm{x}}\_{t+1:t+n}
⊳\triangleright Concatenate the prefix with the nn tokens

7:

8:Compute in parallel for 1≤i≤n1\leq i\leq n:
⊳\triangleright Compute marginals in time 𝒪​(|c|)\mathcal{O}(|c|)

9:  q​(𝐱t+1:t+i∣𝐱≤t)=∑xt+i+1,…,xt+n∈𝒱q​(𝐱t+1:t+n∣𝐱≤t)q(\bm{\mathrm{x}}\_{t+1:t+i}\mid\bm{\mathrm{x}}\_{\leq t})=\sum\_{x\_{t+i+1},\ldots,x\_{t+n}\in\mathcal{V}}q(\bm{\mathrm{x}}\_{t+1:t+n}\mid\bm{\mathrm{x}}\_{\leq t})

10:

11:Compute in parallel for 1≤i≤n+11\leq i\leq n+1:
⊳\triangleright Compute target model conditionals

12:  p​(Xt+i∣𝐱≤t+i−1)=g​(𝐞t+i−1)p(X\_{t+i}\mid\bm{\mathrm{x}}\_{\leq t+i-1})=g(\bm{\mathrm{e}}\_{t+i-1}), where 𝐞t+i−1=f​(𝐱≤t+i−1)\bm{\mathrm{e}}\_{t+i-1}=f(\bm{\mathrm{x}}\_{\leq t+i-1})

13:

14:s←0s\leftarrow 0
⊳\triangleright Determine the number of accepted tokens ss, 0≤s≤n0\leq s\leq n

15:while s<ns<n do

16:  α∼𝒰​(0,1)\alpha\sim\mathcal{U}(0,1)

17:  if s>0s>0 then

18:   q​(xt+s+1∣𝐱≤t+s)←q​(𝐱t+1:t+s+1∣𝐱≤t)/q​(𝐱t+1:t+s∣𝐱≤t)q(x\_{t+s+1}\mid\bm{\mathrm{x}}\_{\leq t+s})\leftarrow q(\bm{\mathrm{x}}\_{t+1:t+s+1}\mid\bm{\mathrm{x}}\_{\leq t})/q(\bm{\mathrm{x}}\_{t+1:t+s}\mid\bm{\mathrm{x}}\_{\leq t})

19:  if α>p​(xt+s+1∣𝐱t+s)/q​(xt+s+1∣𝐱t+s)\alpha>p(x\_{t+s+1}\mid\bm{\mathrm{x}}\_{t+s})/q(x\_{t+s+1}\mid\bm{\mathrm{x}}\_{t+s}) then

20:   exit loop

21:  s←s+1s\leftarrow s+1

22:

23:⊳\triangleright Sample one last token from the autoregressive LLM model

24:if s<ns<n then
⊳\triangleright Adjust the distribution first, if we accept fewer tokens

25:  Let s​(Xt+s+1)=q​(Xt+s+1∣𝐱≤t+s)s(X\_{t+s+1})=q(X\_{t+s+1}\mid\bm{\mathrm{x}}\_{\leq t+s})

26:  Let m​(Xt+s+1)=max⁡(0,p​(Xt+s+1∣𝐱≤t+s)−s​(Xt+s+1))m(X\_{t+s+1})=\max\left(0,\>p(X\_{t+s+1}\mid\bm{\mathrm{x}}\_{\leq t+s})-s(X\_{t+s+1})\right)

27:  r​(Xt+s+1∣𝐱t+s)=m​(Xt+s+1)/Zr(X\_{t+s+1}\mid\bm{\mathrm{x}}\_{t+s})=m(X\_{t+s+1})/Z, with Z=∑x′∈𝒱m​(x′)Z=\sum\_{x^{\prime}\in\mathcal{V}}m(x^{\prime})

28:  xt+s+1∼r​(Xt+s+1∣𝐱≤t+s)x\_{t+s+1}\sim r(X\_{t+s+1}\mid\bm{\mathrm{x}}\_{\leq t+s})

29:else

30:  xt+s+1∼p​(Xt+s+1∣𝐱≤t+s)x\_{t+s+1}\sim p(X\_{t+s+1}\mid\bm{\mathrm{x}}\_{\leq t+s})

31:return 𝐱≤t+s||xt+s+1\bm{\mathrm{x}}\_{\leq t+s}\>||\>x\_{t+s+1}

## Appendix D Additional Results on an RTX 3090

| circuit | rr | μacc\mu\_{\text{acc}} ↑\uparrow | μlat\mu\_{\text{lat}} ↓\downarrow | μtok/s\mu\_{\text{tok/s}} ↑\uparrow | maxtok/s\max\_{\text{tok/s}} |
| --- | --- | --- | --- | --- | --- |
| \rowcolorgray!15 [FF](#S2.Ex1 "Equation FF ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 1 | 5.12 ±\pm 0.03 | 0.0519 ±\pm 0.0003 | 101.2 ±\pm 0.1 | 167.69 |
| [CP](#S2.Ex2 "Equation CP ‣ 2 Speeding up Generation with MTP and Speculative Decoding ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") | 8 | 5.65 ±\pm 0.03 | 0.0545 ±\pm 0.0003 | 106.3 ±\pm 0.4 | 161.49 |
| 16 | 5.78 ±\pm 0.06 | 0.0530 ±\pm 0.0002 | 112.2 ±\pm 1.0 | 159.22 |
| 32 | 5.84 ±\pm 0.04 | 0.0532 ±\pm 0.0004 | 113.1 ±\pm 0.5 | 158.55 |
| 64 | 5.88 ±\pm 0.03 | 0.0532 ±\pm 0.0001 | 113.8 ±\pm 0.8 | 155.04 |
| 128 | 5.94 ±\pm 0.03 | 0.0533 ±\pm 0.0001 | 114.8 ±\pm 0.4 | 153.24 |

  


Table 4: Increasing the mixture components (rr) increases the throughput (μtok/s\mu\_{\text{tok/s}}) as seen for MtPC-cp (n=8n=8) over our baseline on an RTX 3090, EvaByte-MTP (FF) (in gray) where
we report the mean ±\pm std over three sets of 250250 prompts.
MtPC-cp increases throughput as it has a larger acceptance rate (μacc\mu\_{\text{acc}}) and a latency (μlat\mu\_{\text{lat}}) that is constant in rr.



| nn | rr | model | μacc\mu\_{\text{acc}} ↑\uparrow | μlat\mu\_{\text{lat}} ↓\downarrow | μtok/s\mu\_{\text{tok/s}} ↑\uparrow | speed-up ↑\uparrow |
| --- | --- | --- | --- | --- | --- | --- |
| \rowcolorgray!15 1 | 1 | STP | — | 0.047 | 21.4 ±\pm 0.0 | 1.00 |
| \rowcolorgray!15 8 | 1 | FF | 5.12 ±\pm 0.03 | 0.0519 ±\pm 0.0003 | 101.2 ±\pm 0.1 | ×\times 4.73 |
| 32 | HMM | 5.97 ±\pm 0.05 | 0.0594 ±\pm 0.0001 | 103.3 ±\pm 1.0 | ×\times 4.83 |
| 32 | BTREE | 5.94 ±\pm 0.02 | 0.0546 ±\pm 0.0005 | 111.9 ±\pm 1.1 | ×\times 5.23 |
| 32 | CP | 5.84 ±\pm 0.04 | 0.0532 ±\pm 0.0004 | 113.1 ±\pm 0.5 | ×\times 5.29 |
| \rowcolorgray!15 16 | 1 | FF | 5.38 ±\pm 0.03 | 0.0530 ±\pm 0.0003 | 104.3 ±\pm 1.2 | ×\times 4.87 |
| 32 | HMM | 6.81 ±\pm 0.07 | 0.0701 ±\pm 0.0002 | 99.7 ±\pm 0.8 | ×\times 4.66 |
| 32 | CP | 6.13 ±\pm 0.03 | 0.0547 ±\pm 0.0003 | 115.4 ±\pm 1.3 | ×\times 5.39 |
| 32 | BTREE | 6.67 ±\pm 0.07 | 0.0578 ±\pm 0.0005 | 118.9 ±\pm 2.5 | ×\times 5.56 |

  


Table 5: More expressive architectures such as MtPC-btree outperform MtPC-cp on longer windows in terms of throughput for n=8n=8 and n=16n=16 for no LoRA models on an RTX 3090 GPU. We shade the baselines in gray, these are EvaByte-STP and the FF models trained for the same steps as our circuits.



| nn | model | # LoRA | μacc\mu\_{\text{acc}} ↑\uparrow | μlat\mu\_{\text{lat}} ↓\downarrow | μtok/s\mu\_{\text{tok/s}} ↑\uparrow | speed-up↑\uparrow |
| --- | --- | --- | --- | --- | --- | --- |
| \rowcolorgray!15 1 | STP | 0 | — | 0.047 | 21.40 | 1.00 |
| \rowcolorbrown!15 8 | FF | 0 | 5.11 | 0.0538 | 97.2 | 4.54 |
| 8 | FF | 1 | 5.09 | 0.0564 | 92.6 | 4.33 |
| 8 | FF | 2 | 5.11 | 0.0567 | 92.6 | 4.33 |
| 8 | FF | 4 | 5.11 | 0.0601 | 87.1 | 4.07 |
| \rowcolorbrown!15 8 | BTREE | 0 | 6.08 | 0.0568 | 110.0 | 5.14 |
| 8 | BTREE | 1 | 6.15 | 0.0581 | 109.4 | 5.11 |
| 8 | BTREE | 2 | 6.17 | 0.0604 | 105.7 | 4.94 |
| 8 | BTREE | 4 | 6.18 | 0.0625 | 102.3 | 4.78 |
| \rowcolorbrown!15 16 | FF | 0 | 5.48 | 0.0546 | 102.8 | 4.81 |
| 16 | FF | 1 | 5.55 | 0.0559 | 102.1 | 4.77 |
| 16 | FF | 2 | 5.51 | 0.0584 | 97.2 | 4.54 |
| 16 | FF | 4 | 5.63 | 0.0613 | 94.5 | 4.42 |
| \rowcolorbrown!15 16 | BTREE | 0 | 6.92 | 0.0587 | 121.4 | 5.67 |
| 16 | BTREE | 1 | 7.26 | 0.0617 | 122.0 | 5.70 |
| 16 | BTREE | 2 | 7.30 | 0.0627 | 120.9 | 5.65 |
| 16 | BTREE | 4 | 7.47 | 0.0669 | 116.2 | 5.43 |

Table 6: Fine-tuning separate layers in the draft model with LoRA can increase the acceptance rate and speed up [BTree](#S3.Ex4 "Equation BTree ‣ 3.2 PC Architectures for MTP ‣ 3 Probabilistic Circuits for Multi-Token Prediction ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits") MtPCs for n=16n=16 and one LoRA layer by 5.70 over STP on an RTX 3090 GPU. Nevertheless, the increased acceptance rate comes at increased latency, making further throughput boosts via more LoRA layers unviable for EvaByte. We shade the STP baseline in gray and ablated models trained for the additional epoch without LoRA in brown. Interestingly, for the L40S in [Table 3](#S4.T3.fig1 "In 4.4 MtPCs with Adapters ‣ 4 MtPCs in Action: Retrofitting a Byte-Level LLM ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits"), we still got improvements with two LoRA layers, which highlights the importance of carrying out such an analysis across devices.

## Appendix E Hidden Markov Models Setup

In our experiments we use contextual, inhomogeneous hidden Markov models (HMMs) with identity initialisation (see [Table 7](#A5.T7 "In Appendix E Hidden Markov Models Setup ‣ Fast and Expressive Multi-Token Prediction with Probabilistic Circuits")).
We chose the above after preliminary experiments where we assessed the following configuration choices for training our HMMs.

Parameterisation
We can parameterise HMMs to either be contextual, i.e. we can make the transition probabilities depend on the input, or we can make the transition probabilities be independent of the input (non-contextual).

Transition Type
The transition matrix can be the same at each time step (homogeneous) or it can be different (inhomogeneous).
The former would correspond to additional parameter sharing across sum layers in the circuit representation.

Initialisation
A crucial setup is to initialise the HMM with transition matrices that are identity matrices, which make the HMM equivalent to CP at the beginning of training. We achieve this by adding a bias term to allow the HMM model to be initialised to identity matrices. This setting in combination with extending to larger token windows, i.e. n=16n=16 lead to a scenario where HMMs outperform CP.
The other alternative is to initialise the transition matrices uniformly at random (before softmax), but this complicates learning and yields performance that is lower than CP models.

Table 7: Most Successful HMM Configuration

| Parameterisation | | Transition Type | | Initialisation | |
| --- | --- | --- | --- | --- | --- |
| Contextual | Non-Contextual | Homogeneous | Inhomogeneous | Identity Init. | Uniform Init. |
| ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |

## Appendix F Further Experimental Details

To make the comparison between methods fair, we: a) constrained the models to not produce end-of-sequence symbols during generation, as the latency of retrieving KV cache items from memory increases with sequence length (nawrot2024dynamic) and b) we filtered the validation set of the models to only include examples with both prompts and responses in English, as acceptance rates may vary dramatically based on the language chosen for the response.

We compute throughput by generating answers to 250250 prompts and report the mean and std of 33 runs with different prompts.

## Appendix G Alternative Losses

In early versions of this work we also experimented using a Kullback-Leibler divergence (KL) loss as recommended by basharin2024faster.
However, we found that training with the KL loss doubled the training time while requiring a lot more memory, and the benefits in acceptance rate did not outweigh the additional complexity. For completeness we include the loss below.
KL Loss ℒ\mathcal{L}

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ=∑j=1nℒj​γj−1,ℒj=∑i=1N∑t=1LfKL(pθ(xt+j(i)∣𝐱<t+j(i))∥qθ′(xt+j(i)∣𝐱<t+j(i)))N​valid​(i,j)\mathcal{L}=\sum\_{j=1}^{n}\mathcal{L}\_{j}\gamma^{j-1},\quad\mathcal{L}\_{j}=\sum\_{i=1}^{N}\sum\_{t=1}^{L}\frac{f\_{\text{KL}}\left(p\_{\theta}(x\_{t+j}^{(i)}\mid\mathbf{x}\_{<t+j}^{(i)})\,\middle\|\,q\_{\theta^{\prime}}(x\_{t+j}^{(i)}\mid\mathbf{x}\_{<t+j}^{(i)})\right)}{N\text{valid}(i,j)} |  | (5) |

In the above we condition both the draft model, qθ′q\_{\theta^{\prime}}, and the target model, pθp\_{\theta}, on the gold data.
The above is equivalent to the KL term from the word-level distillation loss in (kim2016sequencedistillation).101010While performing sequence-level distillation, *i.e.*, conditioning on data sampled from the teacher model may improve distillation (kim2016sequencedistillation), we did not explore this.

## Appendix H Further Related Work

MTP for byte-level LLMs
gloeckle2024better and evabyte2025 both pretrain byte-level LLMs which predict n=8n=8 future bytes.
This window size was found to be optimal for downstream performance in gloeckle2024better.
Both make conditional independence assumptions, but the approaches are architecturally different.
gloeckle2024better uses a transformer head per token to provide different feature vectors to each head and uses a shared unembedding matrix. On the other hand, evabyte2025 uses a shared feature vector across heads with different unembedding matrices per token.111111It is worth noting that gloeckle2024better also do an ablation in the appendix and find that linear heads were competitive.
However, they only focus on greedy self-speculative decoding, while in our work, we also explore speculative sampling.

Speculative Decoding with MTP drafts
Previous MTP work either ignores speculative sampling and only focuses on greedy self-speculative decoding (gloeckle2024better), or abandons guarantees altogether (possibly at the expense of quality): specifically, cai2024medusa and evabyte2025 use a tree decoding mechanism to consider multiple candidates at each speculative decoding step. Since their approach may accept multiple continuations but only generate the longest accepted one, they bias the distribution and break the guarantees.
wang2024mambabyte use a subword-level draft model to speed up their byte-level STP model via speculative decoding.
Most prior work has introduced sequential dependencies in the draft model through architecture modifications.
Hydra (ankner2024hydra) modifies the Medusa heads such that the predicted probabilities are also a function of the input embeddings of predicted draft tokens.
Eagle (li2024eagle) introduces sequential dependencies by autoregressively predicting future feature representations.
While these works relax the independence assumption, they have no explicit probabilistic model for the dependencies introduced.

An exception is basharin2024faster, who study the effect of relaxing the conditional independence assumption by using a CP factorisation.
While they obtain some first promising results, showing that increasing the rank can increase the acceptance rate of tokens for speculative decoding,
they focus on subword-level models which have very large vocabulary sizes (e.g. v≥v\geq 100k).
This makes CP very expensive, both in terms of the number of parameters needed, and the GPU memory required.
Moreover, they evaluate their models on unrealistic scenarios, i.e. datasets used for pre-training LLMs rather than instruction fine-tuning.
Finally, despite the fact that a lot of previous work exists on MTP for subword-level LLMs,
they use different models from those widely used for benchmarking speculative decoding methods, despite the existence of a benchmark, Spec-Bench (xia-etal-2024) and common models (e.g. Vicuna).
In our case, since there is a limited amount of work on MTP for byte-level LLMs (gloeckle2024better; evabyte2025), we directly compare with the results of the EvaByte model.

There has been increasing interest in multi-token prediction not only for generation speed-up, but also for improved model performance on tasks due to the lookahead offered by MTP.
For example, deepseekai2024 show that MTP for a token window of 2 tokens leads to improvements in benchmark metrics even when MTP is not used at inference time. Furthermore, they report an increase in the throughput of the model by ×1.8\times 1.8 when using speculative decoding.

Token granularity
In MTP a token can vary in granularity from bytes (evabyte2025) to subword tokens provided by tokenisers (basharin2024faster).

In addition to the choice of token granularity, there are generally 3 axes related works differ on:

* •

  Training from scratch vs distilling an existing STP model into a MTP model
* •

  Neural network architectures for the token heads (e.g. Linear, MLP, Transformer)
* •

  Probabilistic modelling assumptions (conditional independence vs more expressive models)

### H.1 Differences in Scenario

Training from Scratch
Evabyte (evabyte2025) train a MTP byte-level model from scratch using n=8n=8.

Retrofitting STP to MTP
Some works explore both training from scratch and retrofitting an STP model into an MTP one  (cai2024medusa; basharin2024faster). In our work, we focus on the second setting.

### H.2 Differences in Architectures

Linear Heads
basharin2024faster use a linear parameterisation for each token head in their distillation experiments.

MLP Heads
cai2024medusa use a MLP with a single hidden layer for each output token head. ankner2024hydra extend this to multiple layers of MLPs per output token.
gloeckle2024better make the heads context-aware by including a transformer head in each token head.

Autoregressive Head
While the main point of having future token heads is to avoid the expensive autoregression of the target model, current state of the art speculative decoding models rely on “cheap” autoregression.
Eagle (li2024eagle), which is the best performing on the speculative decoding benchmark, Spec-bench (xia-etal-2024), fits an autoregressive model to predict future feature vectors of the model (i.e. the inputs to the softmax layer). A similar architecture was also used for the DeepSeekV3 model (deepseekai2024).

MLP Heads
cai2024medusa propose the Medusa model which uses a MLP with a residual connection for each token head.
While in theory they could use many MLP layers, they choose to use MLPs with single hidden layer.
ankner2024hydra explore using deeper MLPs for each token head.

Transformer Heads
gloeckle2024better use a shared unembedding matrix and use a separate transformer for each token head.
More precisely, in order to predict the token at offset ss, i.e. xt+sx\_{t+s}, they compute softmax⁡(𝐖𝐳s)\operatorname{softmax}\left(\mathbf{W}\mathbf{z}\_{s}\right), where 𝐳s\mathbf{z}\_{s} is produced by a separate transformer head for each ss, i.e. 𝐳s=Hs​(𝐱≤t)\mathbf{z}\_{s}=H\_{s}(\mathbf{x}\_{\leq t}).

Sharing the Unembedding layer.
While the decoding of  evabyte2025 is based on Medusa, instead of using a MLP for each token head, they use a different unembedding matrix per token head.121212<https://github.com/OpenEvaByte/evabyte/blob/98d5f48d32197b803e7560a798da35c7a4bdcf4d/evabyte_hf/modeling_evabyte.py#L753>
This modelling choice is possible due to the small vocabulary size of byte-level models, i.e. |𝒱|=320|\mathcal{V}|=320.
In their training from scratch scenario,  basharin2024faster use different unembedding layers 𝐖a(s)\mathbf{W}^{(s)}\_{a} in order to predict xt+sx\_{t+s} for the mixture component with index aa.
As such, they parameterise s×|a|s\times|a| unembedding matrices. This seems non-ideal, since the last layer in LLMs can have a large number of parameters, i.e. (V×dV\times d).
In their distillation scenario they use a shared unembedding matrix.

[◄](/html/2511.11345)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2511.11346)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2511.11346)
[View original  
on arXiv](https://arxiv.org/abs/2511.11346)[►](/html/2511.11347)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Dec 5 19:35:43 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
