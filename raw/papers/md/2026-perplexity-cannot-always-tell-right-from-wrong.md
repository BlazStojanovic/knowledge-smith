---
arxiv: '2601.22950'
authors:
- Petar Veličković
- Federico Barbero
- Christos Perivolaropoulos
- Simon Osindero
- Razvan Pascanu
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: Perplexity Cannot Always Tell Right from Wrong
url: https://arxiv.org/abs/2601.22950
year: 2026
---

# Perplexity Cannot Always Tell Right from Wrong

Petar Veličković
  
Federico Barbero
  
Christos Perivolaropoulos
  
Simon Osindero
  
Razvan Pascanu

###### Abstract

Perplexity—a function measuring a model’s overall level of “surprise” when encountering a particular output—has gained significant traction in recent years, both as a loss function and as a simple-to-compute metric of model quality. Prior studies have pointed out several limitations of perplexity, often from an empirical manner. Here we leverage recent results on Transformer continuity to show in a rigorous manner how perplexity may be an unsuitable metric for model selection. Specifically, we prove that, if there is *any* sequence that a compact decoder-only Transformer model predicts accurately and confidently—a necessary pre-requisite for strong generalisation—it must imply existence of another sequence with very low perplexity, but not predicted correctly by that same model. Further, by analytically studying iso-perplexity plots, we find that perplexity will not always select for the more accurate model—rather, any increase in model confidence must be accompanied by a commensurate rise in accuracy for the new model to be selected.

Machine Learning, ICML

## 1 Introduction

Perplexity is a measure of a model’s “surprise” when observing ground-truth data; assuming a model’s output distribution, QQ, over a space of classes, 𝒞\mathcal{C}, used to approximate a ground-truth distribution, PP, it can be expressed as
exp​∑k∈𝒞−pk​log⁡qk,\exp\sum\_{k\in\mathcal{C}}-p\_{k}\log q\_{k},
and, since it is easy to compute over any classification task (inlcuding tokenised data), it has become a popular function for evaluating sequential machine learning models when no other performance metric is readily computable. However, even though it is simple to use and interpret, in this paper, we provide novel evidence that perplexity should not be blindly trusted as a model selection objective.

This is a result that has been informally observed in several venues (Fang et al. ([2025](#bib.bib1 "What is wrong with perplexity for long-context language modeling?")); Hu et al. ([2024](#bib.bib10 "Can perplexity reflect large language model’s ability in long text understanding?")); Hsieh et al. ([2024](#bib.bib7 "RULER: what’s the real context size of your long-context language models?"))). Intuitively, it stems from the fact that perplexity encodes both confidence in prediction as well as correctness. Subsequently, a model with a lower accuracy (e.g. more answer tokens predicted incorrectly in the context of language models), but with better-calibrated confidence, may have lower perplexity—and thus be preferred.

!(/html/2601.22950/assets/x1.png)

Figure 1: Using the continuity result of Pasten et al. ([2025](#bib.bib13 "Continuity and isolation lead to doubts or dilemmas in large language models")), we show that, if a (compact) Transformer TT is confident in copying *any* long enough sequence 𝜶N\boldsymbol{\alpha}\_{N}, then there must exist 𝜷N\boldsymbol{\beta}\_{N} which TT fails to copy, yet, log-perplexity will tend to zero as NN grows.

Making use of recent important theoretical developments (Pasten et al., [2025](#bib.bib13 "Continuity and isolation lead to doubts or dilemmas in large language models")), in this work
we aim to provide a rigorous treatment of some of these observations, along with a detailed analytic account of how perplexity drives unfavourable “offsets” between confident and performant models in perplexity’s decision space. In doing so, we will prove that any confident model will necessarily introduce inputs which it will likely get wrong, but at a negligible effect to the perplexity measure (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Perplexity Cannot Always Tell Right from Wrong")).

Specifically, our work makes the following contributions, all pertaining to the damaging effect of *miscalibrated confidence* on perplexity’s power as a model selector:

1. 1.

   We prove that, for a wide class of decoder-only Transformer-based language models, should the model be highly confident and correct on a sufficiently long input sequence, this must imply existence of *another* input where the model’s prediction is *wrong*, yet the log-perplexity of that prediction approaches *zero*.
2. 2.

   We empirically validate this observation by studying *bitstring copy tasks*, both for a custom trained decoder-only Transformer over a small vocabulary, and the Gemma 3 4B large language model (Team et al., [2025](#bib.bib17 "Gemma 3 technical report")).
3. 3.

   Under certain assumptions on homogeneity of confidence, we study *iso-perplexity curves* in the confidence/accuracy space. These curves reveal clear “unfavourable regions” where a model gets too confident to justify its own accuracy improvement, and would not be selected against many weaker models.
4. 4.

   By tracking many checkpoints throughout the training lifecycle of a language model for *parity* prediction, we assess the frequency and circumstances under which models end up on the “wrong side” of the iso-perplexity curve in practice. We find that a key driver of undesirable model selection outcomes are *distribution shifts*.

## 2 Related work

While perplexity (or similarly log-likelihood) is the standard metric for evaluating language models, it has long been known in the broader generative model literature that likelihood does not necessarily correlate with sample quality. For instance, Theis et al. ([2015](#bib.bib2 "A note on the evaluation of generative models")) showed that good likelihood scores can be achieved by models that generate poor samples, and conversely, high-quality generators can yield poor likelihoods. Nalisnick et al. ([2019](#bib.bib12 "Do deep generative models know what they don’t know?")) showed that VAEs or flow-based models can assign higher likelihood to images that are *outside* the training distribution and that therefore do not represent the training data.

In the context of language models, Holtzman et al. ([2019](#bib.bib8 "The curious case of neural text degeneration")) showed a similar disconnect between optimising for likelihood and the generation of high-quality samples. In particular, they show that decoding using Nucleus Sampling leads to better generations (with lower likelihoods) compared to likelihood-maximising approaches such as beam search.

#### Failures of Perplexity in Long-Context

Fang et al. ([2025](#bib.bib1 "What is wrong with perplexity for long-context language modeling?")) argue that using perplexity as a metric in long-context is often misleading because useful signal may vanish when averaging perplexity over thousands of tokens. Their work champions the view that the *aggregation* method is the culprit. Our work rigorously proves results related to this observation, while also extending to claim that there is a detrimental, asymmetric relationship between accuracy and model confidence, which complicates the story further.

We highlight that this has been alluded to by other work. Gelberg et al. ([2025](#bib.bib3 "Extending the context of pretrained llms by dropping their positional embeddings"), Figure 5) showed that models can maintain low perplexity even when relevant information is strictly unreachable, which seems to explain the effectiveness of the popular context extension method YaRN (Peng et al., [2023](#bib.bib9 "Yarn: efficient context window extension of large language models")). Similarly, Liu et al. ([2024](#bib.bib4 "Lost in the middle: how language models use long contexts")) and Hsieh et al. ([2024](#bib.bib7 "RULER: what’s the real context size of your long-context language models?")) have shown that models often fail to retrieve information ‘lost in the middle’ of a prompt, despite achieving low overall perplexity scores on those same documents. These findings suggest that perplexity is not necessarily aligned with model performance, especially in long-context regimes.

#### Confidence and Calibration

A core component of our analysis is the role of model confidence. Guo et al. ([2017](#bib.bib6 "On calibration of modern neural networks")) famously showed that modern neural networks tend to be miscalibrated and overconfident. In the LLM era, while some argue models are generally calibrated (Kadavath et al., [2022](#bib.bib5 "Language models (mostly) know what they know")), the incentive to minimise perplexity encourages ‘confident’ predictions *in-distribution*. Our analysis studies how these training dynamics allow models to trade accuracy for confidence, creating ‘unfavourable regions’ where a ‘confident but wrong model’ achieves a better perplexity score than a hesitant but more accurate one.

#### Theoretical Results on Transformers

Our work relies on recent theoretical works regarding the limitations of the Transformer architecture. Barbero et al. ([2024](#bib.bib14 "Transformers need glasses! information over-squashing in language tasks")) identified the phenomenon of representation collapse in decoder-only Transformers. Extending this, Pasten et al. ([2025](#bib.bib13 "Continuity and isolation lead to doubts or dilemmas in large language models")) proved the existence of a ‘concentration’ of infinite sequence collections, such that decoder-only LLMs (under reasonable assumptions on positional embeddings) can model exactly one sequence in each collection. We leverage this continuity result to provide a proof of why perplexity fails: specifically, we show that the existence of a long enough sequence the model predicts accurately and confidently implies the existence of another sequence with very low perplexity that the model still fails to predict over.

## 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero

For the specific case of autoregressive models trained on next-token prediction (such as large language models), we can recombine a few previous results to theoretically strengthen the empirical finding of Fang et al. ([2025](#bib.bib1 "What is wrong with perplexity for long-context language modeling?")).

### 3.1 Preliminaries

As a clean proxy to the points we are going to make, throughout this section we will focus on a task where both perplexity and correctness are easy to define. Specifically, we study the bitstring copy task: a language model is provided a sequence of bits followed by a unique “stop” symbol, |\mathtt{|}, after which it needs to reproduce the given sequence of bits exactly. For example, given 𝟶𝟷𝟶𝟷𝟶|\mathtt{01010|}, the model needs to output 𝟶𝟷𝟶𝟷𝟶\mathtt{01010}. The model’s vocabulary is hence made up of only three symbols: 𝟶\mathtt{0}, 𝟷\mathtt{1} and |\mathtt{|}. It is well known that copying is tricky for modern LLMs to learn robustly (Barbero et al., [2024](#bib.bib14 "Transformers need glasses! information over-squashing in language tasks")), making it an ideal candidate for our study.

Secondly, all of our results rely on the assumption that our language model, TT, is a decoder-only Transformer with compact position embeddings (CPE). This is necessary in order for the key results of Pasten et al. ([2025](#bib.bib13 "Continuity and isolation lead to doubts or dilemmas in large language models")) to apply, and is generally true for the majority of positional embeddings in common use today, such as RoPE (Su et al., [2024](#bib.bib11 "Roformer: enhanced transformer with rotary position embedding")). We denote the output probability distribution of TT as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | T​(𝐱)​(y)=PT​(y|𝐱),T(\mathbf{x})(y)=P\_{T}(y\ |\ \mathbf{x}), |  | (1) |

the probability of emitting symbol yy given input prompt 𝐱\mathbf{x}.

### 3.2 Deterministic sampling

In order to make robust claims about a model’s accuracy and perplexity, we need to assume it will behave deterministically across all possible input prompts. We hence assume that its outputs are sampled via *greedy decoding*:

|  |  |  |  |
| --- | --- | --- | --- |
|  | T!​(𝐱)=arg⁡maxs∈{𝟶,𝟷}⁡T​(𝐱)​(s)T\_{!}(\mathbf{x})=\arg\max\_{s\in\{\mathtt{0},\mathtt{1}\}}T(\mathbf{x})(s) |  | (2) |

assuming all ties are broken consistently, e.g. by always choosing 𝟶\mathtt{0} in such cases.

In this regime, we will always measure the log-perplexity of the language model, TT, on the length-nn input bitstring 𝐛∈[𝟶,𝟷]n\mathbf{b}\in[\mathtt{0},\mathtt{1}]^{n}, defined as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | pplxT​(𝐛)=−1n​∑k=1nlog⁡T​(b1​⋯​bn|o1​⋯​ok−1)​(bk),\mathrm{pplx}\_{T}(\mathbf{b})=-\frac{1}{n}\sum\_{k=1}^{n}\log T(b\_{1}\cdots b\_{n}|o\_{1}\cdots o\_{k-1})(b\_{k}), |  | (3) |

where the symbols oio\_{i} are sampled deterministically:

|  |  |  |  |
| --- | --- | --- | --- |
|  | o1=T!(b1⋯bn|)oi=T!(b1⋯bn|o1⋯oi−1).o\_{1}=T\_{!}(b\_{1}\cdots b\_{n}|)\qquad o\_{i}=T\_{!}(b\_{1}\cdots b\_{n}|o\_{1}\cdots o\_{i-1}). |  | (4) |

This aligns well with the model’s loss function, and it is monotonically related to the perplexity.

Finally, we assume that the model performs all of its computations with appropriate numerical protection, meaning that the obtained values of log⁡T​(𝐱)​(y)\log T(\mathbf{x})(y) will never diverge to −∞-\infty, and remain bounded by log⁡T​(𝐱)​(y)≥log⁡ε\log T(\mathbf{x})(y)\geq\log\varepsilon for some ε>0\varepsilon>0.

###### Lemma 3.1 (Perplexity convergence).

Let TT be a decoder-only Transformer with compact position embeddings (CPE), as defined by Pasten et al. ([2025](#bib.bib13 "Continuity and isolation lead to doubts or dilemmas in large language models")). Assume TT is trained to perform a copy task over bitstrings, and it samples outputs by greedy decoding.

Let 𝛂=α1​α2​⋯​αn​⋯\boldsymbol{\alpha}=\alpha\_{1}\alpha\_{2}\cdots\alpha\_{n}\cdots be an infinite bitstring. Assume TT is capable of correctly copying every finite prefix of 𝛂\boldsymbol{\alpha}; that is, there is an ϵ>0\epsilon>0 such that, for all n∈ℕn\in\mathbb{N} and 1≤k≤n1\leq k\leq n:

|  |  |  |  |
| --- | --- | --- | --- |
|  | T​(α1​⋯​αn|α1​⋯​αk−1)​(αk)>1/2+ϵ.T(\alpha\_{1}\cdots\alpha\_{n}|\alpha\_{1}\cdots\alpha\_{k-1})(\alpha\_{k})>1/2+\epsilon. |  | (5) |

Then, for every ξ>0\xi>0, there must exist n′∈ℕn^{\prime}\in\mathbb{N} such that, for all prefixes 𝛂N=α1​α2​⋯​αN\boldsymbol{\alpha}\_{N}=\alpha\_{1}\alpha\_{2}\cdots\alpha\_{N} with N≥n′N\geq n^{\prime}, there is a bitstring 𝛃N\boldsymbol{\beta}\_{N} such that |pplxT​(𝛂N)−pplxT​(𝛃N)|<ξ|\mathrm{pplx}\_{T}(\boldsymbol{\alpha}\_{N})-\mathrm{pplx}\_{T}(\boldsymbol{\beta}\_{N})|<\xi, and 𝛃N\boldsymbol{\beta}\_{N} is not correctly copied by TT.

Armed with this result (proved in Appendix [A](#A1 "Appendix A Proof of Lemma 3.1. ‣ Perplexity Cannot Always Tell Right from Wrong")), we can now introduce an assumption of TT having a certain (high) confidence in copying 𝜶N\boldsymbol{\alpha}\_{N}, which will shortly bring us to one of our key results.

###### Proposition 3.2 (Collapsing confidence).

Let TT be a decoder-only Transformer with compact position embeddings (CPE), as defined by Pasten et al. ([2025](#bib.bib13 "Continuity and isolation lead to doubts or dilemmas in large language models")). Assume TT is trained to perform a copy task over bitstrings, and it samples outputs by greedy decoding.

Let 𝛂=α1​α2​⋯​αn​⋯\boldsymbol{\alpha}=\alpha\_{1}\alpha\_{2}\cdots\alpha\_{n}\cdots be an infinite bitstring. Assume TT is capable of correctly copying every finite prefix of 𝛂\boldsymbol{\alpha} with confidence (1−γ)(1-\gamma); that is, there is a 0≤γ<1/20\leq\gamma<1/2 such that, for all n∈ℕn\in\mathbb{N} and 1≤k≤n1\leq k\leq n:

|  |  |  |  |
| --- | --- | --- | --- |
|  | T​(α1​⋯​αn|α1​⋯​αk−1)​(αk)≥1−γ.T(\alpha\_{1}\cdots\alpha\_{n}|\alpha\_{1}\cdots\alpha\_{k-1})(\alpha\_{k})\geq 1-\gamma. |  | (6) |

Then, for every ϵ>0\epsilon>0, there must exist n′∈ℕn^{\prime}\in\mathbb{N} such that, for every size N≥n′N\geq n^{\prime}, there is a bitstring 𝛃N=β1​⋯​βN\boldsymbol{\beta}\_{N}=\beta\_{1}\cdots\beta\_{N} such that pplxT​(𝛃N)<−log⁡(1−γ)+ϵ\mathrm{pplx}\_{T}(\boldsymbol{\beta}\_{N})<-\log(1-\gamma)+\epsilon, and 𝛃N\boldsymbol{\beta}\_{N} is not correctly copied by TT.

###### Proof.

This result can be derived by applying Lemma [3.1](#S3.Thmtheorem1 "Lemma 3.1 (Perplexity convergence). ‣ 3.2 Deterministic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"), setting (ϵ=12−γ,ξ=ϵ)(\epsilon=\frac{1}{2}-\gamma,\ \xi=\epsilon), and remarking that pplxT​(𝜶N)≤−1N​∑k=1Nlog⁡(1−γ)=−log⁡(1−γ).\mathrm{pplx}\_{T}(\boldsymbol{\alpha}\_{N})\leq-\frac{1}{N}\sum\_{k=1}^{N}\log(1-\gamma)=-\log(1-\gamma).
∎

###### Corollary 3.3.

If there exists any infinite sequence 𝛂\boldsymbol{\alpha} copied with certainty (γ=0\gamma=0) by TT, then there must exist a family of finite sequences 𝛃N\boldsymbol{\beta}\_{N}, such that limN→+∞pplxT​(𝛃N)=0\lim\limits\_{N\to+\infty}\mathrm{pplx}\_{T}(\boldsymbol{\beta}\_{N})=0, and none of the sequences in 𝛃N\boldsymbol{\beta}\_{N} are correctly copied by TT.

This result demonstrates that, as models get more confident on any input, this necessarily allows for confounding situations where some other inputs get incorrectly processed without a visible impact on perplexity.

### 3.3 Stochastic sampling

One important assumption that allowed for this result to be cleanly derived is greedy decoding (i.e. sampling with temperature θ=0\theta=0). As this setup is less common in contemporary use of decoder-only Transformers, here we briefly remark on the applicability of our theoretical results in the stochastic sampling case. In our context, increasing θ\theta also increases the likelihood of a “random bit-flip” which would lead to incorrect copying of the bitstring 𝜶N\boldsymbol{\alpha}\_{N}.

First, we abstract away the choice of θ\theta by folding it into γ\gamma:

###### Remark 3.4.

Let T​(𝐚)​(σ)=(1−γ)T(\mathbf{a})(\sigma)=(1-\gamma) for an input bitstring 𝐚\mathbf{a} and bit σ∈{0,1}\sigma\in\{0,1\}. Then, assuming we sample with temperature θ>0\theta>0, the sampling probability becomes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Tθ​(𝐚)​(σ)=(1−γ)1/θ(1−γ)1/θ+γ1/θ=1−γ′,T\_{\theta}(\mathbf{a})(\sigma)=\frac{(1-\gamma)^{1/\theta}}{(1-\gamma)^{1/\theta}+\gamma^{1/\theta}}=1-\gamma^{\prime}, |  | (7) |

where γ′\gamma^{\prime} is a function of γ\gamma and θ\theta. Therefore, varying temperature of a (1−γ)(1-\gamma)-confident model may be seen as a model with θ=1\theta=1 but a different confidence level (1−γ′)(1-\gamma^{\prime}). Hence, we may assume θ=1\theta=1 without loss of generality.

!(/html/2601.22950/assets/x2.png)

!(/html/2601.22950/assets/x3.png)

Figure 2: For various sequence lengths, NN, on the copy task, we compute (Left) the L∞L\_{\infty} norm of the difference between the logit distributions across all positions, (Middle) the minimal observed probability of predicting 𝜶k\boldsymbol{\alpha}\_{k}—our conservative estimate of 1−γ1-\gamma—and the maximal observed probability of predicting 𝜷N\boldsymbol{\beta}\_{N}—which can serve as a bound on the probability that the model will copy 𝜷N\boldsymbol{\beta}\_{N} properly. We also plot (Right) the log-perplexity for both 𝜶N\boldsymbol{\alpha}\_{N} and 𝜷N\boldsymbol{\beta}\_{N}. This is done both for (Top) a toy copy environment where a CPE Transformer is trained on sizes up to 16 bits, and (Bottom:) prompting Gemma 3 4B with a copy request.

Firstly, we recall a useful result – Boole’s inequality – which allows us to place a meaningful bound on the probability of bit-flips in a (1−γ(1-\gamma)-confident sampler:

###### Remark 3.5.

Assume TT is capable of correctly copying a length-NN bitstring 𝜶N\boldsymbol{\alpha}\_{N} with confidence (1−γ)(1-\gamma). Then, we can bound the probability of any stochastic copying errors using Boole’s inequality (letting α¯k=1−αk\bar{\alpha}\_{k}=1-\alpha\_{k}):

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1−PT​(𝜶N​|𝜶N|)\displaystyle 1-P\_{T}(\boldsymbol{\alpha}\_{N}\ |\ \boldsymbol{\alpha}\_{N}|) | ≤∑k=1NT​(α1​⋯​αN|α1​⋯​αk−1)​(α¯k)\displaystyle\leq\sum\limits\_{k=1}^{N}T(\alpha\_{1}\cdots\alpha\_{N}|\alpha\_{1}\cdots\alpha\_{k-1})(\bar{\alpha}\_{k}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∑k=1Nγ=N​γ.\displaystyle\leq\sum\_{k=1}^{N}\gamma=N\gamma. |  |

That is, if N​γ≪1N\gamma\ll 1, it is unlikely any flips will happen, in which case the baseline sequence 𝜶N\boldsymbol{\alpha}\_{N} is copied correctly.

In the stochastic sampling regime, we can leverage the results of Pasten et al. ([2025](#bib.bib13 "Continuity and isolation lead to doubts or dilemmas in large language models")) once again, to analyse the probability that TT will produce 𝜶N\boldsymbol{\alpha}\_{N} in response to 𝜷N\boldsymbol{\beta}\_{N}!

###### Proposition 3.6.

Assume TT is capable of correctly copying every finite prefix of 𝛂\boldsymbol{\alpha} with confidence (1−γ)(1-\gamma). Then, for every ϵ>0\epsilon>0 there is an n′∈ℕn^{\prime}\in\mathbb{N} such that, for every size N≥n′N\geq n^{\prime}, under stochastic output sampling,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1−PT​(𝜶N​|𝜷N|)≤N​(γ+ϵ),1-P\_{T}(\boldsymbol{\alpha}\_{N}\ |\ \boldsymbol{\beta}\_{N}|)\leq N(\gamma+\epsilon), |  | (8) |

where 𝛃N\boldsymbol{\beta}\_{N} is derived as in Proposition [3.2](#S3.Thmtheorem2 "Proposition 3.2 (Collapsing confidence). ‣ 3.2 Deterministic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong").

###### Informal proof.

Follow a similar argument to Remark [3.5](#S3.Thmtheorem5 "Remark 3.5. ‣ 3.3 Stochastic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"), but this time consider N>⌈1/δ⌉N>\lceil 1/\delta\rceil, where δ>0\delta>0 is the continuity condition for ϵ\epsilon, and leverage continuity.
∎

This result implies that there are three possible outcomes in the stochastic sampling scenario (where ϵN\epsilon\_{N} is the smallest value of ϵ\epsilon attainable at size NN):

N​γ≪1,N​ϵN≪1N\gamma\ll 1,N\epsilon\_{N}\ll 1: Corresponds well to our greedy-decoding analysis. The model is confident enough to copy the baseline sequence 𝜶N\boldsymbol{\alpha}\_{N} with high probability, but it’s too tethered to 𝜶N\boldsymbol{\alpha}\_{N} (due to continuity). Therefore it will fail to copy 𝜷N\boldsymbol{\beta}\_{N} with high probability (most likely producing 𝜶N\boldsymbol{\alpha}\_{N}).

N​γ≪1,N​ϵN≪̸1N\gamma\ll 1,N\epsilon\_{N}\not\ll 1: The model copies 𝜶N\boldsymbol{\alpha}\_{N} with high probability, and the sequence is not long enough for our theory to apply. In this case, we are unable to make concrete claims about the model’s behaviour on 𝜷N\boldsymbol{\beta}\_{N}.

N​γ≪̸1N\gamma\not\ll 1: The model is not confident enough to reliably copy the baseline sequence 𝜶N\boldsymbol{\alpha}\_{N}, and due to continuity, it will likely fail to copy 𝜷N\boldsymbol{\beta}\_{N} in the same way.

### 3.4 Implications on learning dynamics

In Lemma [3.1](#S3.Thmtheorem1 "Lemma 3.1 (Perplexity convergence). ‣ 3.2 Deterministic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"), we showed that the perplexity of an incorrect sequence 𝜷N\boldsymbol{\beta}\_{N} converges to that of a correct sequence 𝜶N\boldsymbol{\alpha}\_{N} within a margin ϵ\epsilon. We now show that this has learnability implications. In particular, as the loss on 𝜶N\boldsymbol{\alpha}\_{N} goes to 0, this implies that the loss on 𝜷N\boldsymbol{\beta}\_{N} also approaches 0. Consequently, the training signal for the incorrect sample vanishes, and such a sample cannot be jointly learned (proved in Appendix [B](#A2 "Appendix B Proof of Corollary 3.7 ‣ Perplexity Cannot Always Tell Right from Wrong")).

###### Corollary 3.7 (Vanishing gradients on incorrect samples).

Let ℒ​(𝐱;Tθ)=−1M​∑i=1Mlog⁡Tθ​(𝐱<i)​(xi)\mathcal{L}(\mathbf{x};T\_{\theta})=-\frac{1}{M}\sum\_{i=1}^{M}\log T\_{\theta}(\mathbf{x}\_{<i})(x\_{i}) be the standard autoregressive cross-entropy loss for a CPE decoder-only Transformer with parameters θ\theta.

Assume that for the sequence 𝛂N\boldsymbol{\alpha}\_{N}, the model achieves a perfect loss, i.e. ℒ​(𝛂N;Tθ)→0\mathcal{L}(\boldsymbol{\alpha}\_{N};T\_{\theta})\to 0 as N→+∞N\to+\infty. Under the conditions of Proposition [3.2](#S3.Thmtheorem2 "Proposition 3.2 (Collapsing confidence). ‣ 3.2 Deterministic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"), for the sequence 𝛃N\boldsymbol{\beta}\_{N} which is not correctly copied by TθT\_{\theta}, the gradient of the loss with respect to θ\theta vanishes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | limN→+∞‖∇θℒ​(𝜷N;Tθ)‖=0.\lim\_{N\to+\infty}\|\nabla\_{\theta}\mathcal{L}(\boldsymbol{\beta}\_{N};T\_{\theta})\|=0. |  | (9) |

### 3.5 Empirical analysis

We attempt to validate our theoretical results in Figure [2](#S3.F2 "Figure 2 ‣ 3.3 Stochastic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"), both when pre-training a CPE Transformer on solely the copy task, and on a larger, general Gemma 3 4B model, setting 𝜶N=00​⋯​00\boldsymbol{\alpha}\_{N}=00\cdots 00 and 𝜷N=00​⋯​01\boldsymbol{\beta}\_{N}=00\cdots 01. Our observations match our expectations: continuity holds in both regimes, with the gap between the probability distributions on 𝜶N\boldsymbol{\alpha}\_{N} and 𝜷N\boldsymbol{\beta}\_{N} diminishing with increasing NN. Further, the probability of continuing 𝜶N\boldsymbol{\alpha}\_{N} remains high and stable for most input sizes, whereas the probability of successfully continuing 𝜷N\boldsymbol{\beta}\_{N} collapses. All the while, (log-)perplexity indeed gets iteratively closer between the two sequences.

One important caveat with these results is the observed noisy patterns in the Gemma 3 4B experiments. This is due to the fact that, unlike the clear-cut bitstring vocabulary of our theoretical setup, Gemma 3 has a much larger set of possible tokens—and especially due to their failure to count (Barbero et al., [2024](#bib.bib14 "Transformers need glasses! information over-squashing in language tasks")), on certain occasions the model attempts to prematurely predict newline characters and end-of-turn characters. These both cause issues with the computed probability distribution and perplexities, but they do not affect the overall trends of the relevant metrics collapsing, which we visualised using dashed lines.

Almost none of the results derived so far are specific to perplexity’s pointwise form—they, instead, mainly rely on the *averaging* process of the equation. As such, one might be tempted to see this as further evidence of Fang et al. ([2025](#bib.bib1 "What is wrong with perplexity for long-context language modeling?"))’s claim that the perplexity function itself might not be inherently problematic—just the way it’s aggregated. Furthermore, the LongPPL replacement for perplexity which is proposed in Fang et al. ([2025](#bib.bib1 "What is wrong with perplexity for long-context language modeling?")) would not necessarily suffer from the smoothing effects we identify here, as it would significantly shorten the number of tokens for which the metric is computed. That said, we believe there *are* inherent issues in the perplexity function beyond how it’s averaged, and this motivates us to study a *pointwise* setup with only one output, but placing important emphasis on the model confidence values.

## 4 An analytic view into confidence

!(/html/2601.22950/assets/x4.png)

!()

Figure 3: Left: Iso-perplexity curves for the setting with an unreliable base model (a=0.5a=0.5) for varying choices of confidence (1−γ1-\gamma). Right: Iso-perplexity curves for an unconfident base model (γ=0.4\gamma=0.4) for varying choice of base accuracy aa,.

From the theoretical and empirical analysis we presented so far, one variable that clearly stands out is *confidence*. In systems relying on stochastic sampling, high confidence (low γ\gamma) is important for them to generalise mechanistically—as N​γN\gamma needs to be sufficiently small to attenuate the likelihood of failures over long ranges NN. However, our theory implies that any high-confidence prediction in CPE Transformers not only opens the door to guaranteed failures elsewhere, it does so in a way that perplexity may not be able to detect the failure. In what follows, we attempt to answer: Can we establish a more general connection between the level of confidence a model has and its predictive power, in a way that reveals how predictable that power is via perplexity?

The answer is affirmative—in that, whenever confidence of a model increases, the model needs to supplement that confidence with a sufficient boost in predictive power—otherwise, perplexity will be unable to recognise this jump in confidence as positive. Specifically, with the right set of initial assumptions, it is possible to *analytically* solve for the “critical accuracy” needed to justify increased confidence.

### 4.1 Preliminaries

In order to be able to analytically manipulate the expressions we care about, it is important to make simplifying assumptions that will allow us to abstract away our model’s *confidence* (1−γ1-\gamma) and *accuracy* (aa) as scalar variables in [0,1][0,1]. Further, there must be a simple way to relate those scalar variables to the model’s log-perplexity, pplx\mathrm{pplx}.

The specific framework we assume which captures this idea well is a binary classification problem, where the model *always* makes its decisions with identical confidence (1−γ1-\gamma). This means that we can express the log-perplexity over a dataset with accuracy aa as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | pplxa,γ=−a​log⁡(1−γ)−(1−a)​log⁡γ\mathrm{pplx}\_{a,\gamma}=-a\log(1-\gamma)-(1-a)\log\gamma |  | (10) |

### 4.2 Iso-perplexity curves

!(/html/2601.22950/assets/x6.png)

Figure 4: Scatter plots of micro-F1\mathrm{F}\_{1} scores against log-perplexities, LL, for various checkpoints of a Transformer model trained on the Parity problem, as specified by Vitvitskyi et al. ([2025](#bib.bib15 "What makes a good feedforward computational graph?")), for both in-distribution (Left) and out-of-distribution (Right) held-out data. We also colour-code the checkpoints by their averaged Shannon entropy, H¯\bar{H}, provide the Pearson correlation coefficient, rr, and highlight the point with the highest accuracy by using a star (also colour-coded by entropy).

Now, we consider a setting where the model gets *more confident* by Δγ∈[0,γ]\Delta\_{\gamma}\in[0,\gamma]; that is, its confidence when correct jumps to 1−γ+Δγ1-\gamma+\Delta\_{\gamma}, and its confidence in the correct answer when wrong drops, symmetrically, to γ−Δγ\gamma-\Delta\_{\gamma}. If the accuracy doesn’t change, the first term of pplx\mathrm{pplx} will decrease while the second will increase.

But, even though we symmetrically altered the confidence by Δγ\Delta\_{\gamma}, the change in these two terms is *not* symmetrical, as the log\log function has a significantly higher rate of change when its input is close to 0 compared to being close to 11. As such, if we keep accuracy the same when increasing confidence, this will in many cases *increase* perplexity.

Accordingly, a sufficient *rise in accuracy*, a′a^{\prime}, is needed to compensate for this increase in confidence, if we wish perplexity to recognise this improvement in model accuracy.

This “critical point” in accuracy happens when the perplexities of the old and new model become equal:

|  |  |  |  |
| --- | --- | --- | --- |
|  | −a​log⁡(1−γ)\displaystyle-a\log(1-\gamma) | −(1−a)​log⁡γ\displaystyle-(1-a)\log\gamma |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =−a′​log⁡(1−γ+Δγ)\displaystyle=-a^{\prime}\log(1-\gamma+\Delta\_{\gamma}) | −(1−a′)​log⁡(γ−Δγ)\displaystyle-(1-a^{\prime})\log(\gamma-\Delta\_{\gamma}) |  |

Rearranging the terms, we obtain:

|  |  |  |  |
| --- | --- | --- | --- |
|  | a​(log⁡γ−log⁡(1−γ))\displaystyle a(\log\gamma-\log(1-\gamma)) | −log⁡γ\displaystyle-\log\gamma |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =a′​(log⁡(γ−Δγ)−log⁡(1−γ+Δγ))\displaystyle=a^{\prime}(\log(\gamma-\Delta\_{\gamma})-\log(1-\gamma+\Delta\_{\gamma})) | −log⁡(γ−Δγ)\displaystyle-\log(\gamma-\Delta\_{\gamma}) |  |

From here, we can explicitly derive the required accuracy:

|  |  |  |  |
| --- | --- | --- | --- |
|  | a′=pplxa,γ+log⁡(γ−Δγ)log⁡(γ−Δγ)−log⁡(1−γ+Δγ)a^{\prime}=\frac{\mathrm{pplx}\_{a,\gamma}+\log(\gamma-\Delta\_{\gamma})}{\log(\gamma-\Delta\_{\gamma})-\log(1-\gamma+\Delta\_{\gamma})} |  | (11) |

Note that this function depends on both the initial accuracy aa and initial confidence (1−γ)(1-\gamma). Therefore, it gives rise to several types of *iso-perplexity curves*, depending on whether we keep aa fixed and vary γ\gamma, or keep γ\gamma fixed and vary aa. To illustrate what these curves teach us about the reliability of perplexity as a discriminative metric, we focus on two specific cases here:

#### Iso-perplexity at a=0.5a=0.5

In this setting, we assume starting from an entirely unreliable model, but varying the starting confidence, (1−γ)(1-\gamma). We plot the *critical accuracy*, a′a^{\prime}, against the *normalised confidence shift*, Δγ/γ\Delta\_{\gamma}/\gamma; see Figure [3](#S4.F3 "Figure 3 ‣ 4 An analytic view into confidence ‣ Perplexity Cannot Always Tell Right from Wrong") (Left). Note that any model falling under the iso-perplexity curve would *not* be selected as improving perplexity, even though its accuracy may be better than the random chance of the first model – similarly, a model may end up above the iso-perplexity curve even though its accuracy is worse than the baseline. We can make two key observations:

Firstly, for all considered starting confidences, not all better more-confident models will decrease perplexity. The afforded “breathing room” for a′a^{\prime} tends to be greater (in relative terms) the more confident the base model is – there is “less surprise” when making an already confident model more confident. Still, many confidence shifts require increasing accuracy by over 5–10 percentage points, which is very significant.

Secondly, no matter what the starting confidence, truly extraordinary confidence requires truly extraordinary evidence—as Δγ→γ\Delta\_{\gamma}\to\gamma, a′→1a^{\prime}\to 1. Put differently, a perfectly confident model must be perfectly accurate, otherwise it will always be rejected by perplexity.

#### Iso-perplexity at γ=0.4\gamma=0.4

In this setting, we start from an unconfident model, but varying the starting accuracy, aa—once again plotting critical accuracy against normalised confidence shift. The resulting iso-perplexity curves are in Figure [3](#S4.F3 "Figure 3 ‣ 4 An analytic view into confidence ‣ Perplexity Cannot Always Tell Right from Wrong") (Right). The key insight that this regime offers is the existence of “unjustified free lunch” zones, where the iso-perplexity curves are decreasing for positive Δγ\Delta\_{\gamma}. If an unconfident model is already sufficiently accurate, it is possible to improve their perplexity just by making them more confident – even if this leads to significant drops in accuracy (a′<aa^{\prime}<a).

It is evident that there exists a rather non-negligible space under the iso-perplexity curve with a′>aa^{\prime}>a, as well as below it with a′<aa^{\prime}<a; it describes a significant family of models that would not be selected by perplexity, in spite of being better predictors than their baseline. We hypothesise this might have implications in many relevant regimes of AI deployment where accuracy cannot be easily measured, *especially* as the model needs to predict outside of its training distribution, which often requires higher confidence.

### 4.3 How often are we on the wrong iso-perplexity side?

Having exposed that there exist very clear regions of the model confidence/accuracy space where perplexity would not select the more accurate model, what remains to be seen is to what extent will this occur in practice.

One might hypothesise that a model is particularly vulnerable to such failures when the inputs stray *out-of-distribution* (OOD) compared to data the model was exposed to during training. Indeed, as we saw with the copy task example, models might *need* to maintain a low value of γ\gamma in order to even be able to process their baseline sequences properly. However, higher confidence also implies that when any failures do occur, they will be especially painful to perplexity.

Note that our “identical-confidence” model described in Equation [10](#S4.E10 "Equation 10 ‣ 4.1 Preliminaries ‣ 4 An analytic view into confidence ‣ Perplexity Cannot Always Tell Right from Wrong") is substantially *constrained* in order to make iso-perplexity curves analytically derivable—in reality, there may well not be a value of γ\gamma that fits an observed perplexity/accuracy pair (L,a)(L,a) over a real dataset. That is, there often may be no γ∈[0,1]\gamma\in[0,1] such that L=−a​log⁡(1−γ)−(1−a)​log⁡γL=-a\log(1-\gamma)-(1-a)\log\gamma. Furthermore, if any models start to get less confident, iso-perplexity curves approach their singularity point at γ=0.5\gamma=0.5, at which point it gets hard to see the relevant phase transitions on the confidence/accuracy plot.

For all of the above reasons, we abandon plotting the iso-perplexities here, and instead directly plot the (L,a)(L,a) pairs we observed via a scatter plot. Whenever L1<L2L\_{1}<L\_{2} but a1<a2a\_{1}<a\_{2}, the model’s perplexity cannot discriminate properly between these two points, and we can estimate how often this happens by observing the Pearson correlation coefficient, rr. In an ideal setting, where the LL metric exactly orders accuracies, we would recover r≈−1r\approx-1.

Beyond measuring the frequency of incorrect model selections, we also want to ascertain that these issues can be directly related to model confidence. While we cannot map arbitrary logit collections {log⁡pi}i=1n\{\log p\_{i}\}\_{i=1}^{n} to a fixed value of γ\gamma, we *can* compute a proxy for the model’s overall level of certainty by computing the *averaged Shannon entropy*, H¯=−1n​∑i=1npi​log⁡pi\bar{H}=-\frac{1}{n}\sum\_{i=1}^{n}p\_{i}\log p\_{i}. We can then use this quantity to colour-code the individual models we’re studying on the scatter plot—under the assumption that points that will be particular outliers in the OOD setting are the ones where H¯\bar{H} is lower (when the model is on the whole more confident).

### 4.4 Parity task setup

When deciding on which problem to choose to study these effects, it is not only desirable for the task to have a natural OOD regime—it should also be seen as “mechanistically easy but practically hard”. By this, we mean that there is a very clear, simple procedure that generates the ground-truth outputs, yet it is known that reproducing those outputs is hard for contemporary AI systems. The existence of a clear target procedure means that models need to get confident in order to replicate this procedure; the practical hardness means that their confidence will not always be rewarded.

A very good fit is the parity task: given a bitstring, predict the exclusive-or (XOR) of all of its bits (e.g., for 𝟶𝟷𝟶𝟷𝟶\mathtt{01010}, predict 𝟶\mathtt{0}; for 𝟷𝟷𝟶𝟷𝟶\mathtt{11010}, predict 𝟷\mathtt{1}). Parity is well-understood to be difficult when length-generalising with Transformers, for known theoretical reasons (Hahn, [2020](#bib.bib16 "Theoretical limitations of self-attention in neural sequence models")), yet the target formula for computing the output is very simple.

We replicate the Transformer training setup for the Parity task from Vitvitskyi et al. ([2025](#bib.bib15 "What makes a good feedforward computational graph?")), reusing the baseline hyperparameters leveraged there, and training the model for 5,0005,000 gradient steps. Our aim is to appreciate how the model’s performance/confidence profile evolves throughout training, and hence, we save many checkpoints of the model throughout training—one for every 100100 steps of gradient descent taken—and evaluate them on held-out in-distribution (IID) and out-of-distribution (OOD) bitstrings in terms of size. In this case, we train on bitstrings of size up to 1616, and consider an OOD distribution of bitstrings of size 128128.

Overall, this procedure generates a dataset of (L,F1,H¯)(L,\mathrm{F}\_{1},\bar{H}) tuples, where LL is (log-)perplexity, F1\mathrm{F}\_{1} is the micro-F1\mathrm{F}\_{1} score obtained by the model on those sequences, and H¯\bar{H} is the averaged Shannon entropy estimated using the model’s individual parity prediction logits across the entire bitstring.

### 4.5 Results and Discussion

We visualise the corresponding scatter plots of the stored checkpoints, along with other useful data (colour-coding, Pearson correlation) in Figure [4](#S4.F4 "Figure 4 ‣ 4.2 Iso-perplexity curves ‣ 4 An analytic view into confidence ‣ Perplexity Cannot Always Tell Right from Wrong"). We find that these visualisations provide strong evidence for our hypothesis, by making the following observations over the two data distributions:

#### Training progression

While in-distribution the model appears to gradually improve its loss and performance, with a corresponding decrease in entropy as the model gets more confident, the same trajectory cannot be observed in the OOD case. Worse yet, the checkpoint with the optimal OOD accuracy is one of the worst in terms of OOD perplexity.

#### Pearson correlation

The IID evaluations of the checkpoints paint a picture of a model whose perplexity improvements, for the most part, track micro-F1\mathrm{F}\_{1} score improvements; indeed, with r=−0.94r=-0.94, there is a strong anticorrelation between the two variables. No such trend can be observed OOD, in fact, the empirical value of rr is *positive* rather than negative. This neatly translates to the likelihood that we have a pair of incorrectly discriminated points with L1<L2L\_{1}<L\_{2} but a1<a2a\_{1}<a\_{2}: it is very high in the OOD regime.

#### Entropy connection

Lastly, the entropy colour-coding reveals the final piece of the puzzle and matches our hypothesis very well. In-distribution, entropy reduction is a sign of model maturity: the confidence increase follows a clear jump in predictive power and decrease in loss. Out-of-distribution, however, the checkpoints with low entropy can retain predictive power while drastically harming perplexity. In fact, the aforementioned best-performing observed OOD model has one of the lowest entropies in the entire dataset.

All taken together, we can make a clear conclusion: in the right kind of *out-of-distribution* regime, *many* points end up on the *wrong* side of iso-perplexity, and this effect can be directly tied to an *increase in confidence*.

## 5 Conclusions

In their recent important work, Fang et al. ([2025](#bib.bib1 "What is wrong with perplexity for long-context language modeling?")) make a clear stance on the issues behind perplexity on long ranges:

*“…there is growing evidence that LLMs’ perplexity does not indicate their performance on long-context benchmarks. There are two possible sources of this mismatch: either the log-likelihood-based metric is flawed, or the averaged tokens are not representative enough. In this
work, we champion the latter explanation…”*

We provided theoretical evidence in support of the latter source—with all tokens contributing to an averaged loss, this has the potential to lead to *weird* situations, where a model makes confident mistakes on an input, yet its log-perplexity can get arbitrarily close to zero for that input.

However, we also found that the former source cannot be ignored—the perplexity metric itself is inherently skewed, and prone to favouring less confident predictors, *especially* in the long-context settings mentioned above. We found that high model confidence, coupled with a perplexity objective, can be the very reason for being able to construct the weird situations in the former paragraph. We provided additional evidence for this by studying the ample unfavourable regions with respect to *iso-perplexity curves*.

While we do not offer an alternative to perplexity in regimes where accuracy cannot be measured, we hope that our work serves as a useful foundation for exercising appropriate care when using perplexity, as well as offering a few “diagnostic approaches” that can help us estimate in which situations one might need to rethink their model selection protocol.

## Acknowledgements

We would like to thank Alex Vitvitskyi for help with setting up the Parity experiment, and Xiangming Gu and Shakir Mohamed for reviewing the paper prior to submission.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine
Learning. There are many potential societal consequences of our work, none
which we feel must be specifically highlighted here.

## References

* F. Barbero, A. Banino, S. Kapturowski, D. Kumaran, J. Madeira Araújo, O. Vitvitskyi, R. Pascanu, and P. Veličković (2024)
  Transformers need glasses! information over-squashing in language tasks.
  Advances in Neural Information Processing Systems 37,  pp. 98111–98142.
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Theoretical Results on Transformers ‣ 2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§3.1](#S3.SS1.p1.6 "3.1 Preliminaries ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§3.5](#S3.SS5.p2.1 "3.5 Empirical analysis ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong").
* L. Fang, Y. Wang, Z. Liu, C. Zhang, S. Jegelka, J. Gao, B. Ding, and Y. Wang (2025)
  What is wrong with perplexity for long-context language modeling?.
  In The Thirteenth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=fL4qWkSmtM)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Failures of Perplexity in Long-Context ‣ 2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§3.5](#S3.SS5.p3.1 "3.5 Empirical analysis ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§3](#S3.p1.1 "3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§5](#S5.p1.1 "5 Conclusions ‣ Perplexity Cannot Always Tell Right from Wrong").
* Y. Gelberg, K. Eguchi, T. Akiba, and E. Cetin (2025)
  Extending the context of pretrained llms by dropping their positional embeddings.
  arXiv preprint arXiv:2512.12167.
  Cited by: [§2](#S2.SS0.SSS0.Px1.p2.1 "Failures of Perplexity in Long-Context ‣ 2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong").
* C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger (2017)
  On calibration of modern neural networks.
  In International conference on machine learning,
   pp. 1321–1330.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Confidence and Calibration ‣ 2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong").
* M. Hahn (2020)
  Theoretical limitations of self-attention in neural sequence models.
  Transactions of the Association for Computational Linguistics 8,  pp. 156–171.
  Cited by: [§4.4](#S4.SS4.p2.4 "4.4 Parity task setup ‣ 4 An analytic view into confidence ‣ Perplexity Cannot Always Tell Right from Wrong").
* A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi (2019)
  The curious case of neural text degeneration.
  arXiv preprint arXiv:1904.09751.
  Cited by: [§2](#S2.p2.1 "2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong").
* C. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y. Zhang, and B. Ginsburg (2024)
  RULER: what’s the real context size of your long-context language models?.
  arXiv preprint arXiv:2404.06654.
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§2](#S2.SS0.SSS0.Px1.p2.1 "Failures of Perplexity in Long-Context ‣ 2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong").
* Y. Hu, Q. Huang, M. Tao, C. Zhang, and Y. Feng (2024)
  Can perplexity reflect large language model’s ability in long text understanding?.
  In The Second Tiny Papers Track at ICLR 2024,
  External Links: [Link](https://openreview.net/forum?id=Cjp6YKVeAa)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Perplexity Cannot Always Tell Right from Wrong").
* S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson, et al. (2022)
  Language models (mostly) know what they know.
  arXiv preprint arXiv:2207.05221.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Confidence and Calibration ‣ 2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong").
* N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang (2024)
  Lost in the middle: how language models use long contexts.
  Transactions of the association for computational linguistics 12,  pp. 157–173.
  Cited by: [§2](#S2.SS0.SSS0.Px1.p2.1 "Failures of Perplexity in Long-Context ‣ 2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong").
* E. Nalisnick, A. Matsukawa, Y. W. Teh, D. Gorur, and B. Lakshminarayanan (2019)
  Do deep generative models know what they don’t know?.
  In International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=H1xwNhCcYm)
  Cited by: [§2](#S2.p1.1 "2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong").
* H. Pasten, F. Urrutia, H. I. J. Orellana, C. B. Calderon, C. Rojas, and A. Kozachinskiy (2025)
  Continuity and isolation lead to doubts or dilemmas in large language models.
  In The Thirty-ninth Annual Conference on Neural Information Processing Systems,
  External Links: [Link](https://openreview.net/forum?id=dR58v9Dd42)
  Cited by: [Appendix A](#A1.1.p1.6 "Proof. ‣ Appendix A Proof of Lemma 3.1. ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [Lemma A.1](#A1.Thmtheorem1.p1.2.2 "Lemma A.1 (3.1.: Perplexity convergence). ‣ Appendix A Proof of Lemma 3.1. ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [Figure 1](#S1.F1 "In 1 Introduction ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [Figure 1](#S1.F1.10.5 "In 1 Introduction ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§1](#S1.p3.1 "1 Introduction ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§2](#S2.SS0.SSS0.Px3.p1.1 "Theoretical Results on Transformers ‣ 2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§3.1](#S3.SS1.p2.2 "3.1 Preliminaries ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§3.3](#S3.SS3.p4.3 "3.3 Stochastic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [Lemma 3.1](#S3.Thmtheorem1.p1.2.2 "Lemma 3.1 (Perplexity convergence). ‣ 3.2 Deterministic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [Proposition 3.2](#S3.Thmtheorem2.p1.2.2 "Proposition 3.2 (Collapsing confidence). ‣ 3.2 Deterministic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong").
* B. Peng, J. Quesnelle, H. Fan, and E. Shippole (2023)
  Yarn: efficient context window extension of large language models.
  arXiv preprint arXiv:2309.00071.
  Cited by: [§2](#S2.SS0.SSS0.Px1.p2.1 "Failures of Perplexity in Long-Context ‣ 2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong").
* J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu (2024)
  Roformer: enhanced transformer with rotary position embedding.
  Neurocomputing 568,  pp. 127063.
  Cited by: [§3.1](#S3.SS1.p2.2 "3.1 Preliminaries ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong").
* G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin, T. Matejovicova, A. Ramé, M. Rivière, et al. (2025)
  Gemma 3 technical report.
  arXiv preprint arXiv:2503.19786.
  Cited by: [item 2](#S1.I1.i2.p1.1 "In 1 Introduction ‣ Perplexity Cannot Always Tell Right from Wrong").
* L. Theis, A. v. d. Oord, and M. Bethge (2015)
  A note on the evaluation of generative models.
  arXiv preprint arXiv:1511.01844.
  Cited by: [§2](#S2.p1.1 "2 Related work ‣ Perplexity Cannot Always Tell Right from Wrong").
* A. Vitvitskyi, J. G. Araújo, M. Lackenby, and P. Veličković (2025)
  What makes a good feedforward computational graph?.
  arXiv preprint arXiv:2502.06751.
  Cited by: [Figure 4](#S4.F4 "In 4.2 Iso-perplexity curves ‣ 4 An analytic view into confidence ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [Figure 4](#S4.F4.8.4 "In 4.2 Iso-perplexity curves ‣ 4 An analytic view into confidence ‣ Perplexity Cannot Always Tell Right from Wrong"),
  [§4.4](#S4.SS4.p3.4 "4.4 Parity task setup ‣ 4 An analytic view into confidence ‣ Perplexity Cannot Always Tell Right from Wrong").

## Appendix A Proof of Lemma 3.1.

###### Lemma A.1 (3.1.: Perplexity convergence).

Let TT be a decoder-only Transformer with compact position embeddings (CPE), as defined by Pasten et al. ([2025](#bib.bib13 "Continuity and isolation lead to doubts or dilemmas in large language models")). Assume TT is trained to perform a copy task over bitstrings, and it samples outputs by greedy decoding.

Let 𝛂=α1​α2​⋯​αn​⋯\boldsymbol{\alpha}=\alpha\_{1}\alpha\_{2}\cdots\alpha\_{n}\cdots be an infinite bitstring. Assume TT is capable of correctly copying every finite prefix of 𝛂\boldsymbol{\alpha}; that is, there is an ϵ>0\epsilon>0 such that, for all n∈ℕn\in\mathbb{N} and 1≤k≤n1\leq k\leq n:

|  |  |  |  |
| --- | --- | --- | --- |
|  | T​(α1​⋯​αn|α1​⋯​αk−1)​(αk)>1/2+ϵ.T(\alpha\_{1}\cdots\alpha\_{n}|\alpha\_{1}\cdots\alpha\_{k-1})(\alpha\_{k})>1/2+\epsilon. |  | (12) |

Then, for every ξ>0\xi>0, there must exist n′∈ℕn^{\prime}\in\mathbb{N} such that, for all prefixes 𝛂N=α1​α2​⋯​αN\boldsymbol{\alpha}\_{N}=\alpha\_{1}\alpha\_{2}\cdots\alpha\_{N} with N≥n′N\geq n^{\prime}, there is a bitstring 𝛃N\boldsymbol{\beta}\_{N} such that |pplxT​(𝛂N)−pplxT​(𝛃N)|<ξ|\mathrm{pplx}\_{T}(\boldsymbol{\alpha}\_{N})-\mathrm{pplx}\_{T}(\boldsymbol{\beta}\_{N})|<\xi, and 𝛃N\boldsymbol{\beta}\_{N} is not correctly copied by TT.

###### Proof.

Since TT is a CPE decoder-only Transformer, by Pasten et al. ([2025](#bib.bib13 "Continuity and isolation lead to doubts or dilemmas in large language models"))’s continuity theorem, there must exist δ>0\delta>0 such that, for any two equal-length inputs 𝐱\mathbf{x} and 𝐱′\mathbf{x^{\prime}}, if their relativised Hamming distance dH​(𝐱,𝐲)<δd\_{H}(\mathbf{x},\mathbf{y})<\delta and their last symbol is identical, then ‖T​(𝐱)−T​(𝐲)‖∞≤ϵ\|T(\mathbf{x})-T(\mathbf{y})\|\_{\infty}\leq\epsilon.

Whenever nc>⌈1/δ⌉n\_{c}>\lceil 1/\delta\rceil, we can find a 𝜷nc\boldsymbol{\beta}\_{n\_{c}} such that dH​(𝜶nc,𝜷nc)<δd\_{H}(\boldsymbol{\alpha}\_{n\_{c}},\boldsymbol{\beta}\_{n\_{c}})<\delta—simply flip exactly one bit in 𝜶nc\boldsymbol{\alpha}\_{n\_{c}} at an arbitrary position, jj. Coupled with Equation [5](#S3.E5 "Equation 5 ‣ Lemma 3.1 (Perplexity convergence). ‣ 3.2 Deterministic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong")’s assumption, we can deduce

|  |  |  |  |
| --- | --- | --- | --- |
|  | T​(β1​⋯​βnc|α1​⋯​αk−1)​(αk)>1/2,T(\beta\_{1}\cdots\beta\_{n\_{c}}|\alpha\_{1}\cdots\alpha\_{k-1})(\alpha\_{k})>1/2, |  | (13) |

therefore,

|  |  |  |  |
| --- | --- | --- | --- |
|  | T!(β1⋯βnc|)=α1T!(β1⋯βnc|α1⋯αk−1)=αkT\_{!}(\beta\_{1}\cdots\beta\_{n\_{c}}|)=\alpha\_{1}\qquad T\_{!}(\beta\_{1}\cdots\beta\_{n\_{c}}|\alpha\_{1}\cdots\alpha\_{k-1})=\alpha\_{k} |  | (14) |

for all 1≤k≤nc1\leq k\leq n\_{c}. That is, 𝜷nc\boldsymbol{\beta}\_{n\_{c}} is not correctly copied by TT, and its copying log-perplexity is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | pplxT​(𝜷nc)=−1nc​(log⁡T​(β1​⋯​βnc|α1​⋯​αj−1)​(βj)+∑k≠jlog⁡T​(β1​⋯​βnc|α1​⋯​αk−1)​(αk)).\mathrm{pplx}\_{T}(\boldsymbol{\beta}\_{n\_{c}})=-\frac{1}{n\_{c}}(\log T(\beta\_{1}\cdots\beta\_{n\_{c}}|\alpha\_{1}\cdots\alpha\_{j-1})(\beta\_{j})+\sum\_{k\neq j}\log T(\beta\_{1}\cdots\beta\_{n\_{c}}|\alpha\_{1}\cdots\alpha\_{k-1})(\alpha\_{k})). |  | (15) |

Now, once we observe that we can also, analogously, express

|  |  |  |  |
| --- | --- | --- | --- |
|  | pplxT​(𝜶nc)=−1nc​(log⁡T​(α1​⋯​αnc|α1​⋯​αj−1)​(αj)+∑k≠jlog⁡T​(α1​⋯​αnc|α1​⋯​αk−1)​(αk)),\mathrm{pplx}\_{T}(\boldsymbol{\alpha}\_{n\_{c}})=-\frac{1}{n\_{c}}(\log T(\alpha\_{1}\cdots\alpha\_{n\_{c}}|\alpha\_{1}\cdots\alpha\_{j-1})(\alpha\_{j})+\sum\_{k\neq j}\log T(\alpha\_{1}\cdots\alpha\_{n\_{c}}|\alpha\_{1}\cdots\alpha\_{k-1})(\alpha\_{k})), |  | (16) |

we can match the relevant terms in the summations to obtain |pplxT​(𝜶nc)−pplxT​(𝜷nc)|≤−1nc​(log⁡(12+ϵ)−log⁡ε+(nc−1)​ϵ)|\mathrm{pplx}\_{T}(\boldsymbol{\alpha}\_{n\_{c}})-\mathrm{pplx}\_{T}(\boldsymbol{\beta}\_{n\_{c}})|\leq-\frac{1}{n\_{c}}(\log(\frac{1}{2}+\epsilon)-\log\varepsilon+(n\_{c}-1)\epsilon). By algebraic manipulation of this expression we can conclude that, as long as we choose nc>ϵ−log⁡(12+ϵ)+log⁡εξ+ϵn\_{c}>\frac{\epsilon-\log(\frac{1}{2}+\epsilon)+\log\varepsilon}{\xi+\epsilon}, it will hold that |pplxT​(𝜶nc)−pplxT​(𝜷nc)|<ξ|\mathrm{pplx}\_{T}(\boldsymbol{\alpha}\_{n\_{c}})-\mathrm{pplx}\_{T}(\boldsymbol{\beta}\_{n\_{c}})|<\xi. This implies that we can set

|  |  |  |  |
| --- | --- | --- | --- |
|  | n′=max⁡(⌈1/δ⌉⏟continuity,ϵ−log⁡(12+ϵ)+log⁡εξ+ϵ⏟oversmoothing),n^{\prime}=\max\left(\underbrace{\lceil 1/\delta\rceil}\_{\mathrm{continuity}},\underbrace{\frac{\epsilon-\log\left(\frac{1}{2}+\epsilon\right)+\log\varepsilon}{\xi+\epsilon}}\_{\mathrm{oversmoothing}}\right), |  | (17) |

at which point we are guaranteed to obtain both the effects of continuity, misclassifying 𝜷n′\boldsymbol{\beta}\_{n^{\prime}}, *and* smoothing out the perplexity spike obtained by that misclassification.
∎

## Appendix B Proof of Corollary 3.7

###### Corollary B.1 (3.7.: Vanishing gradients on incorrect samples).

Let ℒ​(𝐱;Tθ)=−1M​∑i=1Mlog⁡Tθ​(𝐱<i)​(xi)\mathcal{L}(\mathbf{x};T\_{\theta})=-\frac{1}{M}\sum\_{i=1}^{M}\log T\_{\theta}(\mathbf{x}\_{<i})(x\_{i}) be the standard autoregressive cross-entropy loss for a CPE decoder-only Transformer with parameters θ\theta.

Assume that for the sequence 𝛂N\boldsymbol{\alpha}\_{N}, the model achieves a perfect loss, i.e. ℒ​(𝛂N;Tθ)→0\mathcal{L}(\boldsymbol{\alpha}\_{N};T\_{\theta})\to 0 as N→+∞N\to+\infty. Under the conditions of Proposition [3.2](#S3.Thmtheorem2 "Proposition 3.2 (Collapsing confidence). ‣ 3.2 Deterministic sampling ‣ 3 Log-perplexity of wrong next-token predictors can arbitrarily approach zero ‣ Perplexity Cannot Always Tell Right from Wrong"), for the sequence 𝛃N\boldsymbol{\beta}\_{N} which is not correctly copied by TθT\_{\theta}, the gradient of the loss with respect to θ\theta vanishes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | limN→+∞‖∇θℒ​(𝜷N;Tθ)‖=0.\lim\_{N\to+\infty}\|\nabla\_{\theta}\mathcal{L}(\boldsymbol{\beta}\_{N};T\_{\theta})\|=0. |  | (18) |

###### Proof.

The gradient of the cross-entropy loss with respect to parameters θ\theta is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇θℒ=1N​∑i=1N(𝐩i−𝐲i)⊤​Jθ​(𝐱<i),\nabla\_{\theta}\mathcal{L}=\frac{1}{N}\sum\_{i=1}^{N}(\mathbf{p}\_{i}-\mathbf{y}\_{i})^{\top}J\_{\theta}(\mathbf{x}{<i}), |  | (19) |

where 𝐩i=Tθ​(𝐱<i)\mathbf{p}\_{i}=T\_{\theta}(\mathbf{x}\_{<i}) is the predicted probability distribution, 𝐲i\mathbf{y}\_{i} is the one-hot target vector for 𝜷i\boldsymbol{\beta}\_{i}, and Jθ​(𝐱<i)=∇θTθ​(𝐱<i)J\_{\theta}(\mathbf{x}\_{<i})=\nabla\_{\theta}T\_{\theta}(\mathbf{x}\_{<i}) is the Jacobian of the model logits with respect to the parameters. We can bound the norm of the gradient using the Cauchy-Schwarz inequality:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖∇θℒ‖≤1N​∑i=1N‖𝐩i−𝐲i‖​‖Jθ​(𝐱<i)‖.\|\nabla\_{\theta}\mathcal{L}\|\leq\frac{1}{N}\sum\_{i=1}^{N}\|\mathbf{p}\_{i}-\mathbf{y}\_{i}\|\|J\_{\theta}(\mathbf{x}{<i})\|. |  | (20) |

As we assume that the Transformer is compact, this implies that the norm of the Jacobian is upper bounded by some KK (Lipschitz property), i.e. sup𝐱‖Jθ​(𝐱)‖≤K\sup\_{\mathbf{x}}\|J\_{\theta}(\mathbf{x})\|\leq K. Furthermore, since the cross-entropy loss is minimised only when the predictive distribution matches the target, convergence in loss implies convergence in
the predicted targets:

|  |  |  |  |
| --- | --- | --- | --- |
|  | limN→+∞‖𝐩i−𝐲i‖=0∀i.\lim\_{N\to+\infty}\|\mathbf{p}\_{i}-\mathbf{y}\_{i}\|=0\quad\forall i. |  | (21) |

Substituting the bounds back, we achieve the desired result:

|  |  |  |  |
| --- | --- | --- | --- |
|  | limN→+∞‖∇θℒ‖≤limN→+∞K​1N​∑i=1N‖𝐩i−𝐲i‖=0.\lim\_{N\to+\infty}\|\nabla\_{\theta}\mathcal{L}\|\leq\lim\_{N\to+\infty}K\frac{1}{N}\sum\_{i=1}^{N}\|\mathbf{p}\_{i}-\mathbf{y}\_{i}\|=0. |  | (22) |

∎
