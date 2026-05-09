---
arxiv: '2309.14322'
authors:
- Mitchell Wortsman
- Peter J. Liu
- Lechao Xiao
- Katie Everett
- Alex Alemi
- Ben Adlam
- John D. Co-Reyes
- Izzeddin Gur
- Abhishek Kumar
- Roman Novak
- Jeffrey Pennington
- Jascha Sohl-dickstein
- Kelvin Xu
- Jaehoon Lee
- Justin Gilmer
- Simon Kornblith
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: Small-scale proxies for large-scale Transformer training instabilities
url: https://arxiv.org/abs/2309.14322
year: 2023
---

[2309.14322] Small-scale proxies for large-scale Transformer training instabilities














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



# Small-scale proxies for large-scale Transformer training instabilities

Mitchell Wortsman   
Peter J. Liu   
Lechao Xiao   
Katie Everett   
  
Alex Alemi   
Ben Adlam   
John D. Co-Reyes  
Izzeddin Gur   
Abhishek Kumar   
  
Roman Novak   
Jeffrey Pennington   
Jascha Sohl-dickstein   
Kelvin Xu   
  
Jaehoon Lee\*   
Justin Gilmer\*   
Simon Kornblith\*
  
Google DeepMind

###### Abstract

Teams that have trained large Transformer-based models have reported training instabilities at large scale that did not appear when training with the same hyperparameters at smaller scales.
Although the causes of such instabilities are of scientific interest, the amount of resources required to reproduce them has made investigation difficult.
In this work, we seek ways to reproduce and study training stability and instability at smaller scales.
First, we focus on two sources of training instability described in previous work: the growth of logits in attention layers (Dehghani et al., 2023) and divergence of the output logits from the log probabilities (Chowdhery et al., 2022).
By measuring the relationship between learning rate and loss across scales, we show that these instabilities also appear in small models when training at high learning rates, and that mitigations previously employed at large scales are equally effective in this regime.
This prompts us to investigate the extent to which other known optimizer and model interventions influence the sensitivity of the final loss to changes in the learning rate.
To this end, we study methods such as warm-up, weight decay, and the μ𝜇\muParam (Yang et al., 2022),
and combine techniques to train small models that achieve similar losses across orders of magnitude of learning rate variation.
Finally, to conclude our exploration we study two cases where instabilities can be predicted before they emerge by examining the scaling behavior of model activation and gradient norms.

00footnotetext: Equal contribution.

## 1 Introduction

![Refer to caption](/html/2309.14322/assets/x1.png)


Figure 1: Qk-layernorm [[11](#bib.bib11)] enables stable training across three orders of magnitude of learning rate (LR) variation.
(Top) For transformers with N𝑁N parameters, we plot the effect of learning rate on final evaluation loss.
(Bottom) We use LR sensitivity to summarize the top plot. LR sensitivity measures the expected deviation from optimal when varying learning rate across three orders of magnitude.
Qk-layernorm reduces LR sensitivity, but LR sensitivity still increases with model scale.

Scaling up transformers has led to remarkable progress from chat models to image generation.
However, not every training run is successful.
When training large Transformers, researchers have reported instabilities which slow or destabilize learning [[6](#bib.bib6), [11](#bib.bib11), [53](#bib.bib53), [35](#bib.bib35), [8](#bib.bib8)].
As the resources required for large runs continue to grow,
it is important to examine the ways that Transformer training can fail.

In this report we reproduce, study, and predict training instability in Transformer models.
We find that measuring the relationship between learning rate and loss across scales is a useful tool to identify instability (e.g., Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Small-scale proxies for large-scale Transformer training instabilities")).
Therefore, we introduce learning rate (LR) sensitivity, which serves as a useful summary statistic for learning rate vs. loss curves.
LR sensitivity measures the deviation from optimal performance when varying LR across orders of magnitude.

We show that two sources of instability, which have previously been described at scale, can be reproduced in small Transformers.111We focus on instabilities which lead to slow divergence, not loss spikes (see Section [4](#S4 "4 Related work ‣ Small-scale proxies for large-scale Transformer training instabilities")).
This enables their study without access to large resource pools.
In particular, we examine the growth of logits in attention layers [[11](#bib.bib11), [16](#bib.bib16), [51](#bib.bib51)] and divergence of the output logits from the log probabilities [[6](#bib.bib6)].
As evident from the learning rate vs. loss curves and by inspecting model characteristics, both instabilities appear at high learning rates in small models.
Moreover, interventions which have previously been employed at scale are also successful in this regime (e.g., Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Small-scale proxies for large-scale Transformer training instabilities")).
These interventions—qk-layernorm [[11](#bib.bib11)]222Based off currently unpublished investigations of Gilmer et al. [[16](#bib.bib16)].
and z-loss regularization [[6](#bib.bib6)]—reduce LR sensitivity and enable successful training across three orders of magnitude of LR variation.

These observations raise the question of how other known optimizer and model interventions affect the shape of the learning rate vs. loss curves across scales.
Therefore, we study the effect of techniques such as warm-up, weight decay, and μ𝜇\muParam [[50](#bib.bib50)] in this context. When employing qk-layernorm and z-loss regularization, these other techniques usually have little impact on the range of learning rates at which models can be stably trained, but do affect the sensitivity to learning rate within this range.
In line with previous work, we find that longer warm-up reduces learning rate sensitivity, as does the independent scaling of learning rate and weight decay recommended by Loshchilov and Hutter [[33](#bib.bib33)].
One interesting finding is that scaling depth increases LR sensitivity at a faster rate than scaling width.

The remainder of our investigation centers on the scaling behavior for model characteristics such as activation and gradient norms.
Using the attention logit growth instability as an example, we show that it is possible to predict an instability before it emerges.
This is in contrast to prior works on scaling which primarily focus on scaling trends related to loss [[27](#bib.bib27), [22](#bib.bib22)].

We conclude by using the scaling behavior of model characteristics to search for instabilities that are currently not well documented.
Our investigation shows that gradient norms decrease with both scale and learning rate, such that the default AdamW [[33](#bib.bib33)] epsilon hyperparameter is too large.
This causes updates that are too small.
We connect this phenomenon and the attention logit growth instability to parameter norm growth [[34](#bib.bib34), [29](#bib.bib29)].

Overall, we believe our work presents new scientific opportunities for studying training stability without access to large resource pools.

## 2 Experimental methodology

This section details our experimental set-up (Section [2.1](#S2.SS1 "2.1 Experimental set-up ‣ 2 Experimental methodology ‣ Small-scale proxies for large-scale Transformer training instabilities")) and useful tools employed by our analysis: (i) measuring the relationship between learning rate and loss across scales (Section [2.2](#S2.SS2 "2.2 LR vs. loss curves and learning rate sensitivity ‣ 2 Experimental methodology ‣ Small-scale proxies for large-scale Transformer training instabilities")) and (ii) examining scaling trends for model characteristics (Section [2.3](#S2.SS3 "2.3 Scaling trends for model characteristics ‣ 2 Experimental methodology ‣ Small-scale proxies for large-scale Transformer training instabilities")).

### 2.1 Experimental set-up

We train small Transformer models [[45](#bib.bib45)] with a similar experimental set-up as GPT-2 [[38](#bib.bib38)] implemented in Flax [[20](#bib.bib20)]:
the models are decoder-only [[31](#bib.bib31)] and trained with an auto-regressive loss (refer to Section [A](#A1 "Appendix A Additional infrastructure details ‣ Small-scale proxies for large-scale Transformer training instabilities") for more infrastructure details).
While we experimentally manipulate many of the following hyperparameters, this section provides their default values, which we use unless otherwise specified.

By default, we use AdamW [[33](#bib.bib33)] with β1=0.9subscript𝛽10.9\beta\_{1}=0.9, β2=0.95subscript𝛽20.95\beta\_{2}=0.95, ϵ=italic-ϵabsent\epsilon= 1e-8, and gradient clipping at global norm 1.
The default warmup is 5e3 steps, and the default number of total steps is 1e5.
We use a linear schedule for warmup and and a cosine-decay [[32](#bib.bib32)] schedule for the remainder, with minimum learning rate 1e-5.
We use an independent weight decay of 1e-4 and auxiliary z-loss [[6](#bib.bib6)] with coefficient 1e-4.
Sections [3.2.2](#S3.SS2.SSS2 "3.2.2 Independent weight decay ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") and [3.1.2](#S3.SS1.SSS2 "3.1.2 Output logit divergence ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") respectively provide additional information and ablations on decoupled weight decay and z-loss.
We use pre-normalization [[38](#bib.bib38)] Transformers with qk-layernorm [[11](#bib.bib11)] (see Section [3.1.1](#S3.SS1.SSS1 "3.1.1 Attention logit growth ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") for information).
We do not use any biases following Chowdhery et al. [[6](#bib.bib6)], and the layernorm [[1](#bib.bib1)] ϵitalic-ϵ\epsilon remains at the default value in Flax [[20](#bib.bib20)] of 1e-6.
We jointly scale up the embedding size, depth, and number of heads when scaling parameters.
We do not use weight tying of the first and last layer [[37](#bib.bib37)], and when reporting the number of parameters we exclude the embedding and head (as in Kaplan et al. [[27](#bib.bib27)]).
We use rotary positional embeddings [[43](#bib.bib43)], and for training data we use C4 [[39](#bib.bib39)].
Letting d𝑑d refer to the model dimension (i.e., the embedding size),
the feed-forward component of the Transformer is an MLP with hidden dimension of 4d𝑑d and gelu [[21](#bib.bib21)] activations.
As in Vaswani et al. [[45](#bib.bib45)] we use factor 1/d1𝑑1/\sqrt{d} scaling in the self-attention.
The embedding initialization is the default in Flax, which is normally distributed with standard deviation 1/d1𝑑1/\sqrt{d}.
The remainder of the weights are initialized with a truncated normal distribution with inverse root fan-in standard deviation [[18](#bib.bib18)].
The default batch size is 256, where each batch element has a sequence length of 512 tokens.
Sequences are packed so that no padding is required.
Finally, we use the vocabulary from Raffel et al. [[40](#bib.bib40)] which has size 32101 and uses a SentencePiece [[28](#bib.bib28)] tokenizer.
We train on TPUs [[26](#bib.bib26)] in bfloat16 precision using Flax [[20](#bib.bib20)] and JAX [[4](#bib.bib4)].

### 2.2 LR vs. loss curves and learning rate sensitivity

To investigate how model instability emerges with scale, it is useful to plot the relationship between learning rate (LR) and loss for models of different sizes.
For instance, an instability is often characterized by an explosion in the loss at high learning rates. LR vs. loss curves can reveal how the lowest unstable learning rate changes as a function of model size.

To summarize LR vs. loss curves, we use LR sensitivity.
LR sensitivity measures the deviation in final validation loss from optimal when sweeping LR across three orders of magnitude.
If a model fails to train at high learning rates, then LR sensitivity will be high.
There are cases where LR vs. loss curves and LR sensitivity are no longer meaningful, for instance if an intervention changes the meaning of learning rate—see Appendix [B](#A2 "Appendix B When is learning rate sensitivity a useful metric ‣ Small-scale proxies for large-scale Transformer training instabilities") for a detailed discussion.

Let θ=𝒜​(η)𝜃𝒜𝜂\theta=\mathcal{A}(\eta) denote the model weights θ𝜃\theta obtained when training with learning rate η𝜂\eta, and let ℓ​(θ)ℓ𝜃\ell(\theta) denote the validation loss when using weights θ𝜃\theta. For a learning rate range [a,b]𝑎𝑏[a,b], let ℓ∗superscriptℓ\ell^{\*} denote the loss obtained with the best learning rate, i.e., ℓ∗=minη∈[a,b]⁡ℓ​(𝒜​(η))superscriptℓsubscript𝜂𝑎𝑏ℓ𝒜𝜂\ell^{\*}=\min\_{\eta\in[a,b]}\ell\left(\mathcal{A}(\eta)\right).
Moreover, let ℓ0subscriptℓ0\ell\_{0} denote loss at initialization. Then, LR sensitivity is defined as 𝔼η∈[a,b]​[min⁡(ℓ​(𝒜​(η)),ℓ0)−ℓ∗]subscript𝔼𝜂𝑎𝑏delimited-[]ℓ𝒜𝜂subscriptℓ0superscriptℓ\mathbb{E}\_{\eta\in[a,b]}\left[\min\left(\ell\left(\mathcal{A}\left(\eta\right)\right),\ell\_{0}\right)-\ell^{\*}\right].

Unless otherwise mentioned, we use the learning rate range 3e-4 to 3e-1 with AdamW [[33](#bib.bib33)] to measure LR sensitivity,
where LR refers to the maximum value in a cosine decay schedule with warm-up [[32](#bib.bib32)].
We consider LRs in {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1} when computing the minimum and expectation.

### 2.3 Scaling trends for model characteristics

To study instability, we also find it useful to examine scaling trends for model characteristics such as gradient or activation norms. This method is helpful for predicting instabilities
and contrasts with previous work on scaling, which primarily focuses on trends relating model scale and loss [[27](#bib.bib27), [22](#bib.bib22)].

## 3 Results

This section presents our results on training stability for small Transformers.
Equipped with LR sensitivity (Section [2.2](#S2.SS2 "2.2 LR vs. loss curves and learning rate sensitivity ‣ 2 Experimental methodology ‣ Small-scale proxies for large-scale Transformer training instabilities")), we study two known instabilities and their corresponding mitigation at small scale (Section [3.1](#S3.SS1 "3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities")).
This raises the question of how other model and optimizer interventions effect sensitivity of final loss to learning rate, which we investigate in Section [3.2](#S3.SS2 "3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities").
Finally, we examine whether instabilities can be reliably predicted before they emerge:
Section [3.3](#S3.SS3 "3.3 Predicting attention logit growth instability from scaling behavior of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") predicts when the logit growth instability may cause divergence in a larger model, while Section [3.4](#S3.SS4 "3.4 Searching for new instabilities via scaling trends of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") aims to find other issues that may occur when scaling up with our default hyperparameters.

### 3.1 Reproducing two known instabilities at small scale

![Refer to caption](/html/2309.14322/assets/x2.png)


Figure 2: The attention logit growth instability [[11](#bib.bib11), [51](#bib.bib51)] appears in small models at high learning rates.
The mitigation of applying qk-layernorm proposed by Dehghani et al. [[11](#bib.bib11)] is equally effective in the small-scale regime.
The max attention logit is reported for layer 0, which we typically observe to have the largest logit values.

![Refer to caption](/html/2309.14322/assets/x3.png)


Figure 3: The effect of the output logit divergence instability [[6](#bib.bib6)] and the z-loss mitigation [[6](#bib.bib6)] (Section [3.1.2](#S3.SS1.SSS2 "3.1.2 Output logit divergence ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities")). Models in this experiment have qk-layernorm [[11](#bib.bib11)].

![Refer to caption](/html/2309.14322/assets/x4.png)


Figure 4: An example of the output logit divergence instability [[6](#bib.bib6)] (Section [3.1.2](#S3.SS1.SSS2 "3.1.2 Output logit divergence ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities")) in a 2.4M parameter Transformer at learning rate 0.1.

Here, we examine two instabilities that have previously been described at scale: the growth of logits in attention layers [[11](#bib.bib11), [16](#bib.bib16), [51](#bib.bib51)] and divergence of the output logits from the log probabilities [[6](#bib.bib6)].
By examining LR vs. loss curves, we show that these instabilities can be reproduced in small models by using high learning rates and that mitigations employed at scale are effective in this regime.

#### 3.1.1 Attention logit growth

Researchers have previously documented that Transformer training fails when the attention logits become large [[11](#bib.bib11), [51](#bib.bib51)].
In Dehghani et al. [[11](#bib.bib11)], this issue emerged when training a ViT model [[14](#bib.bib14)] with 22 billion parameters.

In the self-attention layer of a Transformer [[45](#bib.bib45)], queries qisubscript𝑞𝑖q\_{i} and keys kisubscript𝑘𝑖k\_{i} are combined to compute the attention logits zi​j=⟨qi,kj⟩/dhsubscript𝑧𝑖𝑗

subscript𝑞𝑖subscript𝑘𝑗
subscript𝑑ℎz\_{ij}=\langle q\_{i},k\_{j}\rangle/\sqrt{d\_{h}}, where dhsubscript𝑑ℎd\_{h} is the head dimension.
Next, the attention logits are passed through a softmax to produce attention weights, which are used to combine values visubscript𝑣𝑖v\_{i}. Dehghani et al. [[11](#bib.bib11)] observed that the attention logits z𝑧z became large, which they refered to as attention logit growth.
As a result, the attention weights collapse to one-hot vectors, which was named attention entropy collapse by Zhai et al. [[51](#bib.bib51)].
To resolve this issue, Dehghani et al. [[11](#bib.bib11)] proposed qk-layernorm, which applies LayerNorm [[1](#bib.bib1)] to the queries and keys before computing the attention logits.

In our experiments, we find that models need not be large to exhibit instability related to attention logit growth. As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Small-scale proxies for large-scale Transformer training instabilities"), the maximum learning rate at which small models can be trained increases when using qk-layernorm.
Without qk-layernorm, the learning rate at which models diverge becomes smaller with increasing model size.
By contrast, models with qk-layernorm exhibit considerably lower LR sensitivity and train to low loss at high learning rates.
As a highlight, qk-layernorm allows training a model with 1.2B parameters at learning rate 0.3.
Both with and without qk-layernorm, LR sensitivity increases with scale.

Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") displays the loss and max attention logit for two model scales that differ by three orders of magnitude.
In both cases, the loss diverges without qk-layernorm.
Our results in Appendix Figure [E.1](#A5.F1 "Figure E.1 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities") suggest that attention logit growth is due to growth in the queries and keys, not due to an increase in their alignment.
Instead, we hypothesize this instability could result from the quadratic dependence of attention logits on parameter norms.

#### 3.1.2 Output logit divergence

Another instability reported by researchers training large models is divergence in the output logits from the log probabilities [[6](#bib.bib6)].
Just as before, we reproduce this instability with small models at large learning rates, and the proposed mitigation ameliorates the issue.
Overall, Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") summarizes the effect.

Let y𝑦y denote the model’s output logits, which are used to compute class probabilities pisubscript𝑝𝑖p\_{i} via a softmax pi=eyi/Zsubscript𝑝𝑖superscript𝑒subscript𝑦𝑖𝑍p\_{i}=e^{y\_{i}}/Z where Z=∑jeyj𝑍subscript𝑗superscript𝑒subscript𝑦𝑗Z=\sum\_{j}e^{y\_{j}}.
This instability occurs when the logits diverge and become very negative, as illustrated in Figure [4](#S3.F4 "Figure 4 ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") for a 2.4M parameter model at learning rate 0.1.
In contrast to the attention logit growth instability, this divergence occurs towards the end of training. The mitigation proposed by Chowdhery et al. [[6](#bib.bib6)] is to encourage log⁡Z𝑍\log Z to remain close to zero.
They add an auxiliary loss log2⁡Zsuperscript2𝑍\log^{2}Z, referred to as z-loss, with coefficient 1e-4.

As illustrated in Figures [3](#S3.F3 "Figure 3 ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") and [4](#S3.F4 "Figure 4 ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities"), we find that instability related to output logit divergence occurs in models with no weight decay regardless of scale, and z-loss resolves this instability. Weight decay also mitigates this instability for the larger models we test.

### 3.2 Measuring the effect of other known interventions

The previous section used the relationship between learning rate and loss as a useful tool for examining two known instabilities and their mitigation.
This raises the question of how other known model and optimizer interventions affect the shape of LR vs. loss curves across scales.
In particular, can LR sensitivity help identify additional issues or resolutions when scaling?
This section aims to answer this question for common techniques such as warm-up, weight decay, and μ𝜇\muParam [[50](#bib.bib50)].

#### 3.2.1 Warm-up

As illustrated by Figure [5](#S3.F5 "Figure 5 ‣ 3.2.1 Warm-up ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities"), a longer warm-up period reduces LR sensitivity.
This is most clear for the larger models, which are not stable at LR 3e-1 without long warm-up.
The number of total steps is fixed to 1e5 in this experiment, and all models use qk-layernorm.
The importance of warm-up for stability has previously been highlighted [[17](#bib.bib17), [42](#bib.bib42), [30](#bib.bib30)], although these works do not measure scaling behavior.

![Refer to caption](/html/2309.14322/assets/x5.png)


Figure 5: The effect of warm-up length for different model sizes.
Longer warm-up reduces LR sensitivity and loss, especially for the larger models we test.
Models in this experiment use qk-layernorm [[11](#bib.bib11)].

![Refer to caption](/html/2309.14322/assets/x6.png)


Figure 6: Independently scaling LR without also scaling weight decay reduces LR sensitivity.
While this was recommended by Loshchilov and Hutter [[33](#bib.bib33)], it is not common practice in the default AdamW implementations in popular libraries.
Refer to Section [3.2.2](#S3.SS2.SSS2 "3.2.2 Independent weight decay ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") for more information.
Models in this experiment use qk-layernorm [[11](#bib.bib11)].

![Refer to caption](/html/2309.14322/assets/x7.png)


Figure 7: Independently scaling depth increases LR sensitivity at a faster rate than scaling width, though also produces a model with lower loss at the largest scale we test.
Refer to Appendix Figure [E.2](#A5.F2 "Figure E.2 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities") for this experiment without qk-layernorm.

![Refer to caption](/html/2309.14322/assets/x8.png)


Figure 8: Measuring the effect of μ𝜇\muParam on LR sensitivity for models with qk-layernorm [[11](#bib.bib11)].
In our setting μ𝜇\muParam succeeds in stabilizing the optimal LR, though it does not improve loss or reduce LR sensitivity.
For more information refer to Section [3.2.4](#S3.SS2.SSS4 "3.2.4 𝜇Param ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities").

#### 3.2.2 Independent weight decay

Parameterizing weight decay independently of learning rate reduces LR sensitivity, as illustrated in Figure [6](#S3.F6 "Figure 6 ‣ 3.2.1 Warm-up ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities"). While this was recommended by Loshchilov and Hutter [[33](#bib.bib33)], it is not common practice in the default AdamW implementations of PyTorch [[36](#bib.bib36)] or Optax [[2](#bib.bib2)].
We explain the differences below.

For parameters θ𝜃\theta, let Δ=v/(u+ϵ)Δ𝑣𝑢italic-ϵ\Delta=v/\left(\sqrt{u}+\epsilon\right) denote the AdamW update without learning rate or weight decay.
For weight decay coefficient λ𝜆\lambda, max learning rate η𝜂\eta, and schedule st∈[0,1]subscript𝑠𝑡01s\_{t}\in[0,1], Loshchilov and Hutter [[33](#bib.bib33)] recommend the update θ←θ−st​(η​Δ−λ​θ)←𝜃𝜃subscript𝑠𝑡𝜂Δ𝜆𝜃\theta\leftarrow\theta-s\_{t}(\eta\Delta-\lambda\theta), which we refer to as independent decay.
On the other hand, the default implementation in PyTorch or Optax applies the update
θ←θ−st​η​(Δ−λ​θ)←𝜃𝜃subscript𝑠𝑡𝜂Δ𝜆𝜃\theta\leftarrow\theta-s\_{t}\eta(\Delta-\lambda\theta), i.e., η𝜂\eta now scales both terms.

When reporting LR sensitivity without independent decay in Figure [6](#S3.F6 "Figure 6 ‣ 3.2.1 Warm-up ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities"), we report the minimum LR sensitivity over ranges [1e-4, 1e-1] and [3e-4, 3e-1] because the former is sometimes better centered on the minimum.
The default setting in this paper is to use independent decay.
When using independent decay we set λ𝜆\lambda=1e-4, and without independent decay we set λ𝜆\lambda=0.1.
A sweep on weight decay values is conducted in Figure [E.10](#A5.F10 "Figure E.10 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities").

#### 3.2.3 Scaling width vs. depth

We have so far consistently observed that increasing the number of parameters increases LR sensitivity.
We now examine which part of scaling is most responsible.

Our results, illustrated by Figure [7](#S3.F7 "Figure 7 ‣ 3.2.1 Warm-up ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities"), indicate that scaling depth increases LR sensitivity at a faster rate than scaling width.
However, at the largest scale we test, independently scaling depth produces a model with lower validation loss.
A validation loss comparison between width scaling, depth scaling, and joint scaling is in Appendix Figure [E.3](#A5.F3 "Figure E.3 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities").
The standard practice of joint scaling performs best at the largest scale and also has a more reliable scaling prediction when extrapolating.

When scaling depth, we use d=512𝑑512d=512, and when scaling width, we use 6 layers.
The number of heads is scaled proportionally with width, so that the head dimension remains the same.

Figure [E.2](#A5.F2 "Figure E.2 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities") repeats this experiment without qk-layernorm, finding that the attention logit growth instability occurs more frequently at scale regardless of whether width or depth are scaled.

#### 3.2.4 μ𝜇\muParam

Yang and Hu [[49](#bib.bib49)] introduced the μ𝜇\muParam method for parameterizing a neural network. As a product, the optimal LR remains consistent when scaling model width [[50](#bib.bib50)].
This section tests the effect of μ𝜇\muParam on LR sensitivity, and examines whether μ𝜇\muParam alleviates the need for qk-layernorm [[11](#bib.bib11)].

As illustrated by Figure [8](#S3.F8 "Figure 8 ‣ 3.2.1 Warm-up ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities"), μ𝜇\muParam does succeed in stabilizing the optimal LR at the scale we test.
However, μ𝜇\muParam does not improve loss or reduce LR sensitivity in our experiments.
Appendix Figure [E.4](#A5.F4 "Figure E.4 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities") repeats this experiment without qk-layernorm.
Our results indicate that μ𝜇\muParam does not alleviate the need for this intervention at high learning rates.
We note that from a practical perspective, reducing LR sensitivity is not important if the optimal LR does not change.

We refer to the variant of μ𝜇\muParam that we use in these experiments as μ𝜇\muParam (simple) because it maintains only the core feature of μ𝜇\muParam.
We add additional features from Yang et al. [[50](#bib.bib50)] in Appendix Figure [E.5](#A5.F5 "Figure E.5 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities") without measurable improvement at the largest scale we test.
For μ𝜇\muParam (simple) we make the following changes from our standard baseline: scale the LR for linear layers by base-fan-in/fan-inbase-fan-infan-in\text{base-fan-in}/\text{fan-in}.
For μ𝜇\muParam (full) there are three additional changes:
(i) initialize the head with standard deviation base-fan-in/fan-inbase-fan-infan-in\sqrt{\text{base-fan-in}}/\text{fan-in};
(ii) change the 1/dh1subscript𝑑ℎ1/\sqrt{d\_{h}} scaling factor in attention layers to 1/dh1subscript𝑑ℎ1/d\_{h} where dhsubscript𝑑ℎd\_{h} is the head dimension; and
(iii) initialize the query projection weights with zeros.
For base-fan-in we use the fan-in values for the smallest model we test, which has width 256.

We comment briefly on the aforementioned changes (ii) and (iii).
First, we ablate on change (ii) in isolation in Appendix Figure [E.6](#A5.F6 "Figure E.6 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities").
While this intervention reduces loss slightly at the smallest scale we test, the reverse is true for the largest scale we test.
Also, removing the square root from the scaling factor in attention layers does not alleviate the need for qk-layernorm.
Finally, with regards to change (iii), we note that in preliminary experiments this change had no noticeable effect.

#### 3.2.5 Additional interventions

This section recreates the previous plots with additional interventions or hyperparameter changes.
Corresponding figures are displayed in the appendix.

* •

  Changing the number of training steps from 1e5 to 5e4 or 2e5 does not meaningfully change LR sensitivity (Appendix Figure [E.7](#A5.F7 "Figure E.7 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities")).
* •

  We try applying qk-layernorm across the whole model dimension instead of individually per-head with shared parameters. As illustrated in Appendix Figure [E.8](#A5.F8 "Figure E.8 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities"), the latter performs better. We use per-head qk-layernorm as the default in all other experiments.
* •

  Increasing the batch size from 256 to 512 or 1024 does not meaningfully change LR sensitivity (Appendix Figure [E.9](#A5.F9 "Figure E.9 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities"), each batch element contains 512 tokens). When increasing batch size we decrease the number of training steps so that the amount of data seen is constant.
  We believe a similar effect would be observed if instead we held the number of steps constant because changing the number of steps has no impact on LR sensitivity at batch size 256 (Appendix Figure [E.7](#A5.F7 "Figure E.7 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities")).
* •

  The effect of changing the weight decay from 1e-4 is illustrated in Figure [E.10](#A5.F10 "Figure E.10 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities").
  Increasing decay appears to slightly shift the optimal LR right.
* •

  We find that the logit growth instability is not due to the softmax in the self-attention layer, as it still occurs with a pointwise variant of attention (Appendix Figure [E.11](#A5.F11 "Figure E.11 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities")).

### 3.3 Predicting attention logit growth instability from scaling behavior of model characteristics

![Refer to caption](/html/2309.14322/assets/x9.png)


Figure 9: Predicting the attention logit growth instability via scaling behavior of model characteristics.
We extrapolate to predict that a larger model will become unstable at LR 1e-2, and run an experiment to confirm the prediction.
Refer to Section [3.3](#S3.SS3 "3.3 Predicting attention logit growth instability from scaling behavior of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") for more information.

![Refer to caption](/html/2309.14322/assets/x10.png)


Figure 10: Enforcing a max attention logit of approximately κ𝜅\kappa in a small model to determine which value of κ𝜅\kappa inhibits learning.

A central question when studying instabilities is whether they can be predicted. We now examine whether it is possible to predict the logit growth instability before it occurs.
We track the attention logit maximums across model scales and fit a curve to the data.
We use this to predict that a 4.8B parameter model will be unstable at LR 1e-2 without qk-layernorm and run an experiment to confirm this prediction.

Figure [9](#S3.F9 "Figure 9 ‣ 3.3 Predicting attention logit growth instability from scaling behavior of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") plots the number of parameters vs. max attention logit at different learning rate values.333We use block 0, which typically has the largest logits, and consider the value at step 2e3.
Much earlier than 2e3 was uninformative, and much later the unstable points had long past diverged. At each learning rate, we fit a quadratic to predict how the max attention logit will change with model scale.

We first noticed that all points with attention logits above 1e4 diverged.
Moreover, the quadratic fit predicted that for LR 1e-2 the next model scale would also cross that value.
Based on this prediction, we trained a new 4.8B parameter model at LR 1e-2. This model diverged as predicted.
Not only do we predict the divergence, but our fit closely extrapolates to predict the value of the max attention logit.

One question unresolved by our analysis so far is whether we could have predicted that instability arises when the max attention logit exceeds 1e4 without manipulating learning rate and model size.
We take initial steps towards an answer by transplanting different values of max attention logit into a small network with 10M parameters.
For different constants κ𝜅\kappa we pass the queries and keys through g​(z)=κ⋅z/𝔼i​[zi2]𝑔𝑧⋅𝜅𝑧subscript𝔼𝑖delimited-[]superscriptsubscript𝑧𝑖2g(z)=\sqrt{\kappa}\cdot z/\sqrt{\mathbb{E}\_{i}[z\_{i}^{2}]} before computing the attention logits.
Results are illustrated in Figure [10](#S3.F10 "Figure 10 ‣ 3.3 Predicting attention logit growth instability from scaling behavior of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities").
Loss deteriorates around κ=𝜅absent\kappa=1e3, and by κ=𝜅absent\kappa=1e4 the loss exceeds that of a zero-layer bigram model consisting of the Transformer we use without any self-attention or MLP layers.

![Refer to caption](/html/2309.14322/assets/x11.png)


Figure 11: Predicting a potential instability from the scaling behavior of model characteristics. The gradient root mean square (RMS) decreases with num params (left) and learning rate (middle).
These trends indicate that hyperparameter adjustment may be required to successfully scale further, as the RMS is approaching the default AdamW ϵitalic-ϵ\epsilon hyperparameter.
If the gradient RMS becomes too small without adjusting ϵitalic-ϵ\epsilon or weight decay, a layer may collapse.
The gradient RMS in the left and middle plot is reported for the first MLP layer of block 0, but we observe similar trends for other layers (e.g., Appendix Figure [E.12](#A5.F12 "Figure E.12 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities")).
Gradient RMS across different blocks is also reported (right).
Gradient and update RMS are averaged over the final 500 steps, refer to Appendix Figure [E.13](#A5.F13 "Figure E.13 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities") for the data during training.

### 3.4 Searching for new instabilities via scaling trends of model characteristics

This section examines whether the scaling behavior of model characteristics can be used to predict new issues with the default model and hyperparameter settings.

In Figure [11](#S3.F11 "Figure 11 ‣ 3.3 Predicting attention logit growth instability from scaling behavior of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") we examine scaling trends for the gradient root mean square RMS​(g)=𝔼i​[gi2]RMS𝑔subscript𝔼𝑖delimited-[]superscriptsubscript𝑔𝑖2\text{RMS}(g)=\sqrt{\mathbb{E}\_{i}\left[g\_{i}^{2}\right]}.
This figure reports the RMS for the first layer of the MLP, though we observe similar trends for other layers (Appendix Figure [E.12](#A5.F12 "Figure E.12 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities")).

As models get larger, the value that grad RMS approaches is cause for concern.
At the largest scale and learning rate we test, grad RMS is around the default AdamW ϵitalic-ϵ\epsilon hyperparameter.
Recall that the unscaled AdamW update is Δ=v/(u+ϵ)Δ𝑣𝑢italic-ϵ\Delta=v/\left(\sqrt{u}+\epsilon\right), where v𝑣v and u𝑢u are the first and second gradient moment EMA, respectively.
If the grad RMS is on the same order as ϵitalic-ϵ\epsilon, then ΔΔ\Delta will decrease in magnitude as illustrated by Figure [13](#S3.F13 "Figure 13 ‣ 3.4 Searching for new instabilities via scaling trends of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities"), and parameters will not receive learning signals as intended.

An obvious mitigation for this issue is to simply lower the AdamW ϵitalic-ϵ\epsilon hyperparameter from its default of 1e-8.
We conduct this experiment for a 4.8B parameter model at LR 0.3 and present the results in Figure [12](#S3.F12 "Figure 12 ‣ 3.4 Searching for new instabilities via scaling trends of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities").
Decreasing ϵitalic-ϵ\epsilon to 1e-15 improves loss and mitigates a collapse in grad RMS. We believe this improvement will only increase at scale.
On the other hand, increasing ϵitalic-ϵ\epsilon to 1e-6 results in an instability (shown in Figure [E.15](#A5.F15 "Figure E.15 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities")).

Figure [13](#S3.F13 "Figure 13 ‣ 3.4 Searching for new instabilities via scaling trends of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") expands on this result by illustrating the grad and update RMS throughout training at the largest scale and learning rate we test. When the grad RMS reaches ϵitalic-ϵ\epsilon, the update RMS becomes small.
Figure [E.13](#A5.F13 "Figure E.13 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities") presents data from an analogous experiment at many different scales and LRs, demonstrating that this issue is most apparent for the larger models and LRs we test.

Although we identified the instability above by empirically measuring the scaling behavior of the gradients, a mechanistic explanation exists.
For larger networks and learning rates, the Transformer output RMS entering the final layernorm may grow. Since the layernorm gradients are scaled by the inverse of their input RMS, the gradient received by the Transformer will shrink.
Refer to Appendix [C](#A3 "Appendix C Output norm growth ‣ Small-scale proxies for large-scale Transformer training instabilities") for a more detailed discussion.

![Refer to caption](/html/2309.14322/assets/x12.png)


Figure 12: Decreasing the AdamW ϵitalic-ϵ\epsilon from its default value of 1e-8 to 1e-15 improves loss for a 4.8B parameter model at LR 0.3.
When increasing ϵitalic-ϵ\epsilon to 1e-6, loss diverged.
Grad RMS is averaged over the final 500 steps for the first layer in the MLP; refer to Figure [13](#S3.F13 "Figure 13 ‣ 3.4 Searching for new instabilities via scaling trends of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") for data throughout training.

![Refer to caption](/html/2309.14322/assets/x13.png)


Figure 13: The top row displays the root mean square (RMS) of the gradient for the first MLP layer at different blocks throughout the network. When the grad RMS drops below the AdamW ϵitalic-ϵ\epsilon hyperparameter, the magnitude of the update decreases, as illustrated by the bottom row.
Experiment conducted with a 4.8B parameter model trained with LR 0.3.
The experiment with ϵitalic-ϵ\epsilon = 1e-6 was stopped when loss diverged.

## 4 Related work

This paper mainly focuses on the effect of known interventions and instabilities, and so related work has been primarily discussed when relevant.
This includes the attention growth instability observed by Dehghani et al. [[11](#bib.bib11)], Zhai et al. [[51](#bib.bib51)], and the final logit divergence issue encountered by Chowdhery et al. [[6](#bib.bib6)], Thilak et al. [[44](#bib.bib44)].
However, we highlight similar experimental methods in previous work.
For instance, Yang et al. [[50](#bib.bib50)] also measure the relationship between LR and loss across scales, but their focus is on centering the optimum (see Section [3.2.4](#S3.SS2.SSS4 "3.2.4 𝜇Param ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities")).
In addition, Zhai et al. [[51](#bib.bib51)] elicit instability in base models by doubling learning rate, and Dettmers et al. [[12](#bib.bib12)] measure the presence of outlier features as a function of scale.

There are also important instabilities and related topics we have not directly discussed so far.
For instance, we have primarily focused on instabilities that lead to a slow divergence, and we now summarize research on fast loss spikes.
This instability is characterized by a quick increase in the loss that often eventually recovers.

The Edge of Stability and fast spikes

The conventional understanding of gradient descent predicts that loss instability only occurs when the learning rate exceeds 2/λmax​(H)2subscript𝜆max𝐻2/\lambda\_{\text{max}}(H), where H𝐻H is the Hessian.
However recent investigations into large batch neural network training dynamics have revealed a more complicated picture via edge of stability (EoS) [[7](#bib.bib7)].
When training neural networks with large batch SGD, the loss curvature constantly evolves via the interaction of two processes: progressive sharpening and self stabilization.
Progressive sharpening is the empirical observation that when LR<2/λmax​(H)LR2subscript𝜆max𝐻\text{LR}<2/\lambda\_{\text{max}}(H), the curvature gradually increases until the stability threshold is violated.
When the learning rate becomes too large relative to the curvature, fast loss spikes occur and the parameters oscillate into a region with smaller λmax​(H)subscript𝜆max𝐻\lambda\_{\text{max}}(H) where stable training and progressive sharpening resumes. The latter process where instability results in smaller λmax​(H)subscript𝜆max𝐻\lambda\_{\text{max}}(H) is self-stabilization, a theoretical model of which is given in Damian et al. [[9](#bib.bib9)]. Gradually shrinking λmax​(H)subscript𝜆max𝐻\lambda\_{\text{max}}(H) via self stabilization was shown to be a primary mechanism behind the success of learning rate warmup in Gilmer et al. [[17](#bib.bib17)], who closely studied the connections between curvature, initialization, architecture and max trainable learning rates.

Cohen et al. [[8](#bib.bib8)] further analyze edge of stability of dynamics with adaptive optimizers, showing that progressive sharpening interacts with both the self-stabilization process and the adaptive optimizer state. This interaction results in the preconditioned sharpness λmax​(P−1​H)subscript𝜆maxsuperscript𝑃1𝐻\lambda\_{\text{max}}(P^{-1}H) oscillating around an optimizer specific threshold (38/LR in the case of Adam with β1subscript𝛽1\beta\_{1}=0.9).
Adaptive EoS (AEoS) can also result in periodic loss spikes when progressive sharpening pushes the preconditioned sharpness above the stability threshold, however the optimizer hyperparameters play a role. In particular, when LR>>38/λmax​(P−1​H)subscript𝜆maxsuperscript𝑃1𝐻\lambda\_{\text{max}}(P^{-1}H), two mechanisms are now in play to resolve the step size being too big—either H𝐻H can shrink or P−1superscript𝑃1P^{-1} can shrink (or both). Cohen et al. [[8](#bib.bib8)] found that when β2subscript𝛽2\beta\_{2} is large, H𝐻H tends to shrink and fast loss spikes result during the process, resembling the self stabilization process observed with gradient descent. However when β2subscript𝛽2\beta\_{2} is small, P−1superscript𝑃1P^{-1} tends to shrink, no loss spikes are observed, and λmax​(H)subscript𝜆max𝐻\lambda\_{\text{max}}(H) tends to gradually increase throughout training.

It is noteworthy that the adaptive edge of stability process (and the role of β2subscript𝛽2\beta\_{2}) studied in Cohen et al. [[8](#bib.bib8)] offers a more complete understanding for loss spikes studied in a body of literature [[42](#bib.bib42), [6](#bib.bib6), [35](#bib.bib35), [47](#bib.bib47), [52](#bib.bib52), [5](#bib.bib5)]. For example, Shazeer and Stern [[42](#bib.bib42)] argue that during training of Transformers with adaptive optimizers the optimizer update can become too big resulting in a loss spike followed by recovery. This is sometimes attributed to the adaptive optimizer state becoming “stale”, which is consistent with the observation the reducing β2subscript𝛽2\beta\_{2} resolves the loss spikes [[42](#bib.bib42), [47](#bib.bib47), [52](#bib.bib52)]. This is perhaps the same observation as Cohen et al. [[8](#bib.bib8)] that reducing β2subscript𝛽2\beta\_{2} allows P−1superscript𝑃1P^{-1} to change quicker to adjust to the process of progressive sharpening. AEoS also offers an explanation for the periodic loss spikes observed when training large transformer models [[35](#bib.bib35)].

##### Parameter-free methods and more parameterizations.

While our work has studied sensitivity to learning rate, there is also research that aims to eliminate the need to specify a learning rate [[24](#bib.bib24), [10](#bib.bib10)].
Based on their analysis, Ivgi et al. [[24](#bib.bib24)] set the step size for iteration t𝑡t to the maximum distance from the initialization divided by the root sum of historical gradient squares.
Moreover, while our work investigated μ𝜇\muParam, there are additional parameterizations for which it would be interesting to explore LR vs. loss [[13](#bib.bib13), [48](#bib.bib48), [3](#bib.bib3), [25](#bib.bib25)].

## 5 Conclusion

As the compute required to train the largest models continues to increase, it becomes increasingly important to understand if training will be stable.
This paper has shown that useful insights on stability can be found when studying small Transformers.
We hope that this opens new opportunities for impactful research which benefits large runs without access to large resource pools.

### Acknowledgements

We thank George Dahl for thorough comments and suggestions,
and Hugo Larochelle and Rif A. Saurous for helpful discussion.
Also, we thank the members of the Google DeepMind PAGI team for their support of this effort,
Noah Fiedel, Noah Constant, Aaron Parisi, Alex Rizkowsky, Avi Singh, Azade Nova, Bernd Bohnet, Daniel Freeman, Gamaleldin Elsayed, Hanie Sedghi, Isabelle Simpson, James Harrison, Jiri Hron, Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura Culp, Max Bileschi, Merrie Morris, Rosanne Liu, Yundi Qian, Sharad Vikram, Tris Warkentin.

## References

* Ba et al. [2016]

  Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton.
  Layer normalization.
  *arXiv preprint arXiv:1607.06450*, 2016.
* Babuschkin et al. [2020]

  Igor Babuschkin, Kate Baumli, Alison Bell, Surya Bhupatiraju, Jake Bruce, Peter
  Buchlovsky, David Budden, Trevor Cai, Aidan Clark, Ivo Danihelka, Antoine
  Dedieu, Claudio Fantacci, Jonathan Godwin, Chris Jones, Ross Hemsley, Tom
  Hennigan, Matteo Hessel, Shaobo Hou, Steven Kapturowski, Thomas Keck, Iurii
  Kemaev, Michael King, Markus Kunesch, Lena Martens, Hamza Merzic, Vladimir
  Mikulik, Tamara Norman, George Papamakarios, John Quan, Roman Ring, Francisco
  Ruiz, Alvaro Sanchez, Laurent Sartran, Rosalia Schneider, Eren Sezener,
  Stephen Spencer, Srivatsan Srinivasan, Miloš Stanojević, Wojciech
  Stokowiec, Luyu Wang, Guangyao Zhou, and Fabio Viola.
  The DeepMind JAX Ecosystem, 2020.
  URL <http://github.com/deepmind>.
* Bordelon and Pehlevan [2023]

  Blake Bordelon and Cengiz Pehlevan.
  Dynamics of finite width kernel and prediction fluctuations in mean
  field neural networks.
  *arXiv preprint arXiv:2304.03408*, 2023.
* Bradbury et al. [2018]

  James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary,
  Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye
  Wanderman-Milne, and Qiao Zhang.
  JAX: composable transformations of Python+NumPy programs,
  2018.
  URL <http://github.com/google/jax>.
* Chen et al. [2021]

  X. Chen, S. Xie, and K. He.
  An empirical study of training self-supervised vision transformers.
  In *2021 IEEE/CVF International Conference on Computer Vision
  (ICCV)*, pages 9620–9629, Los Alamitos, CA, USA, oct 2021. IEEE Computer
  Society.
  doi: 10.1109/ICCV48922.2021.00950.
  URL
  <https://doi.ieeecomputersociety.org/10.1109/ICCV48922.2021.00950>.
* Chowdhery et al. [2022]

  Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra,
  Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian
  Gehrmann, et al.
  Palm: Scaling language modeling with pathways.
  *arXiv preprint arXiv:2204.02311*, 2022.
* Cohen et al. [2021]

  Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar.
  Gradient descent on neural networks typically occurs at the edge of
  stability.
  *arXiv preprint arXiv:2103.00065*, 2021.
* Cohen et al. [2022]

  Jeremy M Cohen, Behrooz Ghorbani, Shankar Krishnan, Naman Agarwal, Sourabh
  Medapati, Michal Badura, Daniel Suo, David Cardoze, Zachary Nado, George E
  Dahl, et al.
  Adaptive gradient methods at the edge of stability.
  *arXiv preprint arXiv:2207.14484*, 2022.
* Damian et al. [2022]

  Alex Damian, Eshaan Nichani, and Jason D Lee.
  Self-stabilization: The implicit bias of gradient descent at the edge
  of stability.
  *arXiv preprint arXiv:2209.15594*, 2022.
* Defazio and Mishchenko [2023]

  Aaron Defazio and Konstantin Mishchenko.
  Learning-rate-free learning by d-adaptation.
  *arXiv preprint arXiv:2301.07733*, 2023.
* Dehghani et al. [2023]

  Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan
  Heek, Justin Gilmer, Andreas Steiner, Mathilde Caron, Robert Geirhos, Ibrahim
  Alabdulmohsin, et al.
  Scaling vision transformers to 22 billion parameters.
  *arXiv preprint arXiv:2302.05442*, 2023.
* Dettmers et al. [2022]

  Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer.
  Llm. int8 (): 8-bit matrix multiplication for transformers at scale.
  *arXiv preprint arXiv:2208.07339*, 2022.
* Dinan et al. [2023]

  Emily Dinan, Sho Yaida, and Susan Zhang.
  Effective theory of transformers at initialization.
  *arXiv preprint arXiv:2304.02034*, 2023.
* Dosovitskiy et al. [2021]

  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
  Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
  Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.
  An image is worth 16x16 words: Transformers for image recognition at
  scale.
  In *International Conference on Learning Representations
  (ICLR)*, 2021.
  <https://arxiv.org/abs/2010.11929>.
* Gaffney et al. [2023]

  Colin Gaffney, Dinghua Li, Ruoxin Sang, Ayush Jain, and Haitang Hu.
  Orbax, 2023.
  URL <http://github.com/google/orbax>.
* [16]

  Justin Gilmer, Andrea Schioppa, and Jeremy Cohen.
  Intriguing properties of transformer training instabilities.
  To appear.
* Gilmer et al. [2021]

  Justin Gilmer, Behrooz Ghorbani, Ankush Garg, Sneha Kudugunta, Behnam
  Neyshabur, David Cardoze, George Dahl, Zachary Nado, and Orhan Firat.
  A loss curvature perspective on training instability in deep
  learning.
  *arXiv preprint arXiv:2110.04369*, 2021.
* Glorot and Bengio [2010]

  Xavier Glorot and Yoshua Bengio.
  Understanding the difficulty of training deep feedforward neural
  networks.
  In *Proceedings of the thirteenth international conference on
  artificial intelligence and statistics*, pages 249–256. JMLR Workshop and
  Conference Proceedings, 2010.
* Google [2023]

  Google.
  Grain - feeding jax models, 2023.
  URL <http://github.com/google/grain>.
* Heek et al. [2023]

  Jonathan Heek, Anselm Levskaya, Avital Oliver, Marvin Ritter, Bertrand
  Rondepierre, Andreas Steiner, and Marc van Zee.
  Flax: A neural network library and ecosystem for JAX, 2023.
  URL <http://github.com/google/flax>.
* Hendrycks and Gimpel [2016]

  Dan Hendrycks and Kevin Gimpel.
  Gaussian error linear units (gelus).
  *arXiv preprint arXiv:1606.08415*, 2016.
* Hoffmann et al. [2022]

  Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor
  Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes
  Welbl, Aidan Clark, et al.
  Training compute-optimal large language models.
  *arXiv preprint arXiv:2203.15556*, 2022.
* Hua et al. [2022]

  Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc Le.
  Transformer quality in linear time.
  In *International Conference on Machine Learning*, pages
  9099–9117. PMLR, 2022.
* Ivgi et al. [2023]

  Maor Ivgi, Oliver Hinder, and Yair Carmon.
  Dog is sgd’s best friend: A parameter-free dynamic step size
  schedule.
  *arXiv preprint arXiv:2302.12022*, 2023.
* Jacot et al. [2018]

  Arthur Jacot, Franck Gabriel, and Clément Hongler.
  Neural tangent kernel: Convergence and generalization in neural
  networks.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2018.
  <https://arxiv.org/abs/1806.07572>.
* Jouppi et al. [2017]

  Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal,
  Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, et al.
  In-datacenter performance analysis of a tensor processing unit.
  In *Proceedings of the 44th annual international symposium on
  computer architecture*, pages 1–12, 2017.
* Kaplan et al. [2020]

  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon
  Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
  Scaling laws for neural language models.
  *arXiv preprint arXiv:2001.08361*, 2020.
* Kudo and Richardson [2018]

  Taku Kudo and John Richardson.
  Sentencepiece: A simple and language independent subword tokenizer
  and detokenizer for neural text processing.
  *arXiv preprint arXiv:1808.06226*, 2018.
* Lee [2023]

  Jaehoon Lee.
  A random walk model of transformer parameter growth, 2023.
* Liu et al. [2019]

  Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng
  Gao, and Jiawei Han.
  On the variance of the adaptive learning rate and beyond.
  *arXiv preprint arXiv:1908.03265*, 2019.
* Liu\* et al. [2018]

  Peter J. Liu\*, Mohammad Saleh\*, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz
  Kaiser, and Noam Shazeer.
  Generating wikipedia by summarizing long sequences.
  In *International Conference on Learning Representations*, 2018.
  URL <https://openreview.net/forum?id=Hyg0vbWC->.
* Loshchilov and Hutter [2016]

  Ilya Loshchilov and Frank Hutter.
  Sgdr: Stochastic gradient descent with warm restarts.
  In *International Conference on Learning Representations
  (ICLR)*, 2016.
  <https://arxiv.org/abs/1608.03983>.
* Loshchilov and Hutter [2019]

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  In *International Conference on Learning Representations
  (ICLR)*, 2019.
  <https://openreview.net/forum?id=Bkg6RiCqY7>.
* Merrill et al. [2020]

  William Merrill, Vivek Ramanujan, Yoav Goldberg, Roy Schwartz, and Noah Smith.
  Effects of parameter norm growth during transformer training:
  Inductive bias from gradient descent.
  *arXiv preprint arXiv:2010.09697*, 2020.
* Molybog et al. [2023]

  Igor Molybog, Peter Albert, Moya Chen, Zachary DeVito, David Esiobu, Naman
  Goyal, Punit Singh Koura, Sharan Narang, Andrew Poulton, Ruan Silva, et al.
  A theory on adam instability in large-scale machine learning.
  *arXiv preprint arXiv:2304.09871*, 2023.
* Paszke et al. [2019]

  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
  Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.
  Pytorch: An imperative style, high-performance deep learning library.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2019.
  <https://arxiv.org/abs/1912.01703>.
* Press and Wolf [2017]

  Ofir Press and Lior Wolf.
  Using the output embedding to improve language models.
  In *Proceedings of the 15th Conference of the European Chapter
  of the Association for Computational Linguistics: Volume 2, Short Papers*,
  pages 157–163, Valencia, Spain, April 2017. Association for Computational
  Linguistics.
  URL <https://aclanthology.org/E17-2025>.
* Radford et al. [2019]

  Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya
  Sutskever.
  Language Models are Unsupervised Multitask Learners, 2019.
  <https://openai.com/blog/better-language-models/>.
* Raffel et al. [2020a]

  Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
  Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.
  Exploring the limits of transfer learning with a unified text-to-text
  transformer.
  *Journal of Machine Learning Research*, 2020a.
  <http://jmlr.org/papers/v21/20-074.html>.
* Raffel et al. [2020b]

  Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
  Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.
  Exploring the limits of transfer learning with a unified text-to-text
  transformer.
  *Journal of Machine Learning Research*, 21(140):1–67, 2020b.
  URL <http://jmlr.org/papers/v21/20-074.html>.
* Ren et al. [2021]

  Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan
  Yang, Minjia Zhang, Dong Li, and Yuxiong He.
  {{\{ZeRO-Offload}}\}: Democratizing {{\{Billion-Scale}}\} model
  training.
  In *2021 USENIX Annual Technical Conference (USENIX ATC 21)*,
  pages 551–564, 2021.
* Shazeer and Stern [2018]

  Noam Shazeer and Mitchell Stern.
  Adafactor: Adaptive learning rates with sublinear memory cost.
  In *International Conference on Machine Learning*, pages
  4596–4604. PMLR, 2018.
* Su et al. [2021]

  Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu.
  Roformer: Enhanced transformer with rotary position embedding.
  *arXiv preprint arXiv:2104.09864*, 2021.
* Thilak et al. [2022]

  Vimal Thilak, Etai Littwin, Shuangfei Zhai, Omid Saremi, Roni Paiss, and Joshua
  Susskind.
  The slingshot mechanism: An empirical study of adaptive optimizers
  and the grokking phenomenon.
  *arXiv preprint arXiv:2206.04817*, 2022.
* Vaswani et al. [2017]

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  *Advances in neural information processing systems*, 30, 2017.
* [46]

  Mitchell Wortsman, Jaehoon Lee, Justin Gilmer, and Simon Kornblith.
  Replacing softmax with relu in vision transformers.
* Wortsman et al. [2023]

  Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari Morcos, Ali Farhadi, and
  Ludwig Schmidt.
  Stable and low-precision training for large-scale vision-language
  models.
  *arXiv preprint arXiv:2304.13013*, 2023.
* Yaida [2022]

  Sho Yaida.
  Meta-principled family of hyperparameter scaling strategies.
  *arXiv preprint arXiv:2210.04909*, 2022.
* Yang and Hu [2021]

  Greg Yang and Edward J Hu.
  Tensor programs iv: Feature learning in infinite-width neural
  networks.
  In *International Conference on Machine Learning*, pages
  11727–11737. PMLR, 2021.
* Yang et al. [2022]

  Greg Yang, Edward J Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David
  Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao.
  Tensor programs v: Tuning large neural networks via zero-shot
  hyperparameter transfer.
  *arXiv preprint arXiv:2203.03466*, 2022.
* Zhai et al. [2023a]

  Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason
  Ramapuram, Yizhe Zhang, Jiatao Gu, and Josh Susskind.
  Stabilizing transformer training by preventing attention entropy
  collapse.
  *arXiv preprint arXiv:2303.06296*, 2023a.
* Zhai et al. [2023b]

  Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer.
  Sigmoid loss for language image pre-training.
  *arXiv preprint arXiv:2303.15343*, 2023b.
* Zhang et al. [2022]

  Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui
  Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al.
  Opt: Open pre-trained transformer language models.
  *arXiv preprint arXiv:2205.01068*, 2022.

## Appendix A Additional infrastructure details

This Section provides more details on the training infrastructure, which is built on Flax [[20](#bib.bib20)], Jax [[4](#bib.bib4)], and TPUs [[26](#bib.bib26)], and we call NanoDO.
To enable larger model training, we shard the model and optimizer states as in FSDP [[41](#bib.bib41)], then specify these shadings when compiling with JIT.
We use Orbax [[15](#bib.bib15)] for checkpointing, and Grain [[19](#bib.bib19)] for deterministic data loading.
When loading data, sequences are packed so that no padding is required—if a sequence is less tokens than the context length hyperparameter, then an end of sequence token is appended, followed by the beginning of a new sequence.

## Appendix B When is learning rate sensitivity a useful metric

There are cases where LR sensitivity (defined in Section [2.2](#S2.SS2 "2.2 LR vs. loss curves and learning rate sensitivity ‣ 2 Experimental methodology ‣ Small-scale proxies for large-scale Transformer training instabilities")) is no longer a useful metric.
This section details these scenarios and justifies the use of LR sensitivity for the interventions in this paper.

Interventions which change the meaning of learning rate

When an intervention changes the meaning of learning rate then comparing LR sensitivity is not useful.
A clear example of this would be taking the square root of the LR before passing it to the optimizer, but there are more subtle cases to be cautious of when using LR sensitivity.

In general, we avoid manipulations where the meaning of LR meaningfully changes.
In some cases, we have good empirical evidence that the meaning of the learning rate has not changed when intervening.
For instance, the LR vs. loss curves are indistinguishable up to some critical learning rate when using qk-layernorm (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Small-scale proxies for large-scale Transformer training instabilities")), adding z-loss (Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities")), or changing warm-up.

In other cases, such as when testing μ𝜇\muParam (Section [3.2.4](#S3.SS2.SSS4 "3.2.4 𝜇Param ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities")), we believe that LR sensitivity is useful despite a per-layer modification of LR.
This is because the per-layer LR is manipulated linearly, and this modification does not change for different points on the LR vs loss curve.

The one experiment in this paper where we believe LR sensitivity is likely not a useful metric is when scaling learning rate by the root mean square of the parameters (Figure [E.14](#A5.F14 "Figure E.14 ‣ Appendix E Additional figures ‣ Small-scale proxies for large-scale Transformer training instabilities")).
Therefore, we do not measure LR sensitivity in that case.

Shifting of the optimal LR

The definition of LR sensitivity in Section [2.2](#S2.SS2 "2.2 LR vs. loss curves and learning rate sensitivity ‣ 2 Experimental methodology ‣ Small-scale proxies for large-scale Transformer training instabilities") does not account for the optimal LR shifting when specifying the LR range [a,b]𝑎𝑏[a,b].
In practice we recommend shifting the three order of magnitude range [a,b]𝑎𝑏[a,b] to correspond with this shift.
For instance, we shift the range in Section [3.2.2](#S3.SS2.SSS2 "3.2.2 Independent weight decay ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities"), as discussed in more detail in the section.
However, our main experiments (e.g., Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Small-scale proxies for large-scale Transformer training instabilities")) do not test at a large enough scale to necessitate this shift.

LR sensitivity is invariant to loss

Another limitation of the LR sensitivity metric is that it is invariant to the scale of the loss.
If the network consistently achieves random performance across learning rates, then LR sensitivity will be zero.
We do not offer a solution to this, and instead recommend that LR sensitivity should always be examined in combination with the LR vs. loss curves as we do in this paper.
It is meant as a useful summary of the LR vs. loss curves, not as a metric to optimize in isolation.

## Appendix C Output norm growth

This section discusses the growth of the output norms during Transformer training as previously studied by Merrill et al. [[34](#bib.bib34)], Lee [[29](#bib.bib29)], and relates this phenomenon to the attention logit growth and AdamW epsilon instabilities (Sections [3.1.1](#S3.SS1.SSS1 "3.1.1 Attention logit growth ‣ 3.1 Reproducing two known instabilities at small scale ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") and [3.4](#S3.SS4 "3.4 Searching for new instabilities via scaling trends of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities"), respectively).
As empirical evidence, Figure [C.1](#A3.F1 "Figure C.1 ‣ Appendix C Output norm growth ‣ Small-scale proxies for large-scale Transformer training instabilities") shows that the RMS of the Transformer block output is mainly determined by learning rate.

We have two hypothesis which relate parameter norm growth [[34](#bib.bib34)] and subsequent output norm growth to instability.
First, we believe that the attention output logits are the first to become large because they are the only feature in the network we test whose magnitude depends quadratically on parameter RMS.
For inputs X𝑋X with unit RMS, a typical matrix multiply X​W𝑋𝑊XW with parameters W𝑊W will result in features Y𝑌Y where RMS​(Y)RMS𝑌\text{RMS}(Y) is a linear function of RMS​(W)RMS𝑊\text{RMS}(W).
On the other hand, the attention logit entries are computed via ⟨X​W1,X​W2⟩

𝑋subscript𝑊1𝑋subscript𝑊2\langle XW\_{1},XW\_{2}\rangle so depend quadratically on RMS​(W)RMS𝑊\text{RMS}(W).
Next, this helps to explain the decreasing trend in gradient scale observed in Section [3.4](#S3.SS4 "3.4 Searching for new instabilities via scaling trends of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") (Figure [11](#S3.F11 "Figure 11 ‣ 3.3 Predicting attention logit growth instability from scaling behavior of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities")).
In a pre-normalization [[38](#bib.bib38)] Transformer [[45](#bib.bib45)] there is an output layernorm layer [[1](#bib.bib1)] after the last Transformer block and before the final linear layer.
The gradient from this output layernorm layer is scaled by the reciprocal of the input RMS.
This RMS is growing with depth because of the residual connections (Figure [C.1](#A3.F1 "Figure C.1 ‣ Appendix C Output norm growth ‣ Small-scale proxies for large-scale Transformer training instabilities")).
As the RMS leaving the last Transformer block grows, the gradient received shrinks.

For completeness we now compute the layernorm gradient to input x𝑥x. We assume the input as mean zero and the layernorm has no bias for simplicity. Let

|  |  |  |  |
| --- | --- | --- | --- |
|  | z=LayerNorm(x)=α⋅x𝔼i​[xi2]+ϵ=α⋅xm1/2𝑧LayerNorm(x)⋅𝛼𝑥subscript𝔼𝑖delimited-[]superscriptsubscript𝑥𝑖2italic-ϵ⋅𝛼𝑥superscript𝑚12\displaystyle z=\text{LayerNorm(x)}=\alpha\cdot\frac{x}{\sqrt{\mathbb{E}\_{i}\left[x\_{i}^{2}\right]+\epsilon}}=\alpha\cdot\frac{x}{m^{1/2}} |  | (1) |

where m=𝔼i​[xi2]+ϵ𝑚subscript𝔼𝑖delimited-[]superscriptsubscript𝑥𝑖2italic-ϵm=\mathbb{E}\_{i}\left[x\_{i}^{2}\right]+\epsilon.

Then

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂ℓ∂xj=∑k∂ℓ∂zk​∂zk∂xjℓsubscript𝑥𝑗subscript𝑘ℓsubscript𝑧𝑘subscript𝑧𝑘subscript𝑥𝑗\displaystyle\frac{\partial\ell}{\partial x\_{j}}=\sum\_{k}\frac{\partial\ell}{\partial z\_{k}}\frac{\partial z\_{k}}{\partial x\_{j}} |  | (2) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =∂ℓ∂zj⋅αjm1/2+∑k∂ℓ∂zk⋅(−12)⋅αk​xkm3/2⋅2n⋅xjabsent⋅ℓsubscript𝑧𝑗subscript𝛼𝑗superscript𝑚12subscript𝑘⋅ℓsubscript𝑧𝑘12subscript𝛼𝑘subscript𝑥𝑘superscript𝑚322𝑛subscript𝑥𝑗\displaystyle=\frac{\partial\ell}{\partial z\_{j}}\cdot\frac{\alpha\_{j}}{m^{1/2}}+\sum\_{k}\frac{\partial\ell}{\partial z\_{k}}\cdot\left(-\frac{1}{2}\right)\cdot\frac{\alpha\_{k}x\_{k}}{m^{3/2}}\cdot\frac{2}{n}\cdot x\_{j} |  | (3) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =1m1/2​(αj​∂ℓ∂zj−xjn​m1/2​∑k∂ℓ∂zk​αk​xk)absent1superscript𝑚12subscript𝛼𝑗ℓsubscript𝑧𝑗subscript𝑥𝑗𝑛superscript𝑚12subscript𝑘ℓsubscript𝑧𝑘subscript𝛼𝑘subscript𝑥𝑘\displaystyle=\frac{1}{m^{1/2}}\left(\alpha\_{j}\frac{\partial\ell}{\partial z\_{j}}-\frac{x\_{j}}{nm^{1/2}}\sum\_{k}\frac{\partial\ell}{\partial z\_{k}}\alpha\_{k}x\_{k}\right) |  | (4) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =1m1/2​(αj​∂ℓ∂zj−xjn​m1/2​⟨∇z,α⋅x⟩)absent1superscript𝑚12subscript𝛼𝑗ℓsubscript𝑧𝑗subscript𝑥𝑗𝑛superscript𝑚12  subscript∇𝑧⋅𝛼𝑥\displaystyle=\frac{1}{m^{1/2}}\left(\alpha\_{j}\frac{\partial\ell}{\partial z\_{j}}-\frac{x\_{j}}{nm^{1/2}}\left\langle\nabla\_{z},\alpha\cdot x\right\rangle\right) |  | (5) |

Equivalently,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇x=1m1/2​(α⊙∇z−⟨∇z,α⊙x⟩n​m1/2⊙x).subscript∇𝑥1superscript𝑚12direct-product𝛼subscript∇𝑧direct-product  subscript∇𝑧direct-product𝛼𝑥 𝑛superscript𝑚12𝑥\displaystyle\nabla\_{x}=\frac{1}{m^{1/2}}\left(\alpha\odot\nabla\_{z}-\frac{\left\langle\nabla\_{z},\alpha\odot x\right\rangle}{nm^{1/2}}\odot x\right). |  | (6) |

![Refer to caption](/html/2309.14322/assets/x14.png)


Figure C.1: The root mean square (RMS) of the Transformer block outputs are roughly consistent with scale (left) but increase with learning rate (center).
RMS increases deeper in the transformer because of the residual connections, which is shown for very high learning rates (right).
The first two plots are for block index two, and RMS is averaged over the final 500 training steps.
Recall RMS(X)=𝔼i​[Xi2]𝑋subscript𝔼𝑖delimited-[]superscriptsubscript𝑋𝑖2(X)=\sqrt{\mathbb{E}\_{i}[X\_{i}^{2}]}.

## Appendix D Author contributions

Mitchell Wortsman led the project, ran the experiments and produced the figures, contributed substantially to the infrastructure for experimentation, the framing and direction, and the writing.

Peter J. Liu led the infrastructure and creation of NanoDO for experimentation, provided key insights and advice on multiple technical areas, and contributed to the writing.

Lechao Xiao and Katie Everett contributed to the infrastructure used for experimentation, provided key insight related to parameterization, and contributed to the writing.

Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-dickstein, and Kelvin Xu were active participants in weekly brainstorming meetings which motivated, influenced, and elucidated technical concepts pertaining to this work.

Jaehoon Lee and Justin Gilmer were senior authors advising on the project, contributed substantially to the framing and direction, provided key insight and advice on multiple technical areas, and contributed to the writing.
Jaehoon led the connection with output norm growth. Justin proposed to plot loss as a function of learning rate for different model sizes, and performed initial experiments demonstrating that attention logit growth could be reproduced at high learning rates in small models.

Simon Kornblith was the lead advisor on the project, contributing substantially to the framing, direction, infrastructure, and writing. Simon initially brainstormed the project with Mitchell, and was Mitchell’s host for the summer internship during which this research was conducted, providing substantial technical support.

## Appendix E Additional figures

This Section contains the additional Figures referenced in the main text.

![Refer to caption](/html/2309.14322/assets/x15.png)


Figure E.1: The logit growth instability [[11](#bib.bib11), [51](#bib.bib51)] occurs when the norm of the query and keys increases, not due to an increase in their cosine similarity.

![Refer to caption](/html/2309.14322/assets/x16.png)


Figure E.2: The effect of scaling width vs. scaling depth without qk-layernorm [[11](#bib.bib11)].

![Refer to caption](/html/2309.14322/assets/x17.png)


Figure E.3: Jointly scaling width and depth leads to lower loss than independently scaling depth or width at the largest scale we test.
It also leads to a more reliable scaling prediction when extrapolating from models with less than 1e8 parameters.
Best loss is reported in a sweep over learning rates.

![Refer to caption](/html/2309.14322/assets/x18.png)


Figure E.4: The effect of μ𝜇\muParam on LR sensitivity for models without qk-layernorm [[11](#bib.bib11)].
μ𝜇\muParam succeeds in stabilizing the optimal LR, but does not alleviate the need for qk-layernorm.
For more information refer to Section [3.2.4](#S3.SS2.SSS4 "3.2.4 𝜇Param ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities").

![Refer to caption](/html/2309.14322/assets/x19.png)


Figure E.5: Comparing μ𝜇\muParam (full), which implements μ𝜇\muParam as described in Yang et al. [[50](#bib.bib50)] with and without qk-layernorm, with μ𝜇\muParam (simple) and μ𝜇\muParam (intermediate).
There are four changes in μ𝜇\muParam (full),
(i) Scale the LR for linear layers by base-fan-in/fan-inbase-fan-infan-in\text{base-fan-in}/\text{fan-in},
(ii) initialize the head with standard deviation base-fan-in/fan-inbase-fan-infan-in\sqrt{\text{base-fan-in}}/\text{fan-in}.
(iii) change the 1/dh1subscript𝑑ℎ1/\sqrt{d\_{h}} scaling factor in attention layers to 1/dh1subscript𝑑ℎ1/d\_{h} where dhsubscript𝑑ℎd\_{h} is the head dimension, and
(iv) initialize the query projection weights with zeros.
μ𝜇\muParam (intermediate) consists of (i) and (ii), while μ𝜇\muParam (simple) is only (i).
With μ𝜇\muParam (full) and qk-layernorm, the model trains without diverging at LR 1.
However at the best LR there is no measurable improvement over μ𝜇\muParam (simple) at the largest scale we test.

![Refer to caption](/html/2309.14322/assets/x20.png)


Figure E.6: Measuring the effect of changing the 1/dh1subscript𝑑ℎ1/\sqrt{d\_{h}} term in attention to 1/dh1subscript𝑑ℎ1/d\_{h}, where dhsubscript𝑑ℎd\_{h} is head dimension.
Vaswani et al. [[45](#bib.bib45)] use 1/dh1subscript𝑑ℎ1/\sqrt{d\_{h}} while Yang et al. [[50](#bib.bib50)] use 1/dh1subscript𝑑ℎ1/d\_{h}.

![Refer to caption](/html/2309.14322/assets/x21.png)


Figure E.7: Changing the number of total training steps from 1e5 to 5e4 or 2e5 does not have a large effect of the shape of the learning rate vs. loss curves at the scales we test.

![Refer to caption](/html/2309.14322/assets/x22.png)


Figure E.8: We achieve slightly better performance when applying qk-layernorm individually per-head instead of across the model dimension.
The per-head variant has only head-dim learnable parameters instead of model-dim parameters.
We use the per-head variant as the default in this paper, and we never use biases.

![Refer to caption](/html/2309.14322/assets/x23.png)


Figure E.9: Increasing the batch size from 256 to 512 or 1024 does not have a large effect on the shape of the learning rate vs. loss curves at the scales we test.
Each batch element contains 512 tokens, and we use 256 as the default.

![Refer to caption](/html/2309.14322/assets/x24.png)


Figure E.10: The effect of weight decay on LR sensitivity. We use independent weight decay as described in Section [3.2.2](#S3.SS2.SSS2 "3.2.2 Independent weight decay ‣ 3.2 Measuring the effect of other known interventions ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") and recommended by [[33](#bib.bib33)].

![Refer to caption](/html/2309.14322/assets/x25.png)


Figure E.11: The logit growth instability occurs even without softmax.
For the pointwise variant of attention here, we replace softmax with squared-relu as described by [[23](#bib.bib23)]. As recommended in [[46](#bib.bib46)] we add a scaling factor which depends on sequence length. In this case, we use inverse square root.

![Refer to caption](/html/2309.14322/assets/x26.png)


Figure E.12: 
Recreating Figure [11](#S3.F11 "Figure 11 ‣ 3.3 Predicting attention logit growth instability from scaling behavior of model characteristics ‣ 3 Results ‣ Small-scale proxies for large-scale Transformer training instabilities") with the kernel projection instead of the first MLP layer.

![Refer to caption](/html/2309.14322/assets/x27.png)


Figure E.13: For various learning rates and model sizes we display the gradient root mean square (RMS), and the unscaled update RMS.
The unscaled udpate is the update returned by the optimizer before scaling by learning rate.
The gradient and update are shown here for the first MLP layer of the Transformer.
The update RMS falls when the grad RMS approaches the AdamW ϵitalic-ϵ\epsilon of 1e-8.

![Refer to caption](/html/2309.14322/assets/x28.png)


Figure E.14: The effect of scaling the learning rate for parameters p𝑝p by max⁡(RMS​(p),1e-3)RMS𝑝1e-3\max\left(\text{RMS}(p),\text{1e-3}\right) as in AdaFactor [[42](#bib.bib42)].
As discussed by Appendix [B](#A2 "Appendix B When is learning rate sensitivity a useful metric ‣ Small-scale proxies for large-scale Transformer training instabilities"), it is not meaningful to compare LR sensitivity in this case as this intervention modifies the meaning of learning rate.
Just as in μ𝜇\muParam [[50](#bib.bib50)], RMS scaling appears to stabilize the optimal LR in the range we test.

![Refer to caption](/html/2309.14322/assets/x29.png)


Figure E.15: Increasing the AdamW ϵitalic-ϵ\epsilon from its default value of 1e-8 to 1e-6 causes a loss divergence for a 4.8B parameter model at LR 0.3.
Grad RMS is for the first layer in the MLP.

[◄](/html/2309.14321)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2309.14322)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2309.14322)
[View original  
on arXiv](https://arxiv.org/abs/2309.14322)[►](/html/2309.14324)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Wed Feb 28 04:36:47 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
