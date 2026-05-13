---
arxiv: '2306.15595'
authors:
- Shouyuan Chen
- Sherman Wong
- Liangjian Chen
- Yuandong Tian
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Extending Context Window of Large Language Models via Positional Interpolation
url: https://arxiv.org/abs/2306.15595
year: 2023
---

[2306.15595] Extending Context Window of Large Language Models via Position Interpolation














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



# Extending Context Window of Large Language Models via Position Interpolation

Shouyuan Chen     Sherman Wong     Liangjian Chen     Yuandong Tian
  
Meta Platforms Inc.
  
{chenshouyuan,shermanwong,clj,yuandong}@meta.com

###### Abstract

We present Position Interpolation (PI) that extends
the context window sizes of RoPE-based (Su et al., [2021](#bib.bib34)) pretrained LLMs such as LLaMA (Touvron et al., [2023](#bib.bib36)) models to up to 32768 with minimal fine-tuning (within 1000 steps), while demonstrating strong empirical results on various tasks that require long context, including passkey retrieval, language modeling, and long document summarization from LLaMA 7B to 65B. Meanwhile, the extended model by Position Interpolation preserve quality relatively well on tasks within its original context window. To achieve this goal, Position Interpolation linearly down-scales the input position indices to match the original context window size, rather than extrapolating beyond the trained context length which may lead to catastrophically high attention scores that completely ruin the self-attention mechanism. Our theoretical study shows that the upper bound of interpolation is at least ∼600×\sim 600\times smaller than that of extrapolation, further demonstrating its stability. Models extended via Position Interpolation retain its original architecture
and can reuse most pre-existing optimization and infrastructure.

## 1 Introduction

Large language models (LLMs) typically come with a pre-defined context window size. For example, inputs to LLaMA models (Touvron et al., [2023](#bib.bib36)) must be fewer than 2048 tokens. This pre-set context window limit is frequently exceeded in applications such as conducting long conversations, summarizing long documents, or executing long-term planning. For these applications, LLMs with longer context windows are preferred. However, training an LLM from scratch with long context windows requires significant investments. This naturally leads to a question: Can we extend the context window of an existing pre-trained LLM?

One straightforward approach is to fine-tune an existing pre-trained Transformer with a longer context window. However, empirically, we found that models trained this way adapt to long context windows very slowly.
After training for more than 10000 batches, the effective context window saw a minimal increase, moving from 2048 to 2560 (Table [4](#S3.T4 "Table 4 ‣ 3.3 Measuring Effective Context Window Size through Passkey Retrieval ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation")). This suggests that such method is inefficient for extending to substantially longer context windows.

While certain techniques such as ALiBi (Press et al., [2022](#bib.bib29)) and LeX (Sun et al., [2022](#bib.bib35)) enable length extrapolation of Transformers, i.e. train on short context windows and inference on longer ones, many existing pre-trained LLMs, including LLaMA (Touvron et al., [2023](#bib.bib36)), use positional encodings that have weak extrapolation properties (e.g., RoPE (Su et al., [2021](#bib.bib34))). Therefore, the applicability of these techniques for extending the context window sizes of such LLMs remains limited.

In this work, we introduce Position Interpolation to enable context window extensions for certain existing pre-trained LLMs, including LLaMA. The key idea is, instead of extrapolation, we directly down-scale the position indices so that the maximum position index matches the previous context window limit in the pre-training stage. See Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Extending Context Window of Large Language Models via Position Interpolation") for an illustration. In other words, to accommodate more input tokens, we interpolate the position encodings at neighboring integer positions, utilizing the fact that position encodings can be applied on non-integer positions, as opposed to extrapolating outside the trained positions, which may lead to catastrophic values. We verify our approach theoretically, by showing that the interpolated attention score has a much smaller upper bound (∼600×\sim 600\times smaller in LLaMA 7B setting) than the extrapolated one, and is thus much more stable. Therefore, interpolated position encodings are easier for the model to adapt.

![Refer to caption](/html/2306.15595/assets/x1.png)


Figure 1: An illustration of our Position Interpolation method. Consider a Llama model pre-trained with a 2048 context window length. Upper left illustrates the normal usage of an LLM model: input position indices (blue dots) are within the pre-trained range. Upper right illustrates length extrapolation where models are required to operate unseen positions (red dots) up to 4096. Lower left illustrates Position Interpolation where we downscale the position indices (blue and green dots) themselves from [0, 4096] to [0, 2048] to force them to reside in the pretrained range.

Empirically, we found that Position Interpolation is highly effective and efficient, requiring only a very short period of fine-tuning for the model to fully adapt to greatly extended context windows. We present experimental results for extending the context window to up to 32768 from the initial 2048 across 7B to 65B LLaMA models using Position Interpolation. Our results show that

1. 1.

   Position Interpolation can easily enable very long context windows (e.g. 32768), requiring only fine-tuning for 1000 steps on the Pile (Gao et al., [2020](#bib.bib11)) to achieve a good quality.
   The cost of fine-tuning is negligible compared to the pre-training costs.
   This confirms our hypothesis that it is relatively easy for the models to adapt to interpolated position encodings.
2. 2.

   Position Interpolation generates strong models that can effectively make use of much extended context window. We show that models extended by Position Interpolation enjoy significant perplexity gains from greatly extended context windows for text modeling, and we show that the perplexity reduces graceful with the enlargement of context windows. We also applied Position Interpolation in a long text summarization task, and demonstrate competitive performances.
3. 3.

   Position Interpolation preserves model quality relatively well for tasks within its original context window sizes. We present a variety of evaluation results for the extended LLaMA models on the original LLaMA benchmark. Compared with original LLaMA models, the extended LLaMA models saw a minor degradation on several standard benchmarks within a 2048 token limit.

Our results highlight the innate ability of Transformer models to “extrapolate to sequence lengths longer than the ones encountered during training” as hypothesized in the seminal work of Vaswani et al. ([2017](#bib.bib37)). We reaffirm this hypothesis and suggest that the previously known weakness of extrapolating to longer sequences for language modeling (Press et al., [2022](#bib.bib29)) may be due to direct extrapolation of positional encodings and it can be largely mitigated by interpolating position encodings instead.

Concurrent work. Right before our release, we are informed with a concurrent blogpost (SuperHOT kaiokendev ([2023](#bib.bib18))) that also interpolates positional encoding in RoPE to extend the context window from 2K to 8K. Recently, open source community picks it up in Reddit post 111<https://www.reddit.com/r/LocalLLaMA/comments/14fgjqj/a_simple_way_to_extending_context_to_8k/> and Github Issues 222<https://github.com/ggerganov/llama.cpp/discussions/1965>, which shows that fine-tuning with LoRA (Hu et al., [2021](#bib.bib14)) also seems to work well. Our paper shows a full fine-tuning with up to 65B model work well with Position Interpolation, and we also give theoretical explanations why interpolation achieves much more stable results than extrapolation, by showing that the upper bound of interplated attention score is much lower than that of extrapolated ones.

## 2 Method

### 2.1 Background: Rotary Position Embedding (RoPE)

Transformer models require explicit positional information to be injected, typically in the form of positional encodings, to represent the order of inputs. We consider Rotary Position Embedding (RoPE) (Su et al., [2021](#bib.bib34)), which is the position encoding used in the LLaMA model (Touvron et al., [2023](#bib.bib36)). Given a position index m∈[0,c)𝑚0𝑐m\in[0,c) and an embedding vector 𝐱:=[x0,x1,…,xd−1]⊤assign𝐱superscript

subscript𝑥0subscript𝑥1…subscript𝑥𝑑1
top\mathbf{x}:=[x\_{0},x\_{1},\ldots,x\_{d-1}]^{\top}, where d𝑑d is the dimension of the attention head, RoPE defines a vector-valued complex function 𝐟​(𝐱,m)𝐟𝐱𝑚\mathbf{f}(\mathbf{x},m) as follows

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐟​(𝐱,m)=[(x0+i​x1)​ei​m​θ0,(x2+i​x3)​ei​m​θ1,…,(xd−2+i​xd−1)​ei​m​θd/2−1]⊤𝐟𝐱𝑚superscript  subscript𝑥0isubscript𝑥1superscript𝑒i𝑚subscript𝜃0subscript𝑥2isubscript𝑥3superscript𝑒i𝑚subscript𝜃1…subscript𝑥𝑑2isubscript𝑥𝑑1superscript𝑒i𝑚subscript𝜃𝑑21 top\mathbf{f}(\mathbf{x},m)=[(x\_{0}+\mathrm{i}x\_{1})e^{\mathrm{i}m\theta\_{0}},(x\_{2}+\mathrm{i}x\_{3})e^{\mathrm{i}m\theta\_{1}},\ldots,(x\_{d-2}+\mathrm{i}x\_{d-1})e^{\mathrm{i}m\theta\_{d/2-1}}]^{\top} |  | (1) |

where i:=−1assigni1\mathrm{i}:=\sqrt{-1} is the imaginary unit and θj=10000−2​j/dsubscript𝜃𝑗superscript100002𝑗𝑑\theta\_{j}=10000^{-2j/d}. Using RoPE, the self-attention score

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | a​(m,n)𝑎𝑚𝑛\displaystyle a(m,n) | =\displaystyle= | Re​⟨𝐟​(𝐪,m),𝐟​(𝐤,n)⟩Re  𝐟𝐪𝑚𝐟𝐤𝑛\displaystyle\mathrm{Re}\langle\mathbf{f}(\mathbf{q},m),\mathbf{f}(\mathbf{k},n)\rangle |  | (2) |
|  |  | =\displaystyle= | Re​[∑j=0d/2−1(q2​j+i​q2​j+1)​(k2​j−i​k2​j+1)​ei​(m−n)​θj]Redelimited-[]superscriptsubscript𝑗0𝑑21subscript𝑞2𝑗isubscript𝑞2𝑗1subscript𝑘2𝑗isubscript𝑘2𝑗1superscript𝑒i𝑚𝑛subscript𝜃𝑗\displaystyle\mathrm{Re}\left[\sum\_{j=0}^{d/2-1}(q\_{2j}+\mathrm{i}q\_{2j+1})(k\_{2j}-\mathrm{i}k\_{2j+1})e^{\mathrm{i}(m-n)\theta\_{j}}\right] |  |
|  |  | =\displaystyle= | ∑j=0d/2−1(q2​j​k2​j+q2​j+1​k2​j+1)​cos⁡((m−n)​θj)+(q2​j​k2​j+1−q2​j+1​k2​j)​sin⁡((m−n)​θj)superscriptsubscript𝑗0𝑑21subscript𝑞2𝑗subscript𝑘2𝑗subscript𝑞2𝑗1subscript𝑘2𝑗1𝑚𝑛subscript𝜃𝑗subscript𝑞2𝑗subscript𝑘2𝑗1subscript𝑞2𝑗1subscript𝑘2𝑗𝑚𝑛subscript𝜃𝑗\displaystyle\sum\_{j=0}^{d/2-1}(q\_{2j}k\_{2j}+q\_{2j+1}k\_{2j+1})\cos((m-n)\theta\_{j})+(q\_{2j}k\_{2j+1}-q\_{2j+1}k\_{2j})\sin((m-n)\theta\_{j}) |  |
|  |  | =:absent:\displaystyle=: | a​(m−n)𝑎𝑚𝑛\displaystyle a(m-n) |  |

is only dependent on relative position m−n𝑚𝑛m-n through trigonometric functions. Here 𝐪𝐪\mathbf{q} and 𝐤𝐤\mathbf{k} are the query and key vector for a specific attention head. At each layer, RoPE is applied on both query and key embeddings for computing attention scores.

### 2.2 Direct Extrapolation

While the attention score in RoPE only depends on the relative positions, which is what we want, its extrapolation performance is not great . In particular, when directly extending to larger context windows unseen in the training, the perplexity may shoot up to very high numbers (i.e., >103absentsuperscript103>10^{3}), comparable to untrained models.

Ideally, we want to see the model trained on a context window of size L=2048𝐿2048L=2048 to still work reasonably well on longer context window, but may not have the capability to leverage information that appears beyond L𝐿L. For example, to answer a question located at 3000, the model trained on maximal window size of L=2048𝐿2048L=2048 cannot leverage evidences provided at location 0, but still can leverage the evidences provided at location 2900. In contrast, in reality we see catastrophic behaviors, i.e., question at location 3000 cannot be answered correctly, even if the evidences are located at location 2900.

![Refer to caption](/html/2306.15595/assets/x2.png)


Figure 2: Extrapolation versus interpolation. Left: a fitted attention score function (in red) in the form of Eqn. [3](#S2.E3 "In 2.2 Direct Extrapolation ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation") with d=dmodel/nhead=4096/32=128𝑑subscript𝑑modelsubscript𝑛head409632128d=d\_{\mathrm{model}}/n\_{\mathrm{head}}=4096/32=128 (setting of LLaMA 7B). Dots are random input points to be fitted and red curve is the fitted score function via least square, which is approximately within [−1,1]11[-1,1]. Middle: While the fitted function seems to be well bounded in [0,L]0𝐿[0,L], where L=2048𝐿2048L=2048, out of this region it may goes beyond 800080008000, causing catastrophic issues in attention computation. Note that here we do not cherry pick at all: almost every learned curve from a set of randomly generated input points within [0,L]0𝐿[0,L] has the extrapolation issue. Right: On the other hand, interpolation is much more stable. Curves in between vertical dotted lines (i.e., integer positional difference) are smooth and well-behaved. Please check Appendix [C.1](#A3.SS1 "C.1 Code for Fig. 2 ‣ Appendix C Code ‣ Extending Context Window of Large Language Models via Position Interpolation") for the source code used to generate the figure.

What is the reason behind? How could this happen if the attention score am−nsubscript𝑎𝑚𝑛a\_{m-n} decays as the relative distance |m−n|𝑚𝑛|m-n| increases, according to Section 3.4.3 of (Su et al., [2021](#bib.bib34)), and content from very far distances should not matter that much? It turns out that the upper bound derived in Section 3.4.3 of (Su et al., [2021](#bib.bib34)) may be too loose: while it indeed decays with respect to |m−n|𝑚𝑛|m-n|, the bound can still be quite large (i.e., the bound can be critically depends on the magnitude of vjsubscript𝑣𝑗v\_{j}) and thus vacuous.
In fact, if we treat all trigonometric functions as basis functions (i.e, ϕj​(s):=ei​s​θjassignsubscriptitalic-ϕ𝑗𝑠superscript𝑒i𝑠subscript𝜃𝑗\phi\_{j}(s):=e^{\mathrm{i}s\theta\_{j}}), and think about Eqn. [2](#S2.E2 "In 2.1 Background: Rotary Position Embedding (RoPE) ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation") as basis expansion as the following:

|  |  |  |  |
| --- | --- | --- | --- |
|  | a​(s)=Re​[∑j=0d/2−1hj​ei​s​θj]𝑎𝑠Redelimited-[]superscriptsubscript𝑗0𝑑21subscriptℎ𝑗superscript𝑒i𝑠subscript𝜃𝑗a(s)=\mathrm{Re}\left[\sum\_{j=0}^{d/2-1}h\_{j}e^{\mathrm{i}s\theta\_{j}}\right] |  | (3) |

where s𝑠s is the positional span between a query and a key and hj:=(q2​j+i​q2​j+1)​(k2​j−i​k2​j+1)assignsubscriptℎ𝑗subscript𝑞2𝑗isubscript𝑞2𝑗1subscript𝑘2𝑗isubscript𝑘2𝑗1h\_{j}:=(q\_{2j}+\mathrm{i}q\_{2j+1})(k\_{2j}-\mathrm{i}k\_{2j+1}) are complex coefficients depending on 𝐪𝐪\mathbf{q} and 𝐤𝐤\mathbf{k} (here the definition of hjsubscriptℎ𝑗h\_{j} is exactly the same as the definition of hjsubscriptℎ𝑗h\_{j} in Sec 3.4.3 in RoPE (Su et al., [2021](#bib.bib34))). Now the the issue becomes clear: as shown in Fig. [2](#S2.F2 "Figure 2 ‣ 2.2 Direct Extrapolation ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation"), assubscript𝑎𝑠a\_{s} can be small in magnitude in the range of [0,2048]02048[0,2048], but gives huge values out of the region. The underlying reason is that the trigonometric family {ϕj}subscriptitalic-ϕ𝑗\{\phi\_{j}\} (with sufficiently large d𝑑d) is a universal approximator and can fit any arbitrary functions. Therefore, for assubscript𝑎𝑠a\_{s}, there always exist coefficients {hj}subscriptℎ𝑗\{h\_{j}\} (i.e. key and query) that corresponds to small function values in [0, 2048] but much larger in regions beyond.

### 2.3 Proposed approach: Position Interpolation (PI)

In Fig. [2](#S2.F2 "Figure 2 ‣ 2.2 Direct Extrapolation ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation"), thanks to the smoothness of bases functions ϕjsubscriptitalic-ϕ𝑗\phi\_{j}
*interpolation* is much more stable and will not lead to wild values. Therefore, instead of extrapolate the attention score in Eqn. [3](#S2.E3 "In 2.2 Direct Extrapolation ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation") to s>L𝑠𝐿s>L, how about we define an attention score a~​(s)=a​(L​s/L′)~𝑎𝑠𝑎𝐿𝑠superscript𝐿′\tilde{a}(s)=a(Ls/L^{\prime}) where L′superscript𝐿′L^{\prime} is the longer context window? Formally, we replace RoPE 𝐟𝐟\mathbf{f} by 𝐟​’𝐟’\mathbf{f}’ defined as follows

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐟​’​(𝐱,m)=𝐟​(𝐱,m​LL′).𝐟’𝐱𝑚𝐟𝐱𝑚𝐿superscript𝐿′\mathbf{f}’(\mathbf{x},m)=\mathbf{f}\left(\mathbf{x},\frac{mL}{L^{\prime}}\right). |  | (4) |

We call this transformation on the position encoding Position Interpolation. In this step, we reduce position indices from [0,L′)0superscript𝐿′[0,L^{\prime}) to [0,L)0𝐿[0,L) to match the original range of indices before computing RoPE. Consequently, as inputs to RoPE, the maximum relative distance between any two tokens has been reduced from L′superscript𝐿′L^{\prime} to L𝐿L. Since we align the ranges of position indices and relative distances before and after extension, we mitigate the effect on attention score computation due to context window extensions, which can allow the model easier to adapt. To further demonstrate this is the case, in the following theorem, we show that the interpolated attention score is well-behaved:

###### Theorem 2.1 (Interpolation bound).

For attention score a​(s)=Re​[∑j=0d/2−1hj​ei​s​θj]𝑎𝑠Redelimited-[]superscriptsubscript𝑗0𝑑21subscriptℎ𝑗superscript𝑒i𝑠subscript𝜃𝑗a(s)=\mathrm{Re}\left[\sum\_{j=0}^{d/2-1}h\_{j}e^{\mathrm{i}s\theta\_{j}}\right], where θj=c−2​j/dsubscript𝜃𝑗superscript𝑐2𝑗𝑑\theta\_{j}=c^{-2j/d}, its interpolation value a​(s)𝑎𝑠a(s) for s∈[s1,s2]𝑠subscript𝑠1subscript𝑠2s\in[s\_{1},s\_{2}] is bounded as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |a​(s)−alinear​(s)|≤d​(maxj⁡|hj|)​(s−s1)​(s2−s)8​ln⁡c𝑎𝑠subscript𝑎linear𝑠𝑑subscript𝑗subscriptℎ𝑗𝑠subscript𝑠1subscript𝑠2𝑠8𝑐|a(s)-a\_{\mathrm{linear}}(s)|\leq d\left(\max\_{j}|h\_{j}|\right)\frac{(s-s\_{1})(s\_{2}-s)}{8\ln c} |  | (5) |

where alinear​(s)subscript𝑎linear𝑠a\_{\mathrm{linear}}(s) is the linear interpolation of two grid point a​(s1)𝑎subscript𝑠1a(s\_{1}) and a​(s2)𝑎subscript𝑠2a(s\_{2}) that are known to behave well, enforced by LLM pre-training:

|  |  |  |  |
| --- | --- | --- | --- |
|  | alinear​(s):=(1−λ​(s))​a​(s1)+λ​(s)​a​(s2),λ​(s):=s−s1s2−s1formulae-sequenceassignsubscript𝑎linear𝑠1𝜆𝑠𝑎subscript𝑠1𝜆𝑠𝑎subscript𝑠2assign𝜆𝑠𝑠subscript𝑠1subscript𝑠2subscript𝑠1a\_{\mathrm{linear}}(s):=(1-\lambda(s))a(s\_{1})+\lambda(s)a(s\_{2}),\quad\quad\lambda(s):=\frac{s-s\_{1}}{s\_{2}-s\_{1}} |  | (6) |

Please check Appendix [A](#A1 "Appendix A Proof ‣ Extending Context Window of Large Language Models via Position Interpolation") for the proof. Intuitively, in LLM pre-training, we know that the attention score a​(s)𝑎𝑠a(s) behaves well on integer grid s1subscript𝑠1s\_{1} and s2subscript𝑠2s\_{2}. Therefore, for any interpolation s∈[s1,s2]𝑠subscript𝑠1subscript𝑠2s\in[s\_{1},s\_{2}], we have (s−s1)​(s2−s)≤1/4𝑠subscript𝑠1subscript𝑠2𝑠14(s-s\_{1})(s\_{2}-s)\leq 1/4. Note that c=10000𝑐10000c=10000, the bound becomes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |a​(s)−alinear​(s)|≤d32​ln⁡c​maxj⁡|hj|≈d​maxj⁡|hj|294.73𝑎𝑠subscript𝑎linear𝑠𝑑32𝑐subscript𝑗subscriptℎ𝑗𝑑subscript𝑗subscriptℎ𝑗294.73|a(s)-a\_{\mathrm{linear}}(s)|\leq\frac{d}{32\ln c}\max\_{j}|h\_{j}|\approx\frac{d\max\_{j}|h\_{j}|}{294.73} |  | (7) |

In comparison, Sec. 3.4.3 in RoPE (Su et al., [2021](#bib.bib34)) yields an extrapolation bound (i.e., it works for all positional distance s𝑠s):

|  |  |  |  |
| --- | --- | --- | --- |
|  | |a​(s)|≤(maxj⁡|hj−hj+1|)​∑k=0d/2−1|Ak+1​(s)|≤2​(maxj⁡|hj|)​∑k=0d/2−1|Ak+1​(s)|,𝑎𝑠subscript𝑗subscriptℎ𝑗subscriptℎ𝑗1superscriptsubscript𝑘0𝑑21subscript𝐴𝑘1𝑠2subscript𝑗subscriptℎ𝑗superscriptsubscript𝑘0𝑑21subscript𝐴𝑘1𝑠|a(s)|\leq\left(\max\_{j}|h\_{j}-h\_{j+1}|\right)\sum\_{k=0}^{d/2-1}|A\_{k+1}(s)|\leq 2\left(\max\_{j}|h\_{j}|\right)\sum\_{k=0}^{d/2-1}|A\_{k+1}(s)|, |  | (8) |

where Ak​(s):=∑j=0k−1ei​s​θjassignsubscript𝐴𝑘𝑠superscriptsubscript𝑗0𝑘1superscript𝑒i𝑠subscript𝜃𝑗A\_{k}(s):=\sum\_{j=0}^{k-1}e^{\mathrm{i}s\theta\_{j}}. While there is no close form for B​(s):=∑k=0d/2−1|Ak+1​(s)|assign𝐵𝑠superscriptsubscript𝑘0𝑑21subscript𝐴𝑘1𝑠B(s):=\sum\_{k=0}^{d/2-1}|A\_{k+1}(s)|, numerically it is at least larger than d𝑑d, and for many positional difference s𝑠s, B​(s)𝐵𝑠B(s) is much larger than d𝑑d (check Appendix [B](#A2 "Appendix B Visualization of quantities in extrapolation bound ‣ Extending Context Window of Large Language Models via Position Interpolation") for the plot). Therefore, the interpolation bound is at least 2⋅294.73∼600×2\cdot 294.73\sim 600\times smaller than the extrapolation bound, and thus the interpolated attention score is much more stable than extrapolated one.

Notably, our method of rescaling of position indices does not introduce extra weight, or modify the model architecture in any way. This makes it attractive in practical applications, since most infrastructure and optimization for the original model can be reused after the extension.

Fine-tuning. We can further fine-tune the interpolated model using the next token prediction task with interpolated position encodings on the extended context window size using a pre-training corpus such as the Pile (Gao et al., [2020](#bib.bib11)). In the next section, we show that our fine-tuning process only needs tens to hundreds thousands of examples. We also find that the result of the fine-tuning is not sensitive to the choice of examples. The reason may be that the model is only adapting to the new context window during the fine-tuning phase, starting from a good initialization, as opposed to acquiring new knowledge.

Other ways to reduce interpolation/extrapolation bound. From the expression of the interpolation (Eqn. [5](#S2.E5 "In Theorem 2.1 (Interpolation bound). ‣ 2.3 Proposed approach: Position Interpolation (PI) ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation")) and extrapolation bound (Eqn. [8](#S2.E8 "In 2.3 Proposed approach: Position Interpolation (PI) ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation")), a common term is maxj⁡|hj|subscript𝑗subscriptℎ𝑗\max\_{j}|h\_{j}|, which is the maximal magnitude of query/key products. If we enforce a regularization on |hj|subscriptℎ𝑗|h\_{j}| during LLM training, it is possible that the catastrophic extrapolation error can be mitigated or even resolved. In fact, if we apply ridge regression with proper regularization to fit a curve in Fig. [2](#S2.F2 "Figure 2 ‣ 2.2 Direct Extrapolation ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation"), the magnitude of extrapolated a​(s)𝑎𝑠a(s) when s>L𝑠𝐿s>L can be comparable to that within [0,L]0𝐿[0,L]. To our knowledge, we are not aware of existing LLM pre-training techniques that leverage this regularization and will leave it for future work.

## 3 Experiments

We show Position Interpolation can effectively extend context window up to 32 times of the original size, and such extension can be done
with only several hundreds of training steps.
We show the resulting models are strong LLMs with fully effective long context windows.
We demonstrate its performance in a number of tasks including language modeling, passkey retrieval, and
long document summarization.
We also present benchmark results of the extended models on the original LLaMA evaluation benchmarks.

### 3.1 Setup

Model Variants. We extended the pre-trained 7B, 13B, 33B and 65B LLaMA models (Touvron et al., [2023](#bib.bib36)) to various context window of sizes up to 32768,
using either direct fine-tuning or Position Interpoloation method.
Except for rescaling the position indices for models extended with Position Interpolation, we did not
modify LLaMA model architectures (Touvron et al., [2023](#bib.bib36)) in any ways.

Training Procedure. We fine-tune all model variants using the next token prediction objective.
We use AdamW (Loshchilov & Hutter, [2019](#bib.bib24)) with β1=0.9subscript𝛽10.9\beta\_{1}=0.9 and β2=0.95subscript𝛽20.95\beta\_{2}=0.95.
We use a linear learning rate warmup of 20 steps starting from 10%percent1010\% of the maximum learning rate.
For 7B and 13B models, we set the learning rate to 2×10−52superscript1052\times 10^{-5} and for 33B and 65B models we set the learning rate to 10−5superscript10510^{-5}.
We set the weight decay to zero.
For extending 7B, 13B and 33B models to the 8192 context window size, we use 32 A100 GPUs and 64 global batch size.
For all other cases we use 128 A100 GPUs and 128 global batch size.
We note that the main need of using more GPUs is memory limitation during fine-tuning, and it is possible
to use fewer GPUs in certain cases.
We train all models using PyTorch (Paszke et al., [2019](#bib.bib28)) with Fully Sharded Data Parallel (Zhao et al., [2023](#bib.bib43)) and Flash Attention (Dao et al., [2022](#bib.bib9)).

If not specified otherwise, for the Position Interpolation method, we fine-tune the models for 1000 steps.
For the direct fine-tuning method, we use 10000 steps.
We primarily fine-tune using the Pile training dataset (Gao et al., [2020](#bib.bib11)). In Section [3.4](#S3.SS4 "3.4 Benchmarks on Original Context Window Size ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation") we also compared
fine-tuning performance on the RedPajama dataset (Computer, [2023](#bib.bib7)).

### 3.2 Long Sequence Language Modeling

We evaluate the long sequence language modeling performance of our extended models and baselines on two datasets: book corpus (PG-19) (Rae et al., [2020](#bib.bib30)) and cleaned Arxiv Math proof-pile dataset (Azerbayev et al., [2022](#bib.bib2)).

We use the test splits of PG19 (Rae et al., [2020](#bib.bib30)) and proof-pile (Azerbayev et al., [2022](#bib.bib2)).
For PG19, we use the whole test split consisting of 100 documents.
For the proof-pile dataset, we use a random subsample of 128 documents with at least 32768 SentencePiece (Kudo & Richardson, [2018](#bib.bib22)) tokens and truncate to the first 32768 tokens for each test document.
We evaluate perplexity at various context window size by using a sliding window approach following Press et al. ([2022](#bib.bib29)) with stride S=256𝑆256S=256.

In Table [1](#S3.T1 "Table 1 ‣ 3.2 Long Sequence Language Modeling ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation") and Table [2](#S3.T2 "Table 2 ‣ 3.2 Long Sequence Language Modeling ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation"), we report the perplexity results for our models and baselines on the datasets. From the results, we found that models extended with our method enjoy a significantly
improved perplexity from longer context window sizes.
By increasing the context window size from 2048 to 16384, we observed -0.28 and -0.5 reductions of perplexity for extending LLaMA 7B models on both datasets,
-0.27 and -0.48 reductions for extending LLaMA 13B models, and -0.14 and -0.42 reductions for extending LLaMA 33B models.
For LLaMA 65B models, we observed -0.12 and -0.3 reductions of perplexity by extending to the 8192 context window size.

In general, we observed a consistent trend of our models achieving better perplexity with
longer context windows.
This indicates our models can effectively make use of the longer context windows to
better predict next tokens in language modeling tasks.
Moreover, we found this trend extends to 32768 window size without diminishing on the PG19 dataset for LLaMA 7B and 13B models.
This indicates that our method may enable extension to even longer context windows.

In contrast, we observed that models extended via the direct fine-tuning method has shown
regression (up to +0.48) or minor improvement (up to -0.12) on the perplexity at longer context windows.
This indicates that models extended this way have limited capability of making
use of context windows longer than their pre-trained settings.

We saw a minor degradation of the perplexity on the original context window of 2048 for our extended models in some cases.
For example, on the Proof-pile dataset, we saw a degradation ranging from 0.01 to 0.05 across all models with extended with Position Interpolation.
A small degradation of performance within original evaluation context window is expected since Position Interpolation forces position encodings in original context window to reside in a much narrower region, which may negatively affect the language model’s performance.
We present more benchmark results on the original context window size in Section [3.4](#S3.SS4 "3.4 Benchmarks on Original Context Window Size ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation").

In Table [3](#S3.T3 "Table 3 ‣ 3.2 Long Sequence Language Modeling ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation") we report the relationship between perplexity and the number of fine-tuning steps
for LLaMA 7B model extending to 8192 and 16384 context window sizes using Position Interpolation evaluated
on the PG19 dataset.
We can see without fine-tuning (at step 0) the model can exhibit certain language modeling capability, as
indicated by <20absent20<20 perplexity for extending to 8192 context window (in contrast, the direct extrapolation method leads to >103absentsuperscript103>10^{3} perplexity).
With fine-tuning, we observed that the perplexity improves quickly.
At 200 steps the models surpassed the original model’s perplexity on 2048 context window size, indicating the models
gaining ability of effectively using sequences longer than the pre-training settings for language modeling.
At 1000 steps, we can see the models have improved steadily and achieve a significantly better perplexity.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model | | | Evaluation Context Window Size | | | | |
| Size | Context Window | Method | 2048 | 4096 | 8192 | 16384 | 32768 |
| 7B | 2048 | None | 7.20 | >103absentsuperscript103>10^{3} | >103absentsuperscript103>10^{3} | >103absentsuperscript103>10^{3} | >103absentsuperscript103>10^{3} |
| 7B | 8192 | FT | 7.21 | 7.34 | 7.69 | - | - |
| 7B | 8192 | PI | 7.13 | 6.96 | 6.95 | - | - |
| 7B | 16384 | PI | 7.11 | 6.93 | 6.82 | 6.83 | - |
| 7B | 32768 | PI | 7.23 | 7.04 | 6.91 | 6.80 | 6.77 |
| 13B | 2048 | None | 6.59 | - | - | - | - |
| 13B | 8192 | FT | 6.56 | 6.57 | 6.69 | - | - |
| 13B | 8192 | PI | 6.55 | 6.42 | 6.42 | - | - |
| 13B | 16384 | PI | 6.56 | 6.42 | 6.31 | 6.32 | - |
| 13B | 32768 | PI | 6.54 | 6.40 | 6.28 | 6.18 | 6.09 |
| 33B | 2048 | None | 5.82 | - | - | - | - |
| 33B | 8192 | FT | 5.88 | 5.99 | 6.21 | - | - |
| 33B | 8192 | PI | 5.82 | 5.69 | 5.71 | - | - |
| 33B | 16384 | PI | 5.87 | 5.74 | 5.67 | 5.68 | - |
| 65B | 2048 | None | 5.49 | - | - | - | - |
| 65B | 8192 | PI | 5.42 | 5.32 | 5.37 | - | - |

Table 1: Evaluation perplexity on PG19 dataset (Rae et al., [2020](#bib.bib30)). FT: Direct Fine-tuning. PI: Position Interpolation. Model fine-tuned with PI shows progressively lower perplexity with longer context window, showing that PI can leverage long context well, while the perplexity of FT increases over longer window. Note that overall the perplexity is higher compared to Table [2](#S3.T2 "Table 2 ‣ 3.2 Long Sequence Language Modeling ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation") since PG19 has very different writing styles.



|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model | | | Evaluation Context Window Size | | | | |
| Size | Context Window | Method | 2048 | 4096 | 8192 | 16384 | 32768 |
| 7B | 2048 | None | 2.77 | - | - | - | - |
| 7B | 8192 | FT | 2.85 | 2.74 | 2.73 | - | - |
| 7B | 8192 | PI | 2.79 | 2.57 | 2.39 | - | - |
| 7B | 16384 | PI | 2.79 | 2.57 | 2.37 | 2.25 | - |
| 7B | 32768 | PI | 2.82 | 2.59 | 2.39 | 2.24 | 2.48 |
| 13B | 2048 | None | 2.66 | - | - | - | - |
| 13B | 8192 | FT | 2.71 | 2.56 | 2.50 | - | - |
| 13B | 8192 | PI | 2.67 | 2.47 | 2.30 | - | - |
| 13B | 16384 | PI | 2.68 | 2.47 | 2.29 | 2.18 | - |
| 13B | 32768 | PI | 2.68 | 2.46 | 2.28 | 2.15 | 2.35 |
| 33B | 2048 | None | 2.49 | - | - | - | - |
| 33B | 8192 | FT | 2.56 | 2.48 | 2.47 | - | - |
| 33B | 8192 | PI | 2.50 | 2.32 | 2.18 | - | - |
| 33B | 16384 | PI | 2.53 | 2.34 | 2.18 | 2.07 | - |
| 65B | 2048 | None | 2.42 | - | - | - | - |
| 65B | 8192 | PI | 2.43 | 2.26 | 2.12 | - | - |

Table 2: Evaluation perplexity on Arxiv Math Proof-pile dataset (Azerbayev et al., [2022](#bib.bib2)). FT: Direct Fine-tuning. PI: Position Interpolation.



|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model | | Number of fine-tuning steps | | | | | |
| Size | Context Window | 0 | 200 | 400 | 600 | 800 | 1000 |
| 7B | 8192 | 16.10 | 7.12 | 7.10 | 7.02 | 6.99 | 6.95 |
| 7B | 16384 | 112.13 | 7.05 | 6.93 | 6.88 | 6.84 | 6.83 |

Table 3: Evaluation perplexity on PG19 dataset (Rae et al., [2020](#bib.bib30)) with respect to the number of fine-tuning steps using Position Interpolation.

### 3.3 Measuring Effective Context Window Size through Passkey Retrieval

We study the effective context window size, i.e. the maximum distance of a token can *effectively* attend to during inference, of our models after extension.
To measure this, we follow a synthetic evaluation task of passkey retrieval proposed by Mohtashami & Jaggi ([2023](#bib.bib26)).
In this task, the models are asked to recover a random passkey hidden in a long document.
See Figure [3](#S3.F3 "Figure 3 ‣ 3.3 Measuring Effective Context Window Size through Passkey Retrieval ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation") for the format of the document.

Given a language model, we estimate the upper and lower bounds of effective context windows as follows.
Suppose the random passkey is k𝑘k tokens away from the end of the input.
When a model persistently fails to retrieve the correct passkey value across several independent attempts,
it suggests that the effective context window size of the model is less than k𝑘k.
Conversely, if a model consistently succeeds in retrieving the correct passkey value,
we deduce that the effective context window size of the model is at least k𝑘k.

We evaluate the 7B and 33B LLaMA model variants that are extended via Position Interpolation or direct fine-tuning.
For each model, we use 32 different k𝑘k uniformly spaced in the targeted context window L′superscript𝐿′L^{\prime} and run the above tests
for 10 times for each k𝑘k, where each time a random passkey of 5 random digits is used.
In Table [4](#S3.T4 "Table 4 ‣ 3.3 Measuring Effective Context Window Size through Passkey Retrieval ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation"), we report kmaxsubscript𝑘k\_{\max} as a function of the number of fine-tuning steps, where kmaxsubscript𝑘k\_{\max} is defined as the maximum k𝑘k such that, for all k′≤ksuperscript𝑘′𝑘k^{\prime}\leq k, the model has a success rate of at least 20% on k′superscript𝑘′k^{\prime}.

We can see that models extended via Position Interpolation all successfully attain their desired extension objectives in terms of effective context window sizes, indicating by the effective context window size reaching maximum kmax=L′subscript𝑘superscript𝐿′k\_{\max}=L^{\prime}, after merely fine-tuning for 200 steps, consistently across both 7B and 33B model sizes and up to 32768 context windows.
In contrast, LLaMA models that are extended via direct fine-tuning only saw a minimal increase of the effective context window size kmaxsubscript𝑘k\_{\max} from 2048 to 2560, even after
fine-tuning for more than 10000 steps, with no clear indication of an acceleration in the increase of window size.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | | | Fine-tuning steps | | | | | |
| Size | Context Window | Method | 200 | 400 | 600 | 800 | 1000 | 10000 |
| 7B | 8192 | FT | 1792 | 2048 | 2048 | 2048 | 2304 | 2560 |
| 33B | 8192 | FT | 1792 | 2048 | 1792 | 2048 | 2304 | - |
| 7B | 8192 | PI | 8192 | 8192 | 8192 | 8192 | 8192 | - |
| 7B | 16384 | PI | 16384 | 16384 | 16384 | 16384 | 16384 | - |
| 7B | 32768 | PI | 32768 | 32768 | 18432 | 32768 | 32768 | - |
| 33B | 8192 | PI | 8192 | 8192 | 8192 | 8192 | 8192 | - |
| 33B | 16384 | PI | 16384 | 16384 | 16384 | 16384 | 16384 | - |

Table 4: Effective context window sizes after fine-tuning. FT: Direct fine-tuning. PI: Position Interpolation.



There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.
  
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. (repeat X times)
  
The pass key is 12345. Remember it. 12345 is the pass key.
  
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. (repeat Y times)
  
What is the pass key? The pass key is

Figure 3: Prompt format for passkey retrieval. We use the exact same prompt as proposed by Mohtashami & Jaggi ([2023](#bib.bib26)). Here the passkey 12345 is replaced with a random 5-digit numbers during test.

### 3.4 Benchmarks on Original Context Window Size

We evaluate the models extended by Position Interpolation on several standard benchmark tasks within the original context window size of 2048.
The evaluation results are listed in Table [5](#S3.T5 "Table 5 ‣ 3.4 Benchmarks on Original Context Window Size ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation").
From the results, we saw that models extended to 8192 produce comparable results on
the original benchmark which is designed for a much smaller context window, with
a degradation of up to 2% on the benchmark tasks, for both 7B and 33B model sizes.
Models extended to longer context windows regressed more on the benchmarks, but still in reasonable ranges for most tasks.
We also note that the choice of fine-tuning datasets does not seem to lead significant difference in the
benchmark performances, which may be due to the limited number of fine-tuning steps used in our method.
The regression on benchmark tasks is consistent with our observation on perplexity regression in Section [3.2](#S3.SS2 "3.2 Long Sequence Language Modeling ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation").

| Model Size | Context Window | Fine-tune on | BoolQ | PIQA | Race-M | Race-H | WinoGrande |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7B | 2048 | None | 76.1 | 78.9 | 55.7 | 42.2 | 69.6 |
| 7B | 8192 | Pile | 73.2 | 78.2 | 53.8 | 41.7 | 69.0 |
| 7B | 16384 | Pile | 69.8 | 77.6 | 53.3 | 40.9 | 67.8 |
| 7B | 32768 | Pile | 64.7 | 77.2 | 50.1 | 39.6 | 66.9 |
| 7B | 8192 | RedPajama | 75.5 | 77.4 | 54.5 | 41.5 | 68.1 |
| 33B | 2048 | None | 81.6 | 80.2 | 61.1 | 45.9 | 76.2 |
| 33B | 8192 | Pile | 80.2 | 80.7 | 60.2 | 45.7 | 75.9 |

Table 5: Zero-shot performance on a subset of LLaMA Benchmarks. Models extended by Position Interpolation comparable performance as the original models, except for BoolQ dataset that may require models to pay close attention to word ordering in a short reference paragraph.

### 3.5 Long Document Summarization

In this task, we evaluate our models’ performance on the long document summarization task.
In particular, we consider the GovReport (Huang et al., [2021](#bib.bib15)) dataset, which contains 17457 documents for training and 972 documents for evaluation.
Each document comes with a human generated summary.
We truncate all input documents to their first 15000 tokens.

We fine-tune the LLaMA models extended with Position Interpolation with a context window
of 16384. Note the rescaling of position indices are still required during this fine-tuning step.
We first format the raw document using the prompt template in Figure [4](#S3.F4 "Figure 4 ‣ 3.5 Long Document Summarization ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation"), and then concatenate the prompt
with the ground-truth summary (truncate to 1000 tokens) associated with each document.
We fine-tune the model using the next token prediction task with the above setup for 10 epochs.
The losses from the input prompt proportion of training examples are excluded during our fine-tuning.

We use a generation temperature of 0.5 and topp=0.95subscripttop𝑝0.95\text{top}\_{p}=0.95 as our inference parameter to generate a summarization of each document in the test set.
The final output is truncated at 1000 tokens.
We used the ROUGE-1/ROUGE-2/ROUGE-L scores (Lin, [2004](#bib.bib23)) as the evaluation metrics to evaluate the models’ outputs vs the ground-truth summaries.

In Table [6](#S3.T6 "Table 6 ‣ 3.5 Long Document Summarization ‣ 3 Experiments ‣ Extending Context Window of Large Language Models via Position Interpolation") we report our evaluation results.
We have also included results from two baselines in existing SCROLLS Leaderboard (Shaham et al., [2022](#bib.bib33); Ainslie et al., [2023](#bib.bib1)).
In general, we have obtained competitive R1 score among other models with minimal tuning of hyper-parameters.
This result suggests our models with 16384 context window can effectively handle the long document summarization task.

Read the following article and then summarize it.
  
# .... Document goes here
  
Now summarize the above article.
  
Summary:

Figure 4: Input format for long doc summarization.



| Model | | Evaluation Score | | |
| --- | --- | --- | --- | --- |
| Model | Context Window | ROUGE-1 | ROUGE-2 | ROUGE-L |
| CoLT5 Base (Ainslie et al., [2023](#bib.bib1)) | 16K | 58.7 | 29.6 | 31.4 |
| CoLT5 XL (Ainslie et al., [2023](#bib.bib1)) | 16K | 61.3 | 32.2 | 33.8 |
| LLaMA-7B Extended | 16K | 60.0 | 28.0 | 29.5 |

Table 6: ROUGE Score on GovReport Dataset.

## 4 Related Work

Retrieval-augmented LLM. One line of work extends LLMs by augmenting it with retrieval modules which fetch related documents and include the retrieval results into the input context of an LLM (Karpukhin et al., [2020](#bib.bib19); Guu et al., [2020](#bib.bib12); Izacard et al., [2022](#bib.bib16); Jiang et al., [2022](#bib.bib17); Khattab et al., [2021](#bib.bib20); Santhanam et al., [2022](#bib.bib32)). Our work is complementary to these works as our extended context window allows more documents being included in the input. In addition, with an unmodified attention mechanism and model architecture, our method may be more versatile as it can natively handle tasks beyond retrieval oriented ones, such as long document summarization, few-shots learning, etc.

Recurrent Transformers and Memory Transformers. Several works add memory capabilities to Transformers through recurrence, which increase the models’ capability of handling very long sequences (Bulatov et al., [2022](#bib.bib4); Wu et al., [2020](#bib.bib39); Dai et al., [2019](#bib.bib8); Wu et al., [2022](#bib.bib40); Martins et al., [2021](#bib.bib25); Mu et al., [2023](#bib.bib27)). One limitation of these works is that they only allow attending to a lossy compressed version of past inputs. Mu et al. ([2023](#bib.bib27)) suggested that this may prevent models from remembering specific details in the past inputs. In contrast, our work allows attending to all previous tokens, preserving all details without compression, albeit with higher inference costs. Mohtashami & Jaggi ([2023](#bib.bib26)) proposed landmark attention which allows full random access to any chunk of the input through introducing landmark tokens. Our work allows full access of the entire input through unmodified attention, which may be useful for tasks such as summarization.

Approximated Multi-head Attention. There is a large body of research that focuses on decreasing the memory and computational complexity of the multi-head attention (MHA) mechanism through approximation or sparsification (Child et al., [2019](#bib.bib5); Zaheer et al., [2020](#bib.bib41); Beltagy et al., [2020](#bib.bib3); Wang et al., [2020](#bib.bib38); Choromanski et al., [2021](#bib.bib6); Kitaev et al., [2020](#bib.bib21); Ren et al., [2021](#bib.bib31)). Although not the focus of this work, as these methods are not used in LLaMA (Touvron et al., [2023](#bib.bib36)), we note that our method is compatible with most of them since our changes are restricted to position encodings, and not attention mechanisms.

Length Extrapolation. A recent line of research aims to train Transformers models on short sequences and inference on longer (Press et al., [2022](#bib.bib29); Sun et al., [2022](#bib.bib35); Haviv et al., [2022](#bib.bib13)). However, these methods have not been applied in some of the largest language models such as LLaMA (Touvron et al., [2023](#bib.bib36)), or OPT (Zhang et al., [2022](#bib.bib42)). This has prevented them from enabling length extrapolation of many pre-existing pre-trained language models. Our work focuses on extending existing LLMs, which can save substantial pre-training costs. In addition, our method preserves the quality of the original models, even for small context window tasks, since it does not deviate far from existing definitions of position encoding or attention mechanisms.

Interpolation. The most related technique to ours is proposed by Dosovitskiy et al. ([2021](#bib.bib10)) in their work on Vision Transformers, where the authors proposed to linearly interpolate learnt position embeddings to support higher resolution, which translates to an increased number of input embeddings, in the fine-tuning stage. The interpolated position embedding weights are used as initialization in the fine-tuning process for the newly added positions. Our work differs from their work in several ways (1) Instead of interpolating position embeddings, our method interpolates position indices, which is more suitable for RoPE like position encodings and may require less training since no trainable parameters are added. (2) We report successful results of extending the context window to 32 times while Dosovitskiy et al. ([2021](#bib.bib10)) explored up to 4 times. Our results extend theirs in exploring the upper limit of context window extension via interpolation. (3) We evaluated and confirmed the effectiveness of Position Interpolation for extending context windows for language models.

We believe our results, in conjunction with (Dosovitskiy et al., [2021](#bib.bib10)), provide empirical evidence on Transformer’s remarkable ability of handling significantly longer sequences beyond training.
Further, we conjecture that a method similar to theirs is directly applicable in LLMs with learnable position embeddings such as OPT (Zhang et al., [2022](#bib.bib42)) and we plan to investigate this in the future.

## 5 Conclusions

Position Interpolation can effectively extend LLaMA models’ context window to be significantly larger, using minimal fine-tuning.
The extended models are fully capable to perform a variety of tasks on the extended context windows, and preserve its original
ability relatively well for tasks within the original extended models, making them good choices of generic language models
for both long and short input prompts.
Further, models extended by Position Interpolation can reuse most pre-existing infrastructure and optimization, making this method
attractive in many practical applications.
We believe that Position Interpolation is a general method that could be apply to other
types of position encodings, which can allow extension for more types of LLMs, and we plan to investigate in such directions in the near future.

## Acknowledgements

We thank Mike Lewis for his input on evaluation.

## References

* Ainslie et al. (2023)

  Joshua Ainslie, Tao Lei, Michiel de Jong, Santiago Ontañón, Siddhartha
  Brahma, Yury Zemlyanskiy, David Uthus, Mandy Guo, James Lee-Thorp, Yi Tay,
  Yun-Hsuan Sung, and Sumit Sanghai.
  Colt5: Faster long-range transformers with conditional computation,
  2023.
* Azerbayev et al. (2022)

  Zhangir Azerbayev, Edward Ayers, and Bartosz Piotrowski.
  Proof-pile, 2022.
  URL <https://github.com/zhangir-azerbayev/proof-pile>.
* Beltagy et al. (2020)

  Iz Beltagy, Matthew E. Peters, and Arman Cohan.
  Longformer: The long-document transformer.
  2020.
* Bulatov et al. (2022)

  Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev.
  Recurrent memory transformer.
  2022.
* Child et al. (2019)

  Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever.
  Generating long sequences with sparse transformers.
  2019.
* Choromanski et al. (2021)

  Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song,
  Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Quincy Davis, Afroz
  Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J. Colwell, and
  Adrian Weller.
  Rethinking attention with performers.
  In *9th International Conference on Learning Representations,
  ICLR 2021*. OpenReview.net, May 2021.
* Computer (2023)

  Together Computer.
  Redpajama: An open source recipe to reproduce llama training dataset,
  2023.
  URL <https://github.com/togethercomputer/RedPajama-Data>.
* Dai et al. (2019)

  Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc Le, and Ruslan
  Salakhutdinov.
  Transformerxl: Attentive language models beyond a fixed-length
  context.
  In *Proceedings of the 57th Annual Meeting of the Association
  for Computational Linguistics*, pp.  2978–2988, Florence, Italy, 2019.
  Association for Computational Linguistics.
  doi: 10.18653/v1/P19-1285.
* Dao et al. (2022)

  Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.
  FlashAttention: Fast and memory-efficient exact attention with
  IO-awareness.
  In *Advances in Neural Information Processing Systems*, 2022.
* Dosovitskiy et al. (2021)

  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
  Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
  Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.
  An image is worth 16x16 words: Transformers for image recognition at
  scale.
  In *International Conference on Learning Representations*, 2021.
  URL <https://openreview.net/forum?id=YicbFdNTTy>.
* Gao et al. (2020)

  Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles
  Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser,
  and Connor Leahy.
  The Pile: An 800gb dataset of diverse text for language modeling.
  *arXiv preprint arXiv:2101.00027*, 2020.
* Guu et al. (2020)

  Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
  Realm: Retrieval-augmented language model pre-training.
  2020.
* Haviv et al. (2022)

  Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, and Omer Levy.
  Transformer language models without positional encodings still learn
  positional information.
  2022.
* Hu et al. (2021)

  Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
  Wang, Lu Wang, and Weizhu Chen.
  Lora: Low-rank adaptation of large language models.
  *arXiv preprint arXiv:2106.09685*, 2021.
* Huang et al. (2021)

  Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang.
  Efficient attentions for long document summarization.
  In *Proceedings of the 2021 Conference of the North American
  Chapter of the Association for Computational Linguistics: Human Language
  Technologies*, pp.  1419–1436, Online, June 2021. Association for
  Computational Linguistics.
  doi: 10.18653/v1/2021.naacl-main.112.
  URL <https://aclanthology.org/2021.naacl-main.112>.
* Izacard et al. (2022)

  Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
  Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
  Grave.
  Atlas: Few-shot learning with retrieval augmented language models.
  2022.
* Jiang et al. (2022)

  Zhengbao Jiang, Luyu Gao, Jun Araki, Haibo Ding, Zhiruo Wang, Jamie Callan, and
  Graham Neubig.
  Retrieval as attention: End-to-end learning of retrieval and reading
  within a single transformer.
  2022.
* kaiokendev (2023)

  kaiokendev.
  Things iḿ learning while training superhot.
  <https://kaiokendev.github.io/til#extending-context-to-8k>, 2023.
* Karpukhin et al. (2020)

  Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
  Edunov, Danqi Chen, and Wen-tau Yih.
  Dense passage retrieval for open-domain question answering.
  In *Proceedings of the 2020 Conference on Empirical Methods in
  Natural Language Processing (EMNLP)*, pp.  6769–6781. Association for
  Computational Linguistics, 2020.
  doi: 10.18653/v1/2020.emnlp-main.550.
* Khattab et al. (2021)

  Omar Khattab, Christopher Potts, and Matei Zaharia.
  Relevance-guided supervision for openqa with colbert.
  *Transactions of the Association for Computational Linguistics*,
  9:929–944, 2021.
  doi: 10.1162/tacl˙a˙00405.
* Kitaev et al. (2020)

  Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya.
  Reformer: The efficient transformer.
  In *8th International Conference on Learning Representations,
  ICLR 2020*. OpenReview.net, April 2020.
* Kudo & Richardson (2018)

  Taku Kudo and John Richardson.
  SentencePiece: A simple and language independent subword
  tokenizer and detokenizer for neural text processing.
  In *Proceedings of the 2018 Conference on Empirical Methods in
  Natural Language Processing: System Demonstrations*, pp.  66–71, Brussels,
  Belgium, November 2018. Association for Computational Linguistics.
  doi: 10.18653/v1/D18-2012.
  URL <https://aclanthology.org/D18-2012>.
* Lin (2004)

  Chin-Yew Lin.
  ROUGE: A package for automatic evaluation of summaries.
  In *Text Summarization Branches Out*, pp.  74–81, Barcelona,
  Spain, July 2004. Association for Computational Linguistics.
  URL <https://aclanthology.org/W04-1013>.
* Loshchilov & Hutter (2019)

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  In *International Conference on Learning Representations*, 2019.
  URL <https://openreview.net/forum?id=Bkg6RiCqY7>.
* Martins et al. (2021)

  Pedro Henrique Martins, Zita Marinho, and André F. T. Martins.
  ∞\infty-former: Infinite memory transformer.
  2021.
* Mohtashami & Jaggi (2023)

  Amirkeivan Mohtashami and Martin Jaggi.
  Landmark attention: Random-access infinite context length for
  transformers.
  *arXiv preprint arXiv:2305.16300*, 2023.
* Mu et al. (2023)

  Jesse Mu, Xiang Lisa Li, and Noah Goodman.
  Learning to compress prompts with gist tokens.
  2023.
* Paszke et al. (2019)

  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
  Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban
  Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan
  Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith
  Chintala.
  *PyTorch: An Imperative Style, High-Performance Deep Learning
  Library*.
  Curran Associates Inc., Red Hook, NY, USA, 2019.
* Press et al. (2022)

  Ofir Press, Noah Smith, and Mike Lewis.
  Train short, test long: Attention with linear biases enables input
  length extrapolation.
  In *International Conference on Learning Representations*, 2022.
  URL <https://openreview.net/forum?id=R8sQPpGCv0>.
* Rae et al. (2020)

  Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and
  Timothy P. Lillicrap.
  Compressive transformers for long-range sequence modelling.
  In *International Conference on Learning Representations*, 2020.
  URL <https://openreview.net/forum?id=SylKikSYDH>.
* Ren et al. (2021)

  Hongyu Ren, Hanjun Dai, Zihang Dai, Mengjiao Yang, Jure Leskovec, Dale
  Schuurmans, and Bo Dai.
  Combiner: Full attention transformer with sparse computation cost.
  In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy
  Liang, and Jennifer Wortman Vaughan (eds.), *Advances in Neural
  Information Processing Systems 34: Annual Conference on Neural Information
  Processing Systems 2021, NeurIPS 2021*, pp.  22470–22482. Curran
  Associates, Inc., 2021.
* Santhanam et al. (2022)

  Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
  Zaharia.
  Colbertv2: Effective and efficient retrieval via lightweight late
  interaction.
  In *Proceedings of the 2022 Conference of the North American
  Chapter of the Association for Computational Linguistics: Human Language
  Technologies*, pp.  3715–3734, Seattle, United States, 2022. Association
  for Computational Linguistics.
  doi: 10.18653/v1/2022.naacl-main.272.
* Shaham et al. (2022)

  Uri Shaham, Elad Segal, Maor Ivgi, Avia Efrat, Ori Yoran, Adi Haviv, Ankit
  Gupta, Wenhan Xiong, Mor Geva, Jonathan Berant, and Omer Levy.
  SCROLLS: Standardized CompaRison over long language sequences.
  In *Proceedings of the 2022 Conference on Empirical Methods in
  Natural Language Processing*, pp.  12007–12021, Abu Dhabi, United Arab
  Emirates, December 2022. Association for Computational Linguistics.
  URL <https://aclanthology.org/2022.emnlp-main.823>.
* Su et al. (2021)

  Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu.
  Roformer: Enhanced transformer with rotary position embedding, 2021.
* Sun et al. (2022)

  Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim,
  Vishrav Chaudhary, Xia Song, and Furu Wei.
  A length-extrapolatable transformer, 2022.
* Touvron et al. (2023)

  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
  Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro,
  Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume
  Lample.
  Llama: Open and efficient foundation language models, 2023.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus,
  S. Vishwanathan, and R. Garnett (eds.), *Advances in Neural Information
  Processing Systems*, volume 30. Curran Associates, Inc., 2017.
  URL
  <https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>.
* Wang et al. (2020)

  Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma.
  Linformer: Self-attention with linear complexity.
  2020.
* Wu et al. (2020)

  Qingyang Wu, Zhenzhong Lan, Kun Qian, Jing Gu, Alborz Geramifard, and Zhou Yu.
  Memformer: A memory-augmented transformer for sequence modeling.
  2020.
* Wu et al. (2022)

  Yuhuai Wu, Markus Norman Rabe, DeLesley Hutchins, and Christian Szegedy.
  Memorizing transformers.
  In *The Tenth International Conference on Learning
  Representations, ICLR 2022*. OpenReview.net, April 2022.
* Zaheer et al. (2020)

  Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris
  Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang,
  Li Yang, and Amr Ahmed.
  Big bird: Transformers for longer sequences.
  In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina
  Balcan, and Hsuan-Tien Lin (eds.), *Advances in Neural Information
  Processing Systems 33: Annual Conference on Neural Information Processing
  Systems 2020, NeurIPS 2020*. Curran Associates, Inc., 2020.
* Zhang et al. (2022)

  Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui
  Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov,
  Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali
  Sridhar, Tianlu Wang, and Luke Zettlemoyer.
  Opt: Open pre-trained transformer language models, 2022.
* Zhao et al. (2023)

  Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less
  Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can
  Balioglu, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, and Shen Li.
  Pytorch fsdp: Experiences on scaling fully sharded data parallel,
  2023.

Appendix

## Appendix A Proof

See [2.1](#S2.Thmtheorem1 "Theorem 2.1 (Interpolation bound). ‣ 2.3 Proposed approach: Position Interpolation (PI) ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation")

###### Proof.

Using Taylor expansion, we have:

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | a​(s1)𝑎subscript𝑠1\displaystyle a(s\_{1}) | =\displaystyle= | a​(s)+a′​(s)​(s−s1)+12​a′′​(ξ1)​(s−s1)2𝑎𝑠superscript𝑎′𝑠𝑠subscript𝑠112superscript𝑎′′subscript𝜉1superscript𝑠subscript𝑠12\displaystyle a(s)+a^{\prime}(s)(s-s\_{1})+\frac{1}{2}a^{\prime\prime}(\xi\_{1})(s-s\_{1})^{2} |  | (9) |
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | a​(s2)𝑎subscript𝑠2\displaystyle a(s\_{2}) | =\displaystyle= | a​(s)+a′​(s)​(s−s2)+12​a′′​(ξ2)​(s−s2)2𝑎𝑠superscript𝑎′𝑠𝑠subscript𝑠212superscript𝑎′′subscript𝜉2superscript𝑠subscript𝑠22\displaystyle a(s)+a^{\prime}(s)(s-s\_{2})+\frac{1}{2}a^{\prime\prime}(\xi\_{2})(s-s\_{2})^{2} |  | (10) |

where ξ1∈[s1,s]subscript𝜉1subscript𝑠1𝑠\xi\_{1}\in[s\_{1},s] and ξ2∈[s,s2]subscript𝜉2𝑠subscript𝑠2\xi\_{2}\in[s,s\_{2}]. Multiplying Eqn. [9](#A1.E9 "In Proof. ‣ Appendix A Proof ‣ Extending Context Window of Large Language Models via Position Interpolation") with s−s2𝑠subscript𝑠2s-s\_{2} and Eqn. [10](#A1.E10 "In Proof. ‣ Appendix A Proof ‣ Extending Context Window of Large Language Models via Position Interpolation") with s−s1𝑠subscript𝑠1s-s\_{1} and subtract, we get:

|  |  |  |  |
| --- | --- | --- | --- |
|  | a​(s)−alinear​(s)=R​(s):=−(s−s1)​(s−s2)2​(s1−s2)​[a′′​(ξ1)​(s−s1)−a′′​(ξ2)​(s−s2)]𝑎𝑠subscript𝑎linear𝑠𝑅𝑠assign𝑠subscript𝑠1𝑠subscript𝑠22subscript𝑠1subscript𝑠2delimited-[]superscript𝑎′′subscript𝜉1𝑠subscript𝑠1superscript𝑎′′subscript𝜉2𝑠subscript𝑠2a(s)-a\_{\mathrm{linear}}(s)=R(s):=-\frac{(s-s\_{1})(s-s\_{2})}{2(s\_{1}-s\_{2})}\left[a^{\prime\prime}(\xi\_{1})(s-s\_{1})-a^{\prime\prime}(\xi\_{2})(s-s\_{2})\right] |  | (11) |

Now we bound the second order derivative a′′​(s)superscript𝑎′′𝑠a^{\prime\prime}(s). Note that for any complex number x𝑥x, |Re​(x)|≤|x|Re𝑥𝑥|\mathrm{Re}(x)|\leq|x| so we have:

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | |a′′​(s)|superscript𝑎′′𝑠\displaystyle|a^{\prime\prime}(s)| | ≤\displaystyle\leq | ∑j=0d/2−1|hj|​|ϕj′′​(s)|≤∑j=0d/2−1|hj|​θj2superscriptsubscript𝑗0𝑑21subscriptℎ𝑗subscriptsuperscriptitalic-ϕ′′𝑗𝑠superscriptsubscript𝑗0𝑑21subscriptℎ𝑗subscriptsuperscript𝜃2𝑗\displaystyle\sum\_{j=0}^{d/2-1}|h\_{j}||\phi^{\prime\prime}\_{j}(s)|\leq\sum\_{j=0}^{d/2-1}|h\_{j}|\theta^{2}\_{j} |  | (12) |
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  | ≤\displaystyle\leq | (maxj⁡|hj|)​∑j=0d/2−1c−4​j/d=(maxj⁡|hj|)​11−c−4/dsubscript𝑗subscriptℎ𝑗superscriptsubscript𝑗0𝑑21superscript𝑐4𝑗𝑑subscript𝑗subscriptℎ𝑗11superscript𝑐4𝑑\displaystyle\left(\max\_{j}|h\_{j}|\right)\sum\_{j=0}^{d/2-1}c^{-4j/d}=\left(\max\_{j}|h\_{j}|\right)\frac{1}{1-c^{-4/d}} |  | (13) |

Note that when x<0𝑥0x<0 and c>1𝑐1c>1, cx≤1+x​ln⁡csuperscript𝑐𝑥1𝑥𝑐c^{x}\leq 1+x\ln c, therefore c−4/d≤1−4/d​ln⁡csuperscript𝑐4𝑑14𝑑𝑐c^{-4/d}\leq 1-4/d\ln c and we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 11−c−4/d≤14/d​ln⁡c=d4​ln⁡c11superscript𝑐4𝑑14𝑑𝑐𝑑4𝑐\frac{1}{1-c^{-4/d}}\leq\frac{1}{4/d\ln c}=\frac{d}{4\ln c} |  | (14) |

So

|  |  |  |  |
| --- | --- | --- | --- |
|  | |a′′(s)|≤(maxj|hj|)d4​ln⁡c=:M|a^{\prime\prime}(s)|\leq\left(\max\_{j}|h\_{j}|\right)\frac{d}{4\ln c}=:M |  | (15) |

Let the above bound to be M𝑀M, we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |R​(s)|≤(s−s1)​(s2−s)2​(s2−s1)​[M​(s−s1)+M​(s2−s)]=M2​(s−s1)​(s2−s)𝑅𝑠𝑠subscript𝑠1subscript𝑠2𝑠2subscript𝑠2subscript𝑠1delimited-[]𝑀𝑠subscript𝑠1𝑀subscript𝑠2𝑠𝑀2𝑠subscript𝑠1subscript𝑠2𝑠|R(s)|\leq\frac{(s-s\_{1})(s\_{2}-s)}{2(s\_{2}-s\_{1})}\left[M(s-s\_{1})+M(s\_{2}-s)\right]=\frac{M}{2}(s-s\_{1})(s\_{2}-s) |  | (16) |

As a result:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |a​(s)−alinear​(s)|=|R​(s)|≤d​(maxj⁡|hj|)​(s−s1)​(s2−s)8​ln⁡c𝑎𝑠subscript𝑎linear𝑠𝑅𝑠𝑑subscript𝑗subscriptℎ𝑗𝑠subscript𝑠1subscript𝑠2𝑠8𝑐|a(s)-a\_{\mathrm{linear}}(s)|=|R(s)|\leq d\left(\max\_{j}|h\_{j}|\right)\frac{(s-s\_{1})(s\_{2}-s)}{8\ln c} |  | (17) |

∎

## Appendix B Visualization of quantities in extrapolation bound

As shown in Eqn. [8](#S2.E8 "In 2.3 Proposed approach: Position Interpolation (PI) ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation"), the extrapolation bound contains the term B​(s):=∑k=0d/2−1|Ak+1​(s)|assign𝐵𝑠superscriptsubscript𝑘0𝑑21subscript𝐴𝑘1𝑠B(s):=\sum\_{k=0}^{d/2-1}|A\_{k+1}(s)| where Ak​(s):=∑j=0k−1ei​s​θjassignsubscript𝐴𝑘𝑠superscriptsubscript𝑗0𝑘1superscript𝑒i𝑠subscript𝜃𝑗A\_{k}(s):=\sum\_{j=0}^{k-1}e^{\mathrm{i}s\theta\_{j}}. Here we check how large the bound is. We use θj=c−2​j/dsubscript𝜃𝑗superscript𝑐2𝑗𝑑\theta\_{j}=c^{-2j/d} with c=10000𝑐10000c=10000 and d=4096/32=128𝑑409632128d=4096/32=128 (LLaMA-7B setting), and Fig. [5](#A2.F5 "Figure 5 ‣ Appendix B Visualization of quantities in extrapolation bound ‣ Extending Context Window of Large Language Models via Position Interpolation") shows that B​(s)/d𝐵𝑠𝑑B(s)/d almost always larger than 111 and in many places it is much larger than 111.

![Refer to caption](/html/2306.15595/assets/x3.png)


Figure 5: The bound B​(s)/d𝐵𝑠𝑑B(s)/d decays with s𝑠s. While the bounds goes down with large positional difference s𝑠s, numerically B​(s)/d≥1𝐵𝑠𝑑1B(s)/d\geq 1 and at many s𝑠s much larger than 111 (the dotted horizontal line). Please check Appendix [C.2](#A3.SS2 "C.2 Code for Fig. 5 ‣ Appendix C Code ‣ Extending Context Window of Large Language Models via Position Interpolation") for the source code used to draw the figure.

## Appendix C Code

### C.1 Code for Fig. [2](#S2.F2 "Figure 2 ‣ 2.2 Direct Extrapolation ‣ 2 Method ‣ Extending Context Window of Large Language Models via Position Interpolation")

```
# build basis function
d = 4096 // 32
theta = 10000
# Frequency computation,
freqs = 1.0 / (theta ** (torch.arange(0, d, 2)[: (d // 2)].float() / d))

# construct basis function
L = 2048

x = torch.zeros(L)
x[:L] = torch.arange(0, L)

# basis functions
xfreq = torch.outer(x, freqs)

y = torch.randn(x.shape[0])

# do linear regression
X = torch.cat([xfreq.sin(), xfreq.cos()], dim=1)

eps = 0.000
coeffs = torch.linalg.solve(X.t() @ X + torch.eye(X.shape[1]) * eps, X.t() @ y)

x2 = torch.arange(0, 2*L)
xfreq2 = torch.outer(x2, freqs)
X2 = torch.cat([xfreq2.sin(), xfreq2.cos()], dim=1)

y2 = X2 @ coeffs

x3 = torch.arange(25, 75, 0.125)
xfreq3 = torch.outer(x3, freqs)
X3 = torch.cat([xfreq3.sin(), xfreq3.cos()], dim=1)

y3 = X3 @ coeffs

plt.figure(figsize=(16,5))

plt.subplot(1, 3, 1)
plt.plot(x2[:L], y2[:L], "r")
plt.scatter(x, y)
plt.ylabel("attention score $a(s)$")
plt.xlabel("Positional difference $s$")

plt.subplot(1, 3, 2)

plt.plot(x2, y2, "r")
plt.scatter(x, y)
plt.axvline(L, color="k", linestyle="--", linewidth=0.5)

plt.title("Effect of Extrapolation")
plt.xlabel("Positional difference $s$")


plt.subplot(1, 3, 3)
plt.plot(x3, y3, "r")
for i in range(25,75):
    plt.axvline(i, color="k", linestyle="--", linewidth=0.5)
plt.title("Effect of Interpolation")
plt.xlabel("Positional difference $s$")
plt.show()
```

### C.2 Code for Fig. [5](#A2.F5 "Figure 5 ‣ Appendix B Visualization of quantities in extrapolation bound ‣ Extending Context Window of Large Language Models via Position Interpolation")

```
L = 2048
x = torch.arange(0, 2*L)
d = 4096 // 32
theta = 10000
freqs = 1.0 / (theta ** (torch.arange(0, d, 2)[: (d // 2)].float() / d))

xfreq = torch.outer(x, freqs)

mags = (xfreq.sin().cumsum(dim=1).pow(2) + xfreq.cos().cumsum(dim=1).pow(2)).sqrt()

plt.plot(mags.sum(dim=1)/d)
plt.axhline(1.0, color=’k’, linestyle="--")
plt.xlabel("Positional difference $s$")
plt.ylabel("$B(s)/d$")
plt.show()
```

[◄](/html/2306.15594)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2306.15595)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2306.15595)
[View original  
on arXiv](https://arxiv.org/abs/2306.15595)[►](/html/2306.15596)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Wed Feb 28 22:00:20 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
