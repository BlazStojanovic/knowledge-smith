---
arxiv: '2604.27077'
authors:
- Boris Shigida
- Boris Hanin
- Andrey Gromov
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Learning Rate Transfer in Normalized Transformers
url: https://arxiv.org/abs/2604.27077
year: 2026
---

# Learning Rate Transfer in Normalized Transformers

[
Affiliation: [
Affiliation: [
Email: [bs1624@princeton.edu](mailto:bs1624@princeton.edu)

(May 5, 2026)

###### Abstract

The Normalized Transformer, or nGPT (Loshchilov et al., [2025](#bib.bib20)) achieves impressive training speedups and does not require weight decay or learning rate warmup. However, despite having hyperparameters that explicitly scale with model size, we observe that nGPT does not exhibit learning rate transfer across model dimension and token horizon. To rectify this, we combine numerical experiments with a principled use of alignment exponents (Everett et al., [2024](#bib.bib11)) to revisit and modify the μ\muP approach to hyperparameter transfer (Yang and Hu, [2021](#bib.bib39)). The result is a novel nGPT parameterization we call ν\nuGPT. Through extensive empirical validation, we find ν\nuGPT exhibits learning rate transfer across width, depth, and token horizon.

## 1 Introduction

Neural network performance is sensitive to a range of optimization hyperparameters (HPs) including initialization scheme, learning rate, batch size, weight decay, and compute budget.
Since direct search for good HPs at large scale is impractical
it is useful in practice to extrapolate from HP sweeps in small models
trained on limited token horizons to performant HPs at much larger scale, using HP transfer techniques.
Our goal in this article is to extend and refine such techniques to the important setting of Normalized Transformers or nGPT
(Loshchilov et al., [2025](#bib.bib20)).
These models constrain weight and activation norms by extensive use of normalization (see [section˜2](#S2 "2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers")).
Along with trainable scale parameters that ensure no loss in expressivity,
this removes the need for weight decay and learning rate warmup while achieving impressive performance and training speedups.
Using a mix of principled heuristics and empirical scaling law fits, we propose a novel parameterization we call ν\nuGPT ([section˜3](#S3 "3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") and [table˜1](#S3.T1 "In 3.1 Summary of the changes ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")),
for transferring HPs across model depth, width, and token horizon.

Specifically:

* •

  Inspired by the theoretical framework in Everett et al. ([2024](#bib.bib11)),
  we check empirically that weight-activation alignments in nGPT do not satisfy the hypotheses that underlie typical μ\muP-type parameterizations.
  Instead, starting with an empirically supported mid alignment assumption (in which weights and activations are partially but not completely aligned),
  we derive and validate ([figure˜2](#S1.F2 "In 1 Introduction ‣ Learning Rate Transfer in Normalized Transformers")) a novel prescription for transferring learning rates when scaling model width.
  We find that our parametrization transfers over width somewhat better than μ\muP ([figure˜7](#S4.F7 "In 4.4 𝜈GPT gives better transfer over width than 𝜇P ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers")).
* •

  Adopting a heuristic from Bordelon et al. ([2024b](#bib.bib7)); Dey et al. ([2025](#bib.bib10)) for HP transfer across depth,
  we show that ν\nuGPT allows for transfer over depth and high stability of deep models ([figure˜3](#S4.F3 "In 4.2 Growing 0⁢𝑝⁢𝑡 at a fixed iteration count ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers")).
* •

  Finally, we find that the optimal learning rate in nGPT scales across token horizon like #​tokens−1/3\#\text{tokens}^{-1/3}, consistent with measurements in Bjorck et al. ([2025](#bib.bib2)) for un-normalized Transformers.
* •

  As a result, our experiments show that ν\nuGPT obtains HP transfer across depth, width, and token horizon, with no loss in performance compared to the original (well-tuned) nGPT baseline ([figure˜1](#S1.F1 "In 1 Introduction ‣ Learning Rate Transfer in Normalized Transformers")).

|  |  |
| --- | --- |
|  |  |

!(/html/2604.27077/assets/x1.png)

(a) nGPT (baseline)

!(/html/2604.27077/assets/x2.png)

(b) ν\nuGPT (ours)

Figure 1: Different model sizes at fixed aspect ratio 0​p​t=nheads0pt=n\_{\text{heads}} (in the legend, with the best loss in parentheses) and “compute-optimal” 20 tokens per parameter.
The baseline nGPT does not show learning rate transfer.
Our parametrization ν\nuGPT does, and performs no worse or slightly better.
The points are averaged over three initialization seeds and no validation loss EMA is used.

|  |  |
| --- | --- |
|  |  |

!(/html/2604.27077/assets/x3.png)

(a) nGPT (baseline)

!(/html/2604.27077/assets/x4.png)

(b) ν\nuGPT (ours)

Figure 2: 
Width sweeps with nheadsn\_{\text{heads}} increasing (in the legend, with the best loss in parentheses), head dimension and 0​p​t=120pt=12 fixed, 80 000 iterations (about 21 B tokens).
Baseline nGPT does not show transfer, our parametrization ν\nuGPT shows essentially perfect transfer with no loss of performance.

### 1.1 Related work

Early approaches to scaling the width (hidden dimension) of neural networks identified “lazy” infinite-width limits in which
the network becomes close to its linearization around initialization (Jacot et al., [2018](#bib.bib14); Arora et al., [2019](#bib.bib1); Chizat et al., [2019](#bib.bib9)),
and in particular there is no feature learning.
It was then discovered that non-lazy limits exist
(Mei et al., [2018](#bib.bib22); Rotskoff and Vanden-Eijnden, [2018](#bib.bib29); Sirignano and Spiliopoulos, [2020](#bib.bib31); Nguyen and Pham, [2023](#bib.bib24)).
The Tensor Programs framework
(Yang, [2019](#bib.bib36), [2020](#bib.bib37); Yang and Littwin, [2021](#bib.bib40); Yang, [2021](#bib.bib38)) allowed Yang and Hu ([2021](#bib.bib39))
to classify large-width limits using a natural class they call “abc-Parametrizations”
and describe a parametrization (stable, non-lazy) with weights updated as much as possible without blowup,
which they called the Maximal Update Parametrization (μ\muP).
This limit was studied with dynamical mean field theory in Bordelon and Pehlevan ([2022](#bib.bib5)).
A simple theoretical perspective deriving μ\muP using spectral norms is provided in
Yang et al. ([2024a](#bib.bib41)).
It was empirically observed that hyperparameters such as the learning rate and initialization variance transfer
across width when using μ\muP (Yang et al., [2021](#bib.bib35); Lingle, [2025](#bib.bib18); Vlassis et al., [2025](#bib.bib33));
rigorous theoretical understanding of this phenomenon
is a subject of recent work (Hayou, [2025](#bib.bib12)).
Everett et al. ([2024](#bib.bib11)) conducted a large-scale empirical evaluation of learning rate transfer over width,
and questioned matrix-vector alignment assumptions in μ\muP leading to another non-equivalent parametrization that is not only non-trivial
but is not worse than
μ\muP in practice.
Relatedly,
Kosson et al. ([2026](#bib.bib15))
argue that such alignment assumptions of μ\muP are incorrect during training,
and correct alignment can be predicted theoretically with high precision if weight decay is used.
Blake et al. ([2024](#bib.bib4)) combine μ\muP with Unit Scaling (Blake et al., [2023](#bib.bib3)) ensuring that activations, weights and gradients have scale one at initialization.

Large width and depth limits for residual networks were identified using the Tensor Programs framework in Yang et al. ([2024b](#bib.bib42)),
using dynamical mean field theory in Bordelon et al. ([2024c](#bib.bib8)), with the latter later extended to Transformers in Bordelon et al. ([2024a](#bib.bib6)).
In that literature, residual networks have the form 𝒉ℓ+1=𝒉ℓ+nlayers−αdepth​ℱℓ​(𝒉ℓ)\bm{h}^{\ell+1}=\bm{h}^{\ell}+n\_{\text{layers}}^{-\alpha\_{\text{depth}}}\mathcal{F}\_{\ell}(\bm{h}^{\ell}), where ℱℓ\mathcal{F}\_{\ell} is the ℓ\ellth block,
and where αdepth\alpha\_{\text{depth}} is the exponent that controls the magnitude of the contribution from the new block nlayers−αdepth​ℱℓ​(𝒉ℓ)n\_{\text{layers}}^{-\alpha\_{\text{depth}}}\mathcal{F}\_{\ell}(\bm{h}^{\ell}) relative to the previous
hidden state 𝒉ℓ\bm{h}^{\ell}; it can be shown that absent techniques like Post-LN, αdepth\alpha\_{\text{depth}} has to be between
0.50.5 and 11 to avoid either blowup or triviality of the forward pass at large depths.
When using parametrizations that admit large width and depth limits, hyperparameters like the learning rate and initialization variance
may or may not transfer over depth (at small widths as well); there is disagreement about this issue in the literature.
Yang et al. ([2024b](#bib.bib42)) argue that αdepth=1/2\alpha\_{\text{depth}}=1/2 is optimal because it admits “feature diversity”,
but is defective if residual blocks themselves have depth more than 11 (where there is weight matrix multiplication inside of ℱℓ\mathcal{F}\_{\ell}), the reason being that whenever αdepth∈[1/2,1)\alpha\_{\text{depth}}\in[1/2,1), the network becomes linearized in the weights of each layer.
In particular,
they do not identify transfer over depth for such realistic block structures.
Dey et al. ([2025](#bib.bib10)) observe transfer over depth for αdepth=1\alpha\_{\text{depth}}=1 but not 1/21/2, and argue that it is because
αdepth=1\alpha\_{\text{depth}}=1 is the only value where the network is not linearized in the weights of each layer.
Mlodozeniec et al. ([2025](#bib.bib23)) observe transfer for both αdepth∈{1/2,1}\alpha\_{\text{depth}}\in\{1/2,1\} and study which corrections should be made when changing token horizons and batch size. In short, transfer over depth for αdepth≠1\alpha\_{\text{depth}}\neq 1 is not a settled issue, whereas transfer for αdepth=1\alpha\_{\text{depth}}=1 appears uncontested (although it may lack feature diversity).
Concurrent work Ren et al. ([2026](#bib.bib28)) applies μ\muP-like derivations to a different variant of normalized models optimized with Muon (for most parameters), in particular noticing that Depth-μ\muP is required in their setting.

There are alternative approaches to scaling the model size while finding near-optimal learning rates, such as viewing the network as consisting of modules and
normalizing the updates per-module (Large et al., [2024](#bib.bib16)).

### 1.2 Organization

The rest of the paper is organized as follows. In order to keep our presentation self-contained, we devote a short [section˜2](#S2 "2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers")
to defining nGPT and all its hyperparameters.
The next [section˜3](#S3 "3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") describes our reparametrization and the motivations behind it,
with more detailed proofs and derivations moved to [section˜6](#S6 "6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers").
Ablations and illustrations of transfer across width, depth, and iteration count are provided in [section˜4](#S4 "4 Experiments ‣ Learning Rate Transfer in Normalized Transformers").

## 2 Normalized Transformers: definitions

The Normalized Transformer, or nGPT (Loshchilov et al., [2025](#bib.bib20))
maps an input sequence of one-hot vectors
𝒙n∈ℝV\bm{x}\_{n}\in\mathbb{R}^{V} (with n∈[1:SeqLen]n\in[1:\mathrm{SeqLen}])
of dimension VV (vocabulary size)
to a corresponding output sequence of vectors
𝒛n∈ℝV\bm{z}\_{n}\in\mathbb{R}^{V} (with n∈[1:SeqLen]n\in[1:\mathrm{SeqLen}])
through a series of normalized residual layers.
More precisely, like most Transformers, each input 𝒙n\bm{x}\_{n} is passed through a trainable
linear embedding layer

|  |  |  |
| --- | --- | --- |
|  | 𝒉n1=𝐄input​𝒙n∈ℝdmodel,𝐄input∈ℝdmodel×V.\bm{h}\_{n}^{1}=\mathbf{E}\_{\text{input}}\bm{x}\_{n}\in\mathbb{R}^{d\_{\text{model}}},\qquad\mathbf{E}\_{\text{input}}\in\mathbb{R}^{d\_{\text{model}}\times V}. |  |

The hidden state is then transformed through nlayersn\_{\text{layers}} residual blocks that alternate cross-token causal self-attention

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒉A,nℓ+1\displaystyle\bm{h}\_{A,n}^{\ell+1} | =Norm​(Attentionn​({𝒉mℓ}m=1SeqLen)),\displaystyle=\text{Norm}\big\lparen\text{Attention}\_{n}(\{\bm{h}\_{m}^{\ell}\}\_{m=1}^{\text{SeqLen}})\big\rparen, |  | (1) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒉nℓ+0.5\displaystyle\bm{h}\_{n}^{\ell+0.5} | =Norm​(𝒉nℓ+αA,initαA,scale​𝜶Aℓ+1⊙(𝒉A,nℓ+1−𝒉nℓ)),\displaystyle=\text{Norm}\bigg\lparen\bm{h}\_{n}^{\ell}+\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}\bm{\alpha}\_{A}^{\ell+1}\odot(\bm{h}\_{A,n}^{\ell+1}-\bm{h}\_{n}^{\ell})\bigg\rparen, |  | (2) |

and token-wise MLPs

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒉M,nℓ+1\displaystyle\bm{h}\_{M,n}^{\ell+1} | =Norm​(MLP​(𝒉nℓ+0.5)),\displaystyle=\text{Norm}\big\lparen\text{MLP}(\bm{h}\_{n}^{\ell+0.5})\big\rparen, |  | (3) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒉nℓ+1\displaystyle\bm{h}\_{n}^{\ell+1} | =Norm​(𝒉nℓ+0.5+αM,initαM,scale​𝜶Mℓ+1⊙(𝒉M,nℓ+1−𝒉nℓ+0.5)),\displaystyle=\text{Norm}\bigg\lparen\bm{h}\_{n}^{\ell+0.5}+\frac{\alpha\_{\text{$M$,init}}}{\alpha\_{\text{$M$,scale}}}\bm{\alpha}\_{M}^{\ell+1}\odot(\bm{h}\_{M,n}^{\ell+1}-\bm{h}\_{n}^{\ell+0.5})\bigg\rparen, |  | (4) |

where we denote

|  |  |  |
| --- | --- | --- |
|  | Norm​(𝒚):=𝒚‖𝒚‖\text{Norm}(\bm{y}):=\frac{\bm{y}}{\|\bm{y}\|} |  |

and ⊙\odot is the component-wise product. The final residual state 𝒉nnlayers+1\bm{h}\_{n}^{n\_{\text{layers}+1}} passes through a linear unembedding

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒛n=sz,initsz,scale​𝒔z⊙𝒛^n∈ℝV,𝒛^n=𝐄output​𝒉nnlayers+1,𝐄output∈ℝV×dmodel.\bm{z}\_{n}=\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}\bm{s}\_{z}\odot\hat{\bm{z}}\_{n}\in\mathbb{R}^{V},\qquad\hat{\bm{z}}\_{n}=\mathbf{E}\_{\text{output}}\bm{h}\_{n}^{n\_{\text{layers}+1}},\qquad\mathbf{E}\_{\text{output}}\in\mathbb{R}^{V\times d\_{\text{model}}}. |  | (5) |

We will suppress the dependence of 𝒉A,n\bm{h}\_{A,n}, 𝒉M,n\bm{h}\_{M,n}, 𝒉n\bm{h}\_{n}, 𝜶A\bm{\alpha}\_{A}, 𝜶M\bm{\alpha}\_{M} on the block number to reduce notational clutter.
The columns of 𝐄input\mathbf{E}\_{\text{input}} and rows 𝐄output\mathbf{E}\_{\text{output}} are normalized before111In Loshchilov et al. ([2025](#bib.bib20)), they are normalized after each step. This difference has no practical relevance. starting each training step.

There are several key distinguishing features of nGPT when compared with a standard pre-LN Transformer:

* •

  Standard RMSNorm​(⋅)\text{RMSNorm}(\cdot) or LayerNorm​(⋅)\mathrm{LayerNorm}(\cdot) are replaced with Norm​(⋅)\text{Norm}(\cdot), which ℓ2\ell\_{2}-normalizes the vector and has no trainable parameters.
* •

  The normalization is performed at the output of Attention and MLP blocks rather than input, which is sometimes called residual-post-norm (Liu et al., [2021](#bib.bib19); OLMo et al., [2025](#bib.bib25)).
* •

  The standard residual connection is replaced in ([2](#S2.E2 "Equation 2 ‣ 2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers")) and ([4](#S2.E4 "Equation 4 ‣ 2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers")) by a “linear interpolation” function (LERP), in which trainable vectors 𝜶A\bm{\alpha}\_{A}, 𝜶M\bm{\alpha}\_{M} allow for a learned componentwise rescaling. The components of 𝜶A\bm{\alpha}\_{A}, 𝜶M\bm{\alpha}\_{M} are constrained to be nonnegative. At initialization, each component of 𝜶A\bm{\alpha}\_{A} is equal to a fixed global hyperparameter αA,scale\alpha\_{\text{$A$,scale}}, so that the value of αA,initαA,scale​𝜶A\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}\bm{\alpha}\_{A} at initialization is equal (in each component) to a fixed global hyperparameter αA,init\alpha\_{\text{$A$,init}}. The ratio αA,initαA,scale\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}} modifies the learning rate of 𝜶A\bm{\alpha}\_{A}. Rescalers 𝜶M\bm{\alpha}\_{M} and 𝒔z\bm{s}\_{z} are treated similarly.

In addition to the explicit normalization layers in the residual stream we normalize also the keys and queries inside the self-attention block. Specifically, the nnth token output of an attention head is given by the standard expression

|  |  |  |  |
| --- | --- | --- | --- |
|  | Headn​({𝒉m}m=1SeqLen)\displaystyle\text{Head}\_{n}(\{\bm{h}\_{m}\}\_{m=1}^{\text{SeqLen}}) | =∑m=1nexp⁡(dkey​𝒒n′⁣𝖳​𝒌m′)∑m~=1nexp⁡(dkey​𝒒n′⁣𝖳​𝒌m~′)​𝒗m,\displaystyle=\sum\_{m=1}^{n}\frac{\exp(\sqrt{d\_{\text{key}}}\bm{q}^{\prime\mkern-1.5mu\mathsf{T}}\_{n}\bm{k}\_{m}^{\prime})}{\sum\_{\tilde{m}=1}^{n}\exp(\sqrt{d\_{\text{key}}}\bm{q}^{\prime\mkern-1.5mu\mathsf{T}}\_{n}\bm{k}^{\prime}\_{\tilde{m}})}\bm{v}\_{m}, |  |

where 𝒒n′\bm{q}^{\prime}\_{n} and 𝒌m′\bm{k}^{\prime}\_{m} are normalized query and key vectors calculated using trainable matrices 𝐖q,𝐖k∈ℝdmodel×dkey\mathbf{W}\_{q},\mathbf{W}\_{k}\in\mathbb{R}^{d\_{\text{model}}\times d\_{\text{key}}}

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℝdkey∋𝒒n=QRotn​(𝐖q𝖳​𝒉n),𝒒n′=𝒒n‖𝒒n‖⊙sq​k,initsq​k,scale​𝒔q​k,\mathbb{R}^{d\_{\text{key}}}\ni\bm{q}\_{n}=\text{QRot}\_{n}(\mathbf{W}\_{q}^{\mkern-1.5mu\mathsf{T}}\bm{h}\_{n}),\quad\bm{q}^{\prime}\_{n}=\frac{\bm{q}\_{n}}{\|\bm{q}\_{n}\|}\odot\frac{s\_{\text{$qk$,init}}}{s\_{\text{$qk$,scale}}}\bm{s}\_{qk}, |  | (6) |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℝdkey∋𝒌m=KRotm​(𝐖k𝖳​𝒉m),𝒌m′=𝒌m‖𝒌m‖⊙sq​k,initsq​k,scale​𝒔q​k.\mathbb{R}^{d\_{\text{key}}}\ni\bm{k}\_{m}=\text{KRot}\_{m}(\mathbf{W}\_{k}^{\mkern-1.5mu\mathsf{T}}\bm{h}\_{m}),\quad\bm{k}^{\prime}\_{m}=\frac{\bm{k}\_{m}}{\|\bm{k}\_{m}\|}\odot\frac{s\_{\text{$qk$,init}}}{s\_{\text{$qk$,scale}}}\bm{s}\_{qk}. |  | (7) |

Here QRotn​(⋅)\text{QRot}\_{n}(\cdot) and KRotm​(⋅)\text{KRot}\_{m}(\cdot) denote rotary positional embedding (Su et al., [2023](#bib.bib32)) maps,
whereas the value vectors are calculated using the trainable matrix 𝐖v∈ℝdmodel×dkey\mathbf{W}\_{v}\in\mathbb{R}^{d\_{\text{model}}\times d\_{\text{key}}}

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℝdkey∋𝒗m=𝐖v𝖳​𝒉m.\mathbb{R}^{d\_{\text{key}}}\ni\bm{v}\_{m}=\mathbf{W}\_{v}^{\mkern-1.5mu\mathsf{T}}\bm{h}\_{m}. |  | (8) |

The meaning of 𝒔q​k\bm{s}\_{qk}, sq​k,inits\_{\text{$qk$,init}}, and sq​k,scales\_{\text{$qk$,scale}} is analogous to 𝜶A\bm{\alpha}\_{A}, αA,scale\alpha\_{\text{$A$,scale}}, αA,init\alpha\_{\text{$A$,init}} discussed above.
In practice, there are nheadsn\_{\text{heads}} different attention heads with separate (𝐖q,𝐖k,𝐖v,𝒔q​k)(\mathbf{W}\_{q},\mathbf{W}\_{k},\mathbf{W}\_{v},\bm{s}\_{qk}). The outputs of all heads (each in ℝdkey\mathbb{R}^{d\_{\text{key}}}) are concatenated (yielding a vector in ℝnheads​dkey\mathbb{R}^{n\_{\text{heads}}d\_{\text{key}}}),
and then a trainable linear transformation 𝐖O∈ℝdmodel×nheads​dkey\mathbf{W}\_{O}\in\mathbb{R}^{d\_{\text{model}}\times n\_{\text{heads}}d\_{\text{key}}} transfers the resulting vector back to ℝdmodel\mathbb{R}^{d\_{\text{model}}}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Attentionn​({𝒉m}m=1SeqLen)=𝐖O​[Headn(1)​({𝒉m}m=1SeqLen)⋮Headn(nheads)​({𝒉m}m=1SeqLen)]⏟∈ℝnheads​dkey.\text{Attention}\_{n}(\{\bm{h}\_{m}\}\_{m=1}^{\text{SeqLen}})=\mathbf{W}\_{O}\underbrace{\begin{bmatrix}\text{Head}\_{n}^{(1)}(\{\bm{h}\_{m}\}\_{m=1}^{\text{SeqLen}})\\ \vdots\\ \text{Head}\_{n}^{(n\_{\text{heads}})}(\{\bm{h}\_{m}\}\_{m=1}^{\text{SeqLen}})\end{bmatrix}}\_{\in\mathbb{R}^{n\_{\text{heads}}d\_{\text{key}}}}. |  | (9) |

Columns of 𝐖q(j),𝐖k(j),𝐖v(j)\mathbf{W}\_{q}^{(j)},\mathbf{W}\_{k}^{(j)},\mathbf{W}\_{v}^{(j)} for each head j∈[1:nheads]j\in[1:n\_{\text{heads}}] and of 𝐖O\mathbf{W}\_{O} are all normalized before starting each training step.

Finally, given a vector 𝒉n∈ℝdmodel\bm{h}\_{n}\in\mathbb{R}^{d\_{\text{model}}} the MLP block computes

|  |  |  |
| --- | --- | --- |
|  | MLP​(𝒉n)=𝐖o​MLP​(SiLU​(𝝂)⊙𝒖),\displaystyle\text{MLP}(\bm{h}\_{n})=\mathbf{W}\_{o\text{MLP}}\big\lparen\text{SiLU}(\bm{\nu})\odot\bm{u}\big\rparen, |  |

where SiLU​(𝝂)=𝝂⊙σ​(𝝂)\text{SiLU}(\bm{\nu})=\bm{\nu}\odot\sigma(\bm{\nu}) with σ​(⋅)\sigma(\cdot) the standard sigmoid (applied componentwise) and

|  |  |  |
| --- | --- | --- |
|  | 𝒖=𝐖u​𝒉n⊙su,initsu,scale​𝒔u∈ℝdMLP,𝝂=𝐖ν​𝒉n⊙sν,initsν,scale​dmodel1/2​𝒔ν∈ℝdMLP.\displaystyle\bm{u}=\mathbf{W}\_{u}\bm{h}\_{n}\odot\frac{s\_{\text{$u$,init}}}{s\_{\text{$u$,scale}}}\bm{s}\_{u}\in\mathbb{R}^{d\_{\text{MLP}}},\quad\bm{\nu}=\mathbf{W}\_{\nu}\bm{h}\_{n}\odot\frac{s\_{\text{$\nu$,init}}}{s\_{\text{$\nu$,scale}}}d\_{\text{model}}^{1/2}\bm{s}\_{\nu}\in\mathbb{R}^{d\_{\text{MLP}}}. |  |

The additional dmodel1/2d\_{\text{model}}^{1/2} factor (explained in detail in Appendix A.1 of Loshchilov et al. ([2025](#bib.bib20))) is introduced to benefit better from the shape of SiLU (since the hidden state has a smaller scale than regular Transformers with RMSNorm​(⋅)\text{RMSNorm}(\cdot)).
The rows of 𝐖ν∈ℝdMLP×dmodel\mathbf{W}\_{\nu}\in\mathbb{R}^{d\_{\text{MLP}}\times d\_{\text{model}}}, 𝐖u∈ℝdMLP×dmodel\mathbf{W}\_{u}\in\mathbb{R}^{d\_{\text{MLP}}\times d\_{\text{model}}} and columns of 𝐖o​MLP∈ℝdmodel×dMLP\mathbf{W}\_{o\text{MLP}}\in\mathbb{R}^{d\_{\text{model}}\times d\_{\text{MLP}}} are normalized before starting each training step.
The meaning of (su,init,su,scale,𝒔u)(s\_{\text{$u$,init}},s\_{\text{$u$,scale}},\bm{s}\_{u}),
(sν,init,sν,scale,𝒔ν)(s\_{\text{$\nu$,init}},s\_{\text{$\nu$,scale}},\bm{s}\_{\nu})
is analogous to (αA,scale,αA,init,𝜶A)(\alpha\_{\text{$A$,scale}},\alpha\_{\text{$A$,init}},\bm{\alpha}\_{A}) discussed above.

## 3 Reparametrization for transfer over width, depth and token horizon

In this section we summarize our proposed ν\nuGPT parameterization for learning rate transfer across width, depth, and token horizon.
We begin in [section˜3.1](#S3.SS1 "3.1 Summary of the changes ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") with a high-level summary of our parameterization.
We then provide in [section˜3.2](#S3.SS2 "3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") a detailed discussion of the heuristics that lead to the depth and width transfer prescriptions in ν\nuGPT. Finally, we discuss in [section˜3.3](#S3.SS3 "3.3 Token horizon corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") transfer across token horizon.

### 3.1 Summary of the changes

{NiceTabular}

Table 1: 
Our re-parametrization of nGPT (defined in [section˜2](#S2 "2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers")).
[Section˜3.1](#S3.SS1 "3.1 Summary of the changes ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") contains the text version of this table and defines
data mdatam\_{\text{data}}, width mwidthm\_{\text{width}} and depth mdepthm\_{\text{depth}} ratios.
Other notations: ηbase\eta\_{\text{base}} is the learning rate used by the optimizer,
ηinput\eta\_{\text{input}} is the learning rate of 𝐄input\mathbf{E}\_{\text{input}} weights (σinput2\sigma^{2}\_{\text{input}} their initialization variance),
ηoutput\eta\_{\text{output}} of 𝐄output\mathbf{E}\_{\text{output}} weights (σoutput2\sigma^{2}\_{\text{output}} their initialization variance),
ηhidden\eta\_{\text{hidden}} of all linear layers in Transformer blocks (σhidden2\sigma^{2}\_{\text{hidden}} their initialization variance).

Let us fix some base number of iterations, depth (number of layers) and width (model dimension). These will be constant throughout, and we define

|  |  |  |
| --- | --- | --- |
|  | mdata:=iter. countbase iter. count,mwidth:=widthbase width,mdepth:=depthbase depthm\_{\text{data}}:=\frac{\text{iter. count}}{\text{base iter. count}},\quad m\_{\text{width}}:=\frac{\text{width}}{\text{base width}},\quad m\_{\text{depth}}:=\frac{\text{depth}}{\text{base depth}} |  |

for the data, width, and depth multipliers of the target model that we wish to train. Our ν\nuGPT parameterization ([table˜1](#S3.T1 "In 3.1 Summary of the changes ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) asks the following:

\lxSVG@picture

1.

Multiply the global learning rate by mdata−1/3m\_{\text{data}}^{-1/3}.
2.

Multiply further the learning rate of embedding weights
ηinput\eta\_{\text{input}}
by mwidth−1/2m\_{\text{width}}^{-1/2}.
3.

Multiply the learning rate ηhidden\eta\_{\text{hidden}} for weights matrices in MLP and Attention blocks and the learning rate ηoutput\eta\_{\text{output}} for unembedding weights by mwidth−3/4m\_{\text{width}}^{-3/4}.
4.

Optionally, multiply ηinput\eta\_{\text{input}}, ηoutput\eta\_{\text{output}} by additional constant factors tuned on the base model.
5.

Do not scale αA,scale\alpha\_{\text{$A$,scale}}, αM,scale\alpha\_{\text{$M$,scale}}, sq​k,scales\_{\text{$qk$,scale}}, sz,scales\_{\text{$z$,scale}}: take them to be constant 0.030.03 rather
than dmodel−1/2d\_{\text{model}}^{-1/2} as in original nGPT.
6.

Scale αA,init\alpha\_{\text{$A$,init}} and αM,init\alpha\_{\text{$M$,init}} with depth: put them at 0.05​mdepth−10.05\,m\_{\text{depth}}^{-1} rather than 0.050.05 as in original nGPT.
\endlxSVG@picture

The initialization of embeddings, unembeddings and hidden weights is Gaussian with variances
σinput2\sigma^{2}\_{\text{input}}, σoutput2\sigma^{2}\_{\text{output}} and σhidden2\sigma^{2}\_{\text{hidden}}
respectively. It is not important what the actual values of these variances are because the corresponding matrices are normalized
before starting each optimization step (including the first one), which is why we write “arbitrary” for their values in the table.

### 3.2 Width and depth corrections

In this section we provide theoretically grounded heuristics for transferring learning rates across width and depth. As in prior work Yang and Hu ([2021](#bib.bib39)) and Dey et al. ([2025](#bib.bib10)) we proceed by formulating two intuitive desiderata constraining initialization and learning rates as functions of depth and width,
meant to give a simplified picture of what a correct parametrization must look like.
We then give heuristic computations that explain how these desiderata can be fulfilled, leading to the ν\nuGPT prescription described above.
Detailed estimation of scales of initial values and updates inside each block are deferred to [section˜6.1](#S6.SS1 "6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers").

#### Notation.

For any vector or matrix in the network 𝐏​(t)\mathbf{P}(t), we denote by 𝐏\mathbf{P} its value 𝐏​(0)\mathbf{P}(0) at initialization and by Δ​𝐏​(t)\Delta\mathbf{P}(t) or just Δ​𝐏\Delta\mathbf{P} its change from initialization 𝐏​(t)−𝐏​(0)\mathbf{P}(t)-\mathbf{P}(0). In this section, the number of optimization steps tt is assumed to not exceed a universal constant (we relax this in [section˜3.3](#S3.SS3 "3.3 Token horizon corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")). For quantities aa and bb depending on width and depth, we write a=O​(b)a=O(b), or a≲ba\lesssim b, or b=Ω​(a)b=\Omega(a) to mean that the ratio a/ba/b “does not explode”
(does not grow unbounded at growing width and depth). The notation a=Θ​(b)a=\Theta(b) and a≍ba\asymp b both mean that simultaneously a≲ba\lesssim b and a≳ba\gtrsim b.

#### HP Transfer Desiderata.

Our first desideratum for HP transfer is borrowed from Yang and Hu ([2021](#bib.bib39)) and asks that all residual layer preactivations are order-11.

###### Desideratum 1 (Stability at initialization).

We call a parametrization stable at initialization if
no preactivations blow up or become trivial,
and the logits do not blow up222It is a technicality of the μ\muP initialization that logits converge to zero at initialization, though they evolve non-trivially, thus becoming Θ​(1)\Theta(1) during training. We do not recommend μ\muP as our primary choice, but we do not have reason to exclude it.:
‖𝒉nℓ‖=Θ​(1)\|\bm{h}^{\ell}\_{n}\|=\Theta(1) for each ℓ∈[1:L]\ell\in[1:L] and ‖𝒛n‖=O​(1)\|\bm{z}\_{n}\|=O(1).

[Desideratum˜1](#Thmdesideratum1 "Desideratum 1 (Stability at initialization). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") is essentially automatic for nGPT because of the
Norm​(⋅)\text{Norm}(\cdot) operations in the forward pass and normalization of matrices at each step.

For our second desideratum we again follow ideas from Yang and Hu ([2021](#bib.bib39)) and seek parameterizations in which preactivations change order-11 from each step of training, independent of depth and width.

###### Desideratum 2 (Stable non-trivial feature learning).

We say that a parametrization achieves stable non-trivial feature learning if the logits and all preactivations
evolve non-trivially without blowing up:
‖Δ​𝒉nℓ‖=Θ​(1)\|\Delta\bm{h}\_{n}^{\ell}\|=\Theta(1) and ‖Δ​𝒛n‖=Θ​(1)\|\Delta\bm{z}\_{n}\|=\Theta(1),
and the learning rates ηinput\eta\_{\text{input}}, ηhidden\eta\_{\text{hidden}}, ηoutput\eta\_{\text{output}} have the largest scales possible without
breaking this constraint.

It is a robust empirical observation that desiderata such as the ones above lead to learning rate transfer, even though it is not completely clear why (Dey et al., [2025](#bib.bib10); Yang and Hu, [2021](#bib.bib39); Everett et al., [2024](#bib.bib11)).
We present a simple if tedious calculation in [section˜6.1](#S6.SS1 "6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") for why [˜2](#Thmdesideratum2 "Desideratum 2 (Stable non-trivial feature learning). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") holds in our ν\nuGPT parameterization at growing model width and fixed depth. While we defer the full details to the appendix, we focus here on illustrating the main ideas, especially the somewhat unusual mwidth−3/4m\_{\text{width}}^{-3/4} scaling of the hidden layer and unembedding learning rates.

#### Summary derivations of width corrections

We start with recalling the important notion of alignment exponents introduced in Everett et al. ([2024](#bib.bib11)).

###### Definition 3.1 (Alignment exponents).

Define αoutput,ωoutput,νoutput∈[0,1]\alpha\_{\text{output}},\omega\_{\text{output}},\nu\_{\text{output}}\in[0,1] to be such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Δ​𝐄output​𝒉‖V\displaystyle\frac{\|\Delta\mathbf{E}\_{\text{output}}\bm{h}\|}{\sqrt{V}} | ≍dmodelαoutput​‖Δ​𝐄output‖FV​dmodel​‖𝒉‖dmodel,\displaystyle\asymp d\_{\text{model}}^{\alpha\_{\text{output}}}\frac{\|\Delta\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{Vd\_{\text{model}}}}\frac{\|\bm{h}\|}{\sqrt{d\_{\text{model}}}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝐄output​Δ​𝒉‖V\displaystyle\frac{\|\mathbf{E}\_{\text{output}}\Delta\bm{h}\|}{\sqrt{V}} | ≍dmodelωoutput​‖𝐄output‖FV​dmodel​‖Δ​𝒉‖dmodel,\displaystyle\asymp d\_{\text{model}}^{\omega\_{\text{output}}}\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{Vd\_{\text{model}}}}\frac{\|\Delta\bm{h}\|}{\sqrt{d\_{\text{model}}}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Δ​𝐄output​Δ​𝒉‖V\displaystyle\frac{\|\Delta\mathbf{E}\_{\text{output}}\Delta\bm{h}\|}{\sqrt{V}} | ≍dmodelνoutput​‖Δ​𝐄output‖FV​dmodel​‖Δ​𝒉‖dmodel.\displaystyle\asymp d\_{\text{model}}^{\nu\_{\text{output}}}\frac{\|\Delta\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{Vd\_{\text{model}}}}\frac{\|\Delta\bm{h}\|}{\sqrt{d\_{\text{model}}}}. |  |

Additionally, define αhidden,ωhidden,νhidden∈[0,1]\alpha\_{\text{hidden}},\omega\_{\text{hidden}},\nu\_{\text{hidden}}\in[0,1] to be such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Δ​𝐖​𝒉‖dout\displaystyle\frac{\|\Delta\mathbf{W}\bm{h}\|}{\sqrt{d\_{\text{out}}}} | ≍dinαhidden​‖Δ​𝐖‖Fdout​din​‖𝒉‖din,\displaystyle\asymp d\_{\text{in}}^{\alpha\_{\text{hidden}}}\frac{\|\Delta\mathbf{W}\|\_{F}}{\sqrt{d\_{\text{out}}\,d\_{\text{in}}}}\frac{\|\bm{h}\|}{\sqrt{d\_{\text{in}}}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝐖​Δ​𝒉‖dout\displaystyle\frac{\|\mathbf{W}\Delta\bm{h}\|}{\sqrt{d\_{\text{out}}}} | ≍dinωhidden​‖𝐖‖Fdout​din​‖Δ​𝒉‖din,\displaystyle\asymp d\_{\text{in}}^{\omega\_{\text{hidden}}}\frac{\|\mathbf{W}\|\_{F}}{\sqrt{d\_{\text{out}}\,d\_{\text{in}}}}\frac{\|\Delta\bm{h}\|}{\sqrt{d\_{\text{in}}}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Δ​𝐖​Δ​𝒉‖dout\displaystyle\frac{\|\Delta\mathbf{W}\Delta\bm{h}\|}{\sqrt{d\_{\text{out}}}} | ≍dinνhidden​‖Δ​𝐖‖Fdout​din​‖Δ​𝒉‖din\displaystyle\asymp d\_{\text{in}}^{\nu\_{\text{hidden}}}\frac{\|\Delta\mathbf{W}\|\_{F}}{\sqrt{d\_{\text{out}}\,d\_{\text{in}}}}\frac{\|\Delta\bm{h}\|}{\sqrt{d\_{\text{in}}}} |  |

for any hidden weight 𝐖∈ℝdout×din\mathbf{W}\in\mathbb{R}^{d\_{\text{out}}\times d\_{\text{in}}}.

Although we suppress this from the notation, alignment exponents in principle depend on step number, token position, layer number.

The actual dynamics of the alignment exponents in large nGPT models is complicated and will be explored in [section˜4.3](#S4.SS3 "4.3 Alignment exponents ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers").
Unsurprisingly, some of them change during training. However, in order for the theory to be tractable, we assume them to take fixed values.
We postpone further discussion of this, and for now just list the recommended values. It is observed decisively that ωhidden=ωoutput=1/2\omega\_{\text{hidden}}=\omega\_{\text{output}}=1/2 (see [figure˜5](#S4.F5 "In 4.3 Alignment exponents ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers")). Further, we introduce the following definition for clarity.

###### Definition 3.2 (Full, no, mid alignment).

We introduce the following assumptions:

* •

  no alignment assumption: max⁡{αhidden,νhidden}=max⁡{αoutput,νoutput}=1/2\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}=\max\{\alpha\_{\text{output}},\nu\_{\text{output}}\}=1/2,
* •

  full alignment assumption: max⁡{αhidden,νhidden}=max⁡{αoutput,νoutput}=1\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}=\max\{\alpha\_{\text{output}},\nu\_{\text{output}}\}=1,
* •

  mid alignment assumption:
  max⁡{αhidden,νhidden}=max⁡{αoutput,νoutput}=3/4\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}=\max\{\alpha\_{\text{output}},\nu\_{\text{output}}\}=3/4.

Our parametrization in [table˜1](#S3.T1 "In 3.1 Summary of the changes ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") assumes mid alignment because it achieves perfect transfer over width and is consistent with measurements ([section˜4.3](#S4.SS3 "4.3 Alignment exponents ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers")). Moreover, we make predictions for HP transfer in Adam by studying signGD. Our motivation is that the Adam update for a scalar weight component ww takes the form

|  |  |  |
| --- | --- | --- |
|  | w​(t+1)=w​(t)−η​1−β11−β1t+1​∑τ=0tβ1t−τ​g​(τ)1−β21−β2t+1​∑τ=0tβ2t−τ​g2​(τ)+ϵ,w(t+1)=w(t)-\eta\frac{\frac{1-\beta\_{1}}{1-\beta\_{1}^{t+1}}\sum\_{\tau=0}^{t}\beta\_{1}^{t-\tau}g(\tau)}{\sqrt{\frac{1-\beta\_{2}}{1-\beta\_{2}^{t+1}}\sum\_{\tau=0}^{t}\beta\_{2}^{t-\tau}g^{2}(\tau)}+\epsilon}, |  |

where gg the corresponding gradient component, η\eta the corresponding learning rate, tt the time step, β1,β2,ϵ\beta\_{1},\beta\_{2},\epsilon the Adam’s hyperparameters. This can be viewed as a smoothed version of the signGD update

|  |  |  |
| --- | --- | --- |
|  | w​(t+1)=w​(t)−η​g​(t)g2​(t)=w​(t)−η​sign⁡g​(t).w(t+1)=w(t)-\eta\frac{g(t)}{\sqrt{g^{2}(t)}}=w(t)-\eta\operatorname{sign}g(t). |  |

In particular, we assume “signGD-like” scaling of updates:

|  |  |  |
| --- | --- | --- |
|  | w​(t+1)−w​(t)≍η.w(t+1)-w(t)\asymp\eta. |  |

For small enough updates (which will be the case in our derivations), the renormalization of matrices at each step does not influence this scale.

The following derivations, designed to avoid violating [˜2](#Thmdesideratum2 "Desideratum 2 (Stable non-trivial feature learning). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers"), motivate the choice of width corrections.

* •

  What should we do to make
  the embedding update contribute non-trivially without blowup, that is,
  for
  ‖Δ​𝐄input​𝒙n‖\|\Delta\mathbf{E}\_{\text{input}}\bm{x}\_{n}\| to have scale Θ​(1)\Theta(1)?
  Since 𝒙n\bm{x}\_{n} is one-hot, Δ​𝐄input​𝒙n\Delta\mathbf{E}\_{\text{input}}\bm{x}\_{n} is a column of Δ​𝐄input\Delta\mathbf{E}\_{\text{input}}, so this norm has
  scale dmodel​ηinput\sqrt{d\_{\text{model}}}\eta\_{\text{input}} because Adam updates have signGD-like scaling (with each component almost ±ηinput\pm\eta\_{\text{input}}).
  This is why we need ηinput\eta\_{\text{input}} to be proportional to dmodel−1/2d\_{\text{model}}^{-1/2}, justifying the mwidth−1/2m\_{\text{width}}^{-1/2} correction in ηinput\eta\_{\text{input}}.
* •

  What should we do to ensure ‖Δ​(𝐖​𝒉n)‖\|\Delta(\mathbf{W}\bm{h}\_{n})\| have scale Θ​(1)\Theta(1) for some hidden weight matrix 𝐖∈ℝdout×din\mathbf{W}\in\mathbb{R}^{d\_{\text{out}}\times d\_{\text{in}}} inside one of the Transformer blocks, where dout≍din≍dmodeld\_{\text{out}}\asymp d\_{\text{in}}\asymp d\_{\text{model}}? Using the definition of αhidden\alpha\_{\text{hidden}}, we have

  |  |  |  |
  | --- | --- | --- |
  |  | ‖Δ​𝐖​𝒉n‖dmodel≍dmodelαhidden​‖Δ​𝐖‖Fdmodel​‖𝒉n‖dmodel≍dmodelαhidden−1/2​ηhidden,\displaystyle\frac{\|\Delta\mathbf{W}\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{\alpha\_{\text{hidden}}}\frac{\|\Delta\mathbf{W}\|\_{F}}{d\_{\text{model}}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{\alpha\_{\text{hidden}}-1/2}\eta\_{\text{hidden}}, |  |

  where the last equivalence is true because the quadratic mean of Δ​𝐖\Delta\mathbf{W} components ‖Δ​𝐖‖F/dmodel\|\Delta\mathbf{W}\|\_{F}/d\_{\text{model}} scales like ηhidden\eta\_{\text{hidden}} (Adam updates have signGD-like scaling). Similarly,

  |  |  |  |
  | --- | --- | --- |
  |  | ‖𝐖​Δ​𝒉n‖dmodel≍dmodelωhidden​‖𝐖‖Fdmodel​‖Δ​𝒉n‖dmodel≍dmodel−1/2,\displaystyle\frac{\|\mathbf{W}\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{\omega\_{\text{hidden}}}\frac{\|\mathbf{W}\|\_{F}}{d\_{\text{model}}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{-1/2}, |  |

  where in the last equivalence we used ωhidden=1/2\omega\_{\text{hidden}}=1/2, the fact that rows of 𝐖\mathbf{W} are of norm 1 and assumed that
  the movement of the previous hidden state matches itself in scale, that is, ‖Δ​𝒉n‖≍‖𝒉n‖≍1\|\Delta\bm{h}\_{n}\|\asymp\|\bm{h}\_{n}\|\asymp 1. Finally,

  |  |  |  |
  | --- | --- | --- |
  |  | ‖Δ​𝐖​Δ​𝒉n‖dmodel≍dmodelνhidden−1/2​ηhidden\displaystyle\frac{\|\Delta\mathbf{W}\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{\nu\_{\text{hidden}}-1/2}\eta\_{\text{hidden}} |  |

  by similar arguments. Combining, we obtain the maximal hidden learning rate ηhidden≍dmodel−max⁡{αhidden,νhidden}=dmodel−3/4\eta\_{\text{hidden}}\asymp d\_{\text{model}}^{-\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}}=d\_{\text{model}}^{-3/4},
  justifying the mwidth−3/4m\_{\text{width}}^{-3/4} (mid alignment) correction in ηhidden\eta\_{\text{hidden}}.
* •

  What should we do if we want to ensure ‖Δ​(𝐄output​𝒉n)‖≍‖𝐄output​𝒉n‖\|\Delta(\mathbf{E}\_{\text{output}}\bm{h}\_{n})\|\asymp\|\mathbf{E}\_{\text{output}}\bm{h}\_{n}\|?
  The alignment exponent at initialization is always 1/21/2 by independence:

  |  |  |  |
  | --- | --- | --- |
  |  | ‖𝐄output​𝒉n‖V≍dmodel​‖𝐄output‖Fdmodel​V​‖𝒉n‖dmodel=‖𝐄output‖Fdmodel​V=dmodel−1/2.\frac{\|\mathbf{E}\_{\text{output}}\bm{h}\_{n}\|}{\sqrt{V}}\asymp\sqrt{d\_{\text{model}}}\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{d\_{\text{model}}V}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{d\_{\text{model}}V}}=d\_{\text{model}}^{-1/2}. |  |

  Next, similarly to the above, we have

  |  |  |  |
  | --- | --- | --- |
  |  | ‖Δ​𝐄output​𝒉n‖V≍dmodelαoutput​ηoutput​‖𝒉n‖dmodel=dmodelαoutput−1/2​ηoutput;\displaystyle\frac{\|\Delta\mathbf{E}\_{\text{output}}\bm{h}\_{n}\|}{\sqrt{V}}\asymp d\_{\text{model}}^{\alpha\_{\text{output}}}\eta\_{\text{output}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=d\_{\text{model}}^{\alpha\_{\text{output}}-1/2}\eta\_{\text{output}}; |  |
  |  |  |  |
  | --- | --- | --- |
  |  | ‖𝐄output​Δ​𝒉n‖V≍dmodelωoutput​‖𝐄output‖FV​dmodel​‖Δ​𝒉n‖dmodel≍dmodelωoutput−1/2​‖𝐄output‖FV​dmodel≍dmodelωoutput−1;\displaystyle\frac{\|\mathbf{E}\_{\text{output}}\Delta\bm{h}\_{n}\|}{\sqrt{V}}\asymp d\_{\text{model}}^{\omega\_{\text{output}}}\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{Vd\_{\text{model}}}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{\omega\_{\text{output}}-1/2}\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{Vd\_{\text{model}}}}\asymp d\_{\text{model}}^{\omega\_{\text{output}}-1}; |  |
  |  |  |  |
  | --- | --- | --- |
  |  | ‖Δ​𝐄output​Δ​𝒉n‖V≲dmodelνoutput​ηoutput​‖Δ​𝒉n‖dmodel=dmodelνoutput−1/2​ηoutput.\displaystyle\frac{\|\Delta\mathbf{E}\_{\text{output}}\Delta\bm{h}\_{n}\|}{\sqrt{V}}\lesssim d\_{\text{model}}^{\nu\_{\text{output}}}\eta\_{\text{output}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=d\_{\text{model}}^{\nu\_{\text{output}}-1/2}\eta\_{\text{output}}. |  |

  Recalling that ωoutput=1/2\omega\_{\text{output}}=1/2, we should set
  ηoutput≲dmodel−max⁡{αoutput,νoutput}\eta\_{\text{output}}\lesssim d\_{\text{model}}^{-\max\{\alpha\_{\text{output}},\nu\_{\text{output}}\}},
  justifying the width correction mwidth−3/4m\_{\text{width}}^{-3/4} (mid alignment) in ηoutput\eta\_{\text{output}}.

#### Depth corrections

The preceding discussion concerned HP transfer across width in nGPT. Let us also briefly discuss how to scale learning rates and parameterization with growing depth. It is common in the literature to consider a residual network like

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒉ℓ+1=𝒉ℓ+nlayers−αdepth​ℱℓ​(𝒉ℓ),\bm{h}^{\ell+1}=\bm{h}^{\ell}+n\_{\text{layers}}^{-\alpha\_{\text{depth}}}\mathcal{F}\_{\ell}(\bm{h}^{\ell}), |  | (10) |

where ℱℓ\mathcal{F}\_{\ell} is the ℓ\ellth block, and αdepth\alpha\_{\text{depth}} is the depth alpha. The updates

|  |  |  |
| --- | --- | --- |
|  | Δ​𝒉ℓ+1=Δ​𝒉ℓ+nlayers−αdepth​Δ​ℱℓ​(𝒉ℓ)\Delta\bm{h}^{\ell+1}=\Delta\bm{h}^{\ell}+n\_{\text{layers}}^{-\alpha\_{\text{depth}}}\Delta\mathcal{F}\_{\ell}(\bm{h}^{\ell}) |  |

accumulate over 0​p​t0pt layers, so one can require

|  |  |  |
| --- | --- | --- |
|  | nlayers−αdepth​Δ​ℱℓ​(𝒉ℓ)≲10​p​tn\_{\text{layers}}^{-\alpha\_{\text{depth}}}\Delta\mathcal{F}\_{\ell}(\bm{h}^{\ell})\lesssim\frac{1}{0pt} |  |

to prevent updates from blowing up ([˜2](#Thmdesideratum2 "Desideratum 2 (Stable non-trivial feature learning). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")). If Δ​ℱℓ​(𝒉ℓ)≍ηhidden\Delta\mathcal{F}\_{\ell}(\bm{h}^{\ell})\asymp\eta\_{\text{hidden}} which is typically the case for Adam
because of signGD-like scaling,
this means we may need to introduce a depth correction into the hidden learning rate:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ηhidden≍nlayersαdepth−1,\eta\_{\text{hidden}}\asymp n\_{\text{layers}}^{\alpha\_{\text{depth}}-1}, |  | (11) |

unless αdepth=1\alpha\_{\text{depth}}=1 (in which case no such correction is required).
The Depth-μ\muP (Yang et al., [2024b](#bib.bib42)) style parametrization sets αdepth=1/2\alpha\_{\text{depth}}=1/2 (and therefore ηhidden≍nlayers−1/2\eta\_{\text{hidden}}\asymp n\_{\text{layers}}^{-1/2})
based on additional desiderata such as feature diversity,
whereas CompleteP (Dey et al., [2025](#bib.bib10)) sets αdepth=1\alpha\_{\text{depth}}=1 based on a different additional desideratum
(“complete” feature learning: hidden layers and model output are not near-linear in any of the model parameters).

The Normalized Transformer differs significantly from ([10](#S3.E10 "Equation 10 ‣ Depth corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")),
so we conduct our own theoretical analysis in [section˜6.2](#S6.SS2 "6.2 Depth scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers").
We find that for a simple network closer to nGPT, the safe hidden learning rate is also given by ([11](#S3.E11 "Equation 11 ‣ Depth corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")).
To keep things simple, we just introduce a correction mdepth−1m\_{\text{depth}}^{-1} into αA,init\alpha\_{\text{$A$,init}} and αM,init\alpha\_{\text{$M$,init}},
which roughly corresponds to αdepth=1\alpha\_{\text{depth}}=1 (at least at initialization).
Interestingly, our ablations in [section˜4](#S4 "4 Experiments ‣ Learning Rate Transfer in Normalized Transformers") show that,
(a) baseline nGPT already shows decent transfer over depth
(perhaps because 𝜶A\bm{\alpha}\_{A} and 𝜶M\bm{\alpha}\_{M} are trainable and initialized with a small enough value, which means 0​p​t0pt needs to be extremely large for depth corrections to matter),
(b) the hidden learning rate should not be changed even if a mdepth−1/2m\_{\text{depth}}^{-1/2} correction in αA,init\alpha\_{\text{$A$,init}} and αM,init\alpha\_{\text{$M$,init}} is used instead. Explaining this phenomenon theoretically can be an interesting future direction.

### 3.3 Token horizon corrections

In [section˜3.2](#S3.SS2 "3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers"), the number of steps was bounded.
In practice, however, the token horizon is typically scaled along with the model size; hence,
we need to account for the growing number of steps.
It is a robust finding in recent literature that the optimal learning rates decrease as the token horizon increases
at a fixed batch size
(Everett et al., [2024](#bib.bib11); Bjorck et al., [2025](#bib.bib2); Mlodozeniec et al., [2025](#bib.bib23)).
There is no well-known and well-tested theoretical framework predicting the correct scaling. The work Mlodozeniec et al. ([2025](#bib.bib23)) uses an SDE perspective from Malladi et al. ([2022](#bib.bib21)) to derive the data correction mdata−1/2m\_{\text{data}}^{-1/2}.
Shulgin et al. ([2026](#bib.bib30)) also predict this data correction via a different theoretical method.
However, we find that the exponent 1/21/2 is too large.
Instead, we base our choice mdata−1/3m\_{\text{data}}^{-1/3} on empirical results from Bjorck et al. ([2025](#bib.bib2))
(cf. β=0.32\beta=0.32 there) and our own independent power law fits ([section˜4.6](#S4.SS6 "4.6 Token count correction ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers")).
The exponent 1/31/3 is also concurrently confirmed in Ren et al. ([2026](#bib.bib28)) in a very different setting.

## 4 Experiments

We train nGPT as defined in [section˜2](#S2 "2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers") with cross-entropy loss on the FineWeb-Edu dataset (Penedo et al., [2024](#bib.bib27)) with sequence length 4096,
fixed batch size 64.
We use the OLMo 2 (OLMo et al., [2025](#bib.bib25)) tokenizer333<https://huggingface.co/allenai/OLMo-2-0425-1B> with vocabulary size 100 352100\,352.
The “fixed iteration count” experiments all use 80 00080\,000 steps, or 4096×64×80 000≈21​B4096\times 64\times 80\,000\approx 21B tokens.
We use our own implementation of nGPT in our fork of TorchTitan (Liang et al., [2025](#bib.bib17)).
The “20 tokens per parameter” experiments all use the token count which is 2020 times the number of non-embedding parameters444or, more specifically, the number of steps corresponding to this token count and rounded up to a multiple of 250.
This is heuristically treated as the compute-optimal number of tokens in the literature (Dey et al., [2025](#bib.bib10); Wen et al., [2025](#bib.bib34)), although
Hoffmann et al. ([2022](#bib.bib13)) include embedding parameters when fitting power laws (see also Pearce and Song ([2024](#bib.bib26)) on this topic).
We use Adam (AdamW with weight decay 0.0)
with β1=0.9\beta\_{1}=0.9, β2=0.95\beta\_{2}=0.95, ϵ=10−16\epsilon=10^{-16} (following Dey et al. ([2025](#bib.bib10))), no warm-up, the learning rate decayed to 10%10\% of its peak (initial) value using a cosine schedule. Our base depth and width are 1010.
Unless averaging over seeds, we use validation loss EMA with β=0.95\beta=0.95.

### 4.1 Growing nheadsn\_{\text{heads}} at a fixed iteration count

We first fix the head dimension at555102=⌊1024/10⌋102=\lfloor 1024/10\rfloor, where 1010 is the base depth and width 102 and scale the number of heads,
fixing the number of blocks (depth) at 0​p​t=120pt=12. As the number of heads grows from 8 to 40,
the total number of parameters grows from 0.26 B to 3.22 B.
The baseline nGPT implementation without our reparametrization does not show transfer of the learning rate over width
([figure˜2(a)](#S1.F2.sf1 "In Figure 2 ‣ 1 Introduction ‣ Learning Rate Transfer in Normalized Transformers")). In the same setting,
our parametrization shows good transfer over the number of heads ([figure˜2(b)](#S1.F2.sf2 "In Figure 2 ‣ 1 Introduction ‣ Learning Rate Transfer in Normalized Transformers")).

\lxSVG@picture

Width corrections (powers of mwidthm\_{\text{width}}) are important for width transfer in ν\nuGPT.
\endlxSVG@picture

### 4.2 Growing 0​p​t0pt at a fixed iteration count

|  |  |
| --- | --- |
|  |  |

!(/html/2604.27077/assets/x5.png)

(a) nGPT (baseline)

!(/html/2604.27077/assets/x6.png)

(b) ν\nuGPT (ours)

Figure 3: 
Depth sweeps with 0​p​t0pt increasing (in the legend, with the best loss in parentheses), head dimension and nheads=12n\_{\text{heads}}=12 fixed, 80 000 iterations (about 21 B tokens).
Both baseline nGPT and our CompleteP-inspired (Dey et al., [2025](#bib.bib10)) parametrization show decent transfer, with ours winning slightly in stability and performance.

In this set of sweeps, we fix the head dimension (at the same value 102 as above) and the number of heads at 12 (so that dmodel=1224d\_{\text{model}}=1224), and scale the number of blocks (depth) 0​p​t0pt. As we vary 0​p​t0pt from 8 to 128, the number of parameters in the model grows from 0.39 B to 2.55 B.

Interestingly, baseline nGPT already shows decent transfer over depth at a fixed iteration count ([figure˜3(a)](#S4.F3.sf1 "In Figure 3 ‣ 4.2 Growing 0⁢𝑝⁢𝑡 at a fixed iteration count ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers")),
although models with the highest depth become somewhat unstable.
Therefore, the minimal depth corrections in [section˜3](#S3 "3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers"), although theoretically well-motivated,
may not be necessary. In addition, we observe that the average 𝜶A\bm{\alpha}\_{A} and 𝜶M\bm{\alpha}\_{M} components
(over all components and all blocks) at the end of training decrease at power laws in depth close to 0​p​t−0.50pt^{-0.5}
(see [figure˜10](#S7.F10 "In 7.1 LERP parameters ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers") for precise formulas),
which suggests that among αdepth∈{0,1/2,1}\alpha\_{\text{depth}}\in\{0,1/2,1\} (defined in [section˜3.2](#S3.SS2 "3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")),
the trained baseline Normalized Transformer prefers αdepth=1/2\alpha\_{\text{depth}}=1/2 on average (it should be noted that the variance across components and blocks is very high,
as seen from error bars).

We also verify that our parametrization gives good transfer over depth in [figure˜3(b)](#S4.F3.sf2 "In Figure 3 ‣ 4.2 Growing 0⁢𝑝⁢𝑡 at a fixed iteration count ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers").

\lxSVG@picture

Both baseline nGPT and our reparametrization show good learning rate transfer over depth.
Our parametrization shows lower learning rate sensitivity
at large depths.
\endlxSVG@picture

### 4.3 Alignment exponents

We show in [section˜6.1](#S6.SS1 "6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") (sketch in [section˜3.2](#S3.SS2 "3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) that if we assume fixed values for alignment exponents ([definition˜3.1](#S3.theoremcnt1 "Definition 3.1 (Alignment exponents). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")), then the maximal stable learning rates should scale as

|  |  |  |
| --- | --- | --- |
|  | ηhidden≍dmodel−max⁡{αhidden,νhidden},ηoutput≲dmodel−max⁡{αoutput,νoutput}.\eta\_{\text{hidden}}\asymp d\_{\text{model}}^{-\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}},\quad\eta\_{\text{output}}\lesssim d\_{\text{model}}^{-\max\{\alpha\_{\text{output}},\nu\_{\text{output}}\}}. |  |

Recall the “full alignment” regime where max⁡{αhidden,νhidden}=max⁡{αoutput,νoutput}=1\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}=\max\{\alpha\_{\text{output}},\nu\_{\text{output}}\}=1 and the “no alignment”
regime where max⁡{αhidden,νhidden}=max⁡{αoutput,νoutput}=1/2\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}=\max\{\alpha\_{\text{output}},\nu\_{\text{output}}\}=1/2.
We measure the alignment exponents in a representative run of ν\nuGPT with 0​p​t=nheads=120pt=n\_{\text{heads}}=12
(see [section˜7.2](#S7.SS2 "7.2 Alignment exponents for a wider model ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers") for a different size) and
see that neither of these two regimes perfectly match observation.
[Figure˜4](#S4.F4 "In 4.3 Alignment exponents ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers") shows that all α\alpha and ν\nu vary significantly during training, with αhidden,νhidden\alpha\_{\text{hidden}},\nu\_{\text{hidden}} and αoutput,νoutput\alpha\_{\text{output}},\nu\_{\text{output}} much higher during a short period at the beginning than at the end (where they settle close to 0.5–0.6).

We find that the exact middle of these two regimes gives essentially perfect transfer over width:

|  |  |  |
| --- | --- | --- |
|  | max⁡{αhidden,νhidden}=max⁡{αoutput,νoutput}=3/4,\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}=\max\{\alpha\_{\text{output}},\nu\_{\text{output}}\}=3/4, |  |

and we make this choice that in our final recommendation.
This setting is close to observed average alignment exponents if we weight them by the loss decrease ([figure˜5](#S4.F5 "In 4.3 Alignment exponents ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers")).
This naturally gives more weight to the beginning of training. Although this is not obviously the right weighting, it is reasonable because loss decrease may parametrize the importance of updates, and it is consistent with the μ\muP methodology which is focused on the first one or two steps.

In addition, the plots reported here show definitively that the correct value of ωhidden\omega\_{\text{hidden}} and ωoutput\omega\_{\text{output}} is one half:

|  |  |  |
| --- | --- | --- |
|  | ωhidden=ωoutput=1/2.\omega\_{\text{hidden}}=\omega\_{\text{output}}=1/2. |  |

This confirms the point made theoretically in Everett et al. ([2024](#bib.bib11))
that the assumption ωoutput=1\omega\_{\text{output}}=1 (uniquely motivating μ\muP)
is too conservative.

!(/html/2604.27077/assets/x7.png)

(a) Alignment exponent α\alpha

!(/html/2604.27077/assets/x8.png)

(b) Alignment exponent ω\omega

!(/html/2604.27077/assets/x9.png)

(c) Alignment exponent ν\nu

Figure 4: Alignment exponents ([definition˜3.1](#S3.theoremcnt1 "Definition 3.1 (Alignment exponents). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) of ν\nuGPT with 0​p​t=nheads=120pt=n\_{\text{heads}}=12 on a fixed validation batch, averaged over layers.

!(/html/2604.27077/assets/x10.png)

(a) Alignment exponent α\alpha

!(/html/2604.27077/assets/x11.png)

(b) Alignment exponent ω\omega

!(/html/2604.27077/assets/x12.png)

(c) Alignment exponent ν\nu

Figure 5: Alignment exponents ([definition˜3.1](#S3.theoremcnt1 "Definition 3.1 (Alignment exponents). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) of ν\nuGPT with 0​p​t=nheads=120pt=n\_{\text{heads}}=12 on a fixed validation batch, averaged over layers,
viewed as a function of loss decrease. The mid alignment assumption ([definition˜3.2](#S3.theoremcnt2 "Definition 3.2 (Full, no, mid alignment). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) matches the observations well.

!(/html/2604.27077/assets/x13.png)

(a) Alignment exponent α\alpha

!(/html/2604.27077/assets/x14.png)

(b) Alignment exponent ω\omega

!(/html/2604.27077/assets/x15.png)

(c) Alignment exponent ν\nu

Figure 6: Alignment exponents ([definition˜3.1](#S3.theoremcnt1 "Definition 3.1 (Alignment exponents). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) of ν\nuGPT with 0​p​t=nheads=120pt=n\_{\text{heads}}=12 on a fixed validation batch vs. layer index, averaged over steps.

### 4.4 ν\nuGPT gives better transfer over width than μ\muP

We ablate our carefully tuned alignment exponents by comparing with standard μ\muP-style
width corrections from prior literature. [Figure˜7](#S4.F7 "In 4.4 𝜈GPT gives better transfer over width than 𝜇P ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers") shows that ν\nuGPT gives somewhat better transfer over width.
We do not interpret this as falsifying μ\muP (it still gives decent transfer)
but rather making some corrections in the theory. In particular, we confirm the point made in Everett et al. ([2024](#bib.bib11))
that μ\muP is by no means the unique maximal update parametrization achieving feature learning and giving
learning rate transfer (in particular, our parametrization is somewhat different from any parametrization in prior work).

!(/html/2604.27077/assets/x16.png)

(a) nGPT with μ\muP-style width corrections (“CompleteP” in [table˜1](#S3.T1 "In 3.1 Summary of the changes ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers"))

!(/html/2604.27077/assets/x17.png)

(b) ν\nuGPT (ours)

Figure 7: The number of heads is swept as in [figure˜2](#S1.F2 "In 1 Introduction ‣ Learning Rate Transfer in Normalized Transformers").
For nGPT with μ\muP-style width corrections (“CompleteP” in [table˜1](#S3.T1 "In 3.1 Summary of the changes ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) the learning rate slightly drifts to the right.
Our parametrization provides essentially perfect transfer of the learning rate over width.
The multipliers for (ηinput,ηoutput)(\eta\_{\text{input}},\eta\_{\text{output}}) are tuned for both parametrizations separately.

\lxSVG@picture

Measuring the alignment exponents leads to corrections into the μ\muP-type theory,
providing somewhat better learning rate transfer over width at a fixed token horizon.
\endlxSVG@picture

The correct value ωoutput=1/2\omega\_{\text{output}}=1/2 instead of ωoutput=1\omega\_{\text{output}}=1 allows the logits to be order-1
both at initialization and during training, rather than of order 1/dmodel1/\sqrt{d\_{\text{model}}} at initialization as in μ\muP.
An ablation of the logits scaling specifically (with other choices fixed) is provided in [section˜7.5](#S7.SS5 "7.5 𝜇P or not 𝜇P, that is the question (logits scaling) ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers").

### 4.5 Depth-μ\muP style correction in ηhidden\eta\_{\text{hidden}} would break transfer over depth

We observe in [figure˜8](#S4.F8 "In 4.5 Depth-𝜇P style correction in 𝜂_\"hidden\" would break transfer over depth ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers") that a depth correction in ηhidden\eta\_{\text{hidden}}
motivated by ([11](#S3.E11 "Equation 11 ‣ Depth corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) with αdepth=1/2\alpha\_{\text{depth}}=1/2, as done in Depth-μ\muP,
does not show transfer over depth: the learning rate strongly drifts to the right suggesting
that ηhidden\eta\_{\text{hidden}} should not be changed.
This is likely an artifact of the architecture: 𝜶A\bm{\alpha}\_{A}, 𝜶M\bm{\alpha}\_{M} are trainable and we only control their (small) initial values
and learning rate.
This finding does not confirm or refute (Dey et al., [2025](#bib.bib10)) that “Depth-μ\muP” style αdepth=1/2\alpha\_{\text{depth}}=1/2 is incorrect
but it does suggest that depth transfer in normalized models
(with or without trainable convex combinations)
is a fruitful subject of theoretical research.

|  |  |
| --- | --- |
|  |  |

!(/html/2604.27077/assets/x18.png)

(a) Depth sweep of “Depth-μ\muP” in [table˜1](#S3.T1 "In 3.1 Summary of the changes ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers").

!(/html/2604.27077/assets/x19.png)

(b) ν\nuGPT (ours)

Figure 8: 
The number of layers is swept as in [figure˜3](#S4.F3 "In 4.2 Growing 0⁢𝑝⁢𝑡 at a fixed iteration count ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers").
A mdepth−1/2m\_{\text{depth}}^{-1/2} correction to ηhidden\eta\_{\text{hidden}} like in Depth-μ\muP would break transfer over depth.

### 4.6 Token count correction

We train a fixed ν\nuGPT model (0​p​t=nheads=120pt=n\_{\text{heads}}=12)
with token count corrections removed and fit a power law describing how the optimal learning rate
decreases with the number of iterations.
We see that it decreases roughly as (iter. count)−1/3(\text{iter. count})^{-1/3}
([figure˜9](#S4.F9 "In 4.6 Token count correction ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers")).

|  |  |
| --- | --- |
|  |  |

!(/html/2604.27077/assets/x20.png)

!(/html/2604.27077/assets/x21.png)

Figure 9: A sweep of a fixed ν\nuGPT model with different number of iterations (in the legend, with the best loss in parentheses) but without token count corrections: optimal peak learning rate decreases at about (iter. count)−1/3(\text{iter. count})^{-1/3}.

This matches Bjorck et al. ([2025](#bib.bib2)) who used non-normalized architectures
(cf. β=0.32\beta=0.32 there).
An additional sweep with different model sizes is provided in [section˜7.3](#S7.SS3 "7.3 Width sweep at 20 tokens per parameter ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers").

\lxSVG@picture

It is a robust finding that the learning rate decreases proportionally to (iter. count)−1/3(\text{iter. count})^{-1/3},
calling for an iteration count correction mdata−1/3m\_{\text{data}}^{-1/3} (as in [table˜1](#S3.T1 "In 3.1 Summary of the changes ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")).
\endlxSVG@picture

## 5 Concluding remarks

We have obtained a parametrization ν\nuGPT of the Normalized Transformer that shows good transfer of the learning rate
over model width, depth and token count as well as their combinations, at no loss of performance.
An important feature of the normalized setting is that it is more controlled, reducing the number of assumptions
(such as matrix norms)
that are to be taken on faith.
For example, the models achieve impressive performance without using weight decay, which is a confounder
for HP transfer theory (e. g. Kosson et al. ([2026](#bib.bib15))).
We have further made significant effort to stay empirically grounded, e. g. by measuring weight-activation
alignment exponents,
avoiding μ\muP-type assumptions commonly used in the literature, and questioning
somewhat imprecise token count power laws theoretically predicted in recent works.
This motivates a few theoretical future directions,
such as a more complete understanding of depthwise
transfer in normalized models, the dynamic understanding of weight decay when it is, in fact, used,
explaining analytically the breakdown of the μ\muP assumptions in toy models,
and predicting the ∼1/3\sim 1/3 exponent in the token count corrections.

## Funding Acknowledgments

BH is supported by a 2024 Sloan Fellowship in Mathematics, NSF grant DMS-2143754, DMS-2133806, and DARPA AIQ grant (HR001124S0029).

\beginappendix

## 6 Proofs and derivations

### 6.1 Width scaling

In this [section](#S6.SS1 "6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers"), we will show that the width corrections
described in [section˜3.1](#S3.SS1 "3.1 Summary of the changes ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") satisfy [˜1](#Thmdesideratum1 "Desideratum 1 (Stability at initialization). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") and [2](#Thmdesideratum2 "Desideratum 2 (Stable non-trivial feature learning). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")
in the setting of fixed depth.
Specifically, in [lemma˜6.3](#S6.theoremcnt3 "Lemma 6.3 (Input). ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") we show that the first hidden state 𝒉n1=𝐄input​𝒙n\bm{h}\_{n}^{1}=\mathbf{E}\_{\text{input}}\bm{x}\_{n} is initialized stably
and evolves stably and non-trivially.
In [lemma˜6.6](#S6.theoremcnt6 "Lemma 6.6 (Attention). ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers"), we show that if this was true of the hidden state before an attention block,
it is also true of the hidden state after this block, and so on. Then, since depth 0​p​t0pt is fixed
and the number of iterations is bounded, by induction, we can conclude that [˜1](#Thmdesideratum1 "Desideratum 1 (Stability at initialization). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") and [2](#Thmdesideratum2 "Desideratum 2 (Stable non-trivial feature learning). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") are satisfied
during the whole training.

#### Notation

We ask the reader to recall the notation from [section˜3.2](#S3.SS2 "3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers").
In this section, depth is considered fixed but the number of heads can vary (and we think of it as large).
We will sometimes use N=dmodelN=d\_{\text{model}} to declutter notations.
We will follow Everett et al. ([2024](#bib.bib11)) in using non-standard O​(⋅)O(\cdot) and similar notation:
specifically, aN≲bNa\_{N}\lesssim b\_{N} or aN=O​(bN)a\_{N}=O(b\_{N}) means |aN/bN||a\_{N}/b\_{N}| is allowed to grow sub-polynomially, that is,
almost surely lim supN→∞logN⁡|aN/bN|≤0\limsup\_{N\to\infty}\log\_{N}|a\_{N}/b\_{N}|\leq 0, whereas
aN≍bNa\_{N}\asymp b\_{N} or aN=Θ​(bN)a\_{N}=\Theta(b\_{N}) means aN≲bNa\_{N}\lesssim b\_{N} and bN≲aNb\_{N}\lesssim a\_{N}.
Note that this definition requires care because properties that more standard O​(⋅)O(\cdot) notation
has can be false. For example, it is false that ∑k=1NO​(1)=O​(N)\sum\_{k=1}^{N}O(1)=O(N) (a counter-example is aN(k)≡ka\_{N}^{(k)}\equiv k for k∈ℤ>0k\in\mathbb{Z}\_{>0}).

#### Alignment

Recall the definition of alignment exponents given in [definition˜3.1](#S3.theoremcnt1 "Definition 3.1 (Alignment exponents). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers").

We make the following classical (Everett et al., [2024](#bib.bib11))
simple assumption that is definitively consistent with observation (as shown in [section˜4.3](#S4.SS3 "4.3 Alignment exponents ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers")):

###### Assumption 6.1.

Assume ωhidden=1/2\omega\_{\text{hidden}}=1/2 (no alignment).

Requiring additionally ωoutput=1\omega\_{\text{output}}=1 corresponds to classical μ\muP,
but ωoutput=1/2\omega\_{\text{output}}=1/2 is consistent with observation ([section˜4.3](#S4.SS3 "4.3 Alignment exponents ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers"))
and allows for non-equivalent maximal-update parametrizations
that achieve feature learning at high widths (Everett et al., [2024](#bib.bib11)).

#### Learning rate of “rescalers”

Recall that trainable “rescalers” such as sq​k,initsq​k,scale​𝒔q​k\frac{s\_{\text{$qk$,init}}}{s\_{\text{$qk$,scale}}}\bm{s}\_{qk}
have two hyperparameters sq​k,inits\_{\text{$qk$,init}} and sq​k,scales\_{\text{$qk$,scale}}.
Since each component of 𝒔q​k\bm{s}\_{qk} is initialized with sq​k,scales\_{\text{$qk$,scale}}, each component of
sq​k,initsq​k,scale​𝒔q​k\frac{s\_{\text{$qk$,init}}}{s\_{\text{$qk$,scale}}}\bm{s}\_{qk} at initialization is sq​k,inits\_{\text{$qk$,init}}.
Thus, the hyperparameter sq​k,inits\_{\text{$qk$,init}} controls the initialization value, whereas sq​k,scales\_{\text{$qk$,scale}} controls
the learning rate. So, for certainty, we will leave the learning rate of the raw 𝒔q​k\bm{s}\_{qk} vector unmodified
(equal to ηbase\eta\_{\text{base}}). In other words, Δ​sq​k,i=Θ​(1)\Delta s\_{qk,i}=\Theta(1) for each coordinate ii.
We do not lose generality, because if we needed the learning rate of sq​k,initsq​k,scale​𝒔q​k\frac{s\_{\text{$qk$,init}}}{s\_{\text{$qk$,scale}}}\bm{s}\_{qk} to scale with width or depth,
we would modify sq​k,scales\_{\text{$qk$,scale}}.

###### Assumption 6.2.

At each step t=O​(1)t=O(1) we have

|  |  |  |
| --- | --- | --- |
|  | Δ​sq​k,i​(t)≍Δ​su,i​(t)≍Δ​sν,i​(t)≍Δ​sz,i​(t)≍Δ​αA,i​(t)≍Δ​αM,i​(t)≍1.\Delta s\_{qk,i}(t)\asymp\Delta s\_{u,i}(t)\asymp\Delta s\_{\nu,i}(t)\asymp\Delta s\_{z,i}(t)\asymp\Delta\alpha\_{A,i}(t)\asymp\Delta\alpha\_{M,i}(t)\asymp 1. |  |

We now proceed with proving that neither block of ν\nuGPT breaks the stability and non-triviality requirements.

###### Lemma 6.3 (Input).

Suppose [˜6.1](#S6.theoremcnt1 "Assumption 6.1. ‣ Alignment ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") holds,
and ηinput≍dmodel−1/2\eta\_{\text{input}}\asymp d\_{\text{model}}^{-1/2}.
Then, for one-hot 𝒙n\bm{x}\_{n}, the vector 𝒉n1​(t):=𝐄input​(t)​𝒙n\bm{h}\_{n}^{1}(t):=\mathbf{E}\_{\text{input}}(t)\bm{x}\_{n} satisfies
‖𝒉n1‖=Θ​(1)\|\bm{h}\_{n}^{1}\|=\Theta(1)
and ‖Δ​𝒉n1‖=Θ​(1)\|\Delta\bm{h}\_{n}^{1}\|=\Theta(1).

###### Proof.

Since 𝒙n\bm{x}\_{n} is one-hot, 𝒉n1\bm{h}\_{n}^{1} is a column of 𝐄input\mathbf{E}\_{\text{input}},
so it is normalized: ‖𝒉n1‖=1\|\bm{h}\_{n}^{1}\|=1.
Next, Δ​𝒉n1\Delta\bm{h}\_{n}^{1} is a column of Δ​𝐄input\Delta\mathbf{E}\_{\text{input}}. Therefore,

|  |  |  |
| --- | --- | --- |
|  | ‖Δ​𝒉n1‖≍ηinput​dmodel≍1.∎\|\Delta\bm{h}\_{n}^{1}\|\asymp\eta\_{\text{input}}\sqrt{d\_{\text{model}}}\asymp 1.\qed |  |

The next [lemma](#S6.theoremcnt4 "Lemma 6.4 (MLP). ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") deals with the stability of forward pass and updates in the MLP block.

###### Lemma 6.4 (MLP).

Suppose [˜6.1](#S6.theoremcnt1 "Assumption 6.1. ‣ Alignment ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") and [6.2](#S6.theoremcnt2 "Assumption 6.2. ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") hold,
the hyperparameters
su,inits\_{\text{$u$,init}} su,scales\_{\text{$u$,scale}}, sν,inits\_{\text{$\nu$,init}}, sν,scales\_{\text{$\nu$,scale}}
are positive constants,
and ηhidden≲dmodel−max⁡{αhidden,νhidden}\eta\_{\text{hidden}}\lesssim d\_{\text{model}}^{-\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}}.
In addition, suppose ‖Δ​𝒉n​(t)‖≍1\|\Delta\bm{h}\_{n}(t)\|\asymp 1.
Then the components666We assume implicitly that components of general vectors are of the same scale. (They in fact become i. i. d. random variables in the infinite-width limit, see e. g. Yang and Hu ([2021](#bib.bib39))). of MLP​(𝒉n)\text{MLP}(\bm{h}\_{n}) (at time 0) and Δ​MLP​(𝒉n)​(t)\Delta\text{MLP}(\bm{h}\_{n})(t) are both Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}).

###### Proof.

The scale of 𝐖u​𝒉n\mathbf{W}\_{u}\bm{h}\_{n} at initialization (no alignment) is given by

|  |  |  |
| --- | --- | --- |
|  | ‖𝐖u​𝒉n‖dMLP≍dmodel​‖𝐖u‖FdMLP​dmodel​‖𝒉n‖dmodel=dmodel−1/2,\frac{\|\mathbf{W}\_{u}\bm{h}\_{n}\|}{\sqrt{d\_{\text{MLP}}}}\asymp\sqrt{d\_{\text{model}}}\frac{\|\mathbf{W}\_{u}\|\_{F}}{\sqrt{d\_{\text{MLP}}d\_{\text{model}}}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=d\_{\text{model}}^{-1/2}, |  |

whereas the movement is

|  |  |  |
| --- | --- | --- |
|  | Δ​(𝐖u​𝒉n)=Δ​𝐖u​𝒉n+𝐖u​Δ​𝒉n+Δ​𝐖u​Δ​𝒉n,\Delta(\mathbf{W}\_{u}\bm{h}\_{n})=\Delta\mathbf{W}\_{u}\bm{h}\_{n}+\mathbf{W}\_{u}\Delta\bm{h}\_{n}+\Delta\mathbf{W}\_{u}\Delta\bm{h}\_{n}, |  |

and

|  |  |  |
| --- | --- | --- |
|  | ‖Δ​𝐖u​𝒉n‖dMLP≍dmodelαhidden​ηhidden​‖𝒉n‖dmodel≍dmodelαhidden−1/2​ηhidden;\frac{\|\Delta\mathbf{W}\_{u}\bm{h}\_{n}\|}{\sqrt{d\_{\text{MLP}}}}\asymp d\_{\text{model}}^{\alpha\_{\text{hidden}}}\eta\_{\text{hidden}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{\alpha\_{\text{hidden}}-1/2}\eta\_{\text{hidden}}; |  |

by ωhidden=1/2\omega\_{\text{hidden}}=1/2 (and using ‖Δ​𝒉n‖≍1\|\Delta\bm{h}\_{n}\|\asymp 1)

|  |  |  |
| --- | --- | --- |
|  | ‖𝐖u​Δ​𝒉n‖dMLP≍dmodel​‖𝐖u‖FdMLP​dmodel​‖Δ​𝒉n‖dmodel≍dmodel−1/2;\frac{\|\mathbf{W}\_{u}\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{MLP}}}}\asymp\sqrt{d\_{\text{model}}}\frac{\|\mathbf{W}\_{u}\|\_{F}}{\sqrt{d\_{\text{MLP}}d\_{\text{model}}}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{-1/2}; |  |

and

|  |  |  |
| --- | --- | --- |
|  | ‖Δ​𝐖u​Δ​𝒉n‖dMLP≍dmodelνhidden​ηhidden​‖Δ​𝒉n‖dmodel≍dmodelνhidden−1/2​ηhidden.\frac{\|\Delta\mathbf{W}\_{u}\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{MLP}}}}\asymp d\_{\text{model}}^{\nu\_{\text{hidden}}}\eta\_{\text{hidden}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{\nu\_{\text{hidden}}-1/2}\eta\_{\text{hidden}}. |  |

Since ηhidden≲dmodel−max⁡{αhidden,νhidden}\eta\_{\text{hidden}}\lesssim d\_{\text{model}}^{-\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}},
we see that the scales of components of 𝐖u​𝒉n\mathbf{W}\_{u}\bm{h}\_{n} and Δ​(𝐖u​𝒉n)\Delta(\mathbf{W}\_{u}\bm{h}\_{n}) stay Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}) during training.

The movement of 𝒖\bm{u} is given by

|  |  |  |
| --- | --- | --- |
|  | Δ​ui=Δ​(𝐖u​𝒉n)i​su,initsu,scale​su,i+(𝐖u​𝒉n)i​su,initsu,scale​Δ​su,i+Δ​(𝐖u​𝒉n)i​su,initsu,scale​Δ​su,i.\Delta u\_{i}=\Delta(\mathbf{W}\_{u}\bm{h}\_{n})\_{i}\frac{s\_{\text{$u$,init}}}{s\_{\text{$u$,scale}}}s\_{u,i}+(\mathbf{W}\_{u}\bm{h}\_{n})\_{i}\frac{s\_{\text{$u$,init}}}{s\_{\text{$u$,scale}}}\Delta s\_{u,i}+\Delta(\mathbf{W}\_{u}\bm{h}\_{n})\_{i}\frac{s\_{\text{$u$,init}}}{s\_{\text{$u$,scale}}}\Delta s\_{u,i}. |  |

Since at initialization su,i=su,scales\_{u,i}=s\_{\text{$u$,scale}},
the first term is of scale dmodel−1/2​su,init≍dmodel−1/2d\_{\text{model}}^{-1/2}s\_{\text{$u$,init}}\asymp d\_{\text{model}}^{-1/2};
the second and third are of scale su,initsu,scale​dmodel−1/2≍dmodel−1/2\frac{s\_{\text{$u$,init}}}{s\_{\text{$u$,scale}}}d\_{\text{model}}^{-1/2}\asymp d\_{\text{model}}^{-1/2}.

By the same logic as above, the scales of components of 𝐖ν​𝒉n\mathbf{W}\_{\nu}\bm{h}\_{n} and Δ​(𝐖ν​𝒉n)\Delta(\mathbf{W}\_{\nu}\bm{h}\_{n}) stay Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}) during training.
The update of 𝝂\bm{\nu} is given by

|  |  |  |
| --- | --- | --- |
|  | Δ​νi=Δ​(𝐖ν​𝒉n)i​sν,initsν,scale​dmodel1/2​sν,i+(𝐖ν​𝒉n)i​sν,initsν,scale​dmodel1/2​Δ​sν,i+Δ​(𝐖ν​𝒉n)i​sν,initsν,scale​dmodel1/2​Δ​sν,i,\Delta\nu\_{i}=\Delta(\mathbf{W}\_{\nu}\bm{h}\_{n})\_{i}\frac{s\_{\text{$\nu$,init}}}{s\_{\text{$\nu$,scale}}}d\_{\text{model}}^{1/2}s\_{\nu,i}+(\mathbf{W}\_{\nu}\bm{h}\_{n})\_{i}\frac{s\_{\text{$\nu$,init}}}{s\_{\text{$\nu$,scale}}}d\_{\text{model}}^{1/2}\Delta s\_{\nu,i}+\Delta(\mathbf{W}\_{\nu}\bm{h}\_{n})\_{i}\frac{s\_{\text{$\nu$,init}}}{s\_{\text{$\nu$,scale}}}d\_{\text{model}}^{1/2}\Delta s\_{\nu,i}, |  |

and the scale of each term is Θ​(1)\Theta(1).

We see that components of 𝝂\bm{\nu} and hence SiLU​(𝝂)\text{SiLU}(\bm{\nu}) are Θ​(1)\Theta(1), so the components of
SiLU​(𝝂)⊙𝒖\text{SiLU}(\bm{\nu})\odot\bm{u} are Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}) along with their updates during training.

Proceeding with the same logic (and the same alignment assumptions), we see that
the components of MLP​(𝒉n)\text{MLP}(\bm{h}\_{n}) along with their updates stay
Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}) during training.
∎

###### Remark 6.5.

For baseline nGPT, the same calculation shows that components of 𝒖\bm{u} are Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2})
at initialization, components of Δ​𝒖\Delta\bm{u} are Θ​(max⁡{dmodel1/2​ηglobal,dmodel−1/2})\Theta\big\lparen\max\{d\_{\text{model}}^{1/2}\eta\_{\text{global}},d\_{\text{model}}^{-1/2}\}\big\rparen during training,
whereas su,initsu,scale​Δ​su,i\frac{s\_{\text{$u$,init}}}{s\_{\text{$u$,scale}}}\Delta s\_{u,i} move at scale ηglobal\eta\_{\text{global}}.
This means that the movement of su,initsu,scale​Δ​su,i\frac{s\_{\text{$u$,init}}}{s\_{\text{$u$,scale}}}\Delta s\_{u,i} becomes negligible at large width.
The same can be said about sν,initsν,scale​Δ​sν,i\frac{s\_{\text{$\nu$,init}}}{s\_{\text{$\nu$,scale}}}\Delta s\_{\nu,i}.

The next [lemma](#S6.theoremcnt6 "Lemma 6.6 (Attention). ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") checks the scaling inside attention blocks.

###### Lemma 6.6 (Attention).

Suppose [˜6.1](#S6.theoremcnt1 "Assumption 6.1. ‣ Alignment ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") and [6.2](#S6.theoremcnt2 "Assumption 6.2. ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") hold,
the hyperparameters
sq​k,inits\_{\text{$qk$,init}}, sq​k,scales\_{\text{$qk$,scale}}
are positive constants,
and ηhidden≲dmodel−max⁡{αhidden,νhidden}\eta\_{\text{hidden}}\lesssim d\_{\text{model}}^{-\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}}.
In addition, suppose ‖Δ​𝒉n​(t)‖≍1\|\Delta\bm{h}\_{n}(t)\|\asymp 1.
Assume that dkeyd\_{\text{key}} is large enough so that the approximation Δ​qn,i‖𝒒n‖≍Δ​qn,i‖𝒒n‖\Delta\frac{q\_{n,i}}{\|\bm{q}\_{n}\|}\asymp\frac{\Delta q\_{n,i}}{\|\bm{q}\_{n}\|} holds.
Then the components of 𝒒n′\bm{q}^{\prime}\_{n} (at time 0) and Δ​𝒒n′​(t)\Delta\bm{q}\_{n}^{\prime}(t) are both Θ​(dkey−1/2)\Theta(d\_{\text{key}}^{-1/2}), where 𝒒n′\bm{q}^{\prime}\_{n} is given by [equation˜6](#S2.E6 "In 2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers");
the components of 𝒌m′\bm{k}^{\prime}\_{m} and Δ​𝒌m′​(t)\Delta\bm{k}\_{m}^{\prime}(t) are both Θ​(dkey−1/2)\Theta(d\_{\text{key}}^{-1/2}), where 𝒌m′\bm{k}^{\prime}\_{m} is given by [equation˜7](#S2.E7 "In 2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers");
the components of 𝒗m\bm{v}\_{m} and Δ​𝒗m​(t)\Delta\bm{v}\_{m}(t) are both Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}), where 𝒗m\bm{v}\_{m} is given by [equation˜8](#S2.E8 "In 2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers").
Further, the quantity dkey​𝒒n′⁣𝖳​𝒌m′\sqrt{d\_{\text{key}}}\bm{q}^{\prime\mkern-1.5mu\mathsf{T}}\_{n}\bm{k}^{\prime}\_{m} is Θ​(1)\Theta(1) at initialization,
whereas its movement dkey​Δ​(𝒒n′⁣𝖳​𝒌m′)\sqrt{d\_{\text{key}}}\Delta\big\lparen\bm{q}^{\prime\mkern-1.5mu\mathsf{T}}\_{n}\bm{k}^{\prime}\_{m}\big\rparen is O​(dkey)O(\sqrt{d\_{\text{key}}}).
Finally, if dkeyd\_{\text{key}} is constant, the components of the attention output
Attentionn​({𝒉m}m=1SeqLen)\text{Attention}\_{n}(\{\bm{h}\_{m}\}\_{m=1}^{\text{SeqLen}}) (given by [equation˜9](#S2.E9 "In 2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers"))
and of their movements Δ​Attentionn​({𝒉m}m=1SeqLen)​(t)\Delta\text{Attention}\_{n}(\{\bm{h}\_{m}\}\_{m=1}^{\text{SeqLen}})(t)
are both Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}).

###### Remark.

In μ\muP Yang et al. ([2021](#bib.bib35)), the multiplier of key-query scalar product is changed because of alignment between keys and queries during training.
Here, such alignment would correspond to Δ​(𝒒n′⁣𝖳​𝒌m′)=Θ​(1)\Delta\big\lparen\bm{q}^{\prime\mkern-1.5mu\mathsf{T}}\_{n}\bm{k}^{\prime}\_{m}\big\rparen=\Theta(1).
If dkeyd\_{\text{key}} were not constant, we would need to remove the multiplier dkey\sqrt{d\_{\text{key}}}
in attention
to prevent blowup.

###### Proof.

For simplicity, we ignore rotary embedding maps here because they do not change the relevant scales.
The scale of a query vector at initialization (no alignment) is

|  |  |  |
| --- | --- | --- |
|  | ‖𝒒n‖dkey=dmodel​‖𝐖q‖Fdkey​dmodel​‖𝒉n‖dmodel=dmodel−1/2.\frac{\|\bm{q}\_{n}\|}{\sqrt{d\_{\text{key}}}}=\sqrt{d\_{\text{model}}}\frac{\|\mathbf{W}\_{q}\|\_{F}}{\sqrt{d\_{\text{key}}d\_{\text{model}}}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=d\_{\text{model}}^{-1/2}. |  |

The movement at time tt is given by

|  |  |  |
| --- | --- | --- |
|  | Δ​𝒒n=Δ​𝐖q𝖳​𝒉n+𝐖q𝖳​Δ​𝒉n+Δ​𝐖q𝖳​Δ​𝒉n,\Delta\bm{q}\_{n}=\Delta\mathbf{W}\_{q}^{\mkern-1.5mu\mathsf{T}}\bm{h}\_{n}+\mathbf{W}\_{q}^{\mkern-1.5mu\mathsf{T}}\Delta\bm{h}\_{n}+\Delta\mathbf{W}\_{q}^{\mkern-1.5mu\mathsf{T}}\Delta\bm{h}\_{n}, |  |

and by the definition of αhidden\alpha\_{\text{hidden}}

|  |  |  |
| --- | --- | --- |
|  | ‖Δ​𝐖q𝖳​𝒉n‖dkey≍dmodelαhidden​ηhidden​‖𝒉n‖dmodel=dmodelαhidden−1/2​ηhidden;\frac{\|\Delta\mathbf{W}\_{q}^{\mkern-1.5mu\mathsf{T}}\bm{h}\_{n}\|}{\sqrt{d\_{\text{key}}}}\asymp d\_{\text{model}}^{\alpha\_{\text{hidden}}}\eta\_{\text{hidden}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=d\_{\text{model}}^{\alpha\_{\text{hidden}}-1/2}\eta\_{\text{hidden}}; |  |

by ωhidden=1/2\omega\_{\text{hidden}}=1/2 (and using ‖Δ​𝒉n‖≍1\|\Delta\bm{h}\_{n}\|\asymp 1)

|  |  |  |
| --- | --- | --- |
|  | ‖𝐖q𝖳​Δ​𝒉n‖dkey≍dmodel​‖𝐖q‖Fdkey​dmodel​‖Δ​𝒉n‖dmodel≍dmodel−1/2;\frac{\|\mathbf{W}\_{q}^{\mkern-1.5mu\mathsf{T}}\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{key}}}}\asymp\sqrt{d\_{\text{model}}}\frac{\|\mathbf{W}\_{q}\|\_{F}}{\sqrt{d\_{\text{key}}d\_{\text{model}}}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{-1/2}; |  |

by the definition of νhidden\nu\_{\text{hidden}} (again using ‖Δ​𝒉n‖≍1\|\Delta\bm{h}\_{n}\|\asymp 1)

|  |  |  |
| --- | --- | --- |
|  | ‖Δ​𝐖q𝖳​Δ​𝒉n‖dkey≍dmodelνhidden​ηhidden​‖Δ​𝒉n‖dmodel≍dmodelνhidden−1/2​ηhidden.\frac{\|\Delta\mathbf{W}\_{q}^{\mkern-1.5mu\mathsf{T}}\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{key}}}}\asymp d\_{\text{model}}^{\nu\_{\text{hidden}}}\eta\_{\text{hidden}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{\nu\_{\text{hidden}}-1/2}\eta\_{\text{hidden}}. |  |

Since ηhidden≲dmodel−max⁡{αhidden,νhidden}\eta\_{\text{hidden}}\lesssim d\_{\text{model}}^{-\max\{\alpha\_{\text{hidden}},\nu\_{\text{hidden}}\}},
we see that the scales of components of 𝒒n\bm{q}\_{n} and Δ​𝒒n\Delta\bm{q}\_{n} are Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}) during training.

The update of 𝒒n′\bm{q}^{\prime}\_{n} is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​qn,i′=(Δ​qn,i‖𝒒n‖)​sq​k,initsq​k,scale​sq​k,i+qn,i‖𝒒n‖​sq​k,initsq​k,scale​Δ​sq​k,i+(Δ​qn,i‖𝒒n‖)​sq​k,initsq​k,scale​Δ​sq​k,i,\Delta q\_{n,i}^{\prime}=\bigg\lparen\Delta\frac{q\_{n,i}}{\|\bm{q}\_{n}\|}\bigg\rparen\frac{s\_{\text{$qk$,init}}}{s\_{\text{$qk$,scale}}}s\_{qk,i}+\frac{q\_{n,i}}{\|\bm{q}\_{n}\|}\frac{s\_{\text{$qk$,init}}}{s\_{\text{$qk$,scale}}}\Delta s\_{qk,i}+\bigg\lparen\Delta\frac{q\_{n,i}}{\|\bm{q}\_{n}\|}\bigg\rparen\frac{s\_{\text{$qk$,init}}}{s\_{\text{$qk$,scale}}}\Delta s\_{qk,i}, |  | (12) |

where

|  |  |  |
| --- | --- | --- |
|  | Δ​qn,i≍dmodel−1/2,‖𝒒n‖≍dkey1/2​dmodel−1/2⇒Δ​qn,i‖𝒒n‖=dkey−1/2\Delta q\_{n,i}\asymp d\_{\text{model}}^{-1/2},\quad\|\bm{q}\_{n}\|\asymp d\_{\text{key}}^{1/2}d\_{\text{model}}^{-1/2}\quad\Rightarrow\quad\frac{\Delta q\_{n,i}}{\|\bm{q}\_{n}\|}=d\_{\text{key}}^{-1/2} |  |

and

|  |  |  |
| --- | --- | --- |
|  | qn,i‖𝒒n‖≍dkey−1/2.\frac{q\_{n,i}}{\|\bm{q}\_{n}\|}\asymp d\_{\text{key}}^{-1/2}. |  |

Then, using Δ​qn,i‖𝒒n‖≍Δ​qn,i‖𝒒n‖\Delta\frac{q\_{n,i}}{\|\bm{q}\_{n}\|}\asymp\frac{\Delta q\_{n,i}}{\|\bm{q}\_{n}\|}, we see that each of the three terms in [equation˜12](#S6.E12 "In Proof. ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") is Θ​(dkey−1/2)\Theta(d\_{\text{key}}^{-1/2}).

The claims about 𝒌m′\bm{k}\_{m}^{\prime} and 𝒗m\bm{v}\_{m} are proven similarly.

At initialization, 𝒒n′\bm{q}^{\prime}\_{n} and 𝒌m′\bm{k}^{\prime}\_{m} are unaligned, so that

|  |  |  |
| --- | --- | --- |
|  | |𝒒n′⁣𝖳​𝒌m′|≍‖𝒒n′‖​‖𝒌m′‖dkey≍1dkey.|\bm{q}^{\prime\mkern-1.5mu\mathsf{T}}\_{n}\bm{k}^{\prime}\_{m}|\asymp\frac{\|\bm{q}^{\prime}\_{n}\|\|\bm{k}^{\prime}\_{m}\|}{\sqrt{d\_{\text{key}}}}\asymp\frac{1}{\sqrt{d\_{\text{key}}}}. |  |

During training, they may become aligned, and the upper bound
|Δ​(𝒒n′⁣𝖳​𝒌m′)|≲1\big|\Delta\big\lparen\bm{q}^{\prime\mkern-1.5mu\mathsf{T}}\_{n}\bm{k}^{\prime}\_{m}\big\rparen\big|\lesssim 1
follows from the Cauchy-Schwarz inequality.

We conclude that, if dkeyd\_{\text{key}} is constant, the components of each head’s output and their movements
are Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}). Hence, the components of Attentionn​({𝒉m}m=1SeqLen)\text{Attention}\_{n}(\{\bm{h}\_{m}\}\_{m=1}^{\text{SeqLen}}) and Δ​Attentionn​({𝒉m}m=1SeqLen)​(t)\Delta\text{Attention}\_{n}(\{\bm{h}\_{m}\}\_{m=1}^{\text{SeqLen}})(t) are both Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}) by the same argument as in [lemma˜6.4](#S6.theoremcnt4 "Lemma 6.4 (MLP). ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers").
∎

The next lemma deals with the LERP after Attention or MLP blocks.

###### Lemma 6.7 (LERP; learning rates of 𝜶A\bm{\alpha}\_{A} and 𝜶M\bm{\alpha}\_{M}).

Suppose [˜6.1](#S6.theoremcnt1 "Assumption 6.1. ‣ Alignment ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") and [6.2](#S6.theoremcnt2 "Assumption 6.2. ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") hold and the hyperparameter αA,scale\alpha\_{\text{$A$,scale}}
is a positive constant. In addition, suppose Δ​hA,n,i​(t)≍dmodel−1/2\Delta h\_{A,n,i}(t)\asymp d\_{\text{model}}^{-1/2}.
Then each component of αA,initαA,scale​𝜶A⊙𝒉A,n\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}\bm{\alpha}\_{A}\odot\bm{h}\_{A,n}
(at initialization)
and its movement at time tt are both of scale Θ​(αA,init​dmodel−1/2)\Theta(\alpha\_{\text{$A$,init}}d\_{\text{model}}^{-1/2}).

###### Proof.

The scale of each component in αA,initαA,scale​𝜶A⊙𝒉A,n\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}\bm{\alpha}\_{A}\odot\bm{h}\_{A,n} at initialization
is αA,init​dmodel−1/2\alpha\_{\text{$A$,init}}d\_{\text{model}}^{-1/2}.
The update is given by

|  |  |  |
| --- | --- | --- |
|  | αA,initαA,scale​Δ​αA,i​hA,n,i+αA,initαA,scale​αA,i​Δ​hA,n,i+αA,initαA,scale​Δ​αA,i​Δ​hA,n,i\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}\Delta\alpha\_{A,i}h\_{A,n,i}+\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}\alpha\_{A,i}\Delta h\_{A,n,i}+\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}\Delta\alpha\_{A,i}\Delta h\_{A,n,i} |  |

where

|  |  |  |
| --- | --- | --- |
|  | Δ​hA,n,i≍dmodel−1/2,hA,n,i≍dmodel−1/2.\Delta h\_{A,n,i}\asymp d\_{\text{model}}^{-1/2},\quad h\_{A,n,i}\asymp d\_{\text{model}}^{-1/2}. |  |

Assuming αA,i≍αA,scale\alpha\_{A,i}\asymp\alpha\_{\text{$A$,scale}},
we have

|  |  |  |
| --- | --- | --- |
|  | αA,initαA,scale​Δ​αA,i​hA,n,i≍αA,initαA,scale​dmodel−1/2​Δ​αA,i,\displaystyle\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}\Delta\alpha\_{A,i}h\_{A,n,i}\asymp\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}d\_{\text{model}}^{-1/2}\Delta\alpha\_{A,i}, |  |
|  |  |  |
| --- | --- | --- |
|  | αA,initαA,scale​αA,i​Δ​hA,n,i≍αA,init​dmodel−1/2,\displaystyle\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}\alpha\_{A,i}\Delta h\_{A,n,i}\asymp\alpha\_{\text{$A$,init}}d\_{\text{model}}^{-1/2}, |  |
|  |  |  |
| --- | --- | --- |
|  | αA,initαA,scale​Δ​αA,i​Δ​hA,n,i≍αA,initαA,scale​dmodel−1/2​Δ​αA,i.∎\displaystyle\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}\Delta\alpha\_{A,i}\Delta h\_{A,n,i}\asymp\frac{\alpha\_{\text{$A$,init}}}{\alpha\_{\text{$A$,scale}}}d\_{\text{model}}^{-1/2}\Delta\alpha\_{A,i}.\qed |  |

For logits scaling,
there are two possible choices.
If we assume ωoutput=1/2\omega\_{\text{output}}=1/2 (Everett et al., [2024](#bib.bib11)),
it is possible for logits to be the same scale at initialization
as during training, although this requires sz,init≍dmodel1/2s\_{\text{$z$,init}}\asymp d\_{\text{model}}^{1/2}.
On the other hand, in classical μ\muP parametrizations, the logits scale as 1/dmodel1/\sqrt{d\_{\text{model}}} at initialization but move Θ​(1)\Theta(1).

###### Lemma 6.8 (Same scaling of logits at initialization and during training).

Suppose [˜6.1](#S6.theoremcnt1 "Assumption 6.1. ‣ Alignment ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") and [6.2](#S6.theoremcnt2 "Assumption 6.2. ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") hold,
ωoutput=1/2\omega\_{\text{output}}=1/2, ηoutput≲dmodel−max⁡{αoutput,νoutput}\eta\_{\text{output}}\lesssim d\_{\text{model}}^{-\max\{\alpha\_{\text{output}},\nu\_{\text{output}}\}}, sz,init≍dmodel1/2s\_{\text{$z$,init}}\asymp d\_{\text{model}}^{1/2},
and sz,scales\_{\text{$z$,scale}} is a positive constant.
Suppose further ‖Δ​𝒉n​(t)‖≍1\|\Delta\bm{h}\_{n}(t)\|\asymp 1.
Then each component of 𝒛n\bm{z}\_{n} (at initialization) and Δ​𝒛n​(t)\Delta\bm{z}\_{n}(t) are both of scale Θ​(1)\Theta(1), where 𝒛n\bm{z}\_{n} is defined by [equation˜5](#S2.E5 "In 2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers").

###### Proof.

The scale of 𝒛^n∈ℝV\hat{\bm{z}}\_{n}\in\mathbb{R}^{V} at initialization (no alignment) is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | z^n,i≍‖𝒛^n‖V≍dmodel​‖𝐄output‖Fdmodel​V​‖𝒉n‖dmodel=‖𝐄output‖Fdmodel​V=dmodel−1/2.\hat{z}\_{n,i}\asymp\frac{\|\hat{\bm{z}}\_{n}\|}{\sqrt{V}}\asymp\sqrt{d\_{\text{model}}}\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{d\_{\text{model}}V}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{d\_{\text{model}}V}}=d\_{\text{model}}^{-1/2}. |  | (13) |

The update of 𝒛^n\hat{\bm{z}}\_{n} is given by

|  |  |  |
| --- | --- | --- |
|  | Δ​𝒛^n=Δ​𝐄output​𝒉n+𝐄output​Δ​𝒉n+Δ​𝐄output​Δ​𝒉n.\Delta\hat{\bm{z}}\_{n}=\Delta\mathbf{E}\_{\text{output}}\bm{h}\_{n}+\mathbf{E}\_{\text{output}}\Delta\bm{h}\_{n}+\Delta\mathbf{E}\_{\text{output}}\Delta\bm{h}\_{n}. |  |

The scale of the first perturbation term is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Δ​𝐄output​𝒉n‖V≍dmodelαoutput​ηoutput​‖𝒉n‖dmodel=dmodelαoutput−1/2​ηoutput;\frac{\|\Delta\mathbf{E}\_{\text{output}}\bm{h}\_{n}\|}{\sqrt{V}}\asymp d\_{\text{model}}^{\alpha\_{\text{output}}}\eta\_{\text{output}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=d\_{\text{model}}^{\alpha\_{\text{output}}-1/2}\eta\_{\text{output}}; |  | (14) |

of the second perturbation term (no alignment ωoutput=1/2\omega\_{\text{output}}=1/2 and ‖Δ​𝒉n‖≍1\|\Delta\bm{h}\_{n}\|\asymp 1)

|  |  |  |
| --- | --- | --- |
|  | ‖𝐄output​Δ​𝒉n‖V≍dmodel​‖𝐄output‖FV​dmodel​‖Δ​𝒉n‖dmodel=‖𝐄output‖FV​dmodel=dmodel−1/2;\frac{\|\mathbf{E}\_{\text{output}}\Delta\bm{h}\_{n}\|}{\sqrt{V}}\asymp\sqrt{d\_{\text{model}}}\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{Vd\_{\text{model}}}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{Vd\_{\text{model}}}}=d\_{\text{model}}^{-1/2}; |  |

and of the third perturbation term (using ‖Δ​𝒉n‖≍1\|\Delta\bm{h}\_{n}\|\asymp 1)

|  |  |  |
| --- | --- | --- |
|  | ‖Δ​𝐄output​Δ​𝒉n‖V≍dmodelνoutput​ηoutput​‖Δ​𝒉n‖dmodel=dmodelνoutput−1/2​ηoutput.\frac{\|\Delta\mathbf{E}\_{\text{output}}\Delta\bm{h}\_{n}\|}{\sqrt{V}}\asymp d\_{\text{model}}^{\nu\_{\text{output}}}\eta\_{\text{output}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=d\_{\text{model}}^{\nu\_{\text{output}}-1/2}\eta\_{\text{output}}. |  |

Then, each component of both 𝒛^n\hat{\bm{z}}\_{n} (at initialization) and Δ​𝒛^n​(t)\Delta\hat{\bm{z}}\_{n}(t) is of scale dmodel−1/2d\_{\text{model}}^{-1/2} during training.

The update of 𝒛^n⊙sz,initsz,scale​𝒔z\hat{\bm{z}}\_{n}\odot\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}\bm{s}\_{z} is given by

|  |  |  |
| --- | --- | --- |
|  | Δ​(z^n,i​sz,initsz,scale​sz,i)=Δ​z^n,i​sz,initsz,scale​sz,i+z^n,i​sz,initsz,scale​Δ​sz,i+sz,initsz,scale​Δ​z^n,i​Δ​sz,i.\Delta\bigg\lparen\hat{z}\_{n,i}\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}s\_{z,i}\bigg\rparen=\Delta\hat{z}\_{n,i}\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}s\_{z,i}+\hat{z}\_{n,i}\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}\Delta s\_{z,i}+\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}\Delta\hat{z}\_{n,i}\Delta s\_{z,i}. |  |

Each term is of scale sz,init​dmodel−1/2≍1s\_{\text{$z$,init}}d\_{\text{model}}^{-1/2}\asymp 1.
∎

Finally,
classical μ\muP leads to logits being negligible at initialization
but not during training.
Moreover, μ\muP would be the only possible (stable and nontrivial) choice if we assumed ωoutput=1\omega\_{\text{output}}=1.
Width-wise learning rate transfer in μ\muP is a robust empirical observation,
making it a safe choice.

###### Lemma 6.9 (Logits are smaller at initialization than during training, e. g. classical μ\muP).

Suppose [˜6.1](#S6.theoremcnt1 "Assumption 6.1. ‣ Alignment ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") and [6.2](#S6.theoremcnt2 "Assumption 6.2. ‣ Learning rate of “rescalers” ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") hold,
ηoutput≍dmodel1/2−max⁡{αoutput,νoutput}\eta\_{\text{output}}\asymp d\_{\text{model}}^{1/2-\max\{\alpha\_{\text{output}},\nu\_{\text{output}}\}}, sz,inits\_{\text{$z$,init}}
and sz,scales\_{\text{$z$,scale}} are positive constants.
Suppose further ‖Δ​𝒉n​(t)‖≍1\|\Delta\bm{h}\_{n}(t)\|\asymp 1.
Then, with no assumption on ωoutput\omega\_{\text{output}}, each component of 𝒛n\bm{z}\_{n} (at initialization)
is Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2})
but Δ​𝒛n​(t)\Delta\bm{z}\_{n}(t) is of scale Θ​(1)\Theta(1), where 𝒛n\bm{z}\_{n} is defined by [equation˜5](#S2.E5 "In 2 Normalized Transformers: definitions ‣ Learning Rate Transfer in Normalized Transformers").

###### Proof.

The scale of 𝒛^n∈ℝV\hat{\bm{z}}\_{n}\in\mathbb{R}^{V} at initialization (no alignment) is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | z^n,i≍‖𝒛^n‖V≍dmodel​‖𝐄output‖Fdmodel​V​‖𝒉n‖dmodel=‖𝐄output‖Fdmodel​V=dmodel−1/2.\hat{z}\_{n,i}\asymp\frac{\|\hat{\bm{z}}\_{n}\|}{\sqrt{V}}\asymp\sqrt{d\_{\text{model}}}\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{d\_{\text{model}}V}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{d\_{\text{model}}V}}=d\_{\text{model}}^{-1/2}. |  | (15) |

The update of 𝒛^n\hat{\bm{z}}\_{n} is given by

|  |  |  |
| --- | --- | --- |
|  | Δ​𝒛^n=Δ​𝐄output​𝒉n+𝐄output​Δ​𝒉n+Δ​𝐄output​Δ​𝒉n.\Delta\hat{\bm{z}}\_{n}=\Delta\mathbf{E}\_{\text{output}}\bm{h}\_{n}+\mathbf{E}\_{\text{output}}\Delta\bm{h}\_{n}+\Delta\mathbf{E}\_{\text{output}}\Delta\bm{h}\_{n}. |  |

The scale of the first perturbation term is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Δ​𝐄output​𝒉n‖V≍dmodelαoutput​ηoutput​‖𝒉n‖dmodel=dmodelαoutput−1/2​ηoutput;\frac{\|\Delta\mathbf{E}\_{\text{output}}\bm{h}\_{n}\|}{\sqrt{V}}\asymp d\_{\text{model}}^{\alpha\_{\text{output}}}\eta\_{\text{output}}\frac{\|\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=d\_{\text{model}}^{\alpha\_{\text{output}}-1/2}\eta\_{\text{output}}; |  | (16) |

of the second perturbation term (using ‖Δ​𝒉n‖≍1\|\Delta\bm{h}\_{n}\|\asymp 1)

|  |  |  |
| --- | --- | --- |
|  | ‖𝐄output​Δ​𝒉n‖V≍dmodelωoutput​‖𝐄output‖FV​dmodel​‖Δ​𝒉n‖dmodel≍dmodelωoutput−1/2​‖𝐄output‖FV​dmodel≍dmodelωoutput−1;\frac{\|\mathbf{E}\_{\text{output}}\Delta\bm{h}\_{n}\|}{\sqrt{V}}\asymp d\_{\text{model}}^{\omega\_{\text{output}}}\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{Vd\_{\text{model}}}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}\asymp d\_{\text{model}}^{\omega\_{\text{output}}-1/2}\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{Vd\_{\text{model}}}}\asymp d\_{\text{model}}^{\omega\_{\text{output}}-1}; |  |

and of the third perturbation term (‖Δ​𝒉n‖≍1\|\Delta\bm{h}\_{n}\|\asymp 1)

|  |  |  |
| --- | --- | --- |
|  | ‖Δ​𝐄output​Δ​𝒉n‖V≲dmodelνoutput​ηoutput​‖Δ​𝒉n‖dmodel=dmodelνoutput−1/2​ηoutput.\frac{\|\Delta\mathbf{E}\_{\text{output}}\Delta\bm{h}\_{n}\|}{\sqrt{V}}\lesssim d\_{\text{model}}^{\nu\_{\text{output}}}\eta\_{\text{output}}\frac{\|\Delta\bm{h}\_{n}\|}{\sqrt{d\_{\text{model}}}}=d\_{\text{model}}^{\nu\_{\text{output}}-1/2}\eta\_{\text{output}}. |  |

Thus, the (pre-)logits start much smaller than they are updated.
From the second forward pass onwards, the scale of (pre-)logits becomes
the (dominant) scale of their updates, that is,

|  |  |  |
| --- | --- | --- |
|  | z^n,i≍dmodel−1/2butz^n,i+Δ​z^n,i≍1.\hat{z}\_{n,i}\asymp d\_{\text{model}}^{-1/2}\quad\text{but}\quad\hat{z}\_{n,i}+\Delta\hat{z}\_{n,i}\asymp 1. |  |

The movement of 𝒛^n⊙sz,initsz,scale​𝒔z\hat{\bm{z}}\_{n}\odot\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}\bm{s}\_{z} is given by

|  |  |  |
| --- | --- | --- |
|  | Δ​(z^n,i​sz,initsz,scale​sz,i)=Δ​z^n,i​sz,initsz,scale​sz,i+z^n,i​sz,initsz,scale​Δ​sz,i+sz,initsz,scale​Δ​z^n,i​Δ​sz,i.\Delta\bigg\lparen\hat{z}\_{n,i}\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}s\_{z,i}\bigg\rparen=\Delta\hat{z}\_{n,i}\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}s\_{z,i}+\hat{z}\_{n,i}\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}\Delta s\_{z,i}+\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}}\Delta\hat{z}\_{n,i}\Delta s\_{z,i}. |  |

The first term is of scale sz,inits\_{\text{$z$,init}},
the second of scale dmodel−1/2​sz,initsz,scaled\_{\text{model}}^{-1/2}\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}},
and the third of scale sz,initsz,scale\frac{s\_{\text{$z$,init}}}{s\_{\text{$z$,scale}}} (always dominating the second term).
The fact that sz,scale≍1s\_{\text{$z$,scale}}\asymp 1 ensures matching scales of the first and third terms
(ensuring that the movement in 𝒔z​(t)\bm{s}\_{z}(t) is not negligible but does not dominate the dynamics).
∎

### 6.2 Depth scaling

As common in the literature (Yang et al., [2024b](#bib.bib42); Dey et al., [2025](#bib.bib10)), we consider a simplification of the network to derive depth
corrections.

###### Definition 6.10 (Simple normalized network).

The linear normalized network maps a one-hot vector 𝒙∈ℝV\bm{x}\in\mathbb{R}^{V} to logits 𝒛​(𝒙;t)∈ℝV\bm{z}(\bm{x};t)\in\mathbb{R}^{V} as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒉1​(𝒙;t)\displaystyle\bm{h}^{1}(\bm{x};t) | =𝐄input​(t)​𝒙,\displaystyle=\mathbf{E}\_{\text{input}}(t)\bm{x}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒉^ℓ+1​(𝒙;t)\displaystyle\hat{\bm{h}}^{\ell+1}(\bm{x};t) | =(1−L−α)​𝒉ℓ​(𝒙;t)+L−α​Norm​(𝐖ℓ​(t)​𝒉ℓ​(𝒙;t)),\displaystyle=\big\lparen 1-L^{-\alpha}\big\rparen\bm{h}^{\ell}(\bm{x};t)+L^{-\alpha}\text{Norm}\big\lparen\mathbf{W}^{\ell}(t)\bm{h}^{\ell}(\bm{x};t)\big\rparen, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒉ℓ+1​(𝒙;t)\displaystyle\bm{h}^{\ell+1}(\bm{x};t) | =Norm​(𝒉^ℓ+1​(𝒙;t)),ℓ=1,…,L−1,\displaystyle=\text{Norm}\big\lparen\hat{\bm{h}}^{\ell+1}(\bm{x};t)\big\rparen,\quad\ell=1,\ldots,L-1, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒛​(𝒙;t)\displaystyle\bm{z}(\bm{x};t) | =𝐄output​(t)​𝒉L​(𝒙;t)\displaystyle=\mathbf{E}\_{\text{output}}(t)\bm{h}^{L}(\bm{x};t) |  |

with each matrix 𝐄input​(t)∈ℝN×V\mathbf{E}\_{\text{input}}(t)\in\mathbb{R}^{N\times V}, 𝐖ℓ​(t)∈ℝN×N\mathbf{W}^{\ell}(t)\in\mathbb{R}^{N\times N} for ℓ∈[1:L−1]\ell\in[1:L-1], 𝐄output​(t)∈ℝV×N\mathbf{E}\_{\text{output}}(t)\in\mathbb{R}^{V\times N} initialized with i. i. d. Gaussian components and normalized along the embedding dimension (columns for 𝐄input\mathbf{E}\_{\text{input}}, rows for 𝐖ℓ\mathbf{W}^{\ell} and 𝐄output\mathbf{E}\_{\text{output}}) before each training step, where α>0\alpha>0 is the “depth alpha” exponent.

The signGD updates Δ​𝐄input\Delta\mathbf{E}\_{\text{input}}, Δ​𝐖ℓ\Delta\mathbf{W}^{\ell} and Δ​𝐄output\Delta\mathbf{E}\_{\text{output}} consist of components ±ηinput\pm\eta\_{\text{input}}, ±ηhidden\pm\eta\_{\text{hidden}} and ±ηoutput\pm\eta\_{\text{output}} respectively.

#### Notation

We will treat the width N≡dmodelN\equiv d\_{\text{model}} and depth LL as varying (and think of them as large), whereas VV is a fixed constant dimension.
Similarly to [section˜6.1](#S6.SS1 "6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers"),
the notation aN,L≲bN,La\_{N,L}\lesssim b\_{N,L} or aN,L=O​(bN,L)a\_{N,L}=O(b\_{N,L})
means lim supN→∞logN⁡|aN,L/bN,L|≤0\limsup\_{N\to\infty}\log\_{N}|a\_{N,L}/b\_{N,L}|\leq 0 for all LL (!),
whereas aN,L≍bN,La\_{N,L}\asymp b\_{N,L} or aN,L=Θ​(bN,L)a\_{N,L}=\Theta(b\_{N,L}) means aN,L≲bN,La\_{N,L}\lesssim b\_{N,L} and bN,L≲aN,Lb\_{N,L}\lesssim a\_{N,L}.

First, stability at initialization ([˜1](#Thmdesideratum1 "Desideratum 1 (Stability at initialization). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) is automatic because the network is normalized. Indeed, since 𝒙\bm{x} is one-hot, 𝐄input​𝒙\mathbf{E}\_{\text{input}}\bm{x}
is a column of 𝐄input\mathbf{E}\_{\text{input}}, which is normalized before each training step, so ‖𝒉1‖=1\|\bm{h}^{1}\|=1.
Next, ‖𝒉ℓ‖=1\|\bm{h}^{\ell}\|=1 for ℓ∈[2:L]\ell\in[2:L] by definition.
Finally, at initialization 𝐄output\mathbf{E}\_{\text{output}} is unaligned with 𝒉L\bm{h}^{L}, so

|  |  |  |
| --- | --- | --- |
|  | ‖𝒛‖≍‖𝒛‖V≍N​‖𝐄output‖FN​V​‖𝒉L‖N=‖𝐄output‖FN​V=N−1/2=O​(1).\|\bm{z}\|\asymp\frac{\|\bm{z}\|}{\sqrt{V}}\asymp\sqrt{N}\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{NV}}\frac{\|\bm{h}^{L}\|}{\sqrt{N}}=\frac{\|\mathbf{E}\_{\text{output}}\|\_{F}}{\sqrt{NV}}=N^{-1/2}=O(1). |  |

Next, we deal with [˜2](#Thmdesideratum2 "Desideratum 2 (Stable non-trivial feature learning). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers"). Consider the linearized update

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​hkℓ+1=∑l=1ℓ∑i​j∂hkℓ+1∂Wi​jl​Δ​Wi​jl+∑i​j∂hkℓ+1∂Einput,i​j​Δ​Einput,i​j+O​(ηhidden2+ηinput2).\Delta h\_{k}^{\ell+1}=\sum\_{l=1}^{\ell}\sum\_{ij}\frac{\partial h\_{k}^{\ell+1}}{\partial W\_{ij}^{l}}\Delta W\_{ij}^{l}+\sum\_{ij}\frac{\partial h\_{k}^{\ell+1}}{\partial E\_{\text{input},ij}}\Delta E\_{\text{input},ij}+O(\eta\_{\text{hidden}}^{2}+\eta\_{\text{input}}^{2}). |  | (17) |

Note that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i​j∂hkℓ+1∂Wi​jl​Δ​Wi​jl\displaystyle\sum\_{ij}\frac{\partial h\_{k}^{\ell+1}}{\partial W\_{ij}^{l}}\Delta W^{l}\_{ij} | =∑i​j∑m∂hkℓ+1∂hml+1​∂hml+1∂Wi​jl​Δ​Wi​jl\displaystyle=\sum\_{ij}\sum\_{m}\frac{\partial h\_{k}^{\ell+1}}{\partial h\_{m}^{l+1}}\frac{\partial h\_{m}^{l+1}}{\partial W\_{ij}^{l}}\Delta W\_{ij}^{l} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i​j∑m∂hkℓ+1∂hml+1​∂hml+1∂(𝐖l​𝒉l)i​hjl​Δ​Wi​jl\displaystyle=\sum\_{ij}\sum\_{m}\frac{\partial h\_{k}^{\ell+1}}{\partial h\_{m}^{l+1}}\frac{\partial h\_{m}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})\_{i}}h\_{j}^{l}\Delta W\_{ij}^{l} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑m∂hkℓ+1∂hml+1​[∂𝒉l+1∂(𝐖l​𝒉l)​(Δ​𝐖l​𝒉l)]m\displaystyle=\sum\_{m}\frac{\partial h\_{k}^{\ell+1}}{\partial h\_{m}^{l+1}}\bigg[\frac{\partial\bm{h}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}(\Delta\mathbf{W}^{l}\bm{h}^{l})\bigg]\_{m} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =[∂𝒉ℓ+1∂𝒉l+1​∂𝒉l+1∂(𝐖l​𝒉l)​(Δ​𝐖l​𝒉l)]k.\displaystyle=\bigg[\frac{\partial\bm{h}^{\ell+1}}{\partial\bm{h}^{l+1}}\frac{\partial\bm{h}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}(\Delta\mathbf{W}^{l}\bm{h}^{l})\bigg]\_{k}. |  |

###### Lemma 6.11.

Suppose [˜6.1](#S6.theoremcnt1 "Assumption 6.1. ‣ Alignment ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") holds, L1−2​α≲NL^{1-2\alpha}\lesssim N, and the matrix ∂𝒉ℓ+1∂𝒉l+1\frac{\partial\bm{h}^{\ell+1}}{\partial\bm{h}^{l+1}} is unaligned with the vector ∂𝒉l+1∂(𝐖l​𝒉l)​Δ​𝐖l​𝒉l\frac{\partial\bm{h}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}\Delta\mathbf{W}^{l}\bm{h}^{l}: specifically,

|  |  |  |
| --- | --- | --- |
|  | maxℓ⁡max1≤l≤ℓ⁡(𝔼​‖∂𝒉ℓ+1∂𝒉l+1​∂𝒉l+1∂(𝐖l​𝒉l)​Δ​𝐖l​𝒉l‖)​(N−1/2​𝔼​‖∂𝒉ℓ+1∂𝒉l+1‖F​𝔼​‖∂𝒉l+1∂(𝐖l​𝒉l)​Δ​𝐖l​𝒉l‖)−1≲1.\max\_{\ell}\max\_{1\leq l\leq\ell}\bigg\lparen\mathbb{E}\bigg\|\frac{\partial\bm{h}^{\ell+1}}{\partial\bm{h}^{l+1}}\frac{\partial\bm{h}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}\Delta\mathbf{W}^{l}\bm{h}^{l}\bigg\|\bigg\rparen\bigg\lparen N^{-1/2}\mathbb{E}\bigg\|\frac{\partial\bm{h}^{\ell+1}}{\partial\bm{h}^{l+1}}\bigg\|\_{F}\mathbb{E}\bigg\|\frac{\partial\bm{h}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}\Delta\mathbf{W}^{l}\bm{h}^{l}\bigg\|\bigg\rparen^{-1}\lesssim 1. |  |

Then

|  |  |  |
| --- | --- | --- |
|  | N−1/2​maxℓ⁡max1≤l≤ℓ⁡𝔼​‖∂𝒉ℓ+1∂𝒉l+1​∂𝒉l+1∂(𝐖l​𝒉l)​Δ​𝐖l​𝒉l‖≲L−α​N1/2​ηhidden.N^{-1/2}\max\_{\ell}\max\_{1\leq l\leq\ell}\mathbb{E}\bigg\|\frac{\partial\bm{h}^{\ell+1}}{\partial\bm{h}^{l+1}}\frac{\partial\bm{h}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}\Delta\mathbf{W}^{l}\bm{h}^{l}\bigg\|\lesssim L^{-\alpha}N^{1/2}\eta\_{\text{hidden}}. |  |

###### Proof.

We start with

|  |  |  |  |
| --- | --- | --- | --- |
|  | N−1/2​‖∂𝒉l+1∂𝒉^l+1​∂𝒉^l+1∂(𝐖l​𝒉l)​(Δ​𝐖l​𝒉l)‖≤(a)L−α​N−1/2​‖Δ​𝐖l​𝒉l‖‖𝒉^l+1‖​‖𝐖l​𝒉l‖≤(b)(1+O​(L−α))​L−α​N−1/2​‖Δ​𝐖l​𝒉l‖‖𝐖l​𝒉l‖\begin{multlined}N^{-1/2}\bigg\|\frac{\partial\bm{h}^{l+1}}{\partial\hat{\bm{h}}^{l+1}}\frac{\partial\hat{\bm{h}}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}(\Delta\mathbf{W}^{l}\bm{h}^{l})\bigg\|\stackrel{{\scriptstyle\mathrm{(\char 97\relax)}}}{{\leq}}L^{-\alpha}N^{-1/2}\frac{\|\Delta\mathbf{W}^{l}\bm{h}^{l}\|}{\big\|\hat{\bm{h}}^{l+1}\big\|\|\mathbf{W}^{l}\bm{h}^{l}\|}\\ \stackrel{{\scriptstyle\mathrm{(\char 98\relax)}}}{{\leq}}(1+O(L^{-\alpha}))L^{-\alpha}N^{-1/2}\frac{\|\Delta\mathbf{W}^{l}\bm{h}^{l}\|}{\|\mathbf{W}^{l}\bm{h}^{l}\|}\end{multlined}N^{-1/2}\bigg\|\frac{\partial\bm{h}^{l+1}}{\partial\hat{\bm{h}}^{l+1}}\frac{\partial\hat{\bm{h}}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}(\Delta\mathbf{W}^{l}\bm{h}^{l})\bigg\|\stackrel{{\scriptstyle\mathrm{(\char 97\relax)}}}{{\leq}}L^{-\alpha}N^{-1/2}\frac{\|\Delta\mathbf{W}^{l}\bm{h}^{l}\|}{\big\|\hat{\bm{h}}^{l+1}\big\|\|\mathbf{W}^{l}\bm{h}^{l}\|}\\ \stackrel{{\scriptstyle\mathrm{(\char 98\relax)}}}{{\leq}}(1+O(L^{-\alpha}))L^{-\alpha}N^{-1/2}\frac{\|\Delta\mathbf{W}^{l}\bm{h}^{l}\|}{\|\mathbf{W}^{l}\bm{h}^{l}\|} |  | (18) |

where ([18](#S6.E18 "Equation 18 ‣ Proof. ‣ Notation ‣ 6.2 Depth scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers")a) is by

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖∂Norm​(𝒂)∂𝒂​𝒃‖=1‖𝒂‖​‖(𝐈−𝒂​𝒂𝖳‖𝒂‖2)​𝒃‖≤‖𝒃‖‖𝒂‖\bigg\|\frac{\partial\text{Norm}(\bm{a})}{\partial\bm{a}}\bm{b}\bigg\|=\frac{1}{\|\bm{a}\|}\bigg\|\bigg\lparen\mathbf{I}-\frac{\bm{a}\bm{a}^{\mkern-1.5mu\mathsf{T}}}{\|\bm{a}\|^{2}}\bigg\rparen\bm{b}\bigg\|\leq\frac{\|\bm{b}\|}{\|\bm{a}\|} |  | (19) |

for any two (non-zero) vectors 𝒂\bm{a} and 𝒃\bm{b}, and ([18](#S6.E18 "Equation 18 ‣ Proof. ‣ Notation ‣ 6.2 Depth scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers")b) is because
‖(1−L−α)​𝒂+L−α​𝒃‖=1−O​(L−α)\|(1-L^{-\alpha})\bm{a}+L^{-\alpha}\bm{b}\|=1-O(L^{-\alpha}) for unit vectors 𝒂\bm{a} and 𝒃\bm{b} regardless of their alignment.

Now, consider the matrix

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒉ℓ+1∂𝒉ℓ\displaystyle\frac{\partial\bm{h}^{\ell+1}}{\partial\bm{h}^{\ell}} | =∂𝒉ℓ+1∂𝒉^ℓ+1​∂𝒉^ℓ+1∂𝒉ℓ\displaystyle=\frac{\partial\bm{h}^{\ell+1}}{\partial\hat{\bm{h}}^{\ell+1}}\frac{\partial\hat{\bm{h}}^{\ell+1}}{\partial\bm{h}^{\ell}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∂𝒉ℓ+1∂𝒉^ℓ+1​((1−L−α)​𝐈+L−α​∂Norm​(𝐖ℓ​𝒉ℓ)∂𝐖ℓ​𝒉ℓ​𝐖ℓ)\displaystyle=\frac{\partial\bm{h}^{\ell+1}}{\partial\hat{\bm{h}}^{\ell+1}}\bigg\lparen(1-L^{-\alpha})\mathbf{I}+L^{-\alpha}\frac{\partial\text{Norm}(\mathbf{W}^{\ell}\bm{h}^{\ell})}{\partial\mathbf{W}^{\ell}\bm{h}^{\ell}}\mathbf{W}^{\ell}\bigg\rparen |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1‖𝒉^ℓ+1‖​(𝐈−𝒉ℓ+1​(𝒉ℓ+1)𝖳)​((1−L−α)​𝐈+L−α​∂Norm​(𝐖ℓ​𝒉ℓ)∂𝐖ℓ​𝒉ℓ​𝐖ℓ).\displaystyle=\frac{1}{\big\|\hat{\bm{h}}^{\ell+1}\big\|}\big\lparen\mathbf{I}-\bm{h}^{\ell+1}(\bm{h}^{\ell+1})^{\mkern-1.5mu\mathsf{T}}\big\rparen\bigg\lparen(1-L^{-\alpha})\mathbf{I}+L^{-\alpha}\frac{\partial\text{Norm}(\mathbf{W}^{\ell}\bm{h}^{\ell})}{\partial\mathbf{W}^{\ell}\bm{h}^{\ell}}\mathbf{W}^{\ell}\bigg\rparen. |  |

Notice that

|  |  |  |
| --- | --- | --- |
|  | 𝔼​‖(𝐈−𝒉ℓ+1​(𝒉ℓ+1)𝖳)​((1−L−α)​𝐈+L−α​∂Norm​(𝐖ℓ​𝒉ℓ)∂𝐖ℓ​𝒉ℓ​𝐖ℓ)​𝐀‖F2\displaystyle\mathbb{E}\bigg\|\big\lparen\mathbf{I}-\bm{h}^{\ell+1}(\bm{h}^{\ell+1})^{\mkern-1.5mu\mathsf{T}}\big\rparen\bigg\lparen(1-L^{-\alpha})\mathbf{I}+L^{-\alpha}\frac{\partial\text{Norm}(\mathbf{W}^{\ell}\bm{h}^{\ell})}{\partial\mathbf{W}^{\ell}\bm{h}^{\ell}}\mathbf{W}^{\ell}\bigg\rparen\mathbf{A}\bigg\|\_{F}^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | ≤𝔼​‖((1−L−α)​𝐈+L−α​∂Norm​(𝐖ℓ​𝒉ℓ)∂𝐖ℓ​𝒉ℓ​𝐖ℓ)​𝐀‖F2\displaystyle\quad\leq\mathbb{E}\bigg\|\bigg\lparen(1-L^{-\alpha})\mathbf{I}+L^{-\alpha}\frac{\partial\text{Norm}(\mathbf{W}^{\ell}\bm{h}^{\ell})}{\partial\mathbf{W}^{\ell}\bm{h}^{\ell}}\mathbf{W}^{\ell}\bigg\rparen\mathbf{A}\bigg\|\_{F}^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | =𝔼​tr⁡𝐀𝖳​((1−L−α)​𝐈+L−α​∂Norm​(𝐖ℓ​𝒉ℓ)∂𝐖ℓ​𝒉ℓ​𝐖ℓ)𝖳​((1−L−α)​𝐈+L−α​∂Norm​(𝐖ℓ​𝒉ℓ)∂𝐖ℓ​𝒉ℓ​𝐖ℓ)​𝐀\displaystyle\quad=\mathbb{E}\operatorname{tr}\mathbf{A}^{\mkern-1.5mu\mathsf{T}}\bigg\lparen(1-L^{-\alpha})\mathbf{I}+L^{-\alpha}\frac{\partial\text{Norm}(\mathbf{W}^{\ell}\bm{h}^{\ell})}{\partial\mathbf{W}^{\ell}\bm{h}^{\ell}}\mathbf{W}^{\ell}\bigg\rparen^{\mkern-1.5mu\mathsf{T}}\bigg\lparen(1-L^{-\alpha})\mathbf{I}+L^{-\alpha}\frac{\partial\text{Norm}(\mathbf{W}^{\ell}\bm{h}^{\ell})}{\partial\mathbf{W}^{\ell}\bm{h}^{\ell}}\mathbf{W}^{\ell}\bigg\rparen\mathbf{A} |  |
|  |  |  |
| --- | --- | --- |
|  | ≤[(1−L−α)2+L−2​α​(1+O​(N−1))]​‖𝐀‖F2\displaystyle\quad\leq\big[(1-L^{-\alpha})^{2}+L^{-2\alpha}(1+O(N^{-1}))\big]\|\mathbf{A}\|\_{F}^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | =[(1−L−α)2+L−2​α]​(1+O​(L−2​α​N−1))​‖𝐀‖F2.\displaystyle\quad=\big[(1-L^{-\alpha})^{2}+L^{-2\alpha}\big]\big\lparen 1+O(L^{-2\alpha}N^{-1})\big\rparen\|\mathbf{A}\|\_{F}^{2}. |  |

where 𝐀\mathbf{A} is assumed independent of 𝐖ℓ\mathbf{W}^{\ell} and the expectation is over 𝐖ℓ\mathbf{W}^{\ell}.
Also,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​1‖𝒉^ℓ+1‖2\displaystyle\mathbb{E}\frac{1}{\big\|\hat{\bm{h}}^{\ell+1}\big\|^{2}} | =𝔼​1(1−L−α)2+L−2​α+2​L−α​(1−L−α)​(𝒉ℓ)𝖳​Norm​(𝐖ℓ​𝒉ℓ)\displaystyle=\mathbb{E}\frac{1}{(1-L^{-\alpha})^{2}+L^{-2\alpha}+2L^{-\alpha}(1-L^{-\alpha})(\bm{h}^{\ell})^{\mkern-1.5mu\mathsf{T}}\text{Norm}(\mathbf{W}^{\ell}\bm{h}^{\ell})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1(1−L−α)2+L−2​α​𝔼​11+2​L−α​(1−L−α)(1−L−α)2+L−2​α​(𝒉ℓ)𝖳​Norm​(𝐖ℓ​𝒉ℓ)\displaystyle=\frac{1}{(1-L^{-\alpha})^{2}+L^{-2\alpha}}\mathbb{E}\frac{1}{1+2\frac{L^{-\alpha}(1-L^{-\alpha})}{(1-L^{-\alpha})^{2}+L^{-2\alpha}}(\bm{h}^{\ell})^{\mkern-1.5mu\mathsf{T}}\text{Norm}(\mathbf{W}^{\ell}\bm{h}^{\ell})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =:1(1−L−α)2+L−2​α𝔼11+s=1(1−L−α)2+L−2​α𝔼(1−s+s21+s)\displaystyle=:\frac{1}{(1-L^{-\alpha})^{2}+L^{-2\alpha}}\mathbb{E}\frac{1}{1+s}=\frac{1}{(1-L^{-\alpha})^{2}+L^{-2\alpha}}\mathbb{E}\bigg\lparen 1-s+\frac{s^{2}}{1+s}\bigg\rparen |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤1(1−L−α)2+L−2​α​𝔼​(1−s+2​s2)\displaystyle\leq\frac{1}{(1-L^{-\alpha})^{2}+L^{-2\alpha}}\mathbb{E}(1-s+2s^{2}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1(1−L−α)2+L−2​α​(1+8​L−2​α​(1−L−α)2N​((1−L−α)2+L−2​α)2)\displaystyle=\frac{1}{(1-L^{-\alpha})^{2}+L^{-2\alpha}}\bigg\lparen 1+8\frac{L^{-2\alpha}(1-L^{-\alpha})^{2}}{N\big\lparen(1-L^{-\alpha})^{2}+L^{-2\alpha}\big\rparen^{2}}\bigg\rparen |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤1(1−L−α)2+L−2​α​(1+O​(L−2​α​N−1)).\displaystyle\leq\frac{1}{(1-L^{-\alpha})^{2}+L^{-2\alpha}}\big\lparen 1+O(L^{-2\alpha}N^{-1})\big\rparen. |  |

By the Cauchy-Schwarz inequality, we can now infer

|  |  |  |
| --- | --- | --- |
|  | 𝔼​1‖𝒉^ℓ+1‖​‖(𝐈−𝒉ℓ+1​(𝒉ℓ+1)𝖳)​((1−L−α)​𝐈+L−α​∂Norm​(𝐖ℓ​𝒉ℓ)∂𝐖ℓ​𝒉ℓ​𝐖ℓ)​𝐀‖F≤(1+O​(L−2​α​N−1))​‖𝐀‖F.\mathbb{E}\frac{1}{\big\|\hat{\bm{h}}^{\ell+1}\big\|}\bigg\|\big\lparen\mathbf{I}-\bm{h}^{\ell+1}(\bm{h}^{\ell+1})^{\mkern-1.5mu\mathsf{T}}\big\rparen\bigg\lparen(1-L^{-\alpha})\mathbf{I}+L^{-\alpha}\frac{\partial\text{Norm}(\mathbf{W}^{\ell}\bm{h}^{\ell})}{\partial\mathbf{W}^{\ell}\bm{h}^{\ell}}\mathbf{W}^{\ell}\bigg\rparen\mathbf{A}\bigg\|\_{F}\\ \leq\big\lparen 1+O(L^{-2\alpha}N^{-1})\big\rparen\|\mathbf{A}\|\_{F}. |  |

Repeated conditioning leads to the estimate of the expected Frobenius norm of the matrix ∂𝒉ℓ+1∂𝒉l+1\frac{\partial\bm{h}^{\ell+1}}{\partial\bm{h}^{l+1}} for ℓ\ell and ll arbitrarily distant from each other:

|  |  |  |
| --- | --- | --- |
|  | 𝔼​‖∂𝒉ℓ+1∂𝒉l+1‖F≤(1+O​(L−2​α​N−1))L​N=O​(N),\mathbb{E}\bigg\|\frac{\partial\bm{h}^{\ell+1}}{\partial\bm{h}^{l+1}}\bigg\|\_{F}\leq\big\lparen 1+O(L^{-2\alpha}N^{-1})\big\rparen^{L}\sqrt{N}=O(\sqrt{N}), |  |

where in the last bound we use that L−2​α​N−1≲L−1L^{-2\alpha}N^{-1}\lesssim L^{-1}.

Then, using low alignment of ∂𝒉ℓ+1∂𝒉l+1\frac{\partial\bm{h}^{\ell+1}}{\partial\bm{h}^{l+1}} and ∂𝒉l+1∂(𝐖l​𝒉l)​Δ​𝐖l​𝒉l\frac{\partial\bm{h}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}\Delta\mathbf{W}^{l}\bm{h}^{l}, we see that
uniformly in ll and ℓ\ell,

|  |  |  |
| --- | --- | --- |
|  | 1N​𝔼​‖∂𝒉ℓ+1∂𝒉l+1​∂𝒉l+1∂(𝐖l​𝒉l)​Δ​𝐖l​𝒉l‖≲1N​𝔼​‖∂𝒉l+1∂(𝐖l​𝒉l)​Δ​𝐖l​𝒉l‖≲L−αN​𝔼​‖Δ​𝐖l​𝒉l‖‖𝐖l​𝒉l‖,\frac{1}{\sqrt{N}}\mathbb{E}\bigg\|\frac{\partial\bm{h}^{\ell+1}}{\partial\bm{h}^{l+1}}\frac{\partial\bm{h}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}\Delta\mathbf{W}^{l}\bm{h}^{l}\bigg\|\lesssim\frac{1}{\sqrt{N}}\mathbb{E}\bigg\|\frac{\partial\bm{h}^{l+1}}{\partial(\mathbf{W}^{l}\bm{h}^{l})}\Delta\mathbf{W}^{l}\bm{h}^{l}\bigg\|\lesssim\frac{L^{-\alpha}}{\sqrt{N}}\mathbb{E}\frac{\|\Delta\mathbf{W}^{l}\bm{h}^{l}\|}{\|\mathbf{W}^{l}\bm{h}^{l}\|}, |  |

where the last inequality is by [equation˜18](#S6.E18 "In Proof. ‣ Notation ‣ 6.2 Depth scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers").
The scale of ‖Δ​𝐖l​𝒉l‖‖𝐖l​𝒉l‖\frac{\|\Delta\mathbf{W}^{l}\bm{h}^{l}\|}{\|\mathbf{W}^{l}\bm{h}^{l}\|} is ηhidden​N\eta\_{\text{hidden}}N (recall that αhidden=1\alpha\_{\text{hidden}}=1 by [˜6.1](#S6.theoremcnt1 "Assumption 6.1. ‣ Alignment ‣ 6.1 Width scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers")).
∎

By [lemma˜6.11](#S6.theoremcnt11 "Lemma 6.11. ‣ Notation ‣ 6.2 Depth scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers"),
the first term in [equation˜17](#S6.E17 "In Notation ‣ 6.2 Depth scaling ‣ 6 Proofs and derivations ‣ Learning Rate Transfer in Normalized Transformers") is of scale no more than ηhidden​ℓ​L−α​N1/2\eta\_{\text{hidden}}\ell L^{-\alpha}N^{1/2}.
Taking into account that 𝒙\bm{x} is one-hot, it is easy to see similarly that ∑i​j∂hkℓ+1∂Einput,i​j​Δ​Einput,i​j\sum\_{ij}\frac{\partial h\_{k}^{\ell+1}}{\partial E\_{\text{input},ij}}\Delta E\_{\text{input},ij}
is of scale not exceeding ηinput\eta\_{\text{input}}.
So, the maximal stable learning rates are found from

|  |  |  |
| --- | --- | --- |
|  | ηhidden​L1−α​N1/2≍N−1/2,ηinput≍N−1/2,\eta\_{\text{hidden}}L^{1-\alpha}N^{1/2}\asymp N^{-1/2},\quad\eta\_{\text{input}}\asymp N^{-1/2}, |  |

that is,

|  |  |  |
| --- | --- | --- |
|  | ηhidden≍Lα−1​N−1,ηinput≍N−1/2.\eta\_{\text{hidden}}\asymp L^{\alpha-1}N^{-1},\quad\eta\_{\text{input}}\asymp N^{-1/2}. |  |

## 7 Additional experiments

### 7.1 LERP parameters

|  |  |
| --- | --- |
|  |  |
|  |  |

!(/html/2604.27077/assets/x22.png)

!(/html/2604.27077/assets/x23.png)

!(/html/2604.27077/assets/x24.png)

!(/html/2604.27077/assets/x25.png)

Figure 10: 
Depth sweep of nGPT (baseline) at a fixed iteration count, power law fit of 𝜶A\bm{\alpha}\_{A} and 𝜶M\bm{\alpha}\_{M} components
(averaged over layers and component indices).

|  |  |
| --- | --- |
|  |  |
|  |  |

!(/html/2604.27077/assets/x26.png)

!(/html/2604.27077/assets/x27.png)

!(/html/2604.27077/assets/x28.png)

!(/html/2604.27077/assets/x29.png)

Figure 11: 
Depth sweep of nGPT with μ\muP-style corrections at a fixed iteration count, power law fit of 𝜶A\bm{\alpha}\_{A} and 𝜶M\bm{\alpha}\_{M} components
(averaged over layers and component indices).

### 7.2 Alignment exponents for a wider model

Analogously to [section˜4.3](#S4.SS3 "4.3 Alignment exponents ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers"), we plot in [figures˜12](#S7.F12 "In 7.2 Alignment exponents for a wider model ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers"), [13](#S7.F13 "Figure 13 ‣ 7.2 Alignment exponents for a wider model ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers") and [14](#S7.F14 "Figure 14 ‣ 7.2 Alignment exponents for a wider model ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers")
the alignment exponents for a model with 0​p​t=120pt=12 and nheads=28n\_{\text{heads}}=28.

!(/html/2604.27077/assets/x30.png)

(a) Alignment exponent α\alpha

!(/html/2604.27077/assets/x31.png)

(b) Alignment exponent ω\omega

!(/html/2604.27077/assets/x32.png)

(c) Alignment exponent ν\nu

Figure 12: Alignment exponents ([definition˜3.1](#S3.theoremcnt1 "Definition 3.1 (Alignment exponents). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) of ν\nuGPT with 0​p​t=120pt=12, nheads=28n\_{\text{heads}}=28 on a fixed validation batch, averaged over layers.

!(/html/2604.27077/assets/x33.png)

(a) Alignment exponent α\alpha

!(/html/2604.27077/assets/x34.png)

(b) Alignment exponent ω\omega

!(/html/2604.27077/assets/x35.png)

(c) Alignment exponent ν\nu

Figure 13: Alignment exponents ([definition˜3.1](#S3.theoremcnt1 "Definition 3.1 (Alignment exponents). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) of ν\nuGPT with 0​p​t=120pt=12, nheads=28n\_{\text{heads}}=28 on a fixed validation batch, averaged over layers,
viewed as a function of loss decrease. The mid alignment assumption ([definition˜3.2](#S3.theoremcnt2 "Definition 3.2 (Full, no, mid alignment). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) matches the observations well.

!(/html/2604.27077/assets/x36.png)

(a) Alignment exponent α\alpha

!(/html/2604.27077/assets/x37.png)

(b) Alignment exponent ω\omega

!(/html/2604.27077/assets/x38.png)

(c) Alignment exponent ν\nu

Figure 14: Alignment exponents ([definition˜3.1](#S3.theoremcnt1 "Definition 3.1 (Alignment exponents). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers")) of ν\nuGPT with 0​p​t=120pt=12, nheads=28n\_{\text{heads}}=28 on a fixed validation batch vs. layer index, averaged over steps.

### 7.3 Width sweep at 20 tokens per parameter

We train our version of ν\nuGPT at 20 tokens per parameter with token count corrections removed, scaling the number of heads.
The power law is again close to (iter. count)−1/3(\text{iter. count})^{-1/3} like in the sweep of a fixed model
([figure˜15](#S7.F15 "In 7.3 Width sweep at 20 tokens per parameter ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers")).

|  |  |
| --- | --- |
|  |  |

!(/html/2604.27077/assets/x39.png)

!(/html/2604.27077/assets/x40.png)

Figure 15: A width sweep of ν\nuGPT models with different nheadsn\_{\text{heads}} (in the legend) at 20 tokens per parameter but without token count corrections: optimal peak learning rate also decreases at a law close to (iter. count)−1/3(\text{iter. count})^{-1/3}, as in [figure˜9](#S4.F9 "In 4.6 Token count correction ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers").

### 7.4 ηoutput\eta\_{\text{output}} is too large

The performance of each parametrization is quite sensitive to the learning rate applied to unembedding weights ηoutput\eta\_{\text{output}}.
Thus, consistently with Everett et al. ([2024](#bib.bib11)), we find that significant improvements can be obtained by tuning the ratios
between ηinput\eta\_{\text{input}}, ηhidden\eta\_{\text{hidden}}, ηoutput\eta\_{\text{output}}.
This tuning can be done on a small model and is not resource intensive (though we conduct and report experiments with different sizes).
The results of such tuning,
reported in [figure˜16](#S7.F16 "In 7.4 𝜂_\"output\" is too large ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers"),
suggest that ηoutput\eta\_{\text{output}} is too large in our setting, which does not necessarily impact learning transfer but does hurt performance.
Based on this, we propose a small modification (designed to be as simple as we could make it): multiplying ηoutput\eta\_{\text{output}} by a tuned coefficient (2−12^{-1} in our case).

|  |
| --- |
|  |

!(/html/2604.27077/assets/x41.png)

(a) 0​p​t=nheads=100pt=n\_{\text{heads}}=10

!(/html/2604.27077/assets/x42.png)

(b) 0​p​t=nheads=160pt=n\_{\text{heads}}=16

Figure 16: Sweep of optimal ηinput\eta\_{\text{input}} (different curves) and ηoutput\eta\_{\text{output}} (x-axis) multipliers for ν\nuGPT,
showing that ηoutput\eta\_{\text{output}} is too large; we multiply it by 2−12^{-1} for performance.

### 7.5 μ\muP or not μ\muP, that is the question (logits scaling)

In this ablation, we investigate whether ’tis nobler to have logits be Θ​(dmodel−1/2)\Theta(d\_{\text{model}}^{-1/2}) at the beginning of training with Θ​(1)\Theta(1) updates, as done
in μ\muP, or if they should be Θ​(1)\Theta(1) both at initialization and during training.
Specifically, we compare the two parametrizations in [table˜2](#S7.T2 "In 7.5 𝜇P or not 𝜇P, that is the question (logits scaling) ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers").
Recall that under ωoutput=1\omega\_{\text{output}}=1, μ\muP is the only parametrization satisfying
[˜1](#Thmdesideratum1 "Desideratum 1 (Stability at initialization). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") and [2](#Thmdesideratum2 "Desideratum 2 (Stable non-trivial feature learning). ‣ HP Transfer Desiderata. ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers") for large-width neural networks;
however, our experiments ([section˜4.3](#S4.SS3 "4.3 Alignment exponents ‣ 4 Experiments ‣ Learning Rate Transfer in Normalized Transformers")) decisively show that the correct assumption is
ωoutput=1/2\omega\_{\text{output}}=1/2, allowing for a different parametrization
in which ηoutput≍ηhidden\eta\_{\text{output}}\asymp\eta\_{\text{hidden}} but sz,inits\_{\text{$z$,init}} needs to be scaled with width.
Although mid alignment ([definition˜3.2](#S3.theoremcnt2 "Definition 3.2 (Full, no, mid alignment). ‣ Summary derivations of width corrections ‣ 3.2 Width and depth corrections ‣ 3 Reparametrization for transfer over width, depth and token horizon ‣ Learning Rate Transfer in Normalized Transformers"))
better matches measured average alignment exponents
and transfers better over width,
here we fix the full alignment assumption as in “CompleteP” to isolate the effect of only changing logits scaling,
and refer to this parametrization ν\nuGPT (full align), as opposed to ν\nuGPT.
The analogue of this parametrization is called “standard” and “NTK”
(not μ\muP) in Everett et al. ([2024](#bib.bib11))777For clarity, the parametrizations they call “standard” and “NTK” are theoretically (under infinite precision) equivalent, and both names are unusual with respect to the literature so we avoid them.
. They observe that such a choice outperforms μ\muP in terms of the best validation loss.
In [figures˜17](#S7.F17 "In 7.5 𝜇P or not 𝜇P, that is the question (logits scaling) ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers") and [18](#S7.F18 "Figure 18 ‣ 7.5 𝜇P or not 𝜇P, that is the question (logits scaling) ‣ 7 Additional experiments ‣ Learning Rate Transfer in Normalized Transformers"),
we observe that performance essentially matches in our setting when either not tuning ηoutput\eta\_{\text{output}} for either parametrization or
tuning it separately for both.
Hence, we take no position as to which choice is better but choose
to scale logits Θ​(1)\Theta(1) at initialization (same as during training)
because the assumption ωoutput=1\omega\_{\text{output}}=1 leading to μ\muP is inconsistent with experiment, and hence there is no principled reason
to scale differently.

{NiceTabular}

Table 2: 
The correct value ωoutput=1/2\omega\_{\text{output}}=1/2 leads to two different ways to scale logits and ηoutput\eta\_{\text{output}}.

|  |  |
| --- | --- |
|  |  |

!(/html/2604.27077/assets/x43.png)

(a) “CompleteP”

!(/html/2604.27077/assets/x44.png)

(b) ν\nuGPT (full align)

Figure 17: Aspect ratio sweep of “CompleteP” and ν\nuGPT (full align) at 20 tokens per parameter

|  |  |
| --- | --- |
|  |  |

!(/html/2604.27077/assets/x45.png)

(a) “CompleteP” with tuned ηoutput\eta\_{\text{output}}

!(/html/2604.27077/assets/x46.png)

(b) ν\nuGPT (full align) with tuned ηoutput\eta\_{\text{output}}

Figure 18: Aspect ratio sweep of “CompleteP” and ν\nuGPT (full align) with tuned ηoutput\eta\_{\text{output}} at 20 tokens per parameter

## References

* Arora et al. (2019)

  Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Russ R Salakhutdinov, and
  Ruosong Wang.
  On exact computation with an infinitely wide neural net.
  In *Advances in Neural Information Processing Systems*,
  volume 32. Curran Associates, Inc., 2019.
  <https://proceedings.neurips.cc/paper_files/paper/2019/file/dbc4d84bfcfe2284ba11beffb853a8c4-Paper.pdf>.
* Bjorck et al. (2025)

  Johan Bjorck, Alon Benhaim, Vishrav Chaudhary, Furu Wei, and Xia Song.
  Scaling optimal LR across token horizons.
  In *The Thirteenth International Conference on Learning
  Representations*, 2025.
  <https://openreview.net/forum?id=WYL4eFLcxG>.
* Blake et al. (2023)

  Charlie Blake, Douglas Orr, and Carlo Luschi.
  Unit scaling: Out-of-the-box low-precision training.
  In *Proceedings of the 40th International Conference on Machine
  Learning*, volume 202 of *Proceedings of Machine Learning Research*.
  PMLR, 2023.
  <https://proceedings.mlr.press/v202/blake23a.html>.
* Blake et al. (2024)

  Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Yuri
  Prince, Björn Deiseroth, Andres Felipe Cruz-Salinas, Carlo Luschi, Samuel
  Weinbach, and Douglas Orr.
  u-μ\mup: The unit-scaled maximal update parametrization.
  In *2nd Workshop on Advancing Neural Network Training:
  Computational Efficiency, Scalability, and Resource Optimization (WANT@ICML
  2024)*, 2024.
  <https://openreview.net/forum?id=44NKKzz1n5>.
* Bordelon and Pehlevan (2022)

  Blake Bordelon and Cengiz Pehlevan.
  Self-consistent dynamical field theory of kernel evolution in wide
  neural networks.
  In *Advances in Neural Information Processing Systems*,
  volume 35. Curran Associates, Inc., 2022.
  <https://proceedings.neurips.cc/paper_files/paper/2022/file/d027a5c93d484a4312cc486d399c62c1-Paper-Conference.pdf>.
* Bordelon et al. (2024a)

  Blake Bordelon, Hamza Chaudhry, and Cengiz Pehlevan.
  Infinite limits of multi-head transformer dynamics.
  In *Advances in Neural Information Processing Systems*,
  volume 37. Curran Associates, Inc., 2024a.
  [10.52202/079017-1130](https:/doi.org/10.52202/079017-1130).
  <https://proceedings.neurips.cc/paper_files/paper/2024/file/3eff068e195daace49955348de9f8398-Paper-Conference.pdf>.
* Bordelon et al. (2024b)

  Blake Bordelon, Hamza Chaudhry, and Cengiz Pehlevan.
  Infinite limits of multi-head transformer dynamics.
  *Advances in Neural Information Processing Systems*,
  37:35824–35878, 2024b.
* Bordelon et al. (2024c)

  Blake Bordelon, Lorenzo Noci, Mufan Bill Li, Boris Hanin, and Cengiz Pehlevan.
  Depthwise hyperparameter transfer in residual networks: Dynamics and
  scaling limit.
  In *The Twelfth International Conference on Learning
  Representations*, 2024c.
  <https://openreview.net/forum?id=KZJehvRKGD>.
* Chizat et al. (2019)

  Lénaïc Chizat, Edouard Oyallon, and Francis Bach.
  On lazy training in differentiable programming.
  In *Advances in Neural Information Processing Systems*,
  volume 32. Curran Associates, Inc., 2019.
  <https://proceedings.neurips.cc/paper_files/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf>.
* Dey et al. (2025)

  Nolan Simran Dey, Bin Claire Zhang, Lorenzo Noci, Mufan Li, Blake Bordelon,
  Shane Bergsma, Cengiz Pehlevan, Boris Hanin, and Joel Hestness.
  Don’t be lazy: Completep enables compute-efficient deep transformers.
  In *The Thirty-ninth Annual Conference on Neural Information
  Processing Systems*, 2025.
  <https://openreview.net/forum?id=lMU2kaMANl>.
* Everett et al. (2024)

  Katie Everett, Lechao Xiao, Mitchell Wortsman, Alexander A. Alemi, Roman Novak,
  Peter J. Liu, Izzeddin Gur, Jascha Sohl-Dickstein, Leslie Pack Kaelbling,
  Jaehoon Lee, and Jeffrey Pennington.
  Scaling exponents across parameterizations and optimizers.
  *arXiv preprint arXiv:2407.05872*, 2024.
  <https://arxiv.org/abs/2407.05872>.
* Hayou (2025)

  Soufiane Hayou.
  A proof of learning rate transfer under μ\mup.
  *arXiv preprint arXiv:2511.01734*, 2025.
  <https://arxiv.org/abs/2511.01734>.
* Hoffmann et al. (2022)

  Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor
  Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes
  Welbl, Aidan Clark, Thomas Hennigan, Eric Noland, Katherine Millican, George
  van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karén
  Simonyan, Erich Elsen, Oriol Vinyals, Jack Rae, and Laurent Sifre.
  An empirical analysis of compute-optimal large language model
  training.
  In *Advances in Neural Information Processing Systems*,
  volume 35, pages 30016–30030. Curran Associates, Inc., 2022.
  <https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf>.
* Jacot et al. (2018)

  Arthur Jacot, Franck Gabriel, and Clement Hongler.
  Neural tangent kernel: Convergence and generalization in neural
  networks.
  In *Advances in Neural Information Processing Systems*,
  volume 31. Curran Associates, Inc., 2018.
  <https://proceedings.neurips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf>.
* Kosson et al. (2026)

  Atli Kosson, Jeremy Welborn, Yang Liu, Martin Jaggi, and Xi Chen.
  Weight decay may matter more than µp for learning rate
  transfer in practice.
  In *The Fourteenth International Conference on Learning
  Representations*, 2026.
  <https://openreview.net/forum?id=PvTxIdZc1E>.
* Large et al. (2024)

  Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, and Jeremy
  Bernstein.
  Scalable optimization in the modular norm.
  In *The Thirty-eighth Annual Conference on Neural Information
  Processing Systems*, 2024.
  <https://openreview.net/forum?id=SFxAjB7UXx>.
* Liang et al. (2025)

  Wanchao Liang, Tianyu Liu, Less Wright, Will Constable, Andrew Gu, Chien-Chin
  Huang, Iris Zhang, Wei Feng, Howard Huang, Junjie Wang, Sanket Purandare,
  Gokul Nadathur, and Stratos Idreos.
  Torchtitan: One-stop pytorch native solution for production ready
  LLM pretraining.
  In *The Thirteenth International Conference on Learning
  Representations*, 2025.
  <https://openreview.net/forum?id=SFN6Wm7YBI>.
* Lingle (2025)

  Lucas Lingle.
  An empirical study of μ\mup learning rate transfer.
  *arXiv preprint arXiv:2404.05728v6*, 2025.
  <https://arxiv.org/abs/2404.05728v6>.
* Liu et al. (2021)

  Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue
  Cao, Zheng Zhang, Li Dong, et al.
  Swin Transformer V2: Scaling Up Capacity and Resolution.
  *arXiv preprint arXiv:2111.09883*, 2021.
* Loshchilov et al. (2025)

  Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, and Boris Ginsburg.
  nGPT: Normalized transformer with representation learning on the
  hypersphere.
  In *The Thirteenth International Conference on Learning
  Representations*, 2025.
  <https://openreview.net/forum?id=se4vjm7h4E>.
* Malladi et al. (2022)

  Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, and Sanjeev Arora.
  On the SDEs and scaling rules for adaptive gradient algorithms.
  In *Advances in Neural Information Processing Systems*, 2022.
  <https://openreview.net/forum?id=F2mhzjHkQP>.
* Mei et al. (2018)

  Song Mei, Andrea Montanari, and Phan-Minh Nguyen.
  A mean field view of the landscape of two-layer neural networks.
  *Proceedings of the National Academy of Sciences*, 115(33), 2018.
* Mlodozeniec et al. (2025)

  Bruno Mlodozeniec, Pierre Ablin, Louis Béthune, Dan Busbridge, Michal Klein,
  Jason Ramapuram, and Marco Cuturi.
  Completed hyperparameter transfer across modules, width, depth, batch
  and duration.
  *arXiv preprint arXiv:2512.22382*, 2025.
  <https://arxiv.org/abs/2512.22382>.
* Nguyen and Pham (2023)

  Phan-Minh Nguyen and Huy Tuan Pham.
  A rigorous framework for the mean field limit of multilayer neural
  networks.
  *Mathematical Statistics and Learning*, 6(3), 2023.
* OLMo et al. (2025)

  Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora,
  Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin
  Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman,
  Christopher Clark, Pradeep Dasigi, Nouha Dziri, Allyson Ettinger, Michal
  Guerquin, David Heineman, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya
  Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler
  Murray, Crystal Nam, Jake Poznanski, Valentina Pyatkin, Aman Rangapur,
  Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael
  Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hannaneh
  Hajishirzi.
  2 olmo 2 furious.
  *arXiv preprint arXiv:2501.00656*, 2025.
  <https://arxiv.org/abs/2501.00656>.
* Pearce and Song (2024)

  Tim Pearce and Jinyeop Song.
  Reconciling kaplan and chinchilla scaling laws.
  *Transactions on Machine Learning Research*, 2024.
  ISSN 2835-8856.
  <https://openreview.net/forum?id=NLoaLyuUUF>.
  Reproducibility Certification.
* Penedo et al. (2024)

  Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov,
  Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf.
  The fineweb datasets: Decanting the web for the finest text data at
  scale.
  In *The Thirty-eight Conference on Neural Information Processing
  Systems Datasets and Benchmarks Track*, 2024.
  <https://openreview.net/forum?id=n6SCkn2QaG>.
* Ren et al. (2026)

  Liliang Ren, Yang Liu, Yelong Shen, and Weizhu Chen.
  Rethinking language model scaling under transferable hypersphere
  optimization.
  *arXiv preprint arXiv:2603.28743*, 2026.
* Rotskoff and Vanden-Eijnden (2018)

  Grant Rotskoff and Eric Vanden-Eijnden.
  Parameters as interacting particles: long time convergence and
  asymptotic error scaling of neural networks.
  In *Advances in Neural Information Processing Systems*,
  volume 31. Curran Associates, Inc., 2018.
  <https://proceedings.neurips.cc/paper_files/paper/2018/file/196f5641aa9dc87067da4ff90fd81e7b-Paper.pdf>.
* Shulgin et al. (2026)

  Egor Shulgin, Dimitri von Rütte, Tianyue H Zhang, Niccolò Ajroldi,
  Bernhard Schölkopf, and Antonio Orvieto.
  Deriving hyperparameter scaling laws via modern optimization theory.
  *arXiv preprint arXiv:2603.15958*, 2026.
* Sirignano and Spiliopoulos (2020)

  Justin Sirignano and Konstantinos Spiliopoulos.
  Mean field analysis of neural networks: A law of large numbers.
  *SIAM Journal on Applied Mathematics*, 80(2), 2020.
  [10.1137/18M1192184](https:/doi.org/10.1137/18M1192184).
  <https://doi.org/10.1137/18M1192184>.
* Su et al. (2023)

  Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu.
  Roformer: Enhanced transformer with rotary position embedding.
  *arXiv preprint arXiv:2104.09864*, 2023.
  <https://arxiv.org/abs/2104.09864>.
* Vlassis et al. (2025)

  Georgios Vlassis, David Belius, and Volodymyr Fomichov.
  A thorough reproduction and evaluation of $\mu$p.
  *Transactions on Machine Learning Research*, 2025.
  ISSN 2835-8856.
  <https://openreview.net/forum?id=AFxEdJwQcp>.
* Wen et al. (2025)

  Kaiyue Wen, David Hall, Tengyu Ma, and Percy Liang.
  Fantastic pretraining optimizers and where to find them.
  *arXiv preprint arXiv:2509.02046*, 2025.
  <https://arxiv.org/abs/2509.02046>.
* Yang et al. (2021)

  Ge Yang, Edward Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi,
  Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao.
  Tuning large neural networks via zero-shot hyperparameter transfer.
  In *Advances in Neural Information Processing Systems*,
  volume 34. Curran Associates, Inc., 2021.
  <https://proceedings.neurips.cc/paper_files/paper/2021/file/8df7c2e3c3c3be098ef7b382bd2c37ba-Paper.pdf>.
* Yang (2019)

  Greg Yang.
  Wide feedforward or recurrent neural networks of any architecture are
  gaussian processes.
  In *Advances in Neural Information Processing Systems*,
  volume 32. Curran Associates, Inc., 2019.
  <https://proceedings.neurips.cc/paper_files/paper/2019/file/5e69fda38cda2060819766569fd93aa5-Paper.pdf>.
* Yang (2020)

  Greg Yang.
  Tensor programs ii: Neural tangent kernel for any architecture.
  *arXiv preprint arXiv:2006.14548*, 2020.
  <https://arxiv.org/abs/2006.14548>.
* Yang (2021)

  Greg Yang.
  Tensor programs iii: Neural matrix laws.
  *arXiv preprint arXiv:2009.10685*, 2021.
  <https://arxiv.org/abs/2009.10685>.
* Yang and Hu (2021)

  Greg Yang and Edward J. Hu.
  Tensor programs iv: Feature learning in infinite-width neural
  networks.
  In *Proceedings of the 38th International Conference on Machine
  Learning*, volume 139 of *Proceedings of Machine Learning Research*.
  PMLR, 2021.
  <https://proceedings.mlr.press/v139/yang21c.html>.
* Yang and Littwin (2021)

  Greg Yang and Etai Littwin.
  Tensor programs iib: Architectural universality of neural tangent
  kernel training dynamics.
  In *Proceedings of the 38th International Conference on Machine
  Learning*, volume 139 of *Proceedings of Machine Learning Research*.
  PMLR, 2021.
  <https://proceedings.mlr.press/v139/yang21f.html>.
* Yang et al. (2024a)

  Greg Yang, James B. Simon, and Jeremy Bernstein.
  A spectral condition for feature learning.
  *arXiv preprint arXiv:2310.17813*, 2024a.
  <https://arxiv.org/abs/2310.17813>.
* Yang et al. (2024b)

  Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou.
  Tensor programs VI: Feature learning in infinite depth neural
  networks.
  In *The Twelfth International Conference on Learning
  Representations*, 2024b.
  <https://openreview.net/forum?id=17pVDnpwwl>.
