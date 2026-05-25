---
arxiv: '2604.08510'
authors:
- Emmy Liu
- Kaiser Sun
- Millicent Li
- Isabelle Lee
- Lindia Tjuatja
- Jen-tse Huang
- Graham Neubig
parser: ar5iv
retrieved: '2026-05-25'
source: paper
title: What do Language Models Learn and When? The Implicit Curriculum Hypothesis
url: https://arxiv.org/abs/2604.08510
year: 2026
---

[2604.08510] What Do Language Models Learn and When? The Implicit Curriculum Hypothesis



# What Do Language Models Learn and When? The Implicit Curriculum Hypothesis

Emmy Liu
Affiliation: Language Technologies Institute, Carnegie Mellon University
  
Kaiser Sun
Affiliation: Department of Computer Science, Data ScienceAI Institute,  Johns Hopkins University
  
Millicent Li
Affiliation: Khoury College of Computer Science, Northeastern University
  
Isabelle Lee
Affiliation: Department of Computer Science, University of Southern California
emmy@cmu.edu
  
Lindia Tjuatja
Affiliation: Language Technologies Institute, Carnegie Mellon University
  
  Jen-tse Huang, Graham Neubig
Affiliation: Language Technologies Institute, Carnegie Mellon University
Affiliation: Department of Computer Science, Data ScienceAI Institute,  Johns Hopkins University

###### Abstract

Large language models (LLMs) can perform remarkably complex tasks, yet the fine-grained details of how these capabilities emerge during pretraining remain poorly understood. Scaling laws on validation loss tell us how much a model improves with additional compute, but not what skills it acquires in which order. To remedy this, we propose the *Implicit Curriculum Hypothesis*: pretraining follows a compositional and predictable curriculum across models and data mixtures. We test this by designing a suite of simple, composable tasks spanning retrieval, morphological transformations, coreference, logical reasoning, and mathematics. Using these tasks, we track emergence points across four model families spanning sizes from 410M–13B parameters. We find that *emergence orderings* of when models reach fixed accuracy thresholds are strikingly consistent (ρ=.81\rho=.81 across 45 model pairs), and that composite tasks most often emerge after their component tasks. Furthermore, we find that this structure is encoded in model representations: tasks with similar function vector representations also tend to follow similar trajectories in training. By using the space of representations derived from our task set, we can effectively predict the training trajectories of simple held-out compositional tasks throughout the course of pretraining (R2=.68R^{2}=.68–.84.84 across models) without previously evaluating them. Together, these results suggest that pretraining is more structured than loss curves reveal: skills emerge in a compositional order that is consistent across models and readable from their internals.111Data and code available at <https://github.com/KaiserWhoLearns/ElementalTask>

## 1 Introduction

Large language models (LLMs) exhibit predictable improvements in performance with scale, a phenomenon characterized by well-established scaling laws (hoffmann2022an; gadre2025language; muennighoff2023scaling).
These scaling laws tell us how much models are expected to improve in predicting the next token on the pretraining distribution given additional compute, but not what skills the model acquires, or when during pretraining it acquires them specifically.
In practice, training runs may cost millions of dollars, yet are primarily monitored through aggregate cross-entropy loss, or through evaluating at intervals on downstream benchmarks such as MMLU (mmlu).
However, neither approach provides actionable diagnostic information.
Cross-entropy loss decreases smoothly even as qualitatively different skills are acquired at sudden transition points (kangaslahti2025hidden).
Downstream benchmarks compose many prerequisite skills, making failures opaque: when GSM8k (gsm8k) performance stalls, it is unclear whether the bottleneck is numerical fluency, multi-step planning, or natural language understanding.
For instance, scoring well on GSM8k may require numerical fluency, multi-step planning, as well as natural language understanding, making it difficult to diagnose which prerequisite skills are missing when performance stalls (meister-cotterell-2021-language).

A growing body of theoretical work suggests that neural networks learn functions sequentially, acquiring simpler patterns before more complex ones (lee2025distinctcomputationsemergecompositional; zhang2026saddletosaddledynamicsexplainssimplicity).
Recent work has built on these insights, hypothesizing that complex behaviors and scaling laws themselves emerge from the combination of more elementary sub-tasks that serve as fundamental building blocks (khandelwal2025languagemodelscomposefunctions) or quanta (michaud2023the).
However, much of this theoretical work has focused on simplified modeling settings, leaving open questions about how these insights translate to large-scale language model pretraining (srivastava2023beyond).
Prior empirical work has shown that certain knowledge categories (e.g., syntactic vs. factual) are acquired at different rates (liu-etal-2021-probing-across), and that grammatical phenomena are learned in a consistent order across architectures (friedman-etal-2022-finding).
However, these studies have not examined whether the ordering reflects compositional dependencies between skills, nor whether it is legible in the model’s internal representations.

Based on this, we propose the Implicit Curriculum Hypothesis: during pretraining, skills emerge in a stable compositional order that is consistent across models.
This is a stronger claim than the quanta hypothesis alone.
It predicts not only that simple precedes complex, but also that the specific ordering is reproducible across models and reflects compositional dependencies between skills.
To test the Implicit Curriculum Hypothesis, we design a suite of simple tasks that probe a wide range of skills.
We track emergence across 9 models from 4 families (410M-13B parameters) and find:

1. 1.

   The emergence ordering is consistent across model families.
   Spearman correlations between emergence orderings range from ρ=.64\rho=.64 to .93.93 (mean .81.81) across all 45 model pairs, including cross-family comparisons. Copying is the first skill to emerge, followed by many simple string operations, fact extraction and coreference, then logic operations, simple world knowledge, then multistep arithmetic and more complex reasoning tasks.
   Composite tasks emerge after their elemental prerequisites.
   However, this consistency holds only when emergence is defined by fixed accuracy thresholds, not relative ones.
2. 2.

   The ordering is legible in model representations.
   Tasks whose internal representations are nearby in the model’s residual stream, measured via function vectors (todd2024function), follow similar learning trajectories.
   This proximity is sufficient to predict the full training trajectory of held-out composite tasks (mean R2R^{2} of .68.68–.84.84 across models, with per-task R2R^{2} exceeding .95.95) without ever evaluating them during training.

Together, these results suggest that pretraining is more structured than loss curves reveal: skills emerge in an order that is consistent across models, respects compositional dependencies, and is readable from model internals.

Figure 1: Emergence order across model families and sizes, smoothed with a Gaussian kernel (σ=1.0\sigma=1.0). Dots represent the point at which the model reaches a fixed 50% accuracy threshold. While the absolute emergence time varies across models, the ordering shows regularity.

## 2 Preliminaries

### 2.1 Background

We provide a summary of the work that we directly build on in this paper.
Further related work can be found in [Appendix B](#A2 "Appendix B Extended Related Work ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis").

#### Scaling Laws

Scaling laws characterize the relationship between a model’s held-out validation loss LL and the compute budget allocated to training, typically decomposed into model size NN and data size DD. These relationships are well-approximated by power laws of the form L​(N,D)∝Nα+D−β+L∞L(N,D)\propto N^{\alpha}+D^{-\beta}+L\_{\infty} (kaplan2020scaling; hoffmann2022an), and hold across many orders of magnitude. However, this aggregate loss curve does not directly correlate with downstream performance (lourie2025scalinglawsunreliabledownstream; isik2026scalinglawsdownstreamtask; liu2026notjustscalinglawsbetterunderstanding), and it is not clear what the model is learning as loss decreases.

Quantization Hypothesis michaud2023the offers a hypothesis that these smooth scaling curves arise from the learning of discrete skills, termed quanta.
Under this framework, a model acquires these quanta in an order optimized to reduce total loss, hiding the discrete transitions that correspond to the model learning.
One practical quandary is that quanta require a post-hoc discovery method, the results of which often do not correspond to interpretable skills (michaud2023the).
While compelling, the Quantization Hypothesis typically treats these skills as independent, additive contributions, leaving the structural dependencies and compositional nature of these skills largely unexplored.

#### Simplicity Bias

Secondly, works show that neural networks trained with gradient-descent-based methods tend to exhibit simplicity bias, a tendency to learn simpler functions before more complex ones (saxe2014exactsolutionsnonlineardynamics; nakkiran2019sgdneuralnetworkslearns; shah2020pitfallssimplicitybiasneural).
In the context of language modeling, this is reflected in models learning lower-order n-grams before higher-order ones (michaelov2025languagemodelbehavioralphases).
However, we note that the notions of simplicity are often underspecified, and moreover, these function-level notions of complexity cannot be used to quantify task complexity.

#### Compositional Skill Structure

A recent line of work has investigated whether skills acquired by language models follow explicit dependency structures. chen2023skillitdatadrivenskillsframework represent skills as directed acyclic graphs (DAGs), where an edge from skill AA to skill BB indicates that training on data associated with AA reduces the amount of data needed to learn BB. Such dependency graphs can be used to design curricula for target skills.
While their work focuses on post-training and defines skills by data clusters, a natural question is whether we can also characterize the dependency structure of general web-data-based pretraining. Theoretically, arora2023theoryemergencecomplexskills also provide a framework relating cross-entropy loss to competence on individual sets of skills, showing that a decrease in loss implies simultaneous improvement in both individual skills and their kk-tuples.

### 2.2 The Implicit Curriculum Hypothesis

The four threads of work above establish that (1) capabilities are discrete and unlock progressively, (2) simpler functions are learned before more complex ones, and (3) skills may have a dependency structure. However, they leave open whether these threads combine in practice: does large-scale pretraining follow a structured, compositional ordering of skill acquisition that is consistent across models?

The Implicit Curriculum Hypothesis

Let 𝒯\mathcal{T} be a set of tasks equipped with a known dependency
relation ≺\prec, where τi≺τj\tau\_{i}\prec\tau\_{j} indicates that task τj\tau\_{j} was
constructed to compositionally depend on τi\tau\_{i}. Let
P​(τ)={τ′∈𝒯:τ′≺τ}P(\tau)=\{\tau^{\prime}\in\mathcal{T}:\tau^{\prime}\prec\tau\} denote the
prerequisite set of task τ\tau under this construction. We emphasize that
≺\prec reflects our design-level task structure, not a claim about
the model’s internal primitives or the existence of any particular dependency
in the model. Let tτ∗​(m)t^{\*}\_{\tau}(m) denote the emergence time of task τ\tau for
model mm, defined as the first training step at which performance exceeds a
threshold θ∈ℝ\theta\in\mathbb{R}. We hypothesize that this design-level
structure is reflected empirically in model training dynamics:

H1.

Compositional ordering. Tasks emerge no later than the
tasks constructed to depend on them:



∀τj∈𝒯,∀τi∈P(τj):tτi∗(m)≤tτj∗(m)\forall\,\tau\_{j}\in\mathcal{T},\;\forall\,\tau\_{i}\in P(\tau\_{j}):\quad t^{\*}\_{\tau\_{i}}(m)\leq t^{\*}\_{\tau\_{j}}(m)
H2.

Cross-model stability. The emergence ordering induces a
partial order ⪯𝒯\preceq\_{\mathcal{T}} over tasks that is consistent across
models: for models m1,m2m\_{1},m\_{2}, the rank correlation between their emergence orderings is significantly higher than chance.
H3.

Representational alignment. Tasks with similar internal
representations exhibit similar learning trajectories. That is, for tasks
τi,τj\tau\_{i},\tau\_{j} with representation vectors vi,vjv\_{i},v\_{j}, ∃ϵ∈ℝ+\exists\,\epsilon\in\mathbb{R}^{+}:



Sim​(vi,vj)​ high⟹d​(aτi​(⋅),aτj​(⋅))<ϵ\mathrm{Sim}(v\_{i},v\_{j})\text{ high}\;\implies\;d\!\left(a\_{\tau\_{i}}(\cdot),\;a\_{\tau\_{j}}(\cdot)\right)<\epsilon
where aτ​(⋅)a\_{\tau}(\cdot) is the learning trajectory of task τ\tau and dd
is a distance over trajectories. This further implies that trajectories
of unseen tasks can be predicted from representational geometry alone,
without evaluating them during training.

## 3 Methodology

### 3.1 Models and Checkpoints

In order to test our hypotheses, we focus on examining open-weight models with publicly-released intermediate pre-training checkpoints.
Because our hypotheses are largely about timing and emergence order, it was also important to select models with relatively dense intermediate checkpoints and larger sizes.
The selected models are:

* •

  OLMo-2 (olmo20242): 1B, 7B, and 13B
  parameter models, providing a within-family scale comparison
  across an order of magnitude.
* •

  OLMo-3 (olmo3): 7B, offering comparison with a newer generation compared to OLMo-2.
* •

  LLM360 (liu2023llm360): Crystal (7B)
  and Amber (7B), trained on very different data mixtures
  (code-oriented and natural-language-oriented, respectively),
  allowing us to study the effect of data composition within
  the same model family.
* •

  Pythia (biderman2023pythia): 410M,
  1.4B, and 12B, offering a comparison with an earlier model generation
  trained on different data. We selected sizes spanning the
  full range of the suite; models below 410M were excluded
  due to poor performance.

In order to keep checkpoint sampling consistent across families, we focused on up to the first 1T tokens of training for each model and sampled approximately 20 checkpoints for each model within this range, giving a granularity of roughly every 20B tokens. We hypothesized that this would capture the period for which most relevant simple skills emerge for the tasks we study, while the granularity would be sufficient to resolve ordering differences.

### 3.2 Task Design

We design tasks with intuitive compositional relationships, diverse operation types, and unambiguous outputs, while keeping them simple enough for models as small as 1B parameters to eventually solve via in-context learning. We therefore evaluate all 91 elemental and composite tasks using exact-match accuracy; the full list is given in [Appendix D](#A4 "Appendix D Full list of Elemental and Composite Tasks ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis").

| Task | Input | Out |
| --- | --- | --- |
| simple\_icl: present\_to\_gerund | run | running |
| fact\_extraction: extract\_number | Passage: “John gave 5 apples to Mary on Tuesday.” How many apples? | 5 |
| logical\_ops: conditional | If it rains, the ground gets wet. It rains. Is the ground wet? | yes |
| Compositional tasks | | |
| comp: gerund\_upper | run | RUNNING |
| comp: translate\_sp\_eng\_reverse | hola | olleh |
| comp: extract\_verify | Passage: Nora gave 3 apples to Ben. Ben gave 1 to Li. Claim: Ben received apples first. Does the claim follow? | True |

Table 1: Example simple and compositional tasks. Compositional tasks require chaining multiple primitive skills.

#### Simple tasks.

We define a set of simple tasks spanning string manipulation (e.g., copy, uppercase, first letter), morphological transformation (e.g., singular to plural, present to gerund), knowledge retrieval (country to capital, country to currency), and translation (e.g., en-fr, en-sp). These were selected to cover distinct operation
types while remaining simple enough to be plausibly atomic. Notably, several of these operations have also been investigated in the interpretability literature (olsson2022incontextlearninginductionheads; hendel-etal-2023-context; todd2024function; todd2026incontextalgebra). We do not claim that these are the true minimal units of model
computation, but they serve as a diverse set of operations from which we can construct composites with known structure. In total, we create 53 simple tasks.

#### Composite tasks: synthetic chains.

We construct composite tasks by chaining elemental operations in sequence. For example, gerund\_upper applies the gerund transformation followed by uppercasing (write →\to WRITING). This mechanical construction guarantees that the compositional prerequisites are known exactly, yielding 38 composite
tasks. The inclusion of translation-based composites (e.g., translate\_eng\_fr\_upper\_reverse) additionally tests whether knowledge-dependent elementals compose in the same way as rule-based ones.

### 3.3 Measuring Emergence

Prior work has proposed several notions of emergence, including scale-based definitions and parametric fits to learning curves (wei2022emergentabilitieslargelanguage; snell2024predictingemergentcapabilitiesfinetuning). For our purposes, however, the key quantity is not the sharpness of emergence but the *relative ordering* of when tasks become feasible. Because many trajectories are noisy or irregular, we use simple threshold-based definitions. We consider two variants:

#### Absolute threshold.

We define emergence time tτ∗​(m)t^{\*}\_{\tau}(m) as the first checkpoint at which task τ\tau exceeds a fixed accuracy threshold θabs\theta\_{\text{abs}}.

#### Relative threshold.

We alternatively define emergence time as the first checkpoint at which performance reaches a fraction α\alpha of the model’s best performance on that task.

### 3.4 Measuring Representational Similarity

To operationalize representational alignment, we require a per-task representation that captures the computation the model performs in order to do the task. Following the methodology from todd2024function, we extract task representations (function vectors) from the models.

#### Extraction.

Let a transformer have LL blocks and hidden dimension dd. For block ℓ\ell, let

|  |  |  |
| --- | --- | --- |
|  | hℓattn=hℓ−1+Attn​(LN​(hℓ−1))h^{\mathrm{attn}}\_{\ell}=h\_{\ell-1}+\mathrm{Attn}\!\left(\mathrm{LN}(h\_{\ell-1})\right) |  |

denote the post-attention hidden state, and let

|  |  |  |
| --- | --- | --- |
|  | hℓ=hℓattn+MLP​(LN​(hℓattn))h\_{\ell}=h^{\mathrm{attn}}\_{\ell}+\mathrm{MLP}\!\left(\mathrm{LN}(h^{\mathrm{attn}}\_{\ell})\right) |  |

denote the block-output hidden state. For each task τ\tau, we construct a set of ICL prompts, perform a forward pass for each prompt, and extract activations at the last non-pad token position tlastt\_{\mathrm{last}} (i.e., the position from which the model begins generating its answer). We retain only prompts on which the model produces the correct answer, ensuring that the extracted representation reflects successful task execution. We consider two extraction methods, and for each model use the one that performs best (see [Appendix G](#A7 "Appendix G Function vector hyperparameters ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis")).

Head-based extraction. We use causal indirect effect (CIE) analysis to identify a sparse set of attention heads ℋ⊆[H]×[L]\mathcal{H}\subseteq[H]\times[L] with the strongest causal effects on task performance. The function vector is then the average of these heads’ outputs across correctly answered prompts:

|  |  |  |
| --- | --- | --- |
|  | vτℋ=1|𝒟τ+|​∑xi∈𝒟τ+∑(h,j)∈ℋahj​(xi),v\_{\tau}^{\mathcal{H}}=\frac{1}{|\mathcal{D}\_{\tau}^{+}|}\sum\_{x\_{i}\in\mathcal{D}\_{\tau}^{+}}\sum\_{(h,j)\in\mathcal{H}}a\_{h}^{j}(x\_{i}), |  |

where ahj​(xi)a\_{h}^{j}(x\_{i}) is the output of attention head hh in block jj, evaluated at position tlastt\_{\mathrm{last}}, and 𝒟τ+\mathcal{D}\_{\tau}^{+} denotes the set of correctly answered prompts. We additionally constrain all selected heads to come from the same block.

Hidden-state extraction. Alternatively, we extract the block-output hidden state at block ℓ\ell and position tlastt\_{\mathrm{last}}:

|  |  |  |
| --- | --- | --- |
|  | vτℓ=1|𝒟τ+|​∑xi∈𝒟τ+hℓ,tlast​(xi),v\_{\tau}^{\ell}=\frac{1}{|\mathcal{D}\_{\tau}^{+}|}\sum\_{x\_{i}\in\mathcal{D}\_{\tau}^{+}}h\_{\ell,t\_{\mathrm{last}}}(x\_{i}), |  |

where hℓ,tlast​(xi)∈ℝdh\_{\ell,t\_{\mathrm{last}}}(x\_{i})\in\mathbb{R}^{d} is the post-MLP hidden state at block ℓ\ell and position tlastt\_{\mathrm{last}}.

#### Task similarity

We measure similarity between tasks via cosine similarity between their task representations. The hypothesis predicts that tasks with higher representational similarity exhibit more similar learning trajectories.

### 3.5 Evaluation Protocol

We evaluate the Implicit Curriculum Hypothesis through two
complementary analyses, corresponding to the behavioral claims
(H1, H2) and the representational claim (H3).

#### Testing compositional ordering (H1).

For each composite task cc with a known set of prerequisite tasks P​(c)P(c), we check whether all prerequisites emerge no later than the composite:

|  |  |  |
| --- | --- | --- |
|  | ∀τ∈P(c):tτ∗(m)≤tc∗(m)\forall\,\tau\in P(c):\quad t^{\*}\_{\tau}(m)\leq t^{\*}\_{c}(m) |  |

We report the violation rate: the fraction of (composite, prerequisite, model) triples for which this ordering is violated across all models. For synthetic chain composites, P​(c)P(c) is known by construction.

#### Testing cross-model stability (H2).

For each pair of models (m1,m2)(m\_{1},m\_{2}), we compute the Spearman rank correlation between their emergence orderings σm1\sigma\_{m\_{1}}, σm2\sigma\_{m\_{2}} over the full task set. We report correlations separately for the absolute and relative threshold definitions. For tasks that remain unemerged by the end of training, we bin them into one bucket at the end.222i.e. their emergence time is considered to be 1001B tokens for a 1T training run

#### Leave-one-out prediction of composite trajectories (H3).

We operationalize H3 through a leave-one-out (LOO) protocol over composite tasks. For a held-out composite task cc, we predict its learning trajectory from the trajectories of its nearest neighbors in FV space.

Before prediction, we interpolate basis task trajectories onto the held-out task’s token grid, apply Gaussian smoothing (σ=1.0\sigma=1.0), and also discard tasks with near-zero trajectory variance.333in practice, this ended up being compositions of the reverse task as they were usually 0 throughout training.

We extract unit-normalized residual stream representations for all tasks at the selected layer and compute pairwise similarities using an RBF kernel:

|  |  |  |
| --- | --- | --- |
|  | K​(vi,vj)=exp⁡(−‖vi−vj‖22​σk2)K(v\_{i},v\_{j})=\exp\!\Big(\!-\frac{\|v\_{i}-v\_{j}\|^{2}}{2\sigma\_{k}^{2}}\Big) |  |

We use kernel ridge regression to learn a predictor for the held-out task performance: Let SS denote the set of training tasks (excluding cc), let KS∈ℝ|S|×|S|K\_{S}\in\mathbb{R}^{|S|\times|S|} be the kernel matrix with entries

|  |  |  |
| --- | --- | --- |
|  | (KS)i​j=K​(vτi,vτj),(K\_{S})\_{ij}=K(v\_{\tau\_{i}},v\_{\tau\_{j}}), |  |

and let

|  |  |  |
| --- | --- | --- |
|  | kc=[K​(vc,vτj)]j∈Sk\_{c}=\big[K(v\_{c},v\_{\tau\_{j}})\big]\_{j\in S} |  |

be the vector of similarities between the held-out task and the training tasks. For each training step tt, we form the vector of training trajectory values

|  |  |  |
| --- | --- | --- |
|  | yt=[aτj​(t)]j∈S.y\_{t}=\big[a\_{\tau\_{j}}(t)\big]\_{j\in S}. |  |

Kernel ridge regression solves

|  |  |  |
| --- | --- | --- |
|  | αt=(KS+λ​I)−1​yt\alpha\_{t}=(K\_{S}+\lambda I)^{-1}y\_{t} |  |

and predicts the held-out trajectory as

|  |  |  |
| --- | --- | --- |
|  | a^c​(t)=kc⊤​αt.\hat{a}\_{c}(t)=k\_{c}^{\top}\alpha\_{t}. |  |

We evaluate prediction quality via per-task Pearson r2r^{2} and MAE against smoothed ground-truth trajectories, and report both per-task results and means across all held-out composites.
To test the composition bottleneck, we compare two conditions for the function vector space:

1. 1.

   All tasks: the basis includes both simple
   and composite tasks excluding the held-out
   target.
2. 2.

   Simple tasks only: the basis is restricted to
   non-composite tasks only.

If prediction quality degrades substantially under the elementals-only basis, this indicates that composite trajectories share structure with one another that is not captured by their elemental components alone, indicating a composition bottleneck.

## 4 Emergence Order Results

Figure 2: Emergence order heatmap of selected tasks across models (absolute threshold = 0.8). Tasks sorted by consensus emergence order. Consistent color gradients across columns indicate stable ordering.

We first test H1 and H2 by examining whether the emergence order of tasks is consistent across models and whether composites emerge after their constituent components. [Figure 2](#S4.F2 "Figure 2 ‣ 4 Emergence Order Results ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") shows the emergence times of all
tasks across models, sorted vertically by consensus emergence order.
From inspection, it is clear that tasks that emerge early in one model tend to do so across all models. Copying and simple coreference resolution emerge early across all models, in line with previous work yin2025which.
These are followed by simple ICL tasks such as uppercasing and lowercasing, then morphological transformations, followed by knowledge-dependent tasks such as translation, and finally
a long tail of more difficult or compositional tasks. Furthermore, we examine whether compositional tasks arise after their components. This is generally the case: among all compositional tasks, 54/76 emerged no earlier than their “parent” tasks.
However, we also observe a number of inversions, where the composite task emerged earlier than one (19 weak inversions) or both (3 strong inversions) parents. Notably, all three strong inversions involve the first\_letter component task.

[Table 2](#S4.T2 "Table 2 ‣ 4 Emergence Order Results ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") quantifies the consistency of emergence order across models. Within the OLMo-2 family, Spearman rank correlations range from .72 to .93.
Cross-family correlations are also high: Amber correlates with OLMo-2 models at 0.82–0.88, while correlations with older and smaller models (e.g., OLMo-2 vs. Pythia-410M) remain substantial, ranging from 0.64 to 0.84.
All correlations are highly significant and remain so after correction for multiple comparisons.
Importantly, this consistency holds only under the absolute threshold definition of emergence. When using relative thresholds, cross-model correlations drop substantially ([Appendix F](#A6 "Appendix F Emergence Order Agreement Under Alternate Definitions ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis")).
We hypothesize that this discrepancy arises because relative thresholds depend on each model’s maximum performance: a weak model may reach a relative threshold early despite lacking meaningful task competence, while a stronger model may never satisfy the same criterion.
In contrast, our absolute thresholds are set above chance for all tasks, effectively capturing the point at which the underlying computation becomes functional.
This plausibly corresponds to the formation of a task-relevant circuit.
In this view, the relative consistency of the absolute order suggests that what is shared across models is the order in which computations become feasible under standard pretraining, even when trained on differing data distributions.

Table 2: Spearman rank correlation (ρ\rho) of emergence orderings between model pairs (absolute threshold = 80%). All 45 correlations are significant (p<10−7p<10^{-7}).

|  | OLMo2 | | | OLMo3 | LLM360 | | Pythia | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 1B | 7B | 13B | 7B | Amber | Crystal | 410M | 1.4B | 2.8B | 12B |
| O2-1B | — | .80 | .72 | .75 | .88 | .76 | .84 | .92 | .78 | .84 |
| O2-7B |  | — | .93 | .93 | .88 | .92 | .71 | .72 | .77 | .84 |
| O2-13B |  |  | — | .93 | .82 | .89 | .64 | .66 | .76 | .83 |
| O3-7B |  |  |  | — | .83 | .92 | .70 | .72 | .79 | .85 |
| Amber |  |  |  |  | — | .87 | .77 | .84 | .87 | .90 |
| Crystal |  |  |  |  |  | — | .70 | .75 | .83 | .87 |
| P-410M |  |  |  |  |  |  | — | .88 | .72 | .79 |
| P-1.4B |  |  |  |  |  |  |  | — | .77 | .82 |
| P-2.8B |  |  |  |  |  |  |  |  | — | .91 |
| P-12B |  |  |  |  |  |  |  |  |  | — |

.64 
   .70 
   .75 
   .80 
   .85 
   .90+

## 5 Representational Similarity and Prediction Results

Having established that skill acquisition during pretraining is both structured and consistent (H1, H2), we next ask whether this structure is reflected in the model’s internal representations (H3). Namely, if two tasks have similar function vectors, do they exhibit similar learning trajectories in pretraining? Rather than testing correlations in isolation, we consider a stronger version: can the learning trajectory of a held-out composite task be predicted solely from its representational similarity to other tasks, without further evaluation during training?

Table 3: Leave-one-out prediction of held-out composite task trajectories (26 tasks). All tasks includes simple and composite tasks. Sim. only includes only simple tasks. Restricting to elementals degrades MAE for every model (mean Δ\DeltaMAE =+.135=+.135), indicating a composition bottleneck.

|  | All tasks | | Sim. only | |
| --- | --- | --- | --- | --- |
| Model | R2R^{2} | MAE | R2R^{2} | MAE |
| Pythia-410M | .681 | .195 | .717 | .301 |
| OLMo2-1B | .723 | .070 | .602 | .289 |
| Pythia-1.4B | .778 | .086 | .755 | .193 |
| Amber (7B) | .751 | .082 | .725 | .205 |
| Crystal (7B) | .676 | .133 | .568 | .315 |
| OLMo2-7B | .767 | .068 | .693 | .208 |
| OLMo3-7B | .692 | .079 | .491 | .215 |
| Pythia-12B | .812 | .136 | .789 | .194 |
| OLMo2-13B | .838 | .099 | .860 | .242 |

[Table 3](#S5.T3 "Table 3 ‣ 5 Representational Similarity and Prediction Results ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") reports leave-one-out prediction results for composite task trajectories using kernel ridge regression in function vector space.
When the basis includes all other tasks (elemental and composite), prediction quality is strong: R2R^{2} ranges from .67 (Crystal) to .838 (OLMo2-13B), with MAE between .068 and .195 on a 0-1 accuracy scale.
These results provide strong evidence that representational geometry is closely linked to learning dynamics, supporting H3.
As a case study of specific predicted trajectories, [Figure 3](#S5.F3 "Figure 3 ‣ 5 Representational Similarity and Prediction Results ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") shows representative predicted trajectories compared to ground truth trajectories for OLMo2-7B from 0-1T tokens. For tasks such as fr\_eng\_upper (R2=.99R^{2}=.99, MAE =.017=.017) and plural\_lower (R2=.89R^{2}=.89, MAE =.028=.028), the predicted curve closely tracks the actual trajectory, capturing both the onset of emergence and the subsequent rate of improvement. However, predictions are weaker for tasks such as eng\_fr\_upper (R2=.51R^{2}=.51, MAE =.068=.068), where the held-out task’s trajectory is less well approximated by its nearest neighbors in representation space.
Full prediction results can be found in [Appendix H](#A8 "Appendix H All Held-out trajectory predictions ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis").

Figure 3: Example composite task predictions for OLMo2-7B between 0-1T tokens.

## 6 Conclusion

In this paper, we unify several threads of discussion on the emergence of LM capabilities during pretraining in the Implicit Curriculum Hypothesis – that skills acquired during pretraining emerge in a stable order driven by concrete compositions. We test this hypothesis empirically across several model families and with models spanning 410M-13B parameters. Our empirical findings support both the behavioural and representational aspects of the hypothesis: emergence orders of tasks under absolute thresholds are quite consistent across models, even across families and in models trained on different data. Furthermore, similarity in the function vector space predicts similarity of learning trajectories, such that it is possible to predict the trajectories of held-out compositional tasks from function vector similarity without evaluating them. This indicates that the developmental structure visible in
behavioral evaluations may also be legible in the model’s internal
representations.

Our results open several avenues for further investigation. One practical application is pretraining monitoring – if emergence orders are stable and predictable, our task suite can serve as a basis for monitoring whether models are developing capabilities ahead of or behind schedule. Furthermore, understanding this task structure could also inform data mixture decisions. More broadly, we hope that the framework of studying pretraining as a structured developmental process will prove useful for understanding, predicting, and ultimately steering what language models learn.

## Acknowledgments

EL was supported by the National Sciences and Engineering Research Council of Canada (NSERC), [funding reference number 578085], as well as the SoftBank-ARM Fellowship. ML is supported by a NSF Graduate Research Fellowship. IL is supported by a Technical AI Safety Research Grant from Coefficient Giving via Berkeley Existential Risk Initiative.

This work used the Delta system at the National Center for Supercomputing Applications [award OAC 2005572] through allocation [CIS250578] from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

## Appendix A LLM Usage Disclosure

Claude Opus 4.6 and Sonnet 4.6 were used to format tables and conduct minor writing edits. All outputs were reviewed, and verified by the authors.

## Appendix B Extended Related Work

#### Skill Emergence and Scaling Laws.

Theoretical work has sought to explain how capabilities emerge with scale.
arora2023theoryemergencecomplexskills propose that scaling laws arise from slingshot generalization, where competence at kk-tuples of skills emerges at the same rate as elementary skills themselves.
Similarly, michaud2023the introduce the quanta hypothesis, modeling skills as discrete units whose power-law frequency distribution explains smooth scaling curves.
Both theories predict that complex behaviors emerge from simpler building blocks, but leave open the question of what these building blocks are and how they compose in practice.
Our work provides empirical grounding for these theories by tracking probe tasks designed to be compositionally combined, finding that compositional skills reliably emerge after their constituent components.

#### Skill Evaluation and Structure.

Several approaches characterize LLM capabilities through evaluation-time analysis.
burnell2023revealingstructurelanguagemodel apply factor analysis across 29 models and 27 tasks, finding three latent factors, reasoning, comprehension, and language modeling, that explain performance variation; maimon2025iqtestllmsevaluation scale this psychometric approach to 60 models and 44 tasks, identifying eight core skills.
Beyond identifying skills, yu2024skillmix directly test compositional ability by evaluating whether models can combine kk-tuples of language skills in novel ways.
polo2025sloth unify these perspectives through skill-based scaling laws where performance is driven by low-dimensional latent skills.
These works analyze fully trained models; we complement them by studying *how skills develop during pretraining* and linking emergence order to representational structure.

#### Training Dynamics and Phase Transitions.

Understanding what models learn during training has gained increasing attention.
chen2023sudden identify sudden drops in loss corresponding to syntax acquisition and other phase transitions; kangaslahti2025hidden show that such breakthroughs occur frequently but are obscured by aggregate loss metrics.
van2025polypythias release 50 additional training runs of Pythia models, finding consistent learning phases across seeds and sizes.
Other work examines specific capabilities: sun-dredze-2025-amuro investigate how downstream performance develops across pretraining checkpoints, ge2025evolution track feature evolution using sparse dictionary learning, and mishranext show that mathematical skills emerge in an order correlated with human curriculum despite random data ordering.
Our work contributes to this literature by demonstrating that emergence orderings are stable across model families and can be predicted from representational geometry.

#### Representations for Task Understanding.

Mechanistic interpretability has revealed compact representations of tasks within model activations.
Both todd2024function and hendel-etal-2023-context discover that in-context learning compresses task demonstrations into single directions, termed function vectors and task vectors respectively, which can trigger task execution even in zero-shot settings.
Subsequent work explores the scope of this phenomenon: todd2026incontextalgebra extend it to symbolic reasoning with variable-based tokens, while khandelwal2025languagemodelscomposefunctions investigate compositional tasks and find both compositional and direct processing mechanisms.
We build on this line of work by using residual-stream representations to predict learning trajectories of compositional tasks, connecting representational geometry to training dynamics.
This complements recent work by prasad2026featuresrewardsscalablesupervision that uses interpretable features for training monitoring in RL.

## Appendix C Full list of Elemental and Composite Tasks

We provide a full list of tasks, categorized into reasoning types, in [Table 4](#A4.T4 "Table 4 ‣ Appendix D Full list of Elemental and Composite Tasks ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") and [Table 5](#A4.T5 "Table 5 ‣ Appendix D Full list of Elemental and Composite Tasks ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis"). TextFRCT tasks are taken from the psychometrics literature (ekstrom1976kit), while other tasks have been studied or were inspired by works in interpretability literature (wang2023interpretability; todd2024function; chen2024parallel; feucht2025dualroute).

## Appendix D Full list of Elemental and Composite Tasks

We provide a full list of tasks, categorized into reasoning types, in [Table 4](#A4.T4 "Table 4 ‣ Appendix D Full list of Elemental and Composite Tasks ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") and [Table 5](#A4.T5 "Table 5 ‣ Appendix D Full list of Elemental and Composite Tasks ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis"). TextFRCT tasks are taken from the psychometrics literature (ekstrom1976kit), while other tasks have been studied or were inspired by works in interpretability literature (wang2023interpretability; todd2024function; chen2024parallel; feucht2025dualroute).

Table 4: All elemental tasks in the evaluation suite with representative examples.

| Task | N | Input | Output |
| --- | --- | --- | --- |
| String Operations | | | |
| copying | 20 | gTpigTHK | gTpigTHK |
| token\_reversal | 20 | cat | tac |
| string\_analogy | 10 | abc →\to abd, ijk →\to ? | ijl |
| simple\_icl:uppercase | 26 | b | B |
| simple\_icl:lowercase | 26 | B | b |
| simple\_icl:first\_letter | 190 | the cat went up the tree | t |
| simple\_icl:last\_letter | 190 | the cat went up the tree | e |
| Morphology | | | |
| simple\_icl:present\_to\_gerund | 179 | run | running |
| simple\_icl:singular\_to\_plural | 165 | child | children |
| Translation | | | |
| simple\_icl:translate\_eng\_fr | 173 | hello | bonjour |
| simple\_icl:translate\_fr\_eng | 175 | bonjour | hello |
| simple\_icl:translate\_eng\_sp | 178 | hello | hola |
| simple\_icl:translate\_sp\_eng | 178 | hola | hello |
| World Knowledge | | | |
| simple\_icl:country\_to\_capital | 184 | Afghanistan | Kabul |
| simple\_icl:country\_to\_currency | 198 | United States | Dollar |
| Arithmetic | | | |
| basic\_arithmetic | 10 | What is 5 + 3? | 8 |
| math | 20 | 4 \* 1 | 4 |
| multistep\_arithmetic:two\_step | 20 | 3 + 4, then multiply by 2 | 14 |
| multistep\_arithmetic:three\_step | 20 | Start with 10, subtract 3, then multiply by 4 | 28 |
| textfrct:RG1 | 30 | In general, brass is made of two parts copper to one part zinc. How many pounds of zinc are needed to produce 45 pounds of brass? (MCQ) | B |
| textfrct:RG2 | 30 | Recipe A uses 1.5 cups of sugar; Recipe B uses 2. Making 8 cakes, how many fewer cups does Recipe A require? (MCQ) | E |
| textfrct:RG3 | 30 | There are 4 quarts in a gallon and 4 cups in a quart. How many cups are in a gallon? (MCQ) | C |
| Logic | | | |
| logical\_ops:negation | 12 | Statement: All robots can move.  Candidate: Some robots cannot move.  Is this a correct logical negation? | True |
| logical\_ops:conjunction | 12 | Fact A is True. Fact B is True.  Claim: A AND B. Is the claim true? | True |
| logical\_ops:conditional | 12 | Rule: If it rains, the ground gets wet.  Fact: It rains. Does the conclusion follow? | True |
| textfrct:RL1 | 30 | All birds have purple tails. All cats are birds. Therefore all cats have purple tails. (MCQ: correct/incorrect) | G |
| textfrct:RL3 | 20 | More fatal accidents occur on highways after dark than during daylight hours. (MCQ: which conclusion follows?) | 3 |
| textfrct:RL4 | 24 | ICL ex.: black sheep = dag kip; white dog = tin bud; black cow = dag stam  Query: white sheep = ? (MCQ) | 2 |
| Reading Comprehension | | | |
| fact\_extraction:extract\_entity | 20 | Passage: “Alice gave five apples to Bob at the park.” Who received the apples? | Bob |
| fact\_extraction:extract\_number | 20 | Passage: “John gave 5 apples to Mary on Tuesday.” How many apples did John give? | 5 |
| fact\_extraction:extract\_location | 20 | Passage: “The cat sat on the red mat in the kitchen.” Where is the mat? | the kitchen |
| coreference:pronoun\_simple | 20 | “Alice told Bob that she would be late.” Who does “she” refer to? | Alice |
| coreference:pronoun\_hard | 20 | “The trophy didn’t fit in the suitcase because it was too big.” What was too big? | the trophy |
| ignoring\_context | 5 | Some text here. X = 5. More text.  Question: What is X? | 5 |
| ioi\_task | 1000 | Instr.: Identify who should be referenced.  Then, Henry and Phil had a lot of fun at the harbor. Henry gave a basket to | [Phil, Henry] |
| part\_of\_speech | 15 | The cat is in the house. The part of speech for “cat” is \_ | noun |
| Verbal Closure (FRCT) | | | |
| textfrct:CV1 | 50 | erte | tree, rete |
| textfrct:CV2 | 40 | EZIRTMODSLOWTSEXQILNECKBWOCJAKX | SLOW, NECK |
| textfrct:CV3 | 36 | \_tam\_ | stamp |
| Induction (FRCT) | | | |
| textfrct:I1 | 30 | Instr.: One of the five letter sets does NOT follow the same pattern as the others. Find it.  1. QPPQ  2. HGHH  3. TTTU  4. DDDE  5. MLMM | 1 |
| textfrct:I2 | 28 | Instr.: Each row marks one position with ‘x’. Identify the pattern and find the correct position in row 5.  ------- x------- ---- --  ---- -x--- -- --- ------  --------------- --x-----  -------- ---x-----------  ----1 2---3-- 4---5----- | 3 |
| Associative Memory (FRCT) | | | |
| textfrct:MA2 | 30 | Instr.: Memorize 30 word–number pairs, then answer retrieval queries.  Query: What number corresponds to ‘coat’? | 49 |
| textfrct:MA3 | 30 | Instr.: Memorize 30 first–last name pairs, then answer retrieval queries.  Query: Last name: Nichols | Edward |
| Verbal Comprehension (FRCT) | | | |
| textfrct:V1 | 36 | Instr.: Choose the best definition (MCQ).  ‘airtight’: (1) firm   (2) light   (3) hermetically sealed   (4) plane sick | 3 |
| textfrct:V2 | 36 | Instr.: Choose the best definition (MCQ).  ‘handicraft’: (1) cunning   (2) fast boat   (3) utility   (4) manual skill   (5) guild | 4 |
| textfrct:V3 | 48 | Instr.: Choose the best definition (MCQ).  ‘cottontail’: (1) squirrel   (2) poplar   (3) boa   (4) marshy plant   (5) rabbit | 5 |
| textfrct:V4 | 36 | Instr.: Choose the best definition (MCQ).  ‘mumble’: (1) speak indistinctly   (2) complain   (3) handle awkwardly   (4) fall   (5) tear apart | 1 |
| textfrct:V5 | 36 | Instr.: Choose the best definition (MCQ).  ‘rancor’: (1) forbearance   (2) ridicule   (3) malice   (4) bravery | 3 |




Table 5: All compositional tasks in the evaluation suite with representative examples.

| Task | N | Input | Output |
| --- | --- | --- | --- |
| Morphology ×\times String Operation | | | |
| compositional:gerund\_lower | 178 | RUN | running |
| compositional:gerund\_upper | 178 | run | RUNNING |
| compositional:gerund\_reverse | 178 | run | gninnur |
| compositional:gerund\_upper\_reverse | 178 | run | GNINNUR |
| compositional:plural\_lower | 165 | CHILD | children |
| compositional:plural\_upper | 165 | child | CHILDREN |
| compositional:plural\_reverse | 165 | child | nerdlihc |
| compositional:plural\_upper\_reverse | 165 | child | NERDLIHC |
| Translation ×\times String Operation | | | |
| compositional:translate\_eng\_fr\_first | 173 | hello | b |
| compositional:translate\_eng\_fr\_last | 173 | hello | r |
| compositional:translate\_eng\_fr\_lower | 173 | HELLO | bonjour |
| compositional:translate\_eng\_fr\_reverse | 173 | hello | ruojnob |
| compositional:translate\_eng\_fr\_upper | 173 | hello | BONJOUR |
| compositional:translate\_eng\_fr\_upper\_reverse | 173 | hello | RUOJNOB |
| compositional:translate\_eng\_sp\_first | 178 | hello | h |
| compositional:translate\_eng\_sp\_last | 178 | hello | a |
| compositional:translate\_eng\_sp\_lower | 178 | HELLO | hola |
| compositional:translate\_eng\_sp\_reverse | 178 | hello | aloh |
| compositional:translate\_eng\_sp\_upper | 178 | hello | HOLA |
| compositional:translate\_eng\_sp\_upper\_reverse | 178 | hello | ALOH |
| compositional:translate\_fr\_eng\_first | 171 | bonjour | h |
| compositional:translate\_fr\_eng\_last | 171 | bonjour | o |
| compositional:translate\_fr\_eng\_lower | 171 | BONJOUR | hello |
| compositional:translate\_fr\_eng\_reverse | 171 | bonjour | olleh |
| compositional:translate\_fr\_eng\_upper | 171 | bonjour | HELLO |
| compositional:translate\_sp\_eng\_first | 178 | hola | h |
| compositional:translate\_sp\_eng\_last | 178 | hola | o |
| compositional:translate\_sp\_eng\_lower | 178 | HOLA | hello |
| compositional:translate\_sp\_eng\_reverse | 178 | hola | olleh |
| compositional:translate\_sp\_eng\_upper | 178 | hola | HELLO |
| Case/Reversal Chains | | | |
| compositional:lower\_first | 971 | AFGHANISTAN | a |
| compositional:lower\_last | 971 | AFGHANISTAN | n |
| compositional:lower\_reverse | 971 | AFGHANISTAN | natsinahgfa |
| compositional:upper\_first | 971 | afghanistan | A |
| compositional:upper\_last | 971 | afghanistan | N |
| compositional:upper\_reverse | 971 | afghanistan | NATSINAHGFA |
| compositional:reverse\_first | 971 | Afghanistan | n |
| compositional:reverse\_last | 971 | Afghanistan | A |

## Appendix E Full learning trajectories by category

Figures [4](#A5.F4 "Figure 4 ‣ Appendix E Full learning trajectories by category ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") – [12](#A5.F12 "Figure 12 ‣ Appendix E Full learning trajectories by category ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") show full learning trajectories of tasks for each model.

Figure 4: Complete trajectories for Pythia-410M over 300B tokens.




Figure 5: Complete trajectories for OLMo2-1B over 1T tokens.




Figure 6: Complete trajectories for Pythia-1.4B over 300B tokens.




Figure 7: Complete trajectories for OLMo-2 7B over 1T tokens.




Figure 8: Complete trajectories for OLMo-3 7B over 1T tokens.




Figure 9: Complete trajectories for Amber (7B) over 1T tokens.




Figure 10: Complete trajectories for CrystalCoder (7B) over 1T tokens.




Figure 11: Complete trajectories for Pythia-12B over 300B tokens. Note that this model exhibits some instabilities compared to others.




Figure 12: Complete trajectories for OLMo2-13b over 1T tokens. Note that this model exhibits some instabilities compared to others.

## Appendix F Emergence Order Agreement Under Alternate Definitions

Table 6: Summary of emergence ordering consistency under different definitions. Absolute thresholds yield substantially higher cross-model correlations than relative thresholds.

| Definition | nn pairs | Mean ρ\rho | Min ρ\rho | Max ρ\rho |
| --- | --- | --- | --- | --- |
| Absolute threshold (θ=0.5\theta=0.5) | 36 | 0.860 | 0.597 | 0.955 |
| Absolute threshold (θ=0.8\theta=0.8, stable for 3 consecutive checkpoints) | 36 | 0.790 | 0.599 | 0.961 |
| Relative threshold (α=0.5\alpha=0.5, fraction of max performance) | 36 | 0.528 | 0.085 | 0.866 |
| Relative threshold (α=0.8\alpha=0.8, fraction of max performance) | 36 | 0.500 | 0.077 | 0.773 |




Table 7: Pairwise Spearman rank correlations: Absolute threshold (θ=0.5\theta=0.5). Mean ρ=0.860\rho=0.860.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model A | Model B | nn tasks | ρ\rho | pp |
| Within-family (OLMo-2) | | | | |
| OLMo2-1B | OLMo2-7B | 106 | 0.889 | 4.5×10−374.5\times 10^{-37} |
| OLMo2-1B | OLMo2-13B | 105 | 0.832 | 4.5×10−284.5\times 10^{-28} |
| OLMo2-7B | OLMo2-13B | 104 | 0.865 | 2.9×10−322.9\times 10^{-32} |
| Within-family (Pythia) | | | | |
| Pythia-1.4B | Pythia-410M | 96 | 0.910 | 1.3×10−371.3\times 10^{-37} |
| Pythia-1.4B | Pythia-12B | 98 | 0.909 | 3.7×10−383.7\times 10^{-38} |
| Pythia-410M | Pythia-12B | 100 | 0.815 | 6.5×10−256.5\times 10^{-25} |
| Within-family (LLM360) | | | | |
| Amber | Crystal | 102 | 0.905 | 6.8×10−396.8\times 10^{-39} |
| Cross-family (OLMo-2 ↔\leftrightarrow OLMo-3) | | | | |
| OLMo2-1B | OLMo3-7B | 106 | 0.925 | 1.4×10−451.4\times 10^{-45} |
| OLMo2-7B | OLMo3-7B | 105 | 0.918 | 4.7×10−434.7\times 10^{-43} |
| OLMo2-13B | OLMo3-7B | 104 | 0.897 | 6.5×10−386.5\times 10^{-38} |
| Cross-family (OLMo-2 ↔\leftrightarrow LLM360) | | | | |
| Amber | OLMo2-1B | 107 | 0.932 | 5.5×10−485.5\times 10^{-48} |
| Amber | OLMo2-7B | 106 | 0.907 | 7.2×10−417.2\times 10^{-41} |
| Amber | OLMo2-13B | 105 | 0.839 | 5.6×10−295.6\times 10^{-29} |
| Crystal | OLMo2-1B | 102 | 0.913 | 1.3×10−401.3\times 10^{-40} |
| Crystal | OLMo2-7B | 101 | 0.910 | 1.4×10−391.4\times 10^{-39} |
| Crystal | OLMo2-13B | 100 | 0.889 | 6.3×10−356.3\times 10^{-35} |
| Cross-family (OLMo-3 ↔\leftrightarrow LLM360) | | | | |
| Amber | OLMo3-7B | 106 | 0.918 | 1.6×10−431.6\times 10^{-43} |
| Crystal | OLMo3-7B | 102 | 0.955 | 8.1×10−558.1\times 10^{-55} |
| Cross-family (OLMo-2 ↔\leftrightarrow Pythia) | | | | |
| Pythia-1.4B | OLMo2-1B | 98 | 0.907 | 9.6×10−389.6\times 10^{-38} |
| Pythia-1.4B | OLMo2-7B | 97 | 0.830 | 8.7×10−268.7\times 10^{-26} |
| Pythia-1.4B | OLMo2-13B | 97 | 0.716 | 1.6×10−161.6\times 10^{-16} |
| Pythia-410M | OLMo2-1B | 100 | 0.834 | 4.8×10−274.8\times 10^{-27} |
| Pythia-410M | OLMo2-7B | 99 | 0.793 | 1.2×10−221.2\times 10^{-22} |
| Pythia-410M | OLMo2-13B | 98 | 0.597 | 8.4×10−118.4\times 10^{-11} |
| Pythia-12B | OLMo2-1B | 102 | 0.856 | 2.1×10−302.1\times 10^{-30} |
| Pythia-12B | OLMo2-7B | 101 | 0.832 | 5.1×10−275.1\times 10^{-27} |
| Pythia-12B | OLMo2-13B | 100 | 0.786 | 3.4×10−223.4\times 10^{-22} |
| Cross-family (OLMo-3 ↔\leftrightarrow Pythia) | | | | |
| Pythia-1.4B | OLMo3-7B | 97 | 0.864 | 4.6×10−304.6\times 10^{-30} |
| Pythia-410M | OLMo3-7B | 99 | 0.799 | 3.9×10−233.9\times 10^{-23} |
| Pythia-12B | OLMo3-7B | 101 | 0.867 | 1.2×10−311.2\times 10^{-31} |
| Cross-family (LLM360 ↔\leftrightarrow Pythia) | | | | |
| Amber | Pythia-1.4B | 98 | 0.935 | 7.4×10−457.4\times 10^{-45} |
| Amber | Pythia-410M | 100 | 0.853 | 1.9×10−291.9\times 10^{-29} |
| Amber | Pythia-12B | 102 | 0.930 | 3.6×10−453.6\times 10^{-45} |
| Crystal | Pythia-1.4B | 93 | 0.824 | 3.8×10−243.8\times 10^{-24} |
| Crystal | Pythia-410M | 96 | 0.751 | 1.2×10−181.2\times 10^{-18} |
| Crystal | Pythia-12B | 97 | 0.850 | 3.3×10−283.3\times 10^{-28} |




Table 8: Pairwise Spearman rank correlations: Absolute threshold (θ=0.8\theta=0.8, stable for 3 consecutive checkpoints). Mean ρ=0.790\rho=0.790.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model A | Model B | nn tasks | ρ\rho | pp |
| Within-family (OLMo-2) | | | | |
| OLMo2-1B | OLMo2-7B | 106 | 0.718 | 4.5×10−184.5\times 10^{-18} |
| OLMo2-1B | OLMo2-13B | 105 | 0.721 | 4.0×10−184.0\times 10^{-18} |
| OLMo2-7B | OLMo2-13B | 104 | 0.934 | 2.0×10−472.0\times 10^{-47} |
| Within-family (Pythia) | | | | |
| Pythia-1.4B | Pythia-410M | 96 | 0.824 | 6.1×10−256.1\times 10^{-25} |
| Pythia-1.4B | Pythia-12B | 98 | 0.792 | 2.9×10−222.9\times 10^{-22} |
| Pythia-410M | Pythia-12B | 100 | 0.689 | 2.3×10−152.3\times 10^{-15} |
| Within-family (LLM360) | | | | |
| Amber | Crystal | 102 | 0.823 | 2.7×10−262.7\times 10^{-26} |
| Cross-family (OLMo-2 ↔\leftrightarrow OLMo-3) | | | | |
| OLMo2-1B | OLMo3-7B | 106 | 0.743 | 7.9×10−207.9\times 10^{-20} |
| OLMo2-7B | OLMo3-7B | 105 | 0.961 | 3.5×10−593.5\times 10^{-59} |
| OLMo2-13B | OLMo3-7B | 104 | 0.953 | 8.3×10−558.3\times 10^{-55} |
| Cross-family (OLMo-2 ↔\leftrightarrow LLM360) | | | | |
| Amber | OLMo2-1B | 107 | 0.785 | 1.6×10−231.6\times 10^{-23} |
| Amber | OLMo2-7B | 106 | 0.877 | 6.2×10−356.2\times 10^{-35} |
| Amber | OLMo2-13B | 105 | 0.875 | 3.0×10−343.0\times 10^{-34} |
| Crystal | OLMo2-1B | 102 | 0.695 | 5.0×10−165.0\times 10^{-16} |
| Crystal | OLMo2-7B | 101 | 0.838 | 8.8×10−288.8\times 10^{-28} |
| Crystal | OLMo2-13B | 100 | 0.868 | 1.5×10−311.5\times 10^{-31} |
| Cross-family (OLMo-3 ↔\leftrightarrow LLM360) | | | | |
| Amber | OLMo3-7B | 106 | 0.877 | 6.5×10−356.5\times 10^{-35} |
| Crystal | OLMo3-7B | 102 | 0.853 | 6.1×10−306.1\times 10^{-30} |
| Cross-family (OLMo-2 ↔\leftrightarrow Pythia) | | | | |
| Pythia-1.4B | OLMo2-1B | 98 | 0.883 | 2.3×10−332.3\times 10^{-33} |
| Pythia-1.4B | OLMo2-7B | 97 | 0.693 | 3.6×10−153.6\times 10^{-15} |
| Pythia-1.4B | OLMo2-13B | 97 | 0.669 | 7.3×10−147.3\times 10^{-14} |
| Pythia-410M | OLMo2-1B | 100 | 0.779 | 1.4×10−211.4\times 10^{-21} |
| Pythia-410M | OLMo2-7B | 99 | 0.686 | 4.4×10−154.4\times 10^{-15} |
| Pythia-410M | OLMo2-13B | 98 | 0.629 | 4.2×10−124.2\times 10^{-12} |
| Pythia-12B | OLMo2-1B | 102 | 0.777 | 7.4×10−227.4\times 10^{-22} |
| Pythia-12B | OLMo2-7B | 101 | 0.786 | 2.1×10−222.1\times 10^{-22} |
| Pythia-12B | OLMo2-13B | 100 | 0.843 | 4.4×10−284.4\times 10^{-28} |
| Cross-family (OLMo-3 ↔\leftrightarrow Pythia) | | | | |
| Pythia-1.4B | OLMo3-7B | 97 | 0.752 | 7.3×10−197.3\times 10^{-19} |
| Pythia-410M | OLMo3-7B | 99 | 0.689 | 3.1×10−153.1\times 10^{-15} |
| Pythia-12B | OLMo3-7B | 101 | 0.839 | 5.7×10−285.7\times 10^{-28} |
| Cross-family (LLM360 ↔\leftrightarrow Pythia) | | | | |
| Amber | Pythia-1.4B | 98 | 0.805 | 1.7×10−231.7\times 10^{-23} |
| Amber | Pythia-410M | 100 | 0.758 | 7.4×10−207.4\times 10^{-20} |
| Amber | Pythia-12B | 102 | 0.889 | 1.3×10−351.3\times 10^{-35} |
| Crystal | Pythia-1.4B | 93 | 0.703 | 4.2×10−154.2\times 10^{-15} |
| Crystal | Pythia-410M | 96 | 0.599 | 1.1×10−101.1\times 10^{-10} |
| Crystal | Pythia-12B | 97 | 0.830 | 7.1×10−267.1\times 10^{-26} |




Table 9: Pairwise Spearman rank correlations: Relative threshold (α=0.5\alpha=0.5, fraction of max performance). Mean ρ=0.528\rho=0.528.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model A | Model B | nn tasks | ρ\rho | pp |
| Within-family (OLMo-2) | | | | |
| OLMo2-1B | OLMo2-7B | 106 | 0.579 | 8.2×10−118.2\times 10^{-11} |
| OLMo2-1B | OLMo2-13B | 105 | 0.433 | 3.9×10−63.9\times 10^{-6} |
| OLMo2-7B | OLMo2-13B | 104 | 0.563 | 4.8×10−104.8\times 10^{-10} |
| Within-family (Pythia) | | | | |
| Pythia-1.4B | Pythia-410M | 96 | 0.866 | 5.3×10−305.3\times 10^{-30} |
| Pythia-1.4B | Pythia-12B | 98 | 0.828 | 8.1×10−268.1\times 10^{-26} |
| Pythia-410M | Pythia-12B | 100 | 0.748 | 3.5×10−193.5\times 10^{-19} |
| Within-family (LLM360) | | | | |
| Amber | Crystal | 102 | 0.489 | 1.8×10−71.8\times 10^{-7} |
| Cross-family (OLMo-2 ↔\leftrightarrow OLMo-3) | | | | |
| OLMo2-1B | OLMo3-7B | 106 | 0.588 | 3.5×10−113.5\times 10^{-11} |
| OLMo2-7B | OLMo3-7B | 105 | 0.702 | 7.1×10−177.1\times 10^{-17} |
| OLMo2-13B | OLMo3-7B | 104 | 0.659 | 2.7×10−142.7\times 10^{-14} |
| Cross-family (OLMo-2 ↔\leftrightarrow LLM360) | | | | |
| Amber | OLMo2-1B | 107 | 0.587 | 3.0×10−113.0\times 10^{-11} |
| Amber | OLMo2-7B | 106 | 0.516 | 1.5×10−81.5\times 10^{-8} |
| Amber | OLMo2-13B | 105 | 0.279 | 3.9×10−33.9\times 10^{-3} |
| Crystal | OLMo2-1B | 102 | 0.743 | 3.9×10−193.9\times 10^{-19} |
| Crystal | OLMo2-7B | 101 | 0.668 | 2.3×10−142.3\times 10^{-14} |
| Crystal | OLMo2-13B | 100 | 0.571 | 5.5×10−105.5\times 10^{-10} |
| Cross-family (OLMo-3 ↔\leftrightarrow LLM360) | | | | |
| Amber | OLMo3-7B | 106 | 0.513 | 1.9×10−81.9\times 10^{-8} |
| Crystal | OLMo3-7B | 102 | 0.764 | 8.9×10−218.9\times 10^{-21} |
| Cross-family (OLMo-2 ↔\leftrightarrow Pythia) | | | | |
| Pythia-1.4B | OLMo2-1B | 98 | 0.492 | 2.7×10−72.7\times 10^{-7} |
| Pythia-1.4B | OLMo2-7B | 97 | 0.476 | 8.6×10−78.6\times 10^{-7} |
| Pythia-1.4B | OLMo2-13B | 97 | 0.085 | 0.410.41 |
| Pythia-410M | OLMo2-1B | 100 | 0.445 | 3.5×10−63.5\times 10^{-6} |
| Pythia-410M | OLMo2-7B | 99 | 0.452 | 2.6×10−62.6\times 10^{-6} |
| Pythia-410M | OLMo2-13B | 98 | 0.129 | 0.210.21 |
| Pythia-12B | OLMo2-1B | 102 | 0.511 | 4.1×10−84.1\times 10^{-8} |
| Pythia-12B | OLMo2-7B | 101 | 0.531 | 1.1×10−81.1\times 10^{-8} |
| Pythia-12B | OLMo2-13B | 100 | 0.304 | 2.1×10−32.1\times 10^{-3} |
| Cross-family (OLMo-3 ↔\leftrightarrow Pythia) | | | | |
| Pythia-1.4B | OLMo3-7B | 97 | 0.314 | 1.8×10−31.8\times 10^{-3} |
| Pythia-410M | OLMo3-7B | 99 | 0.355 | 3.1×10−43.1\times 10^{-4} |
| Pythia-12B | OLMo3-7B | 101 | 0.460 | 1.3×10−61.3\times 10^{-6} |
| Cross-family (LLM360 ↔\leftrightarrow Pythia) | | | | |
| Amber | Pythia-1.4B | 98 | 0.763 | 7.3×10−207.3\times 10^{-20} |
| Amber | Pythia-410M | 100 | 0.702 | 4.0×10−164.0\times 10^{-16} |
| Amber | Pythia-12B | 102 | 0.753 | 6.9×10−206.9\times 10^{-20} |
| Crystal | Pythia-1.4B | 93 | 0.381 | 1.6×10−41.6\times 10^{-4} |
| Crystal | Pythia-410M | 96 | 0.341 | 6.6×10−46.6\times 10^{-4} |
| Crystal | Pythia-12B | 97 | 0.409 | 3.1×10−53.1\times 10^{-5} |




Table 10: Pairwise Spearman rank correlations: Relative threshold (α=0.8\alpha=0.8, fraction of max performance). Mean ρ=0.500\rho=0.500.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model A | Model B | nn tasks | ρ\rho | pp |
| Within-family (OLMo-2) | | | | |
| OLMo2-1B | OLMo2-7B | 106 | 0.491 | 8.9×10−88.9\times 10^{-8} |
| OLMo2-1B | OLMo2-13B | 105 | 0.359 | 1.7×10−41.7\times 10^{-4} |
| OLMo2-7B | OLMo2-13B | 104 | 0.707 | 4.6×10−174.6\times 10^{-17} |
| Within-family (Pythia) | | | | |
| Pythia-1.4B | Pythia-410M | 96 | 0.716 | 2.3×10−162.3\times 10^{-16} |
| Pythia-1.4B | Pythia-12B | 98 | 0.547 | 5.5×10−95.5\times 10^{-9} |
| Pythia-410M | Pythia-12B | 100 | 0.521 | 2.7×10−82.7\times 10^{-8} |
| Within-family (LLM360) | | | | |
| Amber | Crystal | 102 | 0.632 | 1.0×10−121.0\times 10^{-12} |
| Cross-family (OLMo-2 ↔\leftrightarrow OLMo-3) | | | | |
| OLMo2-1B | OLMo3-7B | 106 | 0.498 | 5.7×10−85.7\times 10^{-8} |
| OLMo2-7B | OLMo3-7B | 105 | 0.773 | 4.8×10−224.8\times 10^{-22} |
| OLMo2-13B | OLMo3-7B | 104 | 0.698 | 1.8×10−161.8\times 10^{-16} |
| Cross-family (OLMo-2 ↔\leftrightarrow LLM360) | | | | |
| Amber | OLMo2-1B | 107 | 0.556 | 5.3×10−105.3\times 10^{-10} |
| Amber | OLMo2-7B | 106 | 0.612 | 3.0×10−123.0\times 10^{-12} |
| Amber | OLMo2-13B | 105 | 0.544 | 2.1×10−92.1\times 10^{-9} |
| Crystal | OLMo2-1B | 102 | 0.634 | 8.8×10−138.8\times 10^{-13} |
| Crystal | OLMo2-7B | 101 | 0.714 | 5.0×10−175.0\times 10^{-17} |
| Crystal | OLMo2-13B | 100 | 0.603 | 3.1×10−113.1\times 10^{-11} |
| Cross-family (OLMo-3 ↔\leftrightarrow LLM360) | | | | |
| Amber | OLMo3-7B | 106 | 0.590 | 2.8×10−112.8\times 10^{-11} |
| Crystal | OLMo3-7B | 102 | 0.674 | 8.6×10−158.6\times 10^{-15} |
| Cross-family (OLMo-2 ↔\leftrightarrow Pythia) | | | | |
| Pythia-1.4B | OLMo2-1B | 98 | 0.580 | 4.0×10−104.0\times 10^{-10} |
| Pythia-1.4B | OLMo2-7B | 97 | 0.286 | 4.5×10−34.5\times 10^{-3} |
| Pythia-1.4B | OLMo2-13B | 97 | 0.077 | 0.450.45 |
| Pythia-410M | OLMo2-1B | 100 | 0.502 | 1.0×10−71.0\times 10^{-7} |
| Pythia-410M | OLMo2-7B | 99 | 0.309 | 1.9×10−31.9\times 10^{-3} |
| Pythia-410M | OLMo2-13B | 98 | 0.159 | 0.120.12 |
| Pythia-12B | OLMo2-1B | 102 | 0.523 | 1.7×10−81.7\times 10^{-8} |
| Pythia-12B | OLMo2-7B | 101 | 0.527 | 1.5×10−81.5\times 10^{-8} |
| Pythia-12B | OLMo2-13B | 100 | 0.383 | 8.3×10−58.3\times 10^{-5} |
| Cross-family (OLMo-3 ↔\leftrightarrow Pythia) | | | | |
| Pythia-1.4B | OLMo3-7B | 97 | 0.177 | 0.080.08 |
| Pythia-410M | OLMo3-7B | 99 | 0.298 | 2.8×10−32.8\times 10^{-3} |
| Pythia-12B | OLMo3-7B | 101 | 0.436 | 5.1×10−65.1\times 10^{-6} |
| Cross-family (LLM360 ↔\leftrightarrow Pythia) | | | | |
| Amber | Pythia-1.4B | 98 | 0.517 | 4.9×10−84.9\times 10^{-8} |
| Amber | Pythia-410M | 100 | 0.437 | 5.5×10−65.5\times 10^{-6} |
| Amber | Pythia-12B | 102 | 0.622 | 3.0×10−123.0\times 10^{-12} |
| Crystal | Pythia-1.4B | 93 | 0.399 | 7.3×10−57.3\times 10^{-5} |
| Crystal | Pythia-410M | 96 | 0.376 | 1.6×10−41.6\times 10^{-4} |
| Crystal | Pythia-12B | 97 | 0.512 | 8.5×10−88.5\times 10^{-8} |

## Appendix G Function vector hyperparameters

[Table 11](#A7.T11 "Table 11 ‣ Appendix G Function vector hyperparameters ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") shows the hyperparameters selected for each model’s task representation. Hyperparameters (representation type – between a fixed set of heads and a full residual stream, layers, and number of heads) were chosen via a three-criterion search over candidate configurations. A fixed calibration set of elemental and composite tasks was used, and three criteria were considered: (1) within-task consistency, measured as split-half cosine similarity between FVs extracted from random partitions of correct examples; (2) inter-task discriminability, the ratio of within-task to between-task cosine similarity; and (3) compositional structure, measured as the cosine similarity between each compositional task’s FV and its least-squares reconstruction from component FVs. Final selection used a rank-sum policy over these three criteria, with ties broken by raw metric values. Note that in all cases, only correct examples were used when constructing the final function vectors.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | Representation | Layer | k heads | σ\sigma | λ\lambda |
| LLM360 | | | | | |
| amber | hidden states | 21 | — | 6.02568 | 0.0001 |
| crystal | hidden states | 8 | — | 6.25822 | 0.0001 |
| Pythia | | | | | |
| pythia\_410m | hidden states | 3 | — | 0.33991 | 0.001 |
| pythia\_1.4b | hidden states | 12 | — | 5.93639 | 0.001 |
| pythia\_12b | hidden states | 9 | — | 4.02777 | 0.0001 |
| OLMo-2 | | | | | |
| olmo2\_1b | hidden states | 8 | — | 3.46810 | 0.0001 |
| olmo2\_7b | hidden states | 16 | — | 1.05641 | 0.005 |
| olmo2\_13b | cie\_heads | 10 | 15 | 0.96582 | 0.005 |
| OLMo-3 | | | | | |
| olmo3\_7b | hidden states | 16 | — | 4.37314 | 0.001 |

Table 11: Function vector hyperparameters selected per model. The full residual stream was chosen as the representation for all models besides OLMo2-13B, in which the top 10 heads by causal indirect effect in layer 10 were chosen. σ\sigma and λ\lambda are the parameters used in ridge regression.

## Appendix H All Held-out trajectory predictions

Figures [13](#A8.F13 "Figure 13 ‣ Appendix H All Held-out trajectory predictions ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") – [21](#A8.F21 "Figure 21 ‣ Appendix H All Held-out trajectory predictions ‣ What Do Language Models Learn and When? The Implicit Curriculum Hypothesis") show the results of the leave-one-out prediction setup for each compositional task.

Figure 13: Predictions of held-out compositional tasks for Pythia-410M.




Figure 14: Predictions of held-out compositional tasks for OLMo2-1B.




Figure 15: Predictions of held-out compositional tasks for Pythia-1.4B.




Figure 16: Predictions of held-out compositional tasks for OLMo2-7B.




Figure 17: Predictions of held-out compositional tasks for OLMo3-7B.




Figure 18: Predictions of held-out compositional tasks for Amber.




Figure 19: Predictions of held-out compositional tasks for CrystalCoder.




Figure 20: Predictions of held-out compositional tasks for Pythia-12B.




Figure 21: Predictions of held-out compositional tasks for OLMo2-13B.
