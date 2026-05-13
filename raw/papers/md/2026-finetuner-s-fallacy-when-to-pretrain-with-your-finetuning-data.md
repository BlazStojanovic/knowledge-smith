---
arxiv: '2603.16177'
authors:
- Christina Baek
- Ricardo Pio Monti
- David Schwab
- Amro Abbas
- Rishabh Adiga
- Cody Blakeney
- Maximilian Böther
- Paul Burstein
- Aldo Gael Carranza
- Alvin Deng
- Parth Doshi
- Vineeth Dorna
- Alex Fang
- Tony Jiang
- Siddharth Joshi
- Brett W. Larsen
- Jason Chan Lee
- Katherine L. Mentzer
- Luke Merrick
- Haakon Mongstad
- Fan Pan
- Anshuman Suri
- Darren Teh
- Jason Telanoff
- Jack Urbanek
- Zhengping Wang
- Josh Wills
- Haoli Yin
- Aditi Raghunathan
- J. Zico Kolter
- Bogdan Gaza
- Ari Morcos
- Matthew Leavitt
- Pratyush Maini
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'The Finetuner''s Fallacy: When to Pretrain with Your Finetuning Data'
url: https://arxiv.org/abs/2603.16177
year: 2026
---

[2603.16177] The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data














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



# [Uncaptioned image]   The Finetuner’s Fallacy When to Pretrain with Your Finetuning Data

DatologyAI Team
See Contributions (§ [Contributions](#Sx1 "Contributions ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")) for full author list.

###### Abstract

Real-world model deployments demand strong performance on narrow domains where data is often scarce. Typically, practitioners finetune models to specialize them, but this risks overfitting to the domain and forgetting general knowledge.
We study a simple strategy, *specialized pretraining* (SPT), where a small domain dataset, typically reserved for finetuning, is repeated starting from pretraining as a fraction of the total tokens.
Across three specialized domains (ChemPile, MusicPile, and ProofPile), SPT improves domain performance and preserves general capabilities after finetuning compared to standard pretraining. In our experiments, SPT reduces the pretraining tokens needed to reach a given domain performance by up to 1.75×. These gains grow when the target domain is underrepresented in the pretraining corpus: on domains far from web text, a 1B SPT model outperforms a 3B standard pretrained model. Beyond these empirical gains, we derive *overfitting scaling laws* to guide practitioners in selecting the optimal domain-data repetition for a given pretraining compute budget.
Our observations reveal the *finetuner’s fallacy*: while finetuning may appear to be the cheapest path to domain adaptation, introducing specialized domain data during pretraining stretches its utility.
SPT yields better specialized domain performance (via reduced overfitting across repeated exposures) and better general domain performance (via reduced forgetting during finetuning), ultimately achieving stronger results with fewer parameters and less total compute when amortized over inference.
To get the most out of domain data, incorporate it as early in training as possible.

![Refer to caption](/html/2603.16177/assets/x1.png)

![Refer to caption](/html/2603.16177/assets/figures/fig0_main_schematic.png)

Figure 1: Specialized pretraining (SPT) mixes the finetuning dataset into pretraining as a small fraction of tokens, repeating it 10–50× over the course of training. Compared to general pretraining (dashed), SPT (solid) achieves lower domain test loss (blue) and less forgetting of general knowledge (gold) throughout finetuning. For narrow domains, these gains can overcome differences in model scale.

## 1 Introduction

Consider an organization with proprietary data such as support conversations, legal filings, or clinical notes, that wants to train a domain-specialized model. The conventional recipe is straightforward: start from a strong open-weights model pretrained on web-scale data, then finetune it on the proprietary dataset. Because this data is private and absent from public corpora, finetuning is treated as the natural mechanism for injecting missing domain knowledge. More broadly, modern training pipelines often treat pretraining and finetuning as disjoint phases: first learn general knowledge at scale, then specialize using a small curated dataset. The success of instruction tuning, RLHF, and parameter-efficient finetuning has further reinforced this view (Ouyang et al., [2022](#bib.bib17 "Training language models to follow instructions with human feedback"); Wei et al., [2021](#bib.bib18 "Finetuned language models are zero-shot learners"); Hu et al., [2022](#bib.bib23 "LoRA: low-rank adaptation of large language models")).

Growing evidence across several settings suggests that data encountered during
pretraining shapes model behavior more durably than that introduced
later: incorporating reasoning data into pretraining outperforms introducing it only
during fine-tuning (Akter et al., [2025](#bib.bib47 "Front-loading reasoning: the synergy between pretraining and post-training data"); Hatamizadeh et al., [2025](#bib.bib64 "RLP: reinforcement as a pretraining objective")),
unsafe behaviors learned during pretraining are harder to remove via
post-training (Maini et al., [2025](#bib.bib55 "Safety pretraining: toward the next generation of safe ai"); Sam et al., [2025](#bib.bib65 "When should we introduce safety interventions during pretraining?")), and cross-language transfer
during pretraining improves performance for low-resource
languages (Longpre et al., [2025](#bib.bib63 "ATLAS: adaptive transfer scaling laws for multilingual pretraining, finetuning, and decoding the curse of multilinguality")). Yet the question of
when to introduce domain-specific data, and whether it should be mixed
into pretraining rather than reserved for finetuning, remains largely unexplored.

In this work, we question whether reserving all domain-specific data for the final stage
of training is optimal. When the target domain is poorly represented in the pretraining
corpus, introducing domain data only during finetuning may require large representational
updates, leading to weaker generalization and greater forgetting of general knowledge.
We study a simple alternative: interleave the domain dataset throughout pretraining as a
small fraction of the training tokens (often repeating it up to 50 times), and then
finetune on the same data as usual (Figure [1](#S0.F1 "Figure 1 ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). We refer to this
strategy as *specialized pretraining* (SPT). We observe that interleaving domain tokens with general data allows the model to tolerate far more repetitions before overfitting than it would during finetuning.

![Refer to caption](/html/2603.16177/assets/figures/fig1_perf_across_domains.png)


Figure 2: Specialized pretraining (SPT) outperforms finetuning-only
across domains. We pretrain models with a small fraction (δ\delta) of domain-specific
tokens mixed into general web data, then finetune on the domain dataset. We plot the
best post-finetuning domain loss across pretraining budgets for MusicPile, ChemPile,
and ProofPile (300M tokens each). Even small domain mixtures (δ=1\delta=1–5%5\%,
blue curves) consistently outperform pretraining on general data alone (δ=0%\delta=0\%,
gray) at all token scales.

Across chemistry, symbolic music, and mathematical proofs, SPT consistently improves
post-finetuning performance (§ [2](#S2 "2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). Relative to standard pretraining
followed by finetuning, models trained with SPT achieve lower domain test loss, retain
general pretraining knowledge more effectively during finetuning, and perform better on
downstream tasks (Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). Comparing across pretraining scales,
SPT also requires substantially less pretraining compute to reach the same
post-finetuning domain loss.
Even replay (Parmar et al., [2024](#bib.bib66 "Reuse, don’t retrain: a recipe for continued pretraining of language models")), a common strategy that reintroduces general data during continued pretraining, is not a substitute for early domain exposure: SPT’s advantage persists across all replay settings (§ [4](#S4 "4 Does Specialized Pretraining Help Under Replay as Well? ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). On domains far from web text, a 1B model trained with SPT outperforms a 3B model trained without domain data during pretraining.
SPT also reduces the pretraining tokens needed to reach a given domain loss by up to 1.75×\times (Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")), and these loss improvements translate to downstream task accuracy: at 200B pretraining tokens, SPT improves MATH accuracy by up to 6 percentage points and MusicTheoryBench by up to 4 percentage points over the finetuning-only baseline (Figure [5](#S2.F5 "Figure 5 ‣ Which paradigm achieves lower domain test loss? ‣ 2.2 Experimental Results ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).

We further characterize when SPT is most effective. First, SPT helps most when the target domain is underrepresented in the pretraining corpus, as shown both in a controlled multilingual overlap study and across naturally occurring domain shifts (§ [3.1](#S3.SS1 "3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). Second, its benefit depends on the size of the domain dataset: for sufficiently large domain corpora, even repeating domain data from the beginning of pretraining is beneficial, while in more data-constrained settings, introducing domain data later in pretraining is preferable (§ [3.2](#S3.SS2 "3.2 Domain Dataset Size ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).
Third, the pretraining budget determines the optimal mixture fraction: larger
domain fractions help at shorter training horizons, while smaller fractions become
preferable as training lengthens to avoid overfitting from excessive repetition
(§ [3.4](#S3.SS4 "3.4 Pretraining compute budget ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).

Finally, we derive overfitting scaling laws to model the total overfitting incurred by specialized pretraining and finetuning for different mixture fractions (§ [5](#S5 "5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")), which include new expressions for modeling the effects of repeated data. Specifically, the test loss decomposes as the sum of two parts: the training loss w.r.t. pretraining tokens is modeled as a power law with a negative exponent, while the growing train-test gap is modeled as a power law with a positive exponent. This allows us to predict the optimal domain-data fraction for a given compute budget and forecast when aggressive mixing begins to hurt test loss, without running the full training sweep. In practice, this means practitioners can select the right SPT configuration from a small number of pilot runs rather than exhaustive search.

![Refer to caption](/html/2603.16177/assets/figures/fig13_finetuners_tax.png)


Figure 3: The finetuner’s tax. Training a 1B model with specialized
pretraining (SPT) costs more upfront than finetuning a 3B model on
domain data alone, but the 3×3\times smaller model is cheaper to serve. The
break-even point arrives after approximately 1 trillion inference tokens, after which
SPT saves both compute and money while often delivering comparable or better performance.

Broadly, our results expose *the finetuner’s fallacy*. Finetuning a large off-the-shelf model is commonly assumed to be the cheapest path to domain adaptation, since it avoids the cost of pretraining.
This intuition is often misleading. Those who rely on finetuning alone need a larger model to match domain performance, and amortized over inference, pretraining a smaller model with domain data is cheaper (Figure [3](#S1.F3 "Figure 3 ‣ 1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).
Early integration of domain data during pretraining is one way to reduce this cost, and is compatible with other data curation strategies like synthetic augmentation.
As the cost of pretraining continues to fall, the case for early integration only strengthens. To get the most out of domain data, incorporate it as early in training as possible.

![Refer to caption](/html/2603.16177/assets/figures/fig2_bar_plots.png)


Figure 4: Specialized pretraining (SPT) is more effective than scaling
tokens or parameters. We compare models that include domain data during pretraining
(SPT→\toFT) against models pretrained only on general web data
(NPT→\toFT). *Left:* Relative
gain of SPT→\toFT over NPT→\toFT. *Center:* Compute
multiplier showing how much faster SPT reaches NPT’s best
performance. *Right:* Percentage of the 1B vs. 3B parameter performance gap
closed by SPT. Values above 100% indicate the 1B SPT
outperforms 3B NPT. SPT consistently improves model quality, training speed, and parameter efficiency (i.e. SPT is a pareto improvement in training efficiency) across all three examined domains.

## 2 Specialized Pretraining Drives Domain Specific Capabilities

We study whether mixing domain-specific data into pretraining improves the model that
results after finetuning. We refer to standard pretraining on general web data as
*naive pretraining* (NPT), and to pretraining that includes a small
fraction of domain data as *specialized pretraining* (SPT). Both are
followed by finetuning (FT) on the domain dataset. We compare the two
resulting pipelines, NPT→\toFT and SPT→\toFT, across three
specialized domains.

### 2.1 Notation and Experimental Setup

##### Specialized Pretraining

Let δ∈[0,1]\delta\in[0,1] denote the fraction of pretraining tokens drawn from the domain-specific dataset, with the remaining 1−δ1-\delta fraction drawn from general web data (e.g., δ=0.02\delta=0.02 corresponds to a 2% domain token mixture). Note that δ=0\delta=0 corresponds to naive pretraining. Since domain-specific datasets are typically much smaller than the total pretraining budget, domain examples are repeated as necessary during pretraining. Given TT pretraining tokens, the total epochs of domain-specific data 𝒟dom\mathcal{D}\_{\text{dom}} seen is E=(T⋅δ)/|𝒟dom|.E=(T\cdot\delta)/|\mathcal{D}\_{\text{dom}}|. To measure generalization under this heavy repetition, we hold out a fixed test split from each domain dataset. This split is never included in training, and all domain losses reported in this paper are evaluated on it.

##### OLMo Sandbox

We use a controlled pretraining environment based on OLMo-1B trained on the Dolma corpus. We introduce three specialized domains of equal size, MusicPile (Yuan et al., [2024](#bib.bib37 "ChatMusician: understanding and generating music intrinsically with llm")), ChemPile (Mirza et al., [2025b](#bib.bib38 "ChemPile: a 250gb diverse and curated dataset for chemical foundation models")), and ProofPile (Hoskinson Center for Formal Mathematics, [2022](#bib.bib39 "Proof-pile: a dataset of high quality mathematical text")), each containing roughly 300M tokens. We pretrain for 200B tokens with δ∈{0,0.1%,1%,2%,5%}\delta\in\{0,0.1\%,1\%,2\%,5\%\} fraction drawn from the domain-specific dataset and the remainder (1−δ)(1-\delta) drawn from Dolma. Samples are repeated to satisfy mixture constraints e.g. 5%\% SPT repeats the domain-specific data roughly 33×33\times over the course of pretraining. We match OLMo-1B pretraining settings to the publicly documented configuration where possible, including the optimizer, cosine learning rate schedule, and batch size. We report all hyperparameters in Appendix [B](#A2 "Appendix B Optimization Details ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").

We will also compare SPT and NPT across model sizes, for which we create 300M, 600M, and 3B variants of the OLMo-1B architecture by adjusting the model depth. For the 3B variant, we double the number of layers in the 1B model. For smaller 300M and 600M variants, we halve the number of layers and reduce the dimension of the MLP representations and hidden representations. We train all models using the same pretraining recipe.

##### Finetuning

During finetuning, training proceeds exclusively on the
domain-specific dataset with a WSD learning rate schedule. To ensure a fair comparison
of post-finetuning performance between SPT and NPT models, we impose
no restriction on the number of dataset repetitions. Instead, we apply early stopping:
the data is repeated as long as test loss decreases. This ensures that any loss
improvement observed with SPT is not a result of additional passes over
the dataset. Furthermore, we tune warmup steps and the
learning rate by grid search. We report the lowest domain test loss across all
finetuning configurations.

### 2.2 Experimental Results

We compare two training paradigms. In the first, NPT→\toFT, we pretrain over Dolma for 200B tokens (δ=0\delta=0) then finetune. In the second, SPT→\toFT, we pretrain on repeats of domain-specific data mixed with Dolma for 200B tokens before finetuning.

##### Which paradigm achieves lower domain test loss?

Let
LNPT→FTL^{\texttt{NPT}\to\texttt{FT}} and LSPT​(δ)→FTL^{\texttt{SPT}(\delta)\to\texttt{FT}} denote
the best domain test loss for each paradigm. We quantify improvement by *relative
gain*:

|  |  |  |
| --- | --- | --- |
|  | ℛgain​(δ)= 100⋅LNPT→FT−LSPT​(δ)→FTLNPT→FT,\displaystyle\mathcal{R}\_{\text{gain}}(\delta)\;=\;100\cdot\frac{L^{\texttt{NPT}\to\texttt{FT}}-L^{\texttt{SPT}(\delta)\to\texttt{FT}}}{L^{\texttt{NPT}\to\texttt{FT}}}, |  |

SPT consistently helps across all three domains
(Figure [4](#S1.F4 "Figure 4 ‣ 1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), left). MusicPile, a corpus of symbolic music notation
that bears little resemblance to web text, sees the largest relative gain at 2.0%.
ProofPile, containing formal mathematical proofs, follows with 1.5%. ChemPile shows a
more modest 0.8% improvement, which we attribute to chemistry text being closer to the
Dolma distribution. We return to the impact of distributional overlap in
§ [3.1](#S3.SS1 "3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").

![Refer to caption](/html/2603.16177/assets/figures/fig3_forgetting.png)


(a)

![Refer to caption](/html/2603.16177/assets/figures/fig5b_MusicTheory_acc_vs_tokens.png)

![Refer to caption](/html/2603.16177/assets/figures/fig5b_ChemBench_acc_vs_tokens.png)

![Refer to caption](/html/2603.16177/assets/figures/fig5b_MATH_acc_vs_tokens.png)

(b)

Figure 5: SPT reduces forgetting and improves downstream task
performance. (a) For ChemPile, we plot Dolma loss (general knowledge) against
domain loss for the best post-finetuning checkpoint at each pretraining budget (40B
to 200B tokens) and mixture percentage δ\delta. Larger SPT mixtures
achieve lower domain loss *and* lower general loss, indicating less catastrophic
forgetting. (b) We compare NPT (gray) and 2% SPT (blue) on
downstream tasks matched to each domain: MusicTheoryBench for MusicPile, ChemBench General Chemistry subset for ChemPile, and MATH for ProofPile. All tasks are evaluated in
4-choice MCQA format. For each pretraining budget, we report the best accuracy across
finetuning runs. SPT outperforms NPT across most settings.

##### How much earlier in pretraining does SPT reach a given domain test loss?

We
define the *compute multiplier* as the factor by which SPT reduces the
number of pretraining tokens needed to reach a given post-finetuning domain test loss,
compared to NPT. The efficiency gains are substantial
(Figure [4](#S1.F4 "Figure 4 ‣ 1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), center). On MusicPile, SPT matches
NPT’s best performance at 200B pretraining tokens using 1.75×1.75{\times} fewer tokens. ProofPile shows a
similar 1.56×1.56{\times} advantage. Even ChemPile, despite its smaller performance gap,
achieves a 1.40×1.40{\times} multiplier.

##### Can SPT compensate for model size?

If a smaller model trained with
SPT could match or exceed the performance of a larger NPT model, the
practical implications would be substantial. To test this, we pretrain a 3B model on the
same 200B Dolma tokens and measure how much of the gap between the 1B and 3B
NPT models is closed by the 1B SPT model. The results vary by domain
(Figure [4](#S1.F4 "Figure 4 ‣ 1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), right). On ProofPile, the furthest domain from web
text, the 1B SPT model closes 133% of the gap, meaning it surpasses the 3B
model entirely. MusicPile closes 81% of the gap, nearly matching the larger model.
ChemPile shows more modest gains at 23%, consistent with its smaller relative gain
(§ [3.1.2](#S3.SS1.SSS2 "3.1.2 Correlation Analysis Across Domains ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).

Taken together, SPT delivers better domain performance, faster convergence, and stronger parameter efficiency across all three domains, with no observed tradeoff between these axes.

### 2.3 SPT Learns More and Forgets Less

In addition to lower domain loss, SPT reduces forgetting of general knowledge
during finetuning. Although SPT allocates a small fraction of pretraining
tokens to domain data, this has minimal impact on Dolma loss during pretraining: the
NPT and SPT runs achieve comparable general loss after 200B tokens (Appendix [F](#A6 "Appendix F General Pretraining Loss during SPT ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).
The difference emerges during finetuning. NPT models start with higher domain
loss and therefore require more aggressive optimization to close the gap, which drives up
general loss. SPT models start from a lower domain loss, require less
adaptation, and consequently exhibit less forgetting (Figure [5(a)](#S2.F5.sf1 "Figure 5(a) ‣ Figure 5 ‣ Which paradigm achieves lower domain test loss? ‣ 2.2 Experimental Results ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).

The combination of lower domain loss and lower general loss correlates with improved
downstream task performance. We compare the best
post-finetuning performance of a 2% SPT model against an NPT
baseline on symbolic music questions from MusicTheoryBench
(Yuan et al., [2024](#bib.bib37 "ChatMusician: understanding and generating music intrinsically with llm")), ChemBench general chemistry subset (Mirza et al., [2025a](#bib.bib81 "A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists")), and MATH in
MCQA style (Biderman, [2025](#bib.bib43 "MATH-mcqa: a multiple choice adaptation of the math dataset")). Across all benchmarks and most pretraining budgets,
SPT outperforms NPT by several percentage points (Figure [5(b)](#S2.F5.sf2 "Figure 5(b) ‣ Figure 5 ‣ Which paradigm achieves lower domain test loss? ‣ 2.2 Experimental Results ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).

### 2.4 Specialized Pretraining Reduces Overfitting During Finetuning

Prior work on mixing data during pretraining typically studies regimes in which data is
abundant and rarely repeated; in these settings, data mixing primarily serves to
accelerate optimization. Our setting differs fundamentally: the finetuning datasets are
at least an order of magnitude smaller than the model size (300M tokens versus 1B
parameters), meaning that with reasonable repetitions, the models can overfit to the
training data. In fact, across domains, for both SPT→\toFT and
NPT→\toFT, the domain test loss begins to rise after roughly 5 epochs of
finetuning (Figure [1](#S0.F1 "Figure 1 ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). This suggests that the benefit of specialized
pretraining is better understood as a regularization effect rather than an optimization
one.

To understand why SPT reduces overfitting, we compare how the domain train-test gap evolves across pretraining and finetuning (Figure [6](#S2.F6 "Figure 6 ‣ 2.4 Specialized Pretraining Reduces Overfitting During Finetuning ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). The key insight is that overfitting to domain data is far more of a risk during finetuning than during pretraining. During pretraining, domain tokens make up only a small fraction of each batch, and the surrounding general data acts as a natural regularizer, preventing the model from memorizing the domain corpus even after tens of repetitions.

![Refer to caption](/html/2603.16177/assets/figures/fig4_pt_to_ft_train_vs_test.png)


Figure 6: SPT regularizes finetuning. Domain training loss (x-axis)
versus domain test loss (y-axis) across the pretraining and finetuning stages for
MusicPile, with mixture fractions δ=0%,2%,5%\delta=0\%,2\%,5\%. Dotted lines show the
pretraining stage; solid lines show finetuning. During finetuning, SPT models
achieve lower test loss than NPT models at the same training loss, indicating
that SPT reduces overfitting.
Notably, the δ=5%\delta=5\% model has already seen the domain data over 33×33\times
during pretraining, yet it still overfits less during finetuning than the NPT
model seeing the same data for the first time. This regularization is a consequence of diffused exposure to specialized data during pretraining.

During finetuning, this regularization effect is absent: the model trains exclusively on domain data and overfits rapidly. This is visible in Figure [6](#S2.F6 "Figure 6 ‣ 2.4 Specialized Pretraining Reduces Overfitting During Finetuning ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"): at the same domain training loss, comparing SPT at its initial pretrained checkpoint with NPT after early finetuning steps, the two models generalize comparably but as finetuning continues, the NPT model’s train-test gap widens much faster. Because SPT models enter finetuning with a lower domain loss, they need less adaptation and exit finetuning before overfitting sets in.

### 2.5 Key Takeaways

Overall, mixing domain-specific data into pretraining leads to a stronger model in terms
of *both domain perplexity and downstream task performance*. Specialized pretrained
models generalize better during finetuning, which suggests that mixing the two datasets
together leads to qualitatively different learned representations. Furthermore, we find that retaining both general and domain performance simultaneously is a key
advantage of SPT. Because the domain data is introduced as a small fraction of
pretraining tokens alongside general web data, the model learns domain structure without
sacrificing broad coverage. During finetuning, this translates into less catastrophic
forgetting: SPT models start from a lower domain loss and therefore require
less aggressive adaptation, preserving general capabilities that NPT models
forfeit. This dual benefit is what makes SPT practically attractive, and what
underlies the finetuner’s fallacy we characterize in the remainder of this paper.

## 3 Factors governing the relative gain of specialized pretraining

Section [2](#S2 "2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data") established that SPT yields consistent gains over
NPT across diverse domains, improving domain test loss after finetuning while
also better retaining general knowledge. In this section, we characterize *when* to
expect such gains. We identify several factors that impact the relative improvement,
including the similarity between pretraining and finetuning data,
the size of the domain dataset, and the available pretraining compute.

### 3.1 Domain Similarity

The degree of similarity between the domain-specific data and the general pretraining
corpus is expected to influence the efficacy of SPT. In the limiting case where
the pretraining and target distributions coincide, incorporating domain data during
pretraining introduces no additional signal and therefore should not improve downstream
performance. To examine this dependence, we analyze the effect of distributional overlap
through two complementary approaches: (i) a controlled study in which overlap is
systematically varied, and (ii) a cross-domain correlation analysis across naturally
occurring distribution shifts.

#### 3.1.1 A controlled study of distributional overlap

To isolate the effect of distributional similarity with minimal confounding factors, we
consider an English→\toJapanese translation setting. This task is well studied and
presents substantial linguistic divergence in script, morphology, and word order
(NLLB-Team et al., [2022](#bib.bib45 "No language left behind: scaling human-centered machine translation"); Fan et al., [2021](#bib.bib50 "Beyond english-centric multilingual machine translation"); Liu et al., [2020](#bib.bib51 "Multilingual denoising pre-training for neural machine translation"); Carranza et al., [2026](#bib.bib49 "\” UberWeb: insights from multilingual curation for a 20-trillion-token dataset")). Standard practice
in machine translation is to pretrain on monolingual corpora from the source and target
languages and subsequently finetune on parallel data
(Xu et al., [2024](#bib.bib58 "A paradigm shift in machine translation: boosting translation performance of large language models"); Hangya et al., [2022](#bib.bib60 "Improving low-resource languages in pre-trained multilingual language models")). In our
setting, varying the proportion of Japanese relative to English monolingual text during
pretraining systematically alters the overlap between the pretraining distribution and
the downstream translation task. We compare SPT→\toFT (where parallel
translation data is included during pretraining at mixture fraction δ\delta) against
NPT→\toFT (monolingual pretraining only), while holding the total number of
pretraining tokens fixed.

![Refer to caption](/html/2603.16177/assets/x2.png)


Figure 7: Benefits of SPT increase as pretraining and finetuning
domains diverge. We vary the percentage of Japanese monolingual text in the
pretraining mix for an English→\toJapanese translation task, and plot
ℛgain\mathcal{R}\_{\text{gain}} of SPT→\toFT over NPT→\toFT. With
less Japanese monolingual data (leftwards on x-axis), the distributional gap between
pretraining and finetuning data grows, and the gain from SPT increases,
plateauing at approximately 5%.

We pretrain 160M-parameter LLaMA models on 20B tokens of FineWeb2 English and Japanese
monolingual data (Penedo et al., [2025](#bib.bib57 "FineWeb2: one pipeline to scale them all – adapting pre-training data processing to every language")), together with 1B tokens of
parallel data from JParaCrawl v3.0 (Morishita et al., [2022](#bib.bib59 "JParaCrawl v3.0: a large-scale English-Japanese parallel corpus")). For
SPT, a fraction δ∈{0%,0.1%,1%,5%}\delta\in\{0\%,0.1\%,1\%,5\%\} of pretraining tokens is
drawn from the parallel corpus, with the remaining 1−δ1-\delta drawn from a mixture of
English and Japanese monolingual data. We vary the proportion of Japanese monolingual
data within this mixture over {0%,0.001%,0.01%,0.1%,1%,10%}\{0\%,0.001\%,0.01\%,0.1\%,1\%,10\%\}. All models
are subsequently finetuned on the parallel corpus, and we select checkpoints based on
held-out validation loss.

Results (Fig. [7](#S3.F7 "Figure 7 ‣ 3.1.1 A controlled study of distributional overlap ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). As the proportion of Japanese
monolingual data decreases from 10% to 0.1% in pretraining, we observe that
ℛgain\mathcal{R}\_{\text{gain}} increases from approximately 2% to 5%. Increasing Japanese
monolingual content during pretraining reduces the distributional gap between pretraining
and finetuning (i.e., parallel translation) data, thereby diminishing the marginal
benefit of including parallel data early. These findings provide controlled evidence that
the relative gain from SPT is driven by distributional misalignment: when the
pretraining corpus already contains domain-relevant signal, incorporating domain data
during pretraining yields smaller improvements.

#### 3.1.2 Correlation Analysis Across Domains

We next ask whether standard measures of distributional similarity between Dolma and
each specialized domain can predict ℛgain\mathcal{R}\_{\text{gain}}. We consider five
metrics: unigram, bigram, and trigram Jensen–Shannon divergence (JSD), MAUVE
(Pillutla et al., [2021](#bib.bib44 "MAUVE: measuring the gap between neural text and human text using divergence frontiers")), and the classifier two-sample test (C2ST)
(Lopez-Paz and Oquab, [2016](#bib.bib42 "Revisiting classifier two-sample tests")). We also use a direct proxy: the domain loss of
NPT checkpoints after finetuning, since higher post-finetuning loss indicates
less overlap between the general corpus and the target domain.

In the controlled Japanese overlap sweep, all metrics correlate strongly with
ℛgain\mathcal{R}\_{\text{gain}} (|r|>0.85|r|>0.85). However, when we measure distributional
overlap between Dolma and the three benchmark domains (MusicPile, ChemPile, ProofPile),
the metrics disagree on which domain is closest to Dolma. From
Section [2.2](#S2.SS2 "2.2 Experimental Results ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), we know that ℛgain\mathcal{R}\_{\text{gain}} is highest for
MusicPile (2.0%), followed by ProofPile (1.5%) and ChemPile (0.8%). Of the metrics
we tested, only the post-finetuning domain loss, a direct measure of how well
NPT generalizes to each domain, correctly ranks all three domains (detailed analysis in Appendix [E](#A5 "Appendix E Distribution Similarity Metrics: Detailed Analysis ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).

![Refer to caption](/html/2603.16177/assets/figures/fig4a_data_size_by_rgain.png)

![Refer to caption](/html/2603.16177/assets/figures/fig4b_model_size.png)

Figure 8: Effect of domain data size and model size on
ℛgain\mathcal{R}\_{\text{gain}}. *Left:* Relative gain of SPT over
NPT as a function of domain dataset size ({3M, 30M, 300M} tokens from
MusicPile) at three pretraining scales (40B, 120B, 200B tokens). For the 300M-token
dataset, gains are consistently high across all scales. For smaller datasets,
SPT is less helpful at longer training horizons due to overfitting from
excessive repetition; in these regimes, specialized continued pretraining (star)
is preferable. *Right:* Domain test loss vs. model size for NPT
(δ=0%\delta=0\%) and SPT (δ=2%\delta=2\%) on MusicPile-300M. Relative gain grows with scale, driven not only by decreasing NPT loss but also a widening SPT–NPT gap.

### 3.2 Domain Dataset Size

Even when overlap between pretraining and target data is substantial, repeated exposure
to a limited domain corpus can induce overfitting. We examine how the magnitude of
SPT’s relative gain varies as a function of domain dataset size, evaluating
mixture fractions δ∈{0,0.1,1,2,5}\delta\in\{0,0.1,1,2,5\} across subsets of {3M, 30M, 300M}
tokens from MusicPile. Figure [8](#S3.F8 "Figure 8 ‣ 3.1.2 Correlation Analysis Across Domains ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data") reports the maximum
ℛgain\mathcal{R}\_{\text{gain}} achieved at multiple pretraining token budgets.

At a modest pretraining scale of 40B tokens, SPT yields consistent gains
(ℛgain≥0.5%\mathcal{R}\_{\text{gain}}\geq 0.5\%) across all dataset sizes, with the 30M-token
subset exhibiting the largest improvement. However, as the pretraining budget increases
to 120B tokens and beyond, the regimes diverge. For the 300M-token dataset, relative
gain continues to increase with scale. In contrast, for smaller datasets (3M–30M
tokens), the benefit diminishes and can become negative. In these settings, even small
mixture fractions (e.g., δ=0.1%\delta=0.1\%) induce excessive repetition, leading to
overfitting during pretraining and degraded post-finetuning performance.

In such extremely data-constrained regimes (on the order of tens of millions of tokens or
fewer), we find that *specialized continued pretraining* (SCPT), in which
domain data is introduced at later stages of pretraining, may still be more effective than NPT.
Concretely, we take the 180B-token NPT checkpoint and continue training for
an additional 20B tokens with a 1%1\% domain mixture (pink stars in Figure [8](#S3.F8 "Figure 8 ‣ 3.1.2 Correlation Analysis Across Domains ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). For the 30M-token dataset,
SCPT achieves higher relative gain than full SPT from initialization. For the 3M-token dataset, SCPT at 1%\% still overfits excessively.

These results indicate that the interaction between repetition and dataset size induces
distinct scaling regimes: when the domain corpus is sufficiently large (e.g., 300M
tokens), early integration during pretraining yields sustained benefits; when the corpus
is small, deferring domain mixing to later stages improves generalization.

### 3.3 Model Size

The relative gains from SPT persist across model scales and actually increase
with parameter count. Figure [8](#S3.F8 "Figure 8 ‣ 3.1.2 Correlation Analysis Across Domains ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data") (right) plots domain test loss as a
function of model size for NPT (δ=0%\delta=0\%) and SPT (δ=2%\delta=2\%)
across multiple pretraining budgets on MusicPile-300M. Across all settings, the gap in
test loss between SPT and NPT widens with model size: the 3B model
exhibits the largest reduction in post-finetuning test loss under SPT.

We believe this trend is consistent with the overfitting interpretation. Larger models
have greater representational capacity and are therefore more prone to memorizing a
limited domain corpus during finetuning. Collectively, these findings show that the
benefits of SPT amplify with model scale. As capacity increases, so does the
relative advantage of integrating domain data during pretraining, highlighting that
SPT becomes increasingly effective in the large-model regime.

### 3.4 Pretraining compute budget

The optimal mixture fraction δ\delta also depends on the available pretraining compute.
To study this interaction, we track domain test loss throughout training for mixture
fractions ranging from 0%0\% to 10%10\% on MusicPile (Figure [9](#S3.F9 "Figure 9 ‣ 3.4 Pretraining compute budget ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).

At smaller pretraining budgets (e.g., under 30B tokens), larger mixture fractions perform
best, with δ=10%\delta=10\% achieving the lowest test loss. However, as training progresses,
these large mixtures begin to overfit due to repeated exposure to the domain corpus. By
approximately 50B tokens, the δ=10%\delta=10\% curve begins to degrade while moderate
mixtures (e.g., δ=5%\delta=5\%) continue to improve. At longer training horizons (200B
tokens), the best performance is achieved by smaller mixtures in the range of
δ=2\delta=2–5%5\%. Crucially, regardless of the pretraining budget, SPT with an
appropriate choice of δ\delta outperforms NPT.

These results demonstrate that the optimal degree of domain mixing is compute-dependent.
Larger mixture fractions are advantageous when training budgets are limited, as they
accelerate learning of domain structure. At larger compute scales, however, excessive
repetition leads to overfitting, shifting the optimal regime toward smaller mixture
fractions. Consequently, practitioners should treat δ\delta as a function of training
horizon: higher mixtures for short pretraining runs and lower mixtures for longer ones.

![Refer to caption](/html/2603.16177/assets/figures/fig7_optimal_mix.png)


Figure 9: The optimal mixture shifts with compute budget. Domain test loss on
MusicPile throughout SPT for mixture fractions δ∈{0%,2%,5%,10%}\delta\in\{0\%,2\%,5\%,10\%\}. At small compute budgets (left region), high domain mixtures (δ=10%\delta=10\%)
achieve the lowest test loss. As training progresses, these mixtures overfit and their
loss increases, while moderate mixtures (δ=5%\delta=5\%) become optimal (middle region).
Past 240B tokens, lower mixtures (δ=2%\delta=2\%) are best. Regardless of budget,
SPT with an appropriate δ\delta outperforms NPT (δ=0%\delta=0\%).

## 4 Does Specialized Pretraining Help Under Replay as Well?

Replay is commonly used during finetuning to mitigate forgetting by mixing previously seen data back into training (Parmar et al., [2024](#bib.bib66 "Reuse, don’t retrain: a recipe for continued pretraining of language models"); Blakeney et al., [2024](#bib.bib67 "Does your data spark joy? performance gains from domain upsampling at the end of training"); Kotha and Liang, [2026](#bib.bib73 "Replaying pre-training data improves fine-tuning"); Liu et al., [2025](#bib.bib24 "Midtraining bridges pretraining and posttraining distributions")).
If replay already reintroduces general data during finetuning, an important question is whether replay has a similar regularizing effect or whether early exposure to domain data through SPT still outperforms NPT.
We compare SPT →\to FT against NPT →\to FT, where replay-based FT mixes MusicPile (domain) with a Dolma replay mixture (general data).
We evaluate replay rates {0%,10%,20%}\{0\%,10\%,20\%\} tuning the learning rate separately for each setting, and report MusicPile test loss across the finetuning trajectory (Figure [10](#S4.F10 "Figure 10 ‣ 4 Does Specialized Pretraining Help Under Replay as Well? ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")).

Across all replay settings, SPT →\to FT consistently achieves lower domain test loss than NPT →\to FT, reinforcing the core thesis that *when* domain data is seen matters. Notably, 10% replay helps NPT, but NPT →\to FT falls well short of SPT →\to FT with no replay. We hypothesize that these two forms of data mixing, diffuse domain exposure during pretraining versus general-data replay during finetuning, induce qualitatively different effects. As shown in Section [2.4](#S2.SS4 "2.4 Specialized Pretraining Reduces Overfitting During Finetuning ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), SPT’s benefit is more implicit, surfacing only after finetuning. The precise mechanism behind this asymmetry remains an open question.

![Refer to caption](/html/2603.16177/assets/figures/fig12_musicpile_replay_val_loss_1e5_standalone.png)


Figure 10: Replay is not a substitute for early domain exposure.
MusicPile validation-loss trajectories during replay-based continued pretraining.
Blue lines (SPT →\to CPT) consistently achieve lower domain test loss than purple lines (NPT →\to CPT) across all replay rates (lighter shades indicate higher replay: 0%→10%→20%0\%\rightarrow 10\%\rightarrow 20\%). While higher replay slows overfitting, it does not close the gap between SPT and NPT, confirming that *when* domain data is seen has a lasting impact on the model’s performance on specialized domains.

## 5 Predicting Overfitting with Scaling Laws

Specialized pretrained models see multiple repetitions of finetuning data starting from pretraining. As a result, the optimal mixture percentage δ\delta depends on the pretraining compute budget. While larger δ%\delta\% is optimal at shorter pretraining scales, there exists a threshold beyond which test loss plateaus or degrades due to excessive repetition. This observation motivates the following question:

*Can we predict the domain test loss across specialized pretraining and finetuning as a function of δ\delta?*

Addressing this question is challenging since we explicitly have to model the overfitting regime, which the standard power law cannot anticipate. Furthermore, it’s unclear
how to predict scaling laws post-finetuning. We divide modeling overfitting scaling laws into two steps. First, we separately model how domain train and test losses scale during SPT as a function of δ\delta. Second, we measure the difference in test loss before and after finetuning, which we find also scales reliably with the pretraining tokens.

![Refer to caption](/html/2603.16177/assets/figures/fig9_musicpile_pt_law.png)

![Refer to caption](/html/2603.16177/assets/figures/fig9_musicpile_test_combo.png)

Figure 11: Repetition scaling laws as the sum of two powers. As the domain data is repeated throughout pretraining during specialized pretraining, for high mixture percentages, the model may overfit and the test loss goes back up. A single power law cannot model this overfitting stage. Instead, we fit a power law separately, for the domain training loss and the domain train-test gap. The latter also follows a power law with a positive exponent. Then the test loss can be modeled using the sum of two powers. Furthermore, we model all coefficients as a function of the mixture percentage, according to Equation [4](#S5.E4 "Equation 4 ‣ 5.1 Overfitting during Specialized Pretraining ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").

### 5.1 Overfitting during Specialized Pretraining

When repeating domain-specific data, the test loss improves up to a certain number of repeats, after which the model overfits. Previous works that study scaling laws for repeated data lack an expression that accurately models the overfitting stage (Kaplan et al., [2020](#bib.bib26 "Scaling laws for neural language models"); Goyal et al., [2024](#bib.bib3 "Scaling laws for data filtering– data curation cannot be compute agnostic"); Muennighoff et al., [2023](#bib.bib36 "Scaling data-constrained language models")). These works often express overfitting as a power law whose exponent is itself modeled by a decaying function of the number of repetitions, introducing nested nonlinearities that make the resulting expression difficult to fit from limited data and hard to extrapolate beyond the observed training range.

Instead, we find that a simple alternative model does the trick. Instead of fitting a single power law, we decompose the test loss into the training loss and the train-test gap and model them as power laws separately. While the training loss can be modeled using the usual power law with a negative exponent, the train-test gap monotonically increases with more repeats and can be modeled using a positive exponent. Then the test loss is simply the sum of the two terms. We illustrate this across all our SPT runs over MusicPile in Figure [11](#S5.F11 "Figure 11 ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").

Furthermore, we model each learned coefficient as a function of the mixture fraction δ\delta. In total, we propose a scaling law for the *domain* test loss with respect to the number of pretraining tokens TT:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒtrain​(T,δ)\displaystyle\mathcal{L}\_{\mathrm{train}}(T,\delta) | =Atrain​Tbtrain​(δ)+Ctrain​(δ),\displaystyle=A\_{\mathrm{train}}\,T^{\,b\_{\mathrm{train}}(\delta)}+C\_{\mathrm{train}}(\delta), |  | (1) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒgap​(T,δ)\displaystyle\mathcal{L}\_{\mathrm{gap}}(T,\delta) | =Agap​(δ)​Tbgap​(δ),\displaystyle=A\_{\mathrm{gap}}(\delta)\,T^{\,b\_{\mathrm{gap}}(\delta)}, |  | (2) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒtest​(T,δ)\displaystyle\mathcal{L}\_{\mathrm{test}}(T,\delta) | =ℒtrain​(T,δ)+ℒgap​(T,δ),\displaystyle=\mathcal{L}\_{\mathrm{train}}(T,\delta)+\mathcal{L}\_{\mathrm{gap}}(T,\delta), |  | (3) |

where

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | bx​(δ)\displaystyle b\_{x}(\delta) | =δ​bx,s+(1−δ)​bx,g,x∈{train,gap}\displaystyle=\delta b\_{x,s}+(1-\delta)b\_{x,g},\qquad x\in\{\mathrm{train},\mathrm{gap}\} |  | (4) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Agap​(δ)\displaystyle A\_{\mathrm{gap}}(\delta) | =α1​δα2​exp⁡(α3​δ),\displaystyle=\alpha\_{1}\delta^{\alpha\_{2}}\exp(\alpha\_{3}\delta), |  | (5) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Ctrain​(δ)\displaystyle C\_{\mathrm{train}}(\delta) | =κ0−κ1​log⁡(δ+κ2)−κ3​δ.\displaystyle=\kappa\_{0}-\kappa\_{1}\log(\delta+\kappa\_{2})-\kappa\_{3}\delta. |  | (6) |

For the training loss, we fix AtrainA\_{\mathrm{train}} to be a fixed constant, while we model it the Gamma kernel function for the gap. Furthermore, we use a log linear expression to model Ctrain​(δ)C\_{\mathrm{train}}(\delta), which we find best fits the experimental data. We provide further evidence for these design choices in Appendix [D](#A4 "Appendix D Overfitting Scaling Laws Extended ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").

##### Interpreting the exponent

The exponent in particular has an interpretable form. The rate at which the specialized domain training loss decreases can be modeled by both the general pretraining data and the specialized domain training data having (negative) utility bgb\_{g}. For the training loss, both exponents are negative bg,bs<0b\_{g},b\_{s}<0 while |bg|<|bs||b\_{g}|<|b\_{s}|, to represent that the domain training loss goes down slower for smaller mixture fractions. On the other hand, for the train-test gap, we tradeoff between bg<0b\_{g}<0 and bs>0b\_{s}>0 to represent how higher mixture fractions accelerate overfitting.

The key insight is that btrain​(δ)b\_{\mathrm{train}}(\delta) is strictly negative, indicating that training loss decreases monotonically throughout pretraining. Notably, the train-test gap also follows a power law, but with a positive bg​a​pb\_{gap}, meaning that overfitting increases monotonically with more epochs over the dataset. However,bg​a​pb\_{gap} remains below 1 across all mixture percentages up to 10%, implying that overfitting grows sublinearly over extended pretraining horizons. This accounts for the sustained effectiveness of SPT over prolonged training.




![Refer to caption](/html/2603.16177/assets/figures/fig10_delta_notation.png)

![Refer to caption](/html/2603.16177/assets/figures/fig10_new_delta_scatter.png)

Figure 12: Predicting loss after finetuning. We measure the difference in domain-specific test loss right after pretraining versus the best test loss after finetuning (Δ​Test\Delta\ \mathrm{Test}) as a function of the pretraining tokens. We observe that this difference follows a power law relationship.

### 5.2 Predicting Test Loss After Finetuning

Now that we have established an accurate model characterizing the domain-specific test loss as a function of pretraining compute, we turn our attention to predicting the test loss after finetuning. Directly modeling the post-finetuning test loss proves challenging, as it entangles the contributions of pretraining and finetuning. Instead, we leverage our existing pretraining loss model and separately characterize the marginal effect of finetuning by modeling the difference between the test loss at the pretrained checkpoint and the test loss after subsequent finetuning.

Formally, we define the change in domain test loss after finetuning as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​ℓt​e​s​t=ℓt​e​s​t​(θP​T)−ℓt​e​s​t​(θP​T+F​T),\displaystyle\Delta\ell\_{test}=\ell\_{test}({\theta\_{PT}})-\ell\_{test}({\theta\_{PT+FT}}), |  | (7) |

where ℓt​e​s​t​(θP​T)\ell\_{test}({\theta\_{PT}}) denotes the test loss of the pretrained model and ℓt​e​s​t​(θP​T+F​T)\ell\_{test}({\theta\_{PT+FT}}) denotes the best loss after the model has been further finetuned on the target domain. A positive value of Δ​ℓt​e​s​t\Delta\ell\_{test} thus indicates that finetuning has reduced the test loss relative to the pretrained checkpoint. Since we employ early stopping, the difference will always be above 0.

Empirically, we observe that this change in test loss follows a remarkably consistent power-law relationship as a function of the number of pretraining steps
T: Δ​ℓt​e​s​t=a​Tb+c\Delta\ell\_{test}=aT^{b}+c. Intuitively, this relationship captures how the benefit conferred by finetuning varies depending on the stage of pretraining at which the model is finetuned. We illustrate the power-law relationship in MusicPile in Figure [12](#S5.F12 "Figure 12 ‣ Interpreting the exponent ‣ 5.1 Overfitting during Specialized Pretraining ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data") and provide supporting evidence for the other domains in Appendix [D](#A4 "Appendix D Overfitting Scaling Laws Extended ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").

![Refer to caption](/html/2603.16177/assets/figures/fig11a_predicting_pt_loss.png)


(a) Predicting 10%10\% Pretraining Curve

![Refer to caption](/html/2603.16177/assets/figures/fig11_post_ft_loss_forecast.png)


(b) Post-FT Loss past 120B PT Tokens

Figure 13: Forecasting domain loss and optimal mixture from small-scale runs. We demonstrate two ways our scaling laws can guide the choice of mixture percentage for SPT. (Left) We predict the entire domain test loss trajectory of 10%10\% SPT by fitting our scaling laws (Equation [1](#S5.E1 "Equation 1 ‣ 5.1 Overfitting during Specialized Pretraining ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")) using only the curves from δ∈[0,5]\delta\in[0,5]. The forecast correctly predicts that 10%10\% SPT begins to overfit around 70B tokens. (Right) We extrapolate post-finetuning test loss past 120B pretraining tokens and correctly predict that 2%2\% SPT surpasses 5%5\% SPT around 280B tokens. Together, these forecasts allow practitioners to identify the right mixture fraction without running the full training sweep.

### 5.3 Forecasting Examples

Together, our scaling laws precisely model the domain test loss when repeating a small set of data multiple times across pretraining *and* finetuning. We demonstrate two ways our scaling laws can be used to guide the choice of optimal mixture percentage for SPT.
Recall that we model the domain test loss during SPT with any mixture percentage as the sum of two powers with the exponent for both powers set to (1−δ)​bg+δ​bs(1-\delta)b\_{g}+\delta b\_{s} where bg,bsb\_{g},b\_{s} are learned variables. We learn bgb\_{g} and bsb\_{s} and the other variables for the other terms A​(δ)A(\delta) and C​(δ)C(\delta) using the mixtures δ∈[0,5]\delta\in[0,5], then extrapolate the full domain loss curve of SPT for 200B tokens. In Figure [13(a)](#S5.F13.sf1 "Figure 13(a) ‣ Figure 13 ‣ 5.2 Predicting Test Loss After Finetuning ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), we compare our prediction to the true 10%10\% SPT loss curve. The forecast tracks the observed values closely across the full 200B-token range, correctly predicting that SPT at 10%10\% begins to overfit around 70B tokens.

Moreover, we can combine our pretraining scaling laws with our power law relationship over Δ​ℓtest\Delta\ell\_{\mathrm{test}} to fully extrapolate post-finetuning loss beyond 120B pretraining tokens for δ∈[0,5%]\delta\in[0,5\%]. As shown in Figure [13(b)](#S5.F13.sf2 "Figure 13(b) ‣ Figure 13 ‣ 5.2 Predicting Test Loss After Finetuning ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), our extrapolation correctly predicts that 2%2\% SPT surpasses 5%5\% SPT around 280B tokens (consistent with the full runs in Figure [9](#S3.F9 "Figure 9 ‣ 3.4 Pretraining compute budget ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). In practice, these scaling laws eliminate the need for expensive sweeps over mixture fractions and training horizons. A small number of pilot runs suffices to fit the expressions and select the right SPT configuration for a given compute budget, making specialized pretraining both effective and predictable.

## 6 Discussion

Our work exposes *the finetuner’s fallacy*. The standard approach to domain
adaptation, finetuning a large general-purpose model appears cheap because it avoids
the cost of pretraining. But this accounting ignores inference. A 1B SPT model
costs more to train than finetuning a 3B model, but the 3×3\times smaller model is
cheaper to serve (Figure [3](#S1.F3 "Figure 3 ‣ 1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")). At scale, the break-even point
arrives quickly, after which SPT saves both compute and money while delivering
superior performance. For deployed models, inference dominates total energy consumption,
so enabling the same capability with fewer parameters also reduces the carbon footprint
of domain-adapted language models.

Importantly, the finetuner’s fallacy is not “fine-tuning is always enough” or
“always pretrain your model.” Real systems occupy different points on a
continuum, and the right strategy depends on how much domain data you have, how far your
domain sits from web text, and what your serving constraints look like. Our experiments
provide systematic evidence for where the transitions between these regimes lie.
SPT’s gains are largest when the target domain is far from web text, scale with model size, and
depend on the interaction between domain dataset size and pretraining budget. Our overfitting scaling
laws let practitioners navigate this tradeoff from a small number of pilot runs rather
than exhaustive search.

The broader lesson is simple: scarce and specialized domain data should not be treated as a final-stage
resource. Across three domains ranging from symbolic music to mathematical proofs,
mixing even 1–5% domain data into pretraining consistently outperformed the standard
pipeline of general pretraining followed by finetuning. These gains persisted even under
replay-based continued pretraining, confirming that *when* domain data enters the
training pipeline has a lasting impact on model performance. In 2026, the cost of pretraining
continues to fall, and the case for early integration only strengthens with it. As
organizations increasingly seek to deploy models tailored to proprietary domains, these
findings point toward a shift in how specialized models should be trained: away from
post-training patches applied to generic checkpoints, and toward natively specialized
models that incorporate domain knowledge from the start.

## Contributions

Christina Baek led the project and conducted all the core experiments.
Pratyush Maini provided project direction and contributed to the experimental design and analysis.

David Schwab, Ricardo Monti, Aditi Raghunathan, Zico Kolter, Bogdan Gaza, Ari Morcos, and Matthew Leavitt
provided guidance throughout the project and feedback on the draft.

The Datology team contributed to helpful discussions and provided the infrastructure that supported the experiments in this paper:
Amro Abbas, Rishabh Adiga, Cody Blakeney, Maximilian Böther, Paul Burstein, Aldo Gael Carranza, Alvin Deng, Parth Doshi, Vineeth Dorna, Alex Fang, Tony Jiang, Siddharth Joshi, Brett W. Larsen, Jason Chan Lee, Katherine L. Mentzer, Luke Merrick, Haakon Mongstad, Fan Pan, Anshuman Suri, Darren Teh, Jason Telanoff, Jack Urbanek, Zhengping Wang, Josh Wills, and Haoli Yin.

We thank Liz Gatapia for her help with the logo design.

## Acknowledgements

We’d like to thank Suhas Kotha, Jacob Springer, Gaurav Ghosal, and Lawrence Feng for their insights on finetuning and their feedback on earlier versions of this work.

## References

* S. N. Akter, S. Prabhumoye, E. Nyberg, M. Patwary, M. Shoeybi, Y. Choi, and B. Catanzaro (2025)
  Front-loading reasoning: the synergy between pretraining and post-training data.
  arXiv preprint arXiv:2510.03264.
  External Links: [Link](https://arxiv.org/abs/2510.03264)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* R. Asif (2024)
  LLM pre-training and fine-tuning differences.
  Note: <https://raga.ai/resources/blogs/llm-pretraining>Accessed 2025-11-17
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* AWS ML Blog Team (2024)
  Efficient continual pre-training llms for financial domains.
  Note: <https://aws.amazon.com/blogs/machine-learning/efficient-continual-pre-training-llms-for-financial-domains/>Accessed 2025-11-17
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* D. Baraishuk (2025)
  Composer llm from vibe coding platform cursor 2.0: cool or overhyped?.
  Note: <https://belitsoft.com/news/composer-llm-cursor-ai-20251029>Accessed 2025-11-17
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* S. Biderman (2025)
  Cited by: [§2.3](#S2.SS3.p2.1 "2.3 SPT Learns More and Forgets Less ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* C. Blakeney, M. Paul, B. W. Larsen, S. Owen, and J. Frankle (2024)
  Does your data spark joy? performance gains from domain upsampling at the end of training.
  In First Conference on Language Modeling,
  External Links: [Link](https://openreview.net/forum?id=vwIIAot0ff)
  Cited by: [§A.1](#A1.SS1.p2.1 "A.1 Catastrophic Forgetting and Replay in Domain Adaptation ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§4](#S4.p1.3 "4 Does Specialized Pretraining Help Under Replay as Well? ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* Bloomberg L.P. Team (2023)
  Introducing bloomberggpt: bloomberg’s 50-billion-parameter large language model built for finance.
  Note: <https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/>Accessed 2025-11-17
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* A. G. Carranza, K. Mentzer, R. P. Monti, A. Fang, A. Deng, A. Abbas, A. Suri, B. Larsen, C. Blakeney, D. Teh, et al. (2026)
  \\backslash” UberWeb: insights from multilingual curation for a 20-trillion-token dataset.
  arXiv preprint arXiv:2602.15210.
  Cited by: [§3.1.1](#S3.SS1.SSS1.p1.4 "3.1.1 A controlled study of distributional overlap ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* Character.AI Team (2024)
  Optimizing ai inference at character.ai.
  Note: <https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/>Accessed 2025-11-17
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* Character.AI Team (2025)
  Breaking news: our open-source models are a lot of fun.
  Note: <https://blog.character.ai/breaking-news-our-open-source-models-are-a-lot-of-fun/>Accessed 2025-11-17
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* T. Chu, Y. Zhai, J. Yang, S. Tong, S. Xie, D. Schuurmans, Q. V. Le, S. Levine, and Y. Ma (2025)
  SFT memorizes, RL generalizes: a comparative study of foundation model post-training.
  arXiv preprint arXiv:2501.17161.
  Cited by: [§A.3](#A1.SS3.p1.1 "A.3 The synergy between pretraining and post-training ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* Codecademy Team (2025)
  Cursor 2.0: new ai model explained.
  Note: <https://www.codecademy.com/article/cursor-2-0-new-ai-model-explained>Accessed 2025-11-17
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* A. Fan, S. Bhosale, H. Schwenk, Z. Ma, A. El-Kishky, S. Goyal, M. Baines, O. Celebi, G. Wenzek, V. Chaudhary, et al. (2021)
  Beyond english-centric multilingual machine translation.
  Journal of Machine Learning Research 22 (107),  pp. 1–48.
  Cited by: [§3.1.1](#S3.SS1.SSS1.p1.4 "3.1.1 A controlled study of distributional overlap ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* S. Goyal, P. Maini, Z. C. Lipton, A. Raghunathan, and J. Z. Kolter (2024)
  Scaling laws for data filtering– data curation cannot be compute agnostic.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
   pp. 22702–22711.
  Cited by: [§A.2](#A1.SS2.p1.1 "A.2 Scaling Laws for Data Mixing and Repeated Data ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§5.1](#S5.SS1.p1.1 "5.1 Overfitting during Specialized Pretraining ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* D. Groeneveld, I. Beltagy, P. Walsh, A. Bhagia, R. Kinney, O. Tafjord, A. H. Jha, H. Ivison, I. Magnusson, Y. Wang, S. Arora, D. Atkinson, R. Authur, K. Chandu, A. Cohan, J. Dumas, Y. Elazar, Y. Gu, J. Hessel, T. Khot, W. Merrill, J. Morrison, N. Muennighoff, A. Naik, C. Nam, M. E. Peters, V. Pyatkin, A. Ravichander, D. Schwenk, S. Shah, W. Smith, N. Subramani, M. Wortsman, P. Dasigi, N. Lambert, K. Richardson, J. Dodge, K. Lo, L. Soldaini, N. A. Smith, and H. Hajishirzi (2024)
  OLMo: accelerating the science of language models.
  Preprint.
  Cited by: [§B.1](#A2.SS1.p1.1 "B.1 Pretraining Hyperparameter Configurations ‣ Appendix B Optimization Details ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* K. Gupta, B. Thérien, A. Ibrahim, M. L. Richter, Q. Anthony, E. Belilovsky, I. Rish, and T. Lesort (2023)
  Continual pre-training of large language models: how to (re)warm your model?.
  arXiv preprint arXiv:2308.04014.
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Catastrophic Forgetting and Replay in Domain Adaptation ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* S. Gururangan, A. Marasović, S. Swayamdipta, K. Lo, I. Beltagy, D. Downey, and N. A. Smith (2020)
  Don’t stop pretraining: adapt language models to domains and tasks.
  In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics,
  External Links: [Link](https://aclanthology.org/2020.acl-main.740/)
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Catastrophic Forgetting and Replay in Domain Adaptation ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* V. Hangya, H. S. Saadi, and A. Fraser (2022)
  Improving low-resource languages in pre-trained multilingual language models.
  In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Y. Goldberg, Z. Kozareva, and Y. Zhang (Eds.),
  Abu Dhabi, United Arab Emirates,  pp. 11993–12006.
  External Links: [Link](https://aclanthology.org/2022.emnlp-main.822/),
  [Document](https://dx.doi.org/10.18653/v1/2022.emnlp-main.822)
  Cited by: [§3.1.1](#S3.SS1.SSS1.p1.4 "3.1.1 A controlled study of distributional overlap ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* J. Harmon, A. Hochlehnert, M. Bethge, and A. Prabhu (2025)
  Mapping post-training forgetting in language models at scale.
  arXiv preprint arXiv:2510.17776.
  Cited by: [§A.1](#A1.SS1.p2.1 "A.1 Catastrophic Forgetting and Replay in Domain Adaptation ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* A. Hatamizadeh, S. N. Akter, S. Prabhumoye, J. Kautz, M. Patwary, M. Shoeybi, B. Catanzaro, and Y. Choi (2025)
  RLP: reinforcement as a pretraining objective.
  External Links: 2510.01265,
  [Link](https://arxiv.org/abs/2510.01265)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* Hoskinson Center for Formal Mathematics (2022)
  Proof-pile: a dataset of high quality mathematical text.
  Note: <https://huggingface.co/datasets/hoskinson-center/proof-pile>Hugging Face Datasets
  Cited by: [Appendix C](#A3.SS0.SSS0.Px3.p1.1 "ProofPile ‣ Appendix C Construction of Domain Datasets ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§2.1](#S2.SS1.SSS0.Px2.p1.4 "OLMo Sandbox ‣ 2.1 Notation and Experimental Setup ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen (2022)
  LoRA: low-rank adaptation of large language models.
  In International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=nZeVKeeFYf9)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei (2020)
  Scaling laws for neural language models.
  arXiv preprint arXiv:2001.08361.
  External Links: [Link](https://arxiv.org/abs/2001.08361)
  Cited by: [§A.2](#A1.SS2.p1.1 "A.2 Scaling Laws for Data Mixing and Repeated Data ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§5.1](#S5.SS1.p1.1 "5.1 Overfitting during Specialized Pretraining ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* S. Kotha and P. Liang (2026)
  Replaying pre-training data improves fine-tuning.
  arXiv preprint arXiv:2603.04964.
  Cited by: [§A.1](#A1.SS1.p2.1 "A.1 Catastrophic Forgetting and Replay in Domain Adaptation ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§4](#S4.p1.3 "4 Does Specialized Pretraining Help Under Replay as Well? ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* S. Lai, H. Zhao, R. Feng, C. Ma, W. Liu, H. Zhao, X. Lin, D. Yi, Q. Zhang, H. Liu, G. Meng, and F. Zhu (2025)
  Reinforcement fine-tuning naturally mitigates forgetting in continual post-training.
  arXiv preprint arXiv:2507.05386.
  Cited by: [§A.3](#A1.SS3.p1.1 "A.3 The synergy between pretraining and post-training ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* E. Liu, G. Neubig, and C. Xiong (2025)
  Midtraining bridges pretraining and posttraining distributions.
  arXiv preprint arXiv:2510.14865.
  Cited by: [§4](#S4.p1.3 "4 Does Specialized Pretraining Help Under Replay as Well? ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* Y. Liu, J. Gu, N. Goyal, X. Li, S. Edunov, M. Ghazvininejad, M. Lewis, and L. Zettlemoyer (2020)
  Multilingual denoising pre-training for neural machine translation.
  External Links: 2001.08210,
  [Link](https://arxiv.org/abs/2001.08210)
  Cited by: [§3.1.1](#S3.SS1.SSS1.p1.4 "3.1.1 A controlled study of distributional overlap ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* S. Longpre, S. Kudugunta, N. Muennighoff, I. Hsu, I. Caswell, A. Pentland, S. Arik, C. Lee, and S. Ebrahimi (2025)
  ATLAS: adaptive transfer scaling laws for multilingual pretraining, finetuning, and decoding the curse of multilinguality.
  External Links: 2510.22037,
  [Link](https://arxiv.org/abs/2510.22037)
  Cited by: [§A.3](#A1.SS3.p1.1 "A.3 The synergy between pretraining and post-training ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§1](#S1.p2.1 "1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* D. Lopez-Paz and M. Oquab (2016)
  Revisiting classifier two-sample tests.
  External Links: 1610.06545,
  [Link](https://arxiv.org/abs/1610.06545)
  Cited by: [§E.1](#A5.SS1.SSS0.Px3.p1.1 "Classifier Two-Sample Test (C2ST). ‣ E.1 Metrics Description ‣ Appendix E Distribution Similarity Metrics: Detailed Analysis ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§3.1.2](#S3.SS1.SSS2.p1.1 "3.1.2 Correlation Analysis Across Domains ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* Y. Luo, Z. Yang, F. Meng, Y. Li, J. Zhou, and Y. Zhang (2023)
  An empirical study of catastrophic forgetting in large language models during continual fine-tuning.
  arXiv preprint arXiv:2308.08747.
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Catastrophic Forgetting and Replay in Domain Adaptation ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* P. Maini, S. Goyal, D. Sam, A. Robey, Y. Savani, Y. Jiang, A. Zou, M. Fredrikson, Z. C. Lipton, and J. Z. Kolter (2025)
  Safety pretraining: toward the next generation of safe ai.
  arXiv preprint arXiv:2504.16980.
  Cited by: [§A.3](#A1.SS3.p1.1 "A.3 The synergy between pretraining and post-training ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§1](#S1.p2.1 "1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* M. McCloskey and N. J. Cohen (1989)
  Catastrophic interference in connectionist networks: the sequential learning problem.
  Psychology of Learning and Motivation 24,  pp. 109–165.
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Catastrophic Forgetting and Replay in Domain Adaptation ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* A. Mirza, N. Alampara, S. Kunchapu, M. Ríos-García, B. Emoekabu, A. Krishnan, T. Gupta, M. Schilling-Wilhelmi, M. Okereke, A. Aneesh, M. Asgari, J. Eberhardt, A. M. Elahi, H. M. Elbeheiry, M. V. Gil, C. Glaubitz, M. Greiner, C. T. Holick, T. Hoffmann, A. Ibrahim, L. C. Klepsch, Y. K”oster, F. A. Kreth, J. Meyer, S. Miret, J. M. Peschel, M. Ringleb, N. C. Roesner, J. Schreiber, U. S. Schubert, L. M. Stafast, A. D. D. Wonanke, M. Pieler, P. Schwaller, and K. M. Jablonka (2025a)
  A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists.
  Nature Chemistry.
  External Links: ISSN 1755-4349,
  [Link](http://dx.doi.org/10.1038/s41557-025-01815-x),
  [Document](https://dx.doi.org/10.1038/s41557-025-01815-x)
  Cited by: [§2.3](#S2.SS3.p2.1 "2.3 SPT Learns More and Forgets Less ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* A. Mirza, N. Alampara, M. Ríos-García, M. Abdelalim, J. Butler, B. Connolly, T. Dogan, M. Nezhurina, B. Şen, S. Tirunagari, M. Worrall, A. Young, P. Schwaller, M. Pieler, and K. M. Jablonka (2025b)
  ChemPile: a 250gb diverse and curated dataset for chemical foundation models.
  External Links: 2505.12534,
  [Link](https://arxiv.org/abs/2505.12534)
  Cited by: [Appendix C](#A3.SS0.SSS0.Px2.p1.1 "ChemPile ‣ Appendix C Construction of Domain Datasets ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§2.1](#S2.SS1.SSS0.Px2.p1.4 "OLMo Sandbox ‣ 2.1 Notation and Experimental Setup ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* M. Morishita, K. Chousa, J. Suzuki, and M. Nagata (2022)
  JParaCrawl v3.0: a large-scale English-Japanese parallel corpus.
  In Proceedings of the Thirteenth Language Resources and Evaluation Conference,
  Marseille, France,  pp. 6704–6710.
  External Links: [Link](https://aclanthology.org/2022.lrec-1.721)
  Cited by: [§3.1.1](#S3.SS1.SSS1.p2.3 "3.1.1 A controlled study of distributional overlap ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* N. Muennighoff, A. M. Rush, B. Barak, T. L. Scao, A. Piktus, N. Tazi, S. Pyysalo, T. Wolf, and C. Raffel (2023)
  Scaling data-constrained language models.
  External Links: 2305.16264,
  [Link](https://arxiv.org/abs/2305.16264)
  Cited by: [§A.2](#A1.SS2.p1.1 "A.2 Scaling Laws for Data Mixing and Repeated Data ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§5.1](#S5.SS1.p1.1 "5.1 Overfitting during Specialized Pretraining ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* K. Naminas (2025)
  Pre-training vs fine tuning: choosing the right approach.
  Note: <https://labelyourdata.com/articles/llm-fine-tuning/pre-training-vs-fine-tuning>Accessed 2025-11-17
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* A. Nieto (2025)
  Understanding llm pre-training and custom llms.
  Note: <https://www.databricks.com/blog/llm-pre-training-and-custom-llms>Accessed 2025-11-17
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* NIX Solutions (2023)
  Ex-google developers created character.ai.
  Note: <https://nixsolutions-ai.com/characterai/>Accessed 2025-11-17
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* NLLB-Team, M. R. Costa-jussà, J. Cross, O. Çelebi, M. Elbayad, K. Heafield, K. Heffernan, E. Kalbassi, J. Lam, D. Licht, J. Maillard, A. Sun, S. Wang, G. Wenzek, A. Youngblood, B. Akula, L. Barrault, G. M. Gonzalez, P. Hansanti, J. Hoffman, S. Jarrett, K. R. Sadagopan, D. Rowe, S. Spruit, C. Tran, P. Andrews, N. F. Ayan, S. Bhosale, S. Edunov, A. Fan, C. Gao, V. Goswami, F. Guzmán, P. Koehn, A. Mourachko, C. Ropers, S. Saleem, H. Schwenk, and J. Wang (2022)
  No language left behind: scaling human-centered machine translation.
  External Links: 2207.04672,
  [Link](https://arxiv.org/abs/2207.04672)
  Cited by: [§3.1.1](#S3.SS1.SSS1.p1.4 "3.1.1 A controlled study of distributional overlap ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe (2022)
  Training language models to follow instructions with human feedback.
  arXiv preprint arXiv:2203.02155.
  External Links: [Link](https://arxiv.org/abs/2203.02155)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* J. Parmar, S. Satheesh, M. Patwary, M. Shoeybi, and B. Catanzaro (2024)
  Reuse, don’t retrain: a recipe for continued pretraining of language models.
  External Links: 2407.07263,
  [Link](https://arxiv.org/abs/2407.07263)
  Cited by: [§A.1](#A1.SS1.p2.1 "A.1 Catastrophic Forgetting and Replay in Domain Adaptation ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§1](#S1.p4.1 "1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§4](#S4.p1.3 "4 Does Specialized Pretraining Help Under Replay as Well? ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* G. Penedo, H. Kydlíček, V. Sabolčec, B. Messmer, N. Foroutan, A. H. Kargaran, C. Raffel, M. Jaggi, L. V. Werra, and T. Wolf (2025)
  FineWeb2: one pipeline to scale them all – adapting pre-training data processing to every language.
  External Links: 2506.20920,
  [Link](https://arxiv.org/abs/2506.20920)
  Cited by: [§3.1.1](#S3.SS1.SSS1.p2.3 "3.1.1 A controlled study of distributional overlap ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* K. Pillutla, S. Swayamdipta, R. Zellers, J. Thickstun, S. Welleck, Y. Choi, and Z. Harchaoui (2021)
  MAUVE: measuring the gap between neural text and human text using divergence frontiers.
  External Links: 2102.01454,
  [Link](https://arxiv.org/abs/2102.01454)
  Cited by: [§E.1](#A5.SS1.SSS0.Px2.p1.1 "MAUVE. ‣ E.1 Metrics Description ‣ Appendix E Distribution Similarity Metrics: Detailed Analysis ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§3.1.2](#S3.SS1.SSS2.p1.1 "3.1.2 Correlation Analysis Across Domains ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* H. Que, J. Liu, G. Zheng, C. Huang, X. Gao, W. Zhang, et al. (2024)
  D-CPT law: domain-specific continual pre-training scaling law for large language models.
  In Advances in Neural Information Processing Systems,
  Cited by: [§A.2](#A1.SS2.p1.1 "A.2 Scaling Laws for Data Mixing and Repeated Data ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* D. Sam, S. Goyal, P. Maini, A. Robey, and J. Z. Kolter (2025)
  When should we introduce safety interventions during pretraining?.
  arXiv preprint arXiv:2601.07087.
  External Links: [Link](https://arxiv.org/abs/2601.07087)
  Cited by: [§A.3](#A1.SS3.p1.1 "A.3 The synergy between pretraining and post-training ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§1](#S1.p2.1 "1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* I. Shenfeld, J. Pari, and P. Agrawal (2025)
  RL’s razor: why online reinforcement learning forgets less.
  arXiv preprint arXiv:2509.04259.
  Cited by: [§A.3](#A1.SS3.p1.1 "A.3 The synergy between pretraining and post-training ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* H. Shi, Z. Xu, H. Wang, W. Qin, W. Wang, Y. Wang, Z. Wang, S. Ebrahimi, and H. Wang (2024)
  Continual learning of large language models: a comprehensive survey.
  arXiv preprint arXiv:2404.16789.
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Catastrophic Forgetting and Replay in Domain Adaptation ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* L. Thede, S. Winzeck, Z. Akata, and J. R. Schwarz (2026)
  CapTrack: multifaceted evaluation of forgetting in LLM post-training.
  arXiv preprint arXiv:2603.06610.
  Cited by: [§A.1](#A1.SS1.p2.1 "A.1 Catastrophic Forgetting and Replay in Domain Adaptation ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou (2020)
  Minilm: deep self-attention distillation for task-agnostic compression of pre-trained transformers.
  Advances in neural information processing systems 33,  pp. 5776–5788.
  Cited by: [§E.1](#A5.SS1.SSS0.Px3.p1.1 "Classifier Two-Sample Test (C2ST). ‣ E.1 Metrics Description ‣ Appendix E Distribution Similarity Metrics: Detailed Analysis ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le (2021)
  Finetuned language models are zero-shot learners.
  arXiv preprint arXiv:2109.01652.
  External Links: [Link](https://arxiv.org/abs/2109.01652)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* S. Wu, O. Irsoy, S. Lu, V. Dabravolski, M. Dredze, S. Gehrmann, P. Kambadur, D. Rosenberg, and G. Mann (2023)
  BloombergGPT: a large language model for finance.
  arXiv preprint arXiv:2303.17564.
  External Links: [Link](https://arxiv.org/abs/2303.17564)
  Cited by: [§A.4](#A1.SS4.p1.1 "A.4 Specialized Pretraining in Deployed Systems ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* H. Xu, Y. J. Kim, A. Sharaf, and H. H. Awadalla (2024)
  A paradigm shift in machine translation: boosting translation performance of large language models.
  External Links: 2309.11674,
  [Link](https://arxiv.org/abs/2309.11674)
  Cited by: [§3.1.1](#S3.SS1.SSS1.p1.4 "3.1.1 A controlled study of distributional overlap ‣ 3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* J. Ye, P. Liu, T. Sun, Y. Zhou, J. Zhan, and X. Qiu (2024)
  Data mixing laws: optimizing data mixtures by predicting language modeling performance.
  arXiv preprint arXiv:2403.16952.
  Cited by: [§A.2](#A1.SS2.p1.1 "A.2 Scaling Laws for Data Mixing and Repeated Data ‣ Appendix A Related Works ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").
* R. Yuan, H. Lin, Y. Wang, Z. Tian, S. Wu, T. Shen, G. Zhang, Y. Wu, C. Liu, Z. Zhou, Z. Ma, L. Xue, Z. Wang, Q. Liu, T. Zheng, Y. Li, Y. Ma, Y. Liang, X. Chi, R. Liu, Z. Wang, P. Li, J. Wu, C. Lin, Q. Liu, T. Jiang, W. Huang, W. Chen, E. Benetos, J. Fu, G. Xia, R. Dannenberg, W. Xue, S. Kang, and Y. Guo (2024)
  ChatMusician: understanding and generating music intrinsically with llm.
  External Links: 2402.16153
  Cited by: [Appendix C](#A3.SS0.SSS0.Px1.p1.1 "MusicPile ‣ Appendix C Construction of Domain Datasets ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§2.1](#S2.SS1.SSS0.Px2.p1.4 "OLMo Sandbox ‣ 2.1 Notation and Experimental Setup ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"),
  [§2.3](#S2.SS3.p2.1 "2.3 SPT Learns More and Forgets Less ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data").

## Appendix A Related Works

### A.1 Catastrophic Forgetting and Replay in Domain Adaptation

Catastrophic forgetting, the phenomenon whereby learning new information overwrites
previously acquired knowledge, has been studied since at least McCloskey and Cohen [[1989](#bib.bib68 "Catastrophic interference in connectionist networks: the sequential learning problem")].
In the LLM era, Gururangan et al. [[2020](#bib.bib14 "Don’t stop pretraining: adapt language models to domains and tasks")] showed that continued pretraining on domain-relevant text improves downstream performance but can degrade general capabilities.
Luo et al. [[2023](#bib.bib70 "An empirical study of catastrophic forgetting in large language models during continual fine-tuning")] demonstrated that forgetting intensifies as model scale increases from 1B to 7B parameters during continual finetuning, and that general instruction tuning prior to specialization can mitigate the problem.
Gupta et al. [[2023](#bib.bib71 "Continual pre-training of large language models: how to (re)warm your model?")] studied how to “rewarm” learning rate schedules for continual pretraining without destabilizing the model, a practical consideration our work shares.
Shi et al. [[2024](#bib.bib72 "Continual learning of large language models: a comprehensive survey")] provide a comprehensive survey of continual learning strategies for LLMs, cataloguing replay-based, regularization-based, and architecture-based mitigation approaches.

Replay, mixing previously seen general data back into later training stages, is the most widely adopted defense.
Parmar et al. [[2024](#bib.bib66 "Reuse, don’t retrain: a recipe for continued pretraining of language models")] propose a recipe for continued pretraining that mixes general and domain data to mitigate forgetting, and Blakeney et al. [[2024](#bib.bib67 "Does your data spark joy? performance gains from domain upsampling at the end of training")] show that domain upsampling at the end of training can yield performance gains.
More recently, Kotha and Liang [[2026](#bib.bib73 "Replaying pre-training data improves fine-tuning")] show that replaying generic pretraining data during finetuning not only prevents forgetting of general knowledge but can actually *improve* target-task performance, increasing data efficiency by up to 2.06×2.06\times for mid-training.
They further find that replay helps more when there is less target data present in the pretraining mix, directly complementing our finding that SPT’s advantage is largest when the domain is underrepresented.
On the measurement side, Harmon et al. [[2025](#bib.bib74 "Mapping post-training forgetting in language models at scale")] propose sample-wise metrics revealing that large per-example forgetting can hide beneath stable aggregate accuracy, and Thede et al. [[2026](#bib.bib75 "CapTrack: multifaceted evaluation of forgetting in LLM post-training")] introduce CapTrack, a capability-centric framework showing that post-training forgetting extends well beyond factual knowledge loss to encompass drift in multilingual robustness, instruction following, and calibration.

Our work (§ [4](#S4 "4 Does Specialized Pretraining Help Under Replay as Well? ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")) shows that replay during continued pretraining is not a substitute for early domain exposure: SPT’s gains persist across all replay settings.
This is consistent with the broader message of these works that *when* data appears in the training pipeline matters as much as *whether* it appears.

### A.2 Scaling Laws for Data Mixing and Repeated Data

A separate line of work develops predictive models for how data composition affects final loss.
Kaplan et al. [[2020](#bib.bib26 "Scaling laws for neural language models")] established the first scaling laws relating model size, dataset size, and compute to language modeling loss.
Muennighoff et al. [[2023](#bib.bib36 "Scaling data-constrained language models")] extended these laws to data-constrained regimes where tokens must be repeated, finding that multiple epochs are beneficial up to a point but eventually yield diminishing returns; however, their expressions do not model the overfitting stage that we observe.
Ye et al. [[2024](#bib.bib76 "Data mixing laws: optimizing data mixtures by predicting language modeling performance")], Goyal et al. [[2024](#bib.bib3 "Scaling laws for data filtering– data curation cannot be compute agnostic")] propose *data mixing laws* that predict loss as a function of domain mixture proportions, enabling practitioners to optimize data blends from small pilot runs.
Que et al. [[2024](#bib.bib77 "D-CPT law: domain-specific continual pre-training scaling law for large language models")] derive a domain-specific continual pretraining scaling law that predicts the optimal mixture ratio between general and domain corpora as a function of model size and data budget.

Our overfitting scaling laws (§[5](#S5 "5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")) complement these efforts by separately modeling the training loss and train-test gap as power laws with opposing exponents.
This decomposition captures the non-monotonic test loss trajectory that arises from heavy repetition of a small domain corpus, a regime not addressed by prior scaling law formulations.
We further parametrize all coefficients as functions of the mixture fraction δ\delta, enabling extrapolation across both mixture percentages and training horizons from a small set of pilot runs.

### A.3 The synergy between pretraining and post-training

More broadly, a capability-oriented view of the pretraining-vs-post-training boundary is
emerging across several subfields. In safety alignment, Maini et al. [[2025](#bib.bib55 "Safety pretraining: toward the next generation of safe ai")] and
Sam et al. [[2025](#bib.bib65 "When should we introduce safety interventions during pretraining?")] show that behaviors acquired during pretraining are harder to remove
via post-training than the reverse, implying that the stage at which data is introduced
determines how durably it shapes the model. In multilingual settings, Longpre et al. [[2025](#bib.bib63 "ATLAS: adaptive transfer scaling laws for multilingual pretraining, finetuning, and decoding the curse of multilinguality")]
find that cross-language transfer during pretraining improves low-resource language
performance more effectively than later-stage exposure. And in the post-training literature,
Chu et al. [[2025](#bib.bib78 "SFT memorizes, RL generalizes: a comparative study of foundation model post-training")] demonstrate that RL with outcome-based rewards generalizes to unseen
task variants whereas SFT memorizes training examples, a finding formalized by
Shenfeld et al. [[2025](#bib.bib79 "RL’s razor: why online reinforcement learning forgets less")] as *RL’s Razor*: on-policy RL is implicitly biased toward
the policy closest in KL divergence to the base model, which Lai et al. [[2025](#bib.bib80 "Reinforcement fine-tuning naturally mitigates forgetting in continual post-training")] confirm
mitigates forgetting across sequences of continual post-training tasks. The common thread
across all three settings is that the *manner* in which data is introduced, not just its
quantity, determines whether the model memorizes or generalizes. SPT operationalizes this
principle at the pretraining stage: by interleaving domain tokens among general data, no
single batch is dominated by the scarce domain corpus, producing a regularization effect
analogous to RL’s on-policy sampling.

### A.4 Specialized Pretraining in Deployed Systems

Industry practice illustrates that the choice between pretraining from scratch and finetuning an existing model lies on a continuum indexed by data scale, domain shift, and deployment constraints [Naminas, [2025](#bib.bib1 "Pre-training vs fine tuning: choosing the right approach"), Asif, [2024](#bib.bib2 "LLM pre-training and fine-tuning differences"), AWS ML Blog Team, [2024](#bib.bib4 "Efficient continual pre-training llms for financial domains"), Nieto, [2025](#bib.bib5 "Understanding llm pre-training and custom llms")].
Bloomberg trained BloombergGPT, a 50B-parameter model on roughly 700B tokens combining a 363B-token proprietary finance corpus with general data, arguing that a finance-aware model outperforms a stack of task-specific finetuned models [Bloomberg L.P. Team, [2023](#bib.bib8 "Introducing bloomberggpt: bloomberg’s 50-billion-parameter large language model built for finance"), Wu et al., [2023](#bib.bib7 "BloombergGPT: a large language model for finance")].
Character.AI operates both large custom conversational models trained from scratch [NIX Solutions, [2023](#bib.bib9 "Ex-google developers created character.ai"), Character.AI Team, [2024](#bib.bib10 "Optimizing ai inference at character.ai")] and aggressively finetuned open-source models using SFT, DPO, and RL [Character.AI Team, [2025](#bib.bib11 "Breaking news: our open-source models are a lot of fun")].
Cursor’s Composer illustrates the opposite direction: RL-based post-training atop an existing coding backbone rather than training from scratch [Codecademy Team, [2025](#bib.bib12 "Cursor 2.0: new ai model explained"), Baraishuk, [2025](#bib.bib13 "Composer llm from vibe coding platform cursor 2.0: cool or overhyped?")].
As Cursor pursues increasingly agentic capabilities, it may encounter the finetuner’s fallacy anew, as finetuning a base model that lacks agentic traces will eventually yield diminishing returns.

These cases suggest the finetuner’s fallacy is not simply “fine-tuning is always enough” or “everyone should train their own model.” Real systems occupy different points on a continuum indexed by (i) how much *domain-specific data* they control, (ii) how far their domain is from web-scale distributions, and (iii) *scale and latency* requirements. Yet there is little public evidence on where the transitions between these regimes actually lie. The goal of this paper is to move from anecdotes to a systematic characterization of when additional pretraining is warranted, and when fine-tuning suffices, as a function of data size, domain similarity, and model scale.

## Appendix B Optimization Details

### B.1 Pretraining Hyperparameter Configurations

We match OLMo-1B pretraining settings to the publicly documented configuration [Groeneveld et al., [2024](#bib.bib54 "OLMo: accelerating the science of language models")] where possible, including the optimizer, cosine learning rate schedule, and batch size.

Architecture
:   OLMo-1B

Batch Size
:   2048

Context Length
:   2048

Weight Tying
:   True

Gradient Clipping
:   1.0

Weight Decay
:   0.1

AdamW Betas
:   0.9, 0.95

AdamW Epsilon
:   1e-5

Learning Rate Scheduler
:   Cosine decay schedule from 4​e​-​44e\text{-}4 to 4​e​-​54e\text{-}5 over 2​T2T tokens. We use the same scheduler but cutting off the pretraining stage at 200​B200B tokens.

### B.2 Finetuning Hyperparameter Configurations

During finetuning, we ablate learning rate and warmup steps, while keeping other hyperparameters fixed. We set weight decay to be fairly small and remove weight tying.

Batch Size
:   512

Learning Rate
:   {1e-5, 4e-5, 1e-4}

Warmup
:   {50, 100, 200}

Weight Tying
:   False

Weight Decay
:   1e-7

AdamW Betas
:   0.9, 0.95

AdamW Epsilon
:   1e-5

Learning Rate Scheduler
:   Constant with warmup.

## Appendix C Construction of Domain Datasets

##### MusicPile

We take MusicPile [Yuan et al., [2024](#bib.bib37 "ChatMusician: understanding and generating music intrinsically with llm")] and subsample from music-specific sources:
sander-wood/irishman, Generated with GPT-4, and constructed from OpenChat, IrishMAN and KernScores. A large proportion of the final dataset test for musical composition in ABC notation, in addition to general music knowledge.

##### ChemPile

We take ChemPile [Mirza et al., [2025b](#bib.bib38 "ChemPile: a 250gb diverse and curated dataset for chemical foundation models")] and subsample from the reasoning, instruction, and education domains.

##### ProofPile

We subsample the data from entire ProofPile [Hoskinson Center for Formal Mathematics, [2022](#bib.bib39 "Proof-pile: a dataset of high quality mathematical text")] set.

## Appendix D Overfitting Scaling Laws Extended

### D.1 Justification for AgapA\_{\mathrm{gap}} and CtrainC\_{\mathrm{train}}

In Section [5](#S5 "5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), we modeled scaling laws as a function of the mixture fraction δ\delta. In this section, we validate our choices for AgapA\_{\mathrm{gap}} and CtrainC\_{\mathrm{train}}. AgapA\_{\mathrm{gap}} was modeled using the Gamma kernel (Equation [5](#S5.E5 "Equation 5 ‣ 5.1 Overfitting during Specialized Pretraining ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data")), and CtrainC\_{\mathrm{train}} was modeled using the log-linear function.

These function choices were made by first learning separate coefficients AδA\_{\delta} and CδC\_{\delta} to fit power law curves for each SPT run separately, then identifying the right expression that correctly models these coefficients as a function of δ\delta. Using scipy optimization package, we find power law fits by minimizing the mean squared error over all coefficients Aδ,Cδ​∀δ∈{0,0.1,1,2,5}A\_{\delta},C\_{\delta}\forall\delta\in\{0,0.1,1,2,5\} along with the scalar AtrainA\_{\mathrm{train}} and the additional two parameters in bx​(δ)b\_{\mathrm{x}}(\delta).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒtrain​(T,δ)\displaystyle\mathcal{L}\_{\mathrm{train}}(T,\delta) | =Atrain​Tbtrain​(δ)+Cδ,\displaystyle=A\_{\mathrm{train}}\,T^{\,b\_{\mathrm{train}}(\delta)}+C\_{\delta}, |  | (8) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒgap​(T,δ)\displaystyle\mathcal{L}\_{\mathrm{gap}}(T,\delta) | =Aδ​Tbgap​(δ)\displaystyle=A\_{\delta}\,T^{\,b\_{\mathrm{gap}}(\delta)} |  | (9) |

Below, we plot the learned coefficients for each δ\delta as scatter points and demonstrate that they roughly follow the Gamma kernel function for AδA\_{\delta} and log linear function for CδC\_{\delta}. In Figures [15](#A4.F15 "Figure 15 ‣ D.1 Justification for 𝐴_gap and 𝐶ₜᵣₐᵢₙ ‣ Appendix D Overfitting Scaling Laws Extended ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), [16](#A4.F16 "Figure 16 ‣ D.1 Justification for 𝐴_gap and 𝐶ₜᵣₐᵢₙ ‣ Appendix D Overfitting Scaling Laws Extended ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), and [17](#A4.F17 "Figure 17 ‣ D.1 Justification for 𝐴_gap and 𝐶ₜᵣₐᵢₙ ‣ Appendix D Overfitting Scaling Laws Extended ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), we provide the final scaling law fits, where we directly optimize over our final model from Equation [1](#S5.E1 "Equation 1 ‣ 5.1 Overfitting during Specialized Pretraining ‣ 5 Predicting Overfitting with Scaling Laws ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), jointly learning the A​(δ)A(\delta) and C​(δ)C(\delta) parameters.

![Refer to caption](/html/2603.16177/assets/figures/fig15_musicpile_individual.png)

![Refer to caption](/html/2603.16177/assets/figures/fig15_chempile_individual.png)

![Refer to caption](/html/2603.16177/assets/figures/fig15_proofpile_individual.png)

Figure 14: Gamma and log-linear fits over learned coefficients.

![Refer to caption](/html/2603.16177/assets/figures/fig15_musicpile_joint_prediction.png)


Figure 15: All power law fits over the training loss, train-test gap on MusicPile-300M over the course of SPT. Finally, test loss as the sum of two powers.

![Refer to caption](/html/2603.16177/assets/figures/fig15_chempile_joint_prediction.png)


Figure 16: All power law fits over the training loss, train-test gap on ChemPile-300M over the course of SPT. Finally, test loss as the sum of two powers.

![Refer to caption](/html/2603.16177/assets/figures/fig15_proofpile_joint_prediction.png)


Figure 17: All power law fits over the training loss, train-test gap on ProofPile-300M over the course of SPT. Finally, test loss as the sum of two powers.

### D.2 Modeling Difference in Test Loss Post-Finetuning Across PT Tokens

![Refer to caption](/html/2603.16177/assets/figures/fig10_new_delta_scatter_musicpile.png)


(a) MusicPile

![Refer to caption](/html/2603.16177/assets/figures/fig10_new_delta_scatter_chempile.png)


(b) ChemPile

![Refer to caption](/html/2603.16177/assets/figures/fig10_new_delta_scatter_proofpile.png)


(c) ProofPile

Figure 18: Δ\Delta Test follows a power law relationship with respect to the pretraining steps.

## Appendix E Distribution Similarity Metrics: Detailed Analysis

![Refer to caption](/html/2603.16177/assets/x3.png)


Figure 19: Comparison of Pearson correlations between the Japanese overlap sweep and cross-domain analysis. C2ST AUC flips from r=+0.90r=+0.90 to r=−0.98r=-0.98.

This appendix describes the distributional similarity metrics used in Section [3.1](#S3.SS1 "3.1 Domain Similarity ‣ 3 Factors governing the relative gain of specialized pretraining ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data") and analyzes how they correlate with ℛgain\mathcal{R}\_{\text{gain}} across our two experimental settings.

### E.1 Metrics Description

We evaluate five metrics that capture different aspects of how a domain dataset differs from a reference corpus.

##### Jensen-Shannon Divergence (JSD).

JSD measures divergence between two probability distributions, bounded between 0 (identical) and 1 bit (maximally different). We compute JSD over n-gram distributions at three granularities: unigram (individual tokens, capturing vocabulary differences), bigram (consecutive pairs, capturing local syntax), and trigram (triplets, capturing longer-range patterns). Higher JSD indicates greater distributional difference. For bigram and trigram computation, we use feature hashing with 2202^{20} bins to handle the large vocabulary.

##### MAUVE.

MAUVE [Pillutla et al., [2021](#bib.bib44 "MAUVE: measuring the gap between neural text and human text using divergence frontiers")] compares text distributions using neural embeddings from GPT-2 Large. It computes the area under a divergence frontier curve, producing a score between 0 (completely different) and 1 (identical). We use 5,000 text segments of 256 tokens each per distribution.

##### Classifier Two-Sample Test (C2ST).

C2ST [Lopez-Paz and Oquab, [2016](#bib.bib42 "Revisiting classifier two-sample tests")] trains a binary classifier to distinguish samples from two distributions. We use sentence embeddings from all-MiniLM-L6-v2 [Wang et al., [2020](#bib.bib52 "Minilm: deep self-attention distillation for task-agnostic compression of pre-trained transformers")] and train a logistic regression classifier with 5-fold cross-validation. The AUC score indicates separability: 0.5 means indistinguishable, 1.0 means perfectly separable.

### E.2 Experimental Setup

##### Japanese Overlap Sweep.

For the controlled study, we compare mixed pretraining distributions against a held-out English-Japanese parallel translation corpus. The pretraining mixes contain varying percentages of Japanese monolingual web text (0%, 0.001%, 0.01%, 0.1%, 1%, and 10%), with the remainder being English web text. As Japanese percentage increases, the pretraining distribution becomes more similar to the translation target domain. We sample 1.5M tokens from each distribution and compute all metrics against the translation corpus.

##### Cross-Domain Analysis.

For the cross-domain analysis, we compare each specialized domain (ChemPile, MusicPile, ProofPile) against the Dolma web corpus used for pretraining. Each domain represents a naturally occurring distributional shift from web text: chemistry literature, symbolic music notation in ABC format, and formal mathematical proofs.

### E.3 Results

##### Japanese Overlap Sweep.

Figure [20](#A5.F20 "Figure 20 ‣ Japanese Overlap Sweep. ‣ E.3 Results ‣ Appendix E Distribution Similarity Metrics: Detailed Analysis ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data") shows scatter plots of each metric against ℛgain\mathcal{R}\_{\text{gain}} for the Japanese overlap sweep. All five metrics correlate strongly with ℛgain\mathcal{R}\_{\text{gain}} in the expected direction, with Pearson |r|>0.85|r|>0.85 in every case. The JSD metrics show positive correlations (higher divergence corresponds to higher gain), MAUVE shows negative correlation (lower similarity corresponds to higher gain), and C2ST AUC shows positive correlation (higher separability corresponds to higher gain). These strong correlations validate that the metrics capture meaningful distributional differences relevant to SPT benefit.

![Refer to caption](/html/2603.16177/assets/x4.png)


Figure 20: Japanese overlap sweep: each metric plotted against ℛgain\mathcal{R}\_{\text{gain}}. All metrics show strong correlations (|r|>0.85|r|>0.85) in the expected direction.

##### Cross-Domain Analysis.

Figure [21](#A5.F21 "Figure 21 ‣ Cross-Domain Analysis. ‣ E.3 Results ‣ Appendix E Distribution Similarity Metrics: Detailed Analysis ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data") shows the same analysis for our three benchmark domains. The pattern is strikingly different. JSD metrics show weak positive correlations ranging from r=0.32r=0.32 (unigram) to r=0.57r=0.57 (trigram), with longer n-grams performing better. MAUVE shows weak negative correlation (r=−0.23r=-0.23). Most dramatically, C2ST AUC flips from r=+0.90r=+0.90 on the Japanese sweep to r=−0.98r=-0.98 on cross-domain, meaning it predicts the opposite of what we observe.

![Refer to caption](/html/2603.16177/assets/x5.png)


Figure 21: Cross-domain analysis: each metric plotted against ℛgain\mathcal{R}\_{\text{gain}} for ChemPile, MusicPile, and ProofPile. Correlations are much weaker than the Japanese sweep, and C2ST AUC flips sign.

##### Why Does C2ST Flip?

On the Japanese sweep, C2ST correctly identifies that pretraining mixes with more Japanese text are more similar to the translation target, and this similarity predicts lower ℛgain\mathcal{R}\_{\text{gain}}. But on cross-domain, C2ST suggests that MusicPile (lowest AUC of 0.988) is most similar to Dolma, when in fact MusicPile shows the highest ℛgain\mathcal{R}\_{\text{gain}}. The likely explanation is that C2ST’s sentence embeddings capture surface-level semantic similarity that does not generalize across fundamentally different domain types. Music notation in ABC format may appear “similar” to web text in embedding space because both contain ASCII characters and structured patterns, even though the underlying content is entirely different.

## Appendix F General Pretraining Loss during SPT

We plot the Dolma loss over pretraining scales across all domains and SPT configurations. As we discuss in Section [2.3](#S2.SS3 "2.3 SPT Learns More and Forgets Less ‣ 2 Specialized Pretraining Drives Domain Specific Capabilities ‣ The Finetuner’s FallacyWhen to Pretrain with Your Finetuning Data"), SPT replaces a small fraction of general pretraining tokens with domain-specific data, but this has a marginal impact on the general pretraining loss over Dolma for mixture percentages up to 5%5\%. However, we do observe notable degradation once we push the percentage up to 10%10\%.

![[Uncaptioned image]](/html/2603.16177/assets/figures/appendix_general_loss.png)

[◄](/html/2603.16176)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2603.16177)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2603.16177)
[View original  
on arXiv](https://arxiv.org/abs/2603.16177)[►](/html/2603.16178)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Apr 6 08:14:14 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
