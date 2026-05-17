---
arxiv: '2510.14865'
authors:
- Emmy Liu
- Graham Neubig
- Chenyan Xiong
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Midtraining Bridges Pretraining and Posttraining Distributions
url: https://arxiv.org/abs/2510.14865
year: 2025
---

# Midtraining Bridges Pretraining and Posttraining Distributions

Emmy Liu, Graham Neubig, Chenyan Xiong
  
Language Technologies Institute
  
Carnegie Mellon University
  
Pittsburgh, PA 15213, USA
  
{mengyan3, gneubig, cx}@cs.cmu.edu

###### Abstract

Recently, many language models have been pretrained with a “midtraining” phase, in which higher quality, often instruction-formatted data, is mixed in at the end of pretraining. Despite the popularity of this practice, there is little scientific understanding of this phase of model training or why it is effective. In this work, we conduct the first systematic investigation of midtraining through controlled experiments with language models pretrained from scratch and fine-tuned on supervised finetuning datasets in different domains. We find that when compared after supervised fine-tuning, the effectiveness of midtraining is highest in the math and code domains, where midtraining can best reduce the syntactic gap between pretraining and posttraining data. In these cases, midtraining consistently outperforms continued pretraining in both in-domain validation loss as well as pretraining data forgetting after posttraining. We conduct ablations on the starting time of the midtraining phase and mixture weights of the midtraining data, using code midtraining as a case study, and find that timing has a greater impact than mixture weights, with earlier introduction of specialized data, yielding greater benefits in-domain as well as preserving general language modeling better. These findings establish midtraining as a domain adaptation technique that compared to continued pretraining yields better performance through reduced forgetting.111Data and code are available at <https://github.com/nightingal3/all_in_one_pretraining>.

## 1 Introduction

The success of large language models has mostly been driven by scaling model and data size. Though many interventions seem promising, they may wash out at scale. Therefore, when methodological interventions are simple yet widely adopted across model scales, they merit attention. One such intervention is midtraining: breaking pretraining into two or more stages in which the latter stages incorporate higher-quality data from specialized domains such as mathematics and coding, as well as instruction-formatted data (minicpm; llama3; olmo2). While this approach has shown improved performance after posttraining (minicpm), there is surprisingly little systematic study of when and why midtraining works (wang\_octothinker\_2025).

This raises key questions for practitioners: When is midtraining expected to work better than simply posttraining a model, or doing continued pretraining on a target domain? What types of data are most effective in a midtraining data mixture? Finally, how does midtraining compare to continued pretraining in specialized domains? To address these questions, we conduct the first systematic investigation of midtraining through controlled experiments on models pretrained from scratch with different midtraining settings. We systematically vary the type of midtraining data, the timing of its introduction and mixture weights, followed by supervised fine-tuning on diverse datasets. Our investigation reveals three key findings:

* •

  Midtraining is highly effective at improving downstream loss on a few key domains such as math and code. In particular, it is effective in improving accuracy on domains that are farther from the general pretraining distribution as represented by web text.
* •

  Midtraining reduces catastrophic forgetting compared to both direct fine-tuning of a pre-trained model as well as continued pretraining given the same number of tokens.
* •

  The timing of introducing midtraining data into the training process is important, more so than the mixture weight of the midtraining data source.

Put together, these findings cast a new light on our understanding of midtraining’s role in the LM training process – it can be viewed as effectively “bridging” the distribution gap between pretraining and out-of-domain fine-tuning data, introducing a smooth transition between the two at an appropriate time in the training process.

## 2 Preliminaries

In this section, we define what we refer to as midtraining throughout this paper. While this term has been used colloquially by model developers, it lacks a standard definition, so we establish our working definition for clarity.

### 2.1 Training Sequence Definitions

Modern language model training typically follows a sequential approach where models are trained on different data distributions in a specific order. Rather than attempting to define training phases by their data characteristics or objectives, which often overlap in practice, we adopt a sequence-based framework that defines phases by their order in the training process.

Let θ∈Θ\theta\in\Theta denote model parameters and 𝒳\mathcal{X}, 𝒴\mathcal{Y} be input and output spaces. A training phase is a tuple (D,ℒ,n)(D,\mathcal{L},n) where DD is a distinct data distribution over 𝒳×𝒴\mathcal{X}\times\mathcal{Y}, ℒ:Θ×(𝒳×𝒴)→ℝ\mathcal{L}:\Theta\times(\mathcal{X}\times\mathcal{Y})\rightarrow\mathbb{R} is a loss function, and n∈ℕ+n\in\mathbb{N}^{+} is the number of training steps taken in that phase. A training sequence is an ordered collection S={(Di,ℒi,ni)}i=0NS=\{(D\_{i},\mathcal{L}\_{i},n\_{i})\}\_{i=0}^{N} of training phases, where the parameters θi\theta\_{i} resulting from training up to phase ii serve as the initialization for phase i+1i+1.

Pretraining is the initial training phase (D0,ℒ0,n0)(D\_{0},\mathcal{L}\_{0},n\_{0}) in the sequence. In practice, pretraining often uses large, diverse sources of data, such as from the web. The objective for decoder-based language models, which we examine, is typically next-token prediction, ℒ0​(θ;x)=−∑t=1|x|log⁡pθ​(xt|x<t)\mathcal{L}\_{0}(\theta;x)=-\sum\_{t=1}^{|x|}\log p\_{\theta}(x\_{t}|x\_{<t}). Pretraining typically consumes the majority of steps taken during training, so that ∀i>0,n0>>>ni\forall i>0,n\_{0}>>>n\_{i}.

Midtraining refers to intermediate training phases (Di,ℒi,ni)(D\_{i},\mathcal{L}\_{i},n\_{i}) where 0<i<N0<i<N. Midtraining data is typically more specialized than general pretraining data, often including domain-specific content (code, math) and instruction-formatted data, while maintaining a mixture with general pretraining data. However, like pretraining, midtraining updates all model parameters. Typically, midtraining is a longer phase compared to finetuning, but shorter than the preceding pretraining phase, n0>ni>nNn\_{0}>n\_{i}>n\_{N}. There can potentially be multiple midtraining phases as well in the case of multi-stage pretraining curricula, but we focus on one-stage midtraining in this paper.

Finetuning is the final training phase (DN,ℒN,nN)(D\_{N},\mathcal{L}\_{N},n\_{N}) in the sequence. Finetuning usually uses task-specific datasets and may employ diverse objectives or preference-based methods rather than next-token prediction. Typically, finetuning uses the least number of steps, although this is not necessarily always the case.

Note that our framework reflects current popular practices for language model training, rather than prescribing an optimal training methodology. Additional edge cases include multi-stage finetuning, cyclic training schedules, and mixture schedules where distribution weights vary continuously.

### 2.2 Relationship to Curriculum Learning and Continued Pretraining

#### Curriculum learning

The original definition of curriculum learning focused on gradually increasing the difficulty or diversity of training examples throughout the course of training (elman1993learning; bengio2009curriculum). However, this term has evolved to generally mean any strategic ordering of training data (soviany2022curriculumlearningsurvey). In this general view, it encompasses any training strategy in which the sequence S={(Di,ℒi,ni)}i=0NS=\{(D\_{i},\mathcal{L}\_{i},n\_{i})\}\_{i=0}^{N} is designed according to some principled ordering criterion—whether based on example difficulty, domain progression, task complexity, data quality, or other strategic considerations. Therefore, midtraining can be viewed as a type of curriculum learning that operates at a distributional level rather than over individual examples: instead of ordering specific training instances by difficulty or other criteria, midtraining strategically organizes different data distributions or domain mixtures across discrete training phases. This coarse-grained approach focuses on when to introduce entire types of data during pretraining rather than sequencing of individual examples.

#### Continued pretraining

Continued pretraining typically refers to additional pretraining of an already-pretrained model on domain-specific data, where the data distribution shifts completely to the target domain (dapt; scibert). This approach commits entirely to domain-specific data after a certain point in training, potentially losing general capabilities. In contrast, midtraining maintains mixed distributions throughout the intermediate phase, preserving general pretraining data alongside specialized data. Formally, continued pretraining can be viewed as the limiting case where the midtraining distribution DiD\_{i} consists entirely of domain-specific data (0% original pretraining data, 100% domain data), though our results suggest this pure approach may be suboptimal. Implementation details such as optimizer state handling and learning rate schedule may vary, but most often midtraining continues on the original pretraining schedule, preserving optimizer state, while continued pretraining often operates on a new schedule.

## 3 Experimental Setting

### 3.1 Training Setup

#### Pretraining

We pretrain models from the Pythia family ranging in size from 70M-410M parameters on C4 web data (c4; pythia). In all cases, we train for 128B tokens (approx. 61k steps) with a cosine learning rate schedule with a maximum learning rate of 3e-4 and the AdamW optimizer (adamw). We chose to fix the training budget at a point past Chinchilla-optimality for all models (chinchilla), in order to ensure that models have stabilized by the point at which midtraining data has been introduced, at least for later insertion points of midtraining data. We describe the exact training setup in LABEL:appendix:pretrain\_settings.

#### Midtraining

We use five midtraining mixtures spanning popular domains: code (Starcoder), math, instructions (FLAN), general knowledge/QA, and high-quality web data (DCLM). [Table 1](#S3.T1 "Table 1 ‣ Midtraining ‣ 3.1 Training Setup ‣ 3 Experimental Setting ‣ Midtraining Bridges Pretraining and Posttraining Distributions") details each mixture’s composition and sources. All mixtures are introduced at varying start points (Starcoder: 6k steps, Math: 20k steps, others: 40k steps) based on data availability to prevent repetition. We compare against a control condition continuing C4 pretraining for the same number of tokens, keeping all other training details identical.

Midtrain mix
Num. Tokens (B)
Sources

Starcoder
196
(Starcoder)

Math
12
(mammoth; toshniwal2024openmath)

FLAN
3.5
(flan)

KnowledgeQA
9.6
(hu2024minicpm)

DCLM
51
(dclm)

Table 1: Midtraining mixes used in our experiments and dataset(s) from which they were derived.

##### Starcoder (code)

Our code mixture is a subset of the Starcoder pretraining dataset (Starcoder), which contains code in many languages. Note that we use code from all languages, rather than Python.

##### Math

The math mixture combines mathematical reasoning problems from the MAmmoTH (mammoth) and OpenMathInstruct (toshniwal2024openmath) datasets, featuring step-by-step explanations.

##### FLAN (instructions)

Our instruction-formatted data comes from a processed version of the FLAN collection, which includes diverse task instructions and responses across natural language tasks (flan).

##### KnowledgeQA (general knowledge and QA)

The KnowledgeQA mixture is taken from minicpm’s midtraining mix, and focuses on general knowledge and dialogue. However, to distinguish the midtraining mixes further, the StackOverflow portion of this dataset is removed.

##### DCLM (high-quality web)

Our high-quality web data is a subset of the DCLM pretraining dataset, representing web content with improved quality filtering compared to C4 (dclm).

#### Downstream Evaluation

We fine-tune models on the datasets GSM8k (gsm8k), SciQ (sciq), CodeSearchNet-Python (codesearchnet), and LIMA (lima) – chosen to span the domains covered by our midtraining mixtures. This allows us to test cases where the midtraining mixture is aligned or misaligned with the SFT dataset. We used standard language model supervised fine-tuning for all datasets. For information on the posttraining setup, see LABEL:appendix:posttrain\_settings.

#### Catastrophic Forgetting Evaluation

A key concern with supervised fine-tuning is whether introducing specialized data causes models to forget general capabilities acquired during pretraining. We measure catastrophic forgetting by evaluating cross-entropy loss on the original pretraining distribution by measuring loss on held-out C4 data. This approach follows established practices for measuring forgetting in language models (luo2024empirical; kemker2018measuring; li2024revisiting).

## 4 Which downstream tasks benefit most from midtraining?

To identify which tasks benefit most from midtraining, we evaluate all midtraining mix and SFT dataset combinations. For each combination, we measure validation loss on the target domain (adaptation effectiveness) and C4 validation loss after fine-tuning (forgetting). We average results over 5 seeds after hyperparameter search for each checkpoint.

Midtraining benefits are highly domain-specific: specialization on code yields the largest gains on code tasks, while math-focused midtraining helps mathematical-reasoning tasks. Mismatched midtraining provides minimal benefit, and general instruction mixes (e.g., FLAN) produce little improvement. Full per-dataset results and numerical comparisons are reported in Table [4](#S4 "4 Which downstream tasks benefit most from midtraining? ‣ Midtraining Bridges Pretraining and Posttraining Distributions") and Appendix LABEL:appendix:sft\_c4\_losses\_full.

Table 2: SFT and C4 validation losses for the 410M model across downstream datasets and midtraining mixtures (5 seeds per SFT dataset). Bold indicates best within each dataset. Percentages denote the specialized data proportion mixed with C4 during midtraining.

|  |  |  |  |
| --- | --- | --- | --- |
| Dataset | Midtrain Mix | Validation Loss | |
|  |  | SFT | C4 |
| \rowcolorblack!4     Pycode | | | |
|  | C4 | 2.2642.264 | 5.0585.058 |
|  | Starcoder (20%) | 2.091 | 4.611 |
|  | Math (12%) | 2.2742.274 | 5.0965.096 |
|  | FLAN (5%) | 2.2522.252 | 5.0475.047 |
|  | KnowledgeQA (20%) | 2.2642.264 | 5.0945.094 |
|  | DCLM (20%) | 2.2622.262 | 5.1575.157 |
| \rowcolorblack!4     GSM8K | | | |
|  | C4 | 0.9560.956 | 4.9374.937 |
|  | Starcoder (20%) | 0.9440.944 | 4.9514.951 |
|  | Math (12%) | 0.918 | 5.038 |
|  | FLAN (5%) | 0.9580.958 | 4.9704.970 |
|  | KnowledgeQA (20%) | 0.9610.961 | 4.9664.966 |
|  | DCLM (20%) | 0.9490.949 | 4.9664.966 |
| \rowcolorblack!4     LIMA | | | |
|  | C4 | 3.4903.490 | 3.1663.166 |
|  | Starcoder (20%) | 3.422 | 3.155 |
|  | Math (12%) | 3.5073.507 | 3.1683.168 |
|  | FLAN (5%) | 3.5133.513 | 3.1643.164 |
|  | KnowledgeQA (20%) | 3.5013.501 | 3.1633.163 |
|  | DCLM (20%) | 3.4963.496 | 3.1583.158 |
| \rowcolorblack!4     SciQ | | | |
|  | C4 | 3.2713.271 | 6.8736.873 |
|  | Starcoder (20%) | 3.2613.261 | 6.8316.831 |
|  | Math (12%) | 3.257 | 6.8946.894 |
|  | FLAN (5%) | 3.2773.277 | 6.771 |
|  | KnowledgeQA (20%) | 3.2733.273 | 6.8146.814 |
|  | DCLM (20%) | 3.2703.270 | 6.8746.874 |

## 5 What data is most effective for midtraining?

Having established that midtraining effects are domain-specific, we now ask: what determines the strength of these domain-specific effects? Across midtraining-target pairs, we see improvements ranging from negligible (e.g. FLAN →\rightarrow coding) to strong (e.g. Starcoder →\rightarrow coding).

Figure 1: Example similarity matrix between pre/midtrain and posttraining datasets.
For the complete matrix, see LABEL:appendix:dataset\_similarity.

Although the pairs that work the best are intuitive, we ask whether we can quantify this intuition through measurable proximity between data distributions.

To test whether optimal midtraining data “bridges” the gap between pretraining
and target distributions, we simply use token-distributional similarity between
datasets, which we discuss in more detail in LABEL:appendix:dataset\_similarity.
[Figure 1](#S5.F1 "Figure 1 ‣ 5 What data is most effective for midtraining? ‣ 4 Which downstream tasks benefit most from midtraining? ‣ Midtraining Bridges Pretraining and Posttraining Distributions") shows a reduced subset as an example: we can see
that C4 and Pycode are fairly dissimilar (0.54 similarity), but the blend containing
29% Starcoder data brings this closer to Pycode (0.7 similarity).

Finding 1. Midtraining benefits are highly domain-specific.
Code-focused midtraining (Starcoder) yields large gains on coding tasks,
and math-focused midtraining improves mathematical reasoning.
Mismatched domains provide little benefit,
and broad instruction data (FLAN) also shows minimal effect.

### 5.1 Proximity and Bridging Effects

To understand why certain midtraining mixtures are effective, we test the hypothesis that optimal midtraining data “bridges” the gap between pretraining (C4) and target SFT datasets. Using the simple token similarity metric, we measure the “proximity advantage” of each midtraining mixture—how much closer it brings the model to the target posttraining distribution compared to continuing with C4 alone. Results are shown in [Figure 2](#S5.F2 "Figure 2 ‣ 5.1 Proximity and Bridging Effects ‣ 5 What data is most effective for midtraining? ‣ 4 Which downstream tasks benefit most from midtraining? ‣ Midtraining Bridges Pretraining and Posttraining Distributions").

We find a clear relationship between proximity advantage and downstream performance improvements across model sizes. The correlations are particularly strong for smaller models (r=0.869r=0.869, p<0.001p<0.001 for 70m), suggesting that effective midtraining data serves as a distributional stepping stone from general pretraining to specialized target domains. This bridging effect appears to be most beneficial when the gap between pretraining and target distributions is large, consistent with our hypothesis that midtraining helps models adapt gradually rather than requiring abrupt distributional shifts during fine-tuning.

!(/html/2510.14865/assets/x2.png)

Figure 2: Relationship between proximity advantage and midtraining performance improvements for pairs of midtraining and SFT datasets. Each data point represents a (midtrain, SFT) pair, where the color indicates the SFT dataset and shape represents midtrain dataset. Proximity advantage (dist(C4, SFT) - dist(midtrain, SFT)) indicates how much closer midtraining data brings the model to the target SFT dataset compared to the base pretraining data. Proximity advantage pairs near zero are greyed out for clarity but included in calculations. Relative improvement is measured against the base model pretrained on C4.

Finding 2. Effective midtraining data acts as a distributional bridge
between pretraining and target datasets when considering token patterns.

### 5.2 Midtraining vs. Continued Pretraining

Our results so far suggest that effective midtraining data serves as a bridge between general pretraining and specialized posttraining data. However, a question that follows is why midtraining is necessary: continued pretraining on domain-specific data also aims to adapt the model toward a target domain. Why not simply pretrain normally and then switch to domain-specific data entirely?

To examine this, we compare the effect of midtraining with continued pretraining in which the mixture weight switches to 100% specialized data. For code, we compare the default Starcoder midtraining mix (20% mixture weight, starting from 12.6B tokens) with 100% Starcoder data starting from 83B tokens. For math, we compare the math midtraining mix with 100% math data starting from 105B tokens.222The different starting points are due to data availability, to ensure the midtraining mix does not repeat.

Results in [subsection 5.2](#S5.SS2 "5.2 Midtraining vs. Continued Pretraining ‣ 5 What data is most effective for midtraining? ‣ 4 Which downstream tasks benefit most from midtraining? ‣ Midtraining Bridges Pretraining and Posttraining Distributions") show that midtraining consistently outperforms continued pretraining across both domains and model sizes for both in-domain performance and C4 retention after fine-tuning. As this pattern holds for both code and math domains, this suggests that maintaining some general pretraining data is useful during domain adaptation, even for models specialized for a specific domain. This supports our intuition gained from prior sections that domain adaptation benefits from gradual distributional shifts at the token level rather than abrupt changes.

Table 3: SFT and C4 validation losses for 70M and 160M models comparing default midtraining mixes to continued pretraining on only the midtraining data (100%), averaged across 5 seeds. Bold indicates best performance within each dataset/model combination.
