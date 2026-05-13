---
arxiv: '2504.03635'
authors:
- Xinyi Wang
- Shawn Tan
- Shenbo Xu
- Mingyu Jin
- William Yang Wang
- Rameswar Panda
- Yikang Shen
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Do Larger Language Models Generalize Better? A Scaling Law for Implicit Reasoning
  at Pretraining Time
url: https://arxiv.org/abs/2504.03635
year: 2025
---

[2504.03635] Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning














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



# Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning

Xinyi Wang
  
UC Santa Barbara
  
xinyi\_wang@ucsb.edu
  
&Shawn Tan
  
MIT-IBM Watson AI Lab
  
shawntan@ibm.com
  
&Mingyu Jin
  
Rutgers University
  
mingyu.jin@rutgers.edu
  
&William Yang Wang
  
UC Santa Barbara
  
william@cs.ucsb.edu
  
&Rameswar Panda
  
MIT-IBM Watson AI Lab
  
rpanda@ibm.com
  
&Yikang Shen
  
MIT-IBM Watson AI Lab
  
yikang.shn@gmail.com
Most of this work is done during an internship at the MIT-IBM Watson AI Lab.

###### Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks requiring complex reasoning. However, the effects of scaling on their reasoning abilities remain insufficiently understood. In this paper, we introduce a synthetic multihop reasoning environment designed to closely replicate the structure and distribution of real-world large-scale knowledge graphs. Our reasoning task involves completing missing edges in the graph, which requires advanced multi-hop reasoning and mimics real-world reasoning scenarios. To evaluate this, we pretrain language models (LMs) from scratch solely on triples from the incomplete graph and assess their ability to infer the missing edges. Interestingly, we observe that overparameterization can impair reasoning performance due to excessive memorization. We investigate different factors that affect this U-shaped loss curve, including graph structure, model size, and training steps. To predict the optimal model size for a specific knowledge graph, we find an empirical scaling that linearly maps the knowledge graph search entropy to the optimal model size. This work provides new insights into the relationship between scaling and reasoning in LLMs, shedding light on possible ways to optimize their performance for reasoning tasks.

## 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks. Recently, the reasoning capacity of LLMs has drawn a lot of attention as it is highly correlated with LLMs’ performance on many complex real-world tasks (Wei et al., [2022a](#bib.bib28); Guo et al., [2025](#bib.bib8)). While the reasoning capability is usually enhanced during the post-training stage, it is reasonable to assume that LLMs have already acquired the capability during the pretraining stage, as the post-training is on a significantly smaller scale than pretraining. Some recent work has investigated the possible mechanism of LLMs learning to reason through next-token prediction pretraining (Zhu et al., [2024](#bib.bib37); Wang et al., [2024a](#bib.bib25); [b](#bib.bib27)). However, the impact of pretraining scaling on LLMs’ reasoning ability remains insufficiently understood.

The general scaling behavior of language models has been investigated, including the well-known exponential scaling laws for testing loss and compute proposed by Kaplan et al. ([2020](#bib.bib15)) and the training compute-optimal scaling studied by Hoffmann et al. ([2022a](#bib.bib11)). Recent work has also examined the scaling of specific capabilities like machine translation (Ghorbani et al., [2022](#bib.bib7)) and knowledge capacity/memorization (Allen-Zhu & Li, [2025](#bib.bib1); Lu et al., [2024](#bib.bib17)). According to these existing scaling laws, it is in general believed that larger models imply better testing loss or task performance when trained on more data.

In this paper, however, we find that the reasoning ability of language models under pretraining scaling can behave differently from normal power-law scaling, in a simplified pretraining environment. More specifically, when given enough compute, we find that the testing loss scaling curve is U-shape and there exists an optimal model size that produces the best reasoning performance/testing loss. This implies that overparameterization might hurt reasoning capability during pre-training. We first observe this phenomenon with real-world knowledge graph data, and then systematically study it through synthetically generated data.

We choose to mimic the real-world knowledge structure and distribution with synthetic knowledge graphs (KGs). We define reasoning over world knowledge as completing missing edges in an incomplete knowledge graph, which requires multiple jumps on the knowledge graph according to some pre-defined rules that are latently encoded into the graph generation process. To analyze this, we pretrain LMs from scratch using only triples from the incomplete graph and evaluate their ability to infer missing connections.

We investigate important factors that affect the U-shape scaling of reasoning loss versus language model size. Our important findings can be summarized as follows:

* •

  The minimum reasoning loss/maximum reasoning accuracy that a language model can reach is capped by the training data, regardless of the training steps and model size.
* •

  The optimal model size for a training corpus is largely fixed regardless of the training steps when the number of training steps is large enough.
* •

  When the underlying knowledge graph is fixed, training on more data sampled from the graph increases the optimal model size and reasoning performance.
* •

  More complex knowledge graph implies a larger optimal model size.

As we observed that the optimal model size is likely solely determined by the training knowledge graph, we then aim to find an empirical scaling law that can predict the optimal model size with knowledge graph statistics.
We then discover a linear relationship between the optimal model size and a newly proposed graph search entropy, which measures the entropy of performing random searches on a knowledge graph.
Roughly, 124 additional parameters in the optimal model size are required per 1-bit entropy increase in searching a knowledge graph.

Our work contributes to the broader understanding of LLM reasoning by shedding light on the intricate relationship between scaling and reasoning capability. Our proposed empirical reasoning scaling law provides possible practical insights for optimizing LLMs’ reasoning ability at pretraining time.

## 2 Preliminaries

While the real-world LLMs are pretrained on large scale text corpus, this corpus can be viewed as encoding a wide range of world knowledge. The power of LLMs lies in the fact that they can not only memorize the world knowledge and extract the knowledge when queried, but also reason over the world knowledge and draw novel conclusions. In this paper, we propose to construct a simplified pretraining corpus directly from a knowledge graph.
A knowledge graph is comprised of a set of triples, and we use each knowledge triple as a training example. We test the reasoning capability of a language model trained on such a corpus by testing its accuracy in completing triples that have never been seen in the knowledge graph but can be deduced through latent rules encoded in the graph structure. For example, if we know A is B’s father, and B is C’s father, then we can deduce that A is C’s grandfather.

Formally, a knowledge graph GG consists of |G|=N|G|=N triples (eh,r,et)(e^{h},r,e^{t}), where eh∈ℰe^{h}\in\mathcal{E} is the head entity, et∈ℰe^{t}\in\mathcal{E} is the tail entity, and r∈ℛr\in\mathcal{R} is a relation. A simple example of knowledge triple is (DC, is the capital of, USA).
These knowledge triples naturally form a graph, with nodes as the entities and each edge labeled with a relation type. We denote the total number of entities or nodes by |ℰ|=Ne|\mathcal{E}|=N\_{e} and the total number of edge or relation types by |ℛ|=Nr|\mathcal{R}|=N\_{r}. Then a corpus constructed from this knowledge graph would consist of NN data points. The objective of a language model with parameter θ\theta trained on this corpus is then:

|  |  |  |
| --- | --- | --- |
|  | L​(θ)=arg⁡minθ⁡1N​∑i=1N−log⁡Pθ​(eih,ri,eit).\displaystyle L(\theta)=\arg\min\_{\theta}\frac{1}{N}\sum\_{i=1}^{N}-\log P\_{\theta}(e^{h}\_{i},r\_{i},e^{t}\_{i}). |  |

To eliminate confounding variables and information contained in the lexical form of the entity and relation names, we label each entity and relation with a random ID and tokenize the IDs by characters. We use the LlaMA (Touvron et al., [2023](#bib.bib24)) model architecture to implement LMs of different sizes by adjusting the hidden dimensions and the number of layers. The specific parameter scheme can be found in the [Appendix A](#A1 "Appendix A Experiment Details ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning").

To evaluate the language model’s capability of reasoning over the knowledge graph, we test the language models on a held-out set of triples that are not seen in the training time. Note that all entity and relation types should have been seen during training time and the language model is only tasked to connect missing edges. To eliminate the need to generate the correct form of relation and entity IDs, and to handle the case where multiple correct answers exist, we design the testing set to be 10-option multiple-choice questions: the language model is tasked to choose the correct tail entity given the head entity and the relation. We ensure that there is only one correct answer among the given 10 options. Suppose there are MM questions in the testing set.111We fix M=1000M=1000 for all of our experiments. For a ground truth triple (eh,r,et)(e^{h},r,e^{t}), we design 9 distracting options e(1),e(2),…,e(9)e^{(1)},e^{(2)},...,e^{(9)}. Then we use the test accuracy Acc​(θ,G)\text{Acc}(\theta,G) and testing loss ℓ​(θ,G)\ell(\theta,G) to evaluate the reasoning capability of a language model θ\theta over the knowledge graph GG:

|  |  |  |  |
| --- | --- | --- | --- |
|  | e^i\displaystyle\hat{e}\_{i} | =arg⁡maxe∈{eit,ei(1),ei(2),…,ei(9)}⁡Pθ​(e|eih,ri),\displaystyle=\arg\max\_{e\in\{e^{t}\_{i},e^{(1)}\_{i},e^{(2)}\_{i},...,e^{(9)}\_{i}\}}P\_{\theta}(e|e^{h}\_{i},r\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Acc​(θ,G)=∑i=1M\displaystyle\text{Acc}(\theta,G)=\sum\_{i=1}^{M} | 𝟙​[e^i=eit]/M,ℓ​(θ,G)=∑i=1M−log⁡Pθ​(eit|eih,ri)/M.\displaystyle\mathbbm{1}[\hat{e}\_{i}=e^{t}\_{i}]/M,\qquad\ell(\theta,G)=\sum\_{i=1}^{M}-\log P\_{\theta}(e^{t}\_{i}|e^{h}\_{i},r\_{i})/M. |  |

## 3 Real-world Experiments

![Refer to caption](/html/2504.03635/assets/img/FB15K.png)


Figure 1: The multiple-choice accuracy/loss on unseen triples of different-sized language models trained on a real-word knowledge graph FB15K-237. The left panel (trained with 10k steps) shows that the testing accuracy decreases after a certain model size. The middle panel shows U-shape loss curves of language models trained with different number steps. The right panel shows Note that the model size on x-axis is in log scale.

In our initial sets of experiments, we investigate the reasoning scaling effect using a real-world knowledge graph, FB15K-237 (Toutanova & Chen, [2015](#bib.bib23)). FB15K-237 is sampled from FB15K (Bordes et al., [2013](#bib.bib4)), which is a dataset adapted from the Freebase knowledge base (Bollacker et al., [2007](#bib.bib3)), a web-scale knowledge base released by Google. FB15K-237 contains Ne=14,505N\_{e}=14,505 entities, Nr=237N\_{r}=237 relations, and N=310,116N=310,116 knowledge triples.

In [Figure 1](#S3.F1 "In 3 Real-world Experiments ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning"), we show different-sized language models trained on FB15K-237 with different numbers of steps. When trained with the same number of steps, we observe a slight reasoning performance drop when using larger models. So we then look at the testing loss on these datasets and observe a U-shape trend with respect to the model size. This observation contradicts the previous belief that larger models yield a smaller testing loss. The training loss decreases monotonically with respect to model size.

This implies that a language model can overfit to the training data when it is overparameterized for the underlying reasoning structure. Such deviation from traditional scaling law has also been reported in broken neural scaling law (Caballero et al., [2023](#bib.bib5)) which proposed a double-descent-like (Nakkiran et al., [2020](#bib.bib18)) function form instead of a monotonic power-law form.
There have also been observations of tasks with inverse scaling (Wei et al., [2023](#bib.bib30)) for large language models.

In this paper, we mainly focus on the scaling of model size. Instead of only scaling the size of the training data, we explore different possible ways of generating the knowledge graph and studying the effect of overall graph complexity on the model reasoning performance. In the following sections, we will mostly focus on understanding the ”turning point” of the reasoning loss. More specifically, we want to understand what is the optimal model size that can obtain the smallest possible reasoning testing loss.

As shown in [Figure 1](#S3.F1 "In 3 Real-world Experiments ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning") and in [Section 5](#S5 "5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning"), we find this optimal model size is largely stable when training the model for enough steps.
Note that, at training time, we repeat the training triples for many epochs (e.g. 30 times for FB15K-237) to find the optimal model size.
This graph epoch is different from the real-world cases where we repeat the whole pretraining corpus for certain epochs.
Because we can view each triple in the graph as a piece of factual knowledge (e.g. Barack Obama’s wife is Michelle Obama), this knowledge is usually repeated many times in a pretraining text corpus, in many different forms.
Therefore, although our models have seen the same triple many times during training, the same piece of factual knowledge could also have been repeated several times in one pass of a real-world pretraining corpus.

## 4 Synthetic Data Construction

![Refer to caption](/html/2504.03635/assets/img/node_type.png)


Figure 2: Nine possible node types generated by two logical rules. Each entity position in a rule would create a new entity type. Each relation shared between two rules would also create two new entity types.

To investigate how the underlying knowledge structure influences language models’ reasoning performance, we propose an algorithm to generate synthetic knowledge graphs that mimic real-world knowledge graphs. More specifically, we assume that the knowledge graph generation process is governed by a set of logical rules.

For example, a rule for inferring the locatedIn relation can be (e1e\_{1}, locatedIn, e2e\_{2}) ←\leftarrow (e1e\_{1}, neighborOf, e3e\_{3}) ∧\wedge (e3e\_{3}, locatedIn, e2e\_{2}).
Formally, for a target relation rr, we consider logic rules with conjunctive form. For ∀{ei}i=0n⊂ℰ\forall\{e\_{i}\}\_{i=0}^{n}\subset\mathcal{E},

|  |  |  |
| --- | --- | --- |
|  | (e0,r,en)←(e0,r1,e1)∧…∧(en−1,rn,en),\displaystyle(e\_{0},r,e\_{n})\leftarrow(e\_{0},r\_{1},e\_{1})\wedge...\wedge(e\_{n-1},r\_{n},e\_{n}), |  |

where (ei−1,ri,ei)∈𝒢(e\_{i-1},r\_{i},e\_{i})\in\mathcal{G}. We abbreviate such rule by h​(r)=[r1,r2,…,rn]h(r)=[r\_{1},r\_{2},...,r\_{n}]. We randomly generate a set of logical rules ℋ\mathcal{H} and ensuring there is no cycles in the set.
To grow a graph that follows these rules, we enforce sparsity of the possible relation types connecting to and branching out each entity. More specifically, we define node types based on the possible relation types connecting to and branching out each entity, based on the generated rules, as illustrated in [Figure 2](#S4.F2 "In 4 Synthetic Data Construction ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning").
Such sparsity is also observed in real-world knowledge graphs.

Our random graph generation process is inspired by the preferential attachment process (Barabási & Albert, [1999](#bib.bib2)), which is used for generating scale-free networks with a power-law distribution for the degrees of the nodes.
Intuitively, preferential attachment implies a “the rich get richer” approach to edge placement in the graph.
Each time a new node is added to the graph, there is a ‘preference’ to connect to the nodes that are already highly connected, with a probability proportional to the target node’s degree.
Since we have observed the scale-free property in real-world knowledge graphs and the internet is known to be a scale-free network, we adopt a preferential attachment based graph generation process.
To accommodate different relation types assigned to each edge, we maintain a degree distribution for each relationship and add new edges according to preferential attachment.

The code for our random graph generation algorithm is shown in the [Appendix B](#A2 "Appendix B Synthetic Knowledge Graph Generation Code ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning"). In summary, we first randomly generate a set of rules ℋ\mathcal{H}, with the number of rules |ℋ|=Nh|\mathcal{H}|=N\_{h} and the range of rule length [Lm​i​n,Lm​a​x][L\_{min},L\_{max}] as hyperparameters. Then we generate all possible node types as illustrated in [Figure 2](#S4.F2 "In 4 Synthetic Data Construction ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning"), with the maximum number of relations per node MrM\_{r} as a hyperparameter.
We generate a seed graph by instantiating each rule with a set of new entities.
To this, we incrementally add one new entity until the number of entities reaches NrN\_{r}, by first randomly assigning a node type to it, and then randomly sampling the mm relation types from the set of relations defined by the node type.
We choose the target of these mm new edges by preferential attachment. After adding every KK entities, we search through the current graph to add any edges that can be inferred through the logic rules defined in ℋ\mathcal{H}.
We call the triples that can be deduced through a logic rule by deductible triples, otherwise atomic triples.

Finally, we limit the number of training triples to NN and ensure that the the ratio between the number of deductible triples and atomic triples to γ\gamma by subsampling the generated graph.
We also further ensure that the triples in the held-out test set are all deductible through the training triple.
In this way, we can generate synthetic knowledge graphs with specific sizes and complexity.

## 5 Scaling Laws

In this section, we investigate the scaling law of language models trained on different synthetic knowledge graphs. We conduct controlled experiments to show the effect of individual components of the data generation process. We also propose an information-theoretical way to measure the overall reasoning complexity of a knowledge graph, which we call the graph search entropy, and relate this linearly with the optimal model size. i.e. the model size that obtains the lowest possible testing loss.

### 5.1 Graph Generation Ablation

![Refer to caption](/html/2504.03635/assets/img/synthetic_ablation.png)


Figure 3: We show the effect of different hyperparameters of the synthetic knowledge graph generation process. In each experiment, we keep all other parameters the same and only change one hyperparameter. We show the effect with both the testing accuracy (left) and the testing loss (right) as the y-axis, with different model sizes as the x-axis in log scale.

We study the effects of the following four hyperparameters of graph data generation: the number of triples NN, the number of entities NeN\_{e}, the number of relations NrN\_{r}, and the number of rules NhN\_{h}. We fix all training hyperparameters as specified in the [Appendix A](#A1 "Appendix A Experiment Details ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning") but study the effect of training steps, as according to our preliminary experiments it has the largest effect on the optimal model size. The detailed data generation configuration for each set of experiments can also be found in the [Appendix A](#A1 "Appendix A Experiment Details ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning").

Stable optimal model size with respect to training steps. In [Figure 3](#S5.F3 "In 5.1 Graph Generation Ablation ‣ 5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning") (a), we show the effect of training language models on the same knowledge graph with different numbers of training steps. As mention in the last part of [Section 2](#S2 "2 Preliminaries ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning"), the optimal model size becomes smaller when the number of training steps increases, and then becomes stable after 4k steps. Another observation is regardless of the number of training steps, the maximum accuracy or minimum loss is stable. While we have ensured that all testing triples can be deduced through the training triples, there seems to be a performance cap determined solely by the knowledge graph data, which is unaffected by model size. In all following experiments, we train all models for 10k steps.

More triples implies a larger optimal model size. In [Figure 3](#S5.F3 "In 5.1 Graph Generation Ablation ‣ 5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning") (b), we show the effect of the number of unique triples NN sampled after the same knowledge graph generation process. This setting is arguably the most similar to the real-world pretraining of language models: the underlying world knowledge graph of all the pretraining corpora is largely stable, and training data are realizations of the underlying knowledge graph and so the sizes of different corpora are simply a result of subsampling/upsampling the knowledge in the existing graph.
We can see that a larger number of training triples results in a larger optimal model size and a better reasoning performance.
This observation aligns with the classic scaling laws. However, there exists an optimal model size for the full knowledge graph: after sampling beyond the size of the full knowledge graph, you can only sample previously seen knowledge. In this case, the optimal model size would be stable no matter the training data size.

Number of rules does not impact optimal model size. In [Figure 3](#S5.F3 "In 5.1 Graph Generation Ablation ‣ 5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning") (c), we show the effect of generating knowledge graphs of the same size with different numbers of rules NhN\_{h}.
More rules mean that the testing triples need to be solved in more ways. The number of rules does not have a significant effect on the optimal model size, but affects the reasoning performance.
There appears to be an optimal number of rules (20) that results in the best performance.
This is because more rules increases the complexity of solving the test set while fewer rules increases the ambiguity in the training set. i.e. a relation may be be deduced through correlations outside of the predefined rules.
The reason why the number of rules does not affect the optimal model size is likely because it does not significantly impact the graph search entropy. This will be discussed in detail in [Section 5.2](#S5.SS2 "5.2 Optimal Model Size v.s. Graph Search Entropy ‣ 5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning").

More relations imply a larger optimal model size.
In [Figure 3](#S5.F3 "In 5.1 Graph Generation Ablation ‣ 5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning") (d), we show the effect of generating knowledge graphs of the same size and the same number of rules with different numbers of relations NrN\_{r}.
While the rules used for deducing the testing set remain the same for all experiments, there are additional relations that may not be used by any of the rules.
We construct knowledge graphs with an excessive number of relations by adding additional relation patterns.
In general, more relations improves the best reasoning performance while increasing the optimal model size.
More relations increases the complexity of the knowledge graph, and thus increases the optimal model size.
On the other hand, as discussed in the previous experiment, a small number of rules along with a small number of relations increases the ambiguity in the training set.
By adding dummy relations that are not used for reasoning, the language model can better distinguish between the logic rules and spurious correlations between relations.
Thus the reasoning performance improves with more relations.

The optimal model size increases with the deductible ratio when the ratio is small. In [Figure 3](#S5.F3 "In 5.1 Graph Generation Ablation ‣ 5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning") (e), we show the effect of generating knowledge graphs with different ratios between deductible triples and atomic triples, γ\gamma, while keeping the number of entities and the number of triples unchanged.
A larger ratio implies that the language model can see more rule patterns at training time, thus improving the reasoning performance. The increase in performance and optimal model size stops after a ratio threshold.

More entities imply a larger optimal model size. In [Figure 3](#S5.F3 "In 5.1 Graph Generation Ablation ‣ 5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning") (f), we show the effect of generating knowledge graphs with different numbers of nodes/entities NeN\_{e}. In this experiment, we also scale the number of triples to keep all other hyperparameters unchanged.
Increasing the number of entities increases the optimal model size while also increasing the testing loss.
More entities imply a larger graph which increases the graph complexity, thus increasing the optimal model size.
As in this experiment, we use a small number of rule (Nh=5N\_{h}=5) and relations (Nr=10N\_{r}=10), an excessive number of entities and triples will create more ambiguity thus hurting the reasoning performance.

### 5.2 Optimal Model Size v.s. Graph Search Entropy

![Refer to caption](/html/2504.03635/assets/img/entropy_vs_size.png)


Figure 4: The optimal model size with the lowest possible testing loss v.s. the graph search entropy. The red line is the linear regression line using data from the synthetic experiments (blue squares), with a 95% confidence interval. We also plot the graph search entropy and optimal model size from the real-world FB15K-237 experiment (green dot) to verify the accuracy of the obtained linear scaling law.

From our previous ablation studies, we hypothesize that the optimal model size is positively related to the overall complexity of the knowledge graph.
Thus, we propose that we measure the complexity of a knowledge graph by quantifying the amount of information that can be obtained from the graph by exploring the graph through a random search.
From our task definition, to reason over the knowledge graph, the language model needs to (a) identify the set of logic rules by observing repetitive patterns; (b) traverse the graph using one or more specific logic rules to locate the tail entity. So we define the graph search entropy as the maximum amount of information that can be obtained when randomly traversing the graph.

To simplify the problem, we first focus on the average amount of information we can observe at one node of the graph.
If we consider a random walk over the knowledge graph, then we refer to the entropy produced by each step/node on the walk trace for an infinitely long random walk as the *entropy rate* of this random walk.
For a graph GG, the maximum entropy rate is equal to the log of the largest eigenvalue of the adjacency matrix AA.
Note that only consider the entropy rate with respect to the entity, without considering the entropy rate with respect to the relation.
We can compute the relation entropy rate with the stationary distribution and transition matrix induced by the maximal entropy rate random walk.
If we denote the dominating eigenvalue by λ∈ℝ\lambda\in\mathbb{R} and the corresponding eigenvector by ψ∈ℝNe\psi\in\mathbb{R}^{N\_{e}}, then the stationary distribution ρ∈ℝNe\rho\in\mathbb{R}^{N\_{e}} can be written as:

|  |  |  |
| --- | --- | --- |
|  | ρi=ψi/‖ψ‖22.\displaystyle\rho\_{i}=\psi\_{i}/||\psi||\_{2}^{2}. |  |

The transition matrix S∈ℝNe×NeS\in\mathbb{R}^{N\_{e}\times N\_{e}} of the maximal entropy random walk can be written as:

|  |  |  |
| --- | --- | --- |
|  | Si​j=(Ai​j/λ)​(ψj/ψi).\displaystyle S\_{ij}=(A\_{ij}/\lambda)(\psi\_{j}/\psi\_{i}). |  |

We can then transform the entity-to-entity transition matrix S∈ℝNe×NeS\in\mathbb{R}^{N\_{e}\times N\_{e}} into an entity-to-relation transition matrix Sr∈ℝNe×NrS^{r}\in\mathbb{R}^{N\_{e}\times N\_{r}} by merging the entries with the same relation together:

|  |  |  |
| --- | --- | --- |
|  | Si​jr=∑k=1Ne𝟙​[(i,j,k)∈G]​Si​k.\displaystyle S^{r}\_{ij}=\sum\_{k=1}^{N\_{e}}\mathbb{1}[(i,j,k)\in G]S\_{ik}. |  |

Finally, the relation entropy rate Hr​(G)H^{r}(G) can be written as:

|  |  |  |
| --- | --- | --- |
|  | Hr​(G)=−∑i=1Neρi​∑j=1NrSi​jr​log⁡(Si​jr).\displaystyle H^{r}(G)=-\sum\_{i=1}^{N\_{e}}\rho\_{i}\sum\_{j=1}^{N\_{r}}S^{r}\_{ij}\log(S^{r}\_{ij}). |  |

The overall graph search entropy H​(G)H(G) can then be written as the sum of the entity entropy rate and the relation entropy rate multiplied by the number of nodes:

|  |  |  |
| --- | --- | --- |
|  | H​(G)=Ne​(log⁡(λ)+Hr​(G)).\displaystyle H(G)=N\_{e}(\log(\lambda)+H^{r}(G)). |  |

We empirically investigate the relation between the optimal model and the graph search entropy by plotting them against each other in [Figure 4](#S5.F4 "In 5.2 Optimal Model Size v.s. Graph Search Entropy ‣ 5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning"), and perform linear regression. The optimal model sizes are obtained from the synthetic experiments conducted in the ablation studies. In the ablation studies we only report the results for exponentially increasing model sizes for clarity. In this study to better capture the optimal model size, we make the model sizes near the optimal model size more fine-grain. In all experiments, we keep the training hyperparameter the same, with 10k train steps.

We find a strong linear relation between the optimal model size and the graph search entropy with R2=0.85R^{2}=0.85. Note that there are a few sources of noise for locating the optimal model size for a specific knowledge graph. First, we only train language model with selected sizes due to compute and time limitations, and the quantization of the model size would disrupt the smoothness of the scaling law. Second, the exact location of the optimal model size is dependent on the training steps, which we did not thoroughly traverse but choose to inspect at the training step 10k.

After fitting a linear regression line using the data from our synthetic experiments, we check the validity of this empirical scaling law against our real-world knowledge graph, FB15K-237. We calculate the graph search entropy for FB15K-237, and find the predicted optimal model size is very close to the observed optimal model size, shown as a green dot in [Figure 4](#S5.F4 "In 5.2 Optimal Model Size v.s. Graph Search Entropy ‣ 5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning").

From our scaling law, we can see that roughly 124 additional parameters in the optimal model size are
required per 1-bit entropy increase in the knowledge graph. That is a language model can only reliably (not perfectly) reason over 0.008 bit information per parameter. This is very different from the knowledge capacity scaling law concluded by Allen-Zhu & Li ([2025](#bib.bib1)), which shows that the language model can store 2 bits of knowledge per parameter.
We think this discrepancy is due to two reasons: first, our scaling law is not only about memorizing the knowledge, but also about reasoning over the learned knowledge, which is significantly harder.
Second, the way we compute the graph search entropy is fundamentally different from the way Allen-Zhu & Li ([2025](#bib.bib1)) computes the knowledge entropy.
While Allen-Zhu & Li ([2025](#bib.bib1)) describes the entropy of the knowledge generation process, our graph search entropy describes the entropy of randomly traversing a fixed knowledge graph.
In this way, we did not directly measure the amount of information that a language model needs to memorize, but measuring the complexity of traversing, and therefore, reasoning over a graph.
It is hard, if not impossible, to obtain the data generation process of real-world data, but it is possible to get an estimate of the underlying knowledge graph of a corpus through automated knowledge graph construction algorithms (Zhong et al., [2023](#bib.bib36)).
Thus, it is possible to predict the optimal reasoning model size for real-world pretraining, by first constructing a knowledge graph from the pretraining corpus, and then computing its graph search entropy, and finally using a similar scaling law to calculate the optimal model size.

### 5.3 Limitations

We want to highlight that this study is only conducted on simplified pretraining data from knowledge graphs, and the results are likely not directly applicable to real-world language model pretraining with large text corpus.
The setting of our study provides a reasonable analogy to the real-world language model pretraining, and the obtained insight might be found useful in the real world when the compute is abundant with very large models and very large datasets that exhaustively traverse the underlying knowledge graph. We leave the work of verifying our scaling law in the real word to future research due to its resource-demanding nature.

## 6 Related Work

#### Language Model Scaling Laws

Kaplan et al. ([2020](#bib.bib15)) first observed a power-law relationship between LLM perplexity, model parameter count, and training data size, laying the foundation for scaling law research. Subsequently,  Hoffmann et al. ([2022b](#bib.bib12)) explored optimal training strategies under constrained computational resources and discovered that LLM parameter size and the number of training tokens should scale proportionally to achieve optimal compute efficiency under a fixed budget.
Beyond pretraining performance, researchers further confirmed that downstream task performance can also be reliably predicted based on model size and training data volume (Hernandez et al., [2021](#bib.bib10); Isik et al., [2024](#bib.bib13)).  Allen-Zhu & Li ([2025](#bib.bib1)); Lu et al. ([2024](#bib.bib17)) have turned to exploring more specific capability dimensions, focusing particularly on the scaling laws of factual memory in LLMs and their behavioral patterns when memorizing different types of facts. Most recently, Roberts et al. ([2025](#bib.bib20)) have confirmed that scaling laws are skill-dependent, and found that knowledge-intensive tasks are more parameter-hungry while reasoning-intensive tasks are more data-hungry.  Springer et al. ([2025](#bib.bib21)) challenge a core assumption in scaling research—that more pretraining invariably leads to better downstream performance. Our paper identifies a different U-shaped scaling curve under the specific scenario of knowledge graph reasoning and reveals that the search complexity of the knowledge graph determines the optimal model size. This echoes the discovery of Pandey ([2024](#bib.bib19)) and Yin et al. ([2024](#bib.bib34)) that classic scaling laws are highly dependent on the data complexity or the compression ratio of the data. Havrilla & Liao ([2024](#bib.bib9)) also confirmed from both theoretical and empirical perspectives that the power of the power scaling law depends on the intrinsic dimension of the training data.

#### Language Model Reasoning

Our paper focuses on the reasoning capability of language models which has drawn a lot of attention recently (Zhang et al., [2023](#bib.bib35); Chen et al., [2023](#bib.bib6); Yao et al., [2023a](#bib.bib31); [b](#bib.bib32); Wang et al., [2023](#bib.bib26); Guo et al., [2025](#bib.bib8); Jin et al., [2024](#bib.bib14); Yeo et al., [2025](#bib.bib33); Team et al., [2025](#bib.bib22); Li et al., [2025](#bib.bib16)). LLMs usually reason in a step-by-step manner in real-world tasks like math word problems (Wei et al., [2022b](#bib.bib29)). In our experiments, we do not ask language models to generate a step-by-step solution for its answer, but ask the language model to directly choose the correct answer from the given options, because our pretrain-only language models are not trained to give a step-by-step solution for a query. Our synthetic reasoning environment is the most similar to Wang et al. ([2024b](#bib.bib27)), which also use the knowledge graph completion task as a testbed to understand how language models learn to reason at pretraining time. They propose that language models are able to aggregate random walk paths sampled from the knowledge graph. Wang et al. ([2024a](#bib.bib25)); Zhu et al. ([2024](#bib.bib37)) also employ a graph structure to ground their synthetic reasoning tasks to explain how LLMs reason, but their reasoning is defined as concatenations of relations: A is r1r\_{1} to B and B is r2r\_{2} to C implies A is r1​r2r\_{1}r\_{2} to C. The knowledge graph completion task we employ is more complex than simple concatenation of relations as the language model needs to find out which relation r1​r2r\_{1}r\_{2} corresponds to from the knowledge graph.

## 7 Conclusion

This paper investigates reasoning scaling in language models trained on knowledge graphs. Our results reveal a U-shaped relationship between model size and reasoning performance, where overparameterization leads to excessive memorization and degraded reasoning ability. We identify key factors that determine the optimal model size, such as the number of training triples and graph complexity. Notably, we propose an empirical scaling law linking optimal model size to graph search entropy, offering a quantitative guide for model design. While our experiments are conducted in controlled settings, these insights pave the way for future work in real-world pretraining scenarios and improved reasoning capabilities in LLMs.

## References

* Allen-Zhu & Li (2025)

  Zeyuan Allen-Zhu and Yuanzhi Li.
  Physics of language models: Part 3.3, knowledge capacity scaling laws.
  In *The Thirteenth International Conference on Learning Representations*, 2025.
  URL <https://openreview.net/forum?id=FxNNiUgtfa>.
* Barabási & Albert (1999)

  Albert-László Barabási and Réka Albert.
  Emergence of scaling in random networks.
  *science*, 286(5439):509–512, 1999.
* Bollacker et al. (2007)

  Kurt Bollacker, Robert Cook, and Patrick Tufts.
  Freebase: a shared database of structured general human knowledge.
  In *Proceedings of the 22nd National Conference on Artificial Intelligence - Volume 2*, AAAI’07, pp.  1962–1963. AAAI Press, 2007.
  ISBN 9781577353232.
* Bordes et al. (2013)

  Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko.
  Translating embeddings for modeling multi-relational data.
  In C.J. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K.Q. Weinberger (eds.), *Advances in Neural Information Processing Systems*, volume 26. Curran Associates, Inc., 2013.
  URL <https://proceedings.neurips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf>.
* Caballero et al. (2023)

  Ethan Caballero, Kshitij Gupta, Irina Rish, and David Krueger.
  Broken neural scaling laws.
  In *The Eleventh International Conference on Learning Representations*, 2023.
  URL <https://openreview.net/forum?id=sckjveqlCZ>.
* Chen et al. (2023)

  Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen.
  Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks.
  *Transactions on Machine Learning Research*, 2023.
* Ghorbani et al. (2022)

  Behrooz Ghorbani, Orhan Firat, Markus Freitag, Ankur Bapna, Maxim Krikun, Xavier Garcia, Ciprian Chelba, and Colin Cherry.
  Scaling laws for neural machine translation.
  In *International Conference on Learning Representations*, 2022.
* Guo et al. (2025)

  Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.
  Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.
  *arXiv preprint arXiv:2501.12948*, 2025.
* Havrilla & Liao (2024)

  Alexander Havrilla and Wenjing Liao.
  Understanding scaling laws with statistical and approximation theory for transformer neural networks on intrinsically low-dimensional data.
  In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024.
  URL <https://openreview.net/forum?id=N2wYPMpifA>.
* Hernandez et al. (2021)

  Danny Hernandez, Jared Kaplan, Tom Henighan, and Sam McCandlish.
  Scaling laws for transfer.
  *arXiv preprint arXiv:2102.01293*, 2021.
* Hoffmann et al. (2022a)

  Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al.
  Training compute-optimal large language models.
  *arXiv preprint arXiv:2203.15556*, 2022a.
* Hoffmann et al. (2022b)

  Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al.
  Training compute-optimal large language models.
  In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, pp.  30016–30030, 2022b.
* Isik et al. (2024)

  Berivan Isik, Natalia Ponomareva, Hussein Hazimeh, Dimitris Paparas, Sergei Vassilvitskii, and Sanmi Koyejo.
  Scaling laws for downstream task performance of large language models.
  In *ICLR 2024 Workshop on Navigating and Addressing Data Problems for Foundation Models*, 2024.
* Jin et al. (2024)

  Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du.
  The impact of reasoning step length on large language models.
  In *Findings of the Association for Computational Linguistics ACL 2024*, pp.  1830–1842, 2024.
* Kaplan et al. (2020)

  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
  Scaling laws for neural language models.
  *arXiv preprint arXiv:2001.08361*, 2020.
* Li et al. (2025)

  Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, et al.
  From system 1 to system 2: A survey of reasoning large language models.
  *arXiv preprint arXiv:2502.17419*, 2025.
* Lu et al. (2024)

  Xingyu Lu, Xiaonan Li, Qinyuan Cheng, Kai Ding, Xuan-Jing Huang, and Xipeng Qiu.
  Scaling laws for fact memorization of large language models.
  In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pp.  11263–11282, 2024.
* Nakkiran et al. (2020)

  Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever.
  Deep double descent: Where bigger models and more data hurt.
  In *International Conference on Learning Representations*, 2020.
  URL <https://openreview.net/forum?id=B1g5sA4twr>.
* Pandey (2024)

  Rohan Pandey.
  gzip predicts data-dependent scaling laws.
  *arXiv preprint arXiv:2405.16684*, 2024.
* Roberts et al. (2025)

  Nicholas Roberts, Niladri Chatterji, Sharan Narang, Mike Lewis, and Dieuwke Hupkes.
  Compute optimal scaling of skills: Knowledge vs reasoning.
  *arXiv preprint arXiv:2503.10061*, 2025.
* Springer et al. (2025)

  Jacob Mitchell Springer, Sachin Goyal, Kaiyue Wen, Tanishq Kumar, Xiang Yue, Sadhika Malladi, Graham Neubig, and Aditi Raghunathan.
  Overtrained language models are harder to fine-tune.
  *https://arxiv.org/abs/2503.19206*, 2025.
* Team et al. (2025)

  Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al.
  Kimi k1. 5: Scaling reinforcement learning with llms.
  *arXiv preprint arXiv:2501.12599*, 2025.
* Toutanova & Chen (2015)

  Kristina Toutanova and Danqi Chen.
  Observed versus latent features for knowledge base and text inference.
  In Alexandre Allauzen, Edward Grefenstette, Karl Moritz Hermann, Hugo Larochelle, and Scott Wen-tau Yih (eds.), *Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality*, pp.  57–66, Beijing, China, July 2015. Association for Computational Linguistics.
  doi: 10.18653/v1/W15-4007.
  URL <https://aclanthology.org/W15-4007/>.
* Touvron et al. (2023)

  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.
  Llama: Open and efficient foundation language models.
  *arXiv preprint arXiv:2302.13971*, 2023.
* Wang et al. (2024a)

  Boshi Wang, Xiang Yue, Yu Su, and Huan Sun.
  Grokking of implicit reasoning in transformers: A mechanistic journey to the edge of generalization.
  In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024a.
  URL <https://openreview.net/forum?id=D4QgSWxiOb>.
* Wang et al. (2023)

  Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim.
  Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models.
  In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp.  2609–2634, 2023.
* Wang et al. (2024b)

  Xinyi Wang, Alfonso Amayuelas, Kexun Zhang, Liangming Pan, Wenhu Chen, and William Yang Wang.
  Understanding reasoning ability of language models from the perspective of reasoning paths aggregation.
  In *Forty-first International Conference on Machine Learning*, 2024b.
* Wei et al. (2022a)

  Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus.
  Emergent abilities of large language models.
  *Transactions on Machine Learning Research*, 2022a.
  ISSN 2835-8856.
  URL <https://openreview.net/forum?id=yzkSU5zdwD>.
  Survey Certification.
* Wei et al. (2022b)

  Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.
  Chain-of-thought prompting elicits reasoning in large language models.
  In *Advances in neural information processing systems*, volume 35, pp.  24824–24837, 2022b.
* Wei et al. (2023)

  Jason Wei, Najoung Kim, Yi Tay, and Quoc V Le.
  Inverse scaling can become u-shaped.
  In *The 2023 Conference on Empirical Methods in Natural Language Processing*, 2023.
  URL <https://openreview.net/forum?id=19sGqVUxQw>.
* Yao et al. (2023a)

  Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik R Narasimhan.
  Tree of thoughts: Deliberate problem solving with large language models.
  In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023a.
* Yao et al. (2023b)

  Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
  React: Synergizing reasoning and acting in language models.
  In *International Conference on Learning Representations (ICLR)*, 2023b.
* Yeo et al. (2025)

  Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue.
  Demystifying long chain-of-thought reasoning in llms.
  *arXiv preprint arXiv:2502.03373*, 2025.
* Yin et al. (2024)

  Mingjia Yin, Chuhan Wu, Yufei Wang, Hao Wang, Wei Guo, Yasheng Wang, Yong Liu, Ruiming Tang, Defu Lian, and Enhong Chen.
  Entropy law: The story behind data compression and llm performance.
  *arXiv preprint arXiv:2407.06645*, 2024.
* Zhang et al. (2023)

  Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola.
  Automatic chain of thought prompting in large language models.
  In *The Eleventh International Conference on Learning Representations*, 2023.
* Zhong et al. (2023)

  Lingfeng Zhong, Jia Wu, Qian Li, Hao Peng, and Xindong Wu.
  A comprehensive survey on automatic knowledge graph construction.
  *ACM Computing Surveys*, 56(4):1–62, 2023.
* Zhu et al. (2024)

  Hanlin Zhu, Baihe Huang, Shaolun Zhang, Michael Jordan, Jiantao Jiao, Yuandong Tian, and Stuart J Russell.
  Towards a theoretical understanding of the’reversal curse’via training dynamics.
  *Advances in Neural Information Processing Systems*, 37:90473–90513, 2024.

## Appendix A Experiment Details

| batch size | lr | lr scheduler | warmup ratio | weight decay | max length |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1e-4 | cosine | 0.2 | 0 | 128 |

Table 1: Hyperparameter settings for language model pretraining.



|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | NN | NeN\_{e} | NrN\_{r} | NhN\_{h} | γ\gamma |
| (a) | 100k | 10k | 100 | 50 | 0.5 |
| (b) | 10k/20k/…/100k | 10k | 100 | 50 | 0.5 |
| (c) | 100k | 10k | 100 | 5/10/…/50 | 0.5 |
| (d) | 100k | 10k | 10/20/…/100 | 50 | 0.5 |
| (e) | 100k | 10k | 100 | 50 | 0.1/0.5/…/0.9 |
| (f) | 10k/20k/…/100k | 1k/2k/…/10k | 10 | 5 | 0.5 |

Table 2: Knowledge graph hyperparameter settings for [Figure 3](#S5.F3 "In 5.1 Graph Generation Ablation ‣ 5 Scaling Laws ‣ Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning") experiments. We keep Lm​i​n=2L\_{min}=2 and Lm​a​x=4L\_{max}=4 for all experiments. Here NN denotes the number of triples, NeN\_{e} denotes the number of entities, NrN\_{r} denotes the number of relations, NhN\_{h} denotes the number of rules, γ\gamma denotes the ratio between deductible triples and atomic triples, Lm​i​nL\_{min} denotes the minimum rule length, and Lm​a​xL\_{max} denotes the maximum rule length.

## Appendix B Synthetic Knowledge Graph Generation Code

[⬇](data:text/plain;base64,aW1wb3J0IG5ldHdvcmt4IGFzIG54CmltcG9ydCBudW1weSBhcyBucAppbXBvcnQgcmFuZG9tCmZyb20gY29sbGVjdGlvbnMgaW1wb3J0IGRlZmF1bHRkaWN0CgpkZWYgYWRkX2VkZ2UoRywgaCwgdCwgcik6CiAgICBudW1fZWRnZXMgPSAwCiAgICBpZiBHLmhhc19lZGdlKGgsIHQpOgogICAgICAgIGlmIHIgbm90IGluIEdbaF1bdF1bJ2lkJ106CiAgICAgICAgICAgIEdbaF1bdF1bJ2lkJ10uYXBwZW5kKHIpCiAgICAgICAgICAgIG51bV9lZGdlcyArPSAxCiAgICAgICAgZWxzZToKICAgICAgICAgICAgcHJpbnQoJ2VkZ2UgYWxyZWFkeSBleGlzdHMnKQogICAgZWxzZToKICAgICAgICBHLmFkZF9lZGdlKGgsIHQsIGlkPVtyXSkKICAgICAgICBudW1fZWRnZXMgKz0gMQogICAgcHJpbnQoJ2FkZCBlZGdlOiAnLCAoaCwgciwgdCksICdudW0gZWRnZXM6ICcsIG51bV9lZGdlcykKICAgIHJldHVybiBudW1fZWRnZXMKICAgICAgICAgICAgCiAgICAgICAgICAgIApkZWYgZ2VuZXJhdGVfcnVsZXMocmVsYXRpb25zLCBudW1fcnVsZXMsIExfbWluLCBMX21heCwgd2VpZ2h0ZWQ9RmFsc2UsIHRlbXBlcmF0dXJlPTAuMjUpOgogICAgIyBHZW5lcmF0ZSBLIGFjeWNsaWMgbG9naWMgcnVsZXMgd2l0aCB2YXJ5aW5nIGxlbmd0aHMKICAgIGRlcGVuZGVuY3lfZ3JhcGggPSBkZWZhdWx0ZGljdChzZXQpCiAgICBydWxlcyA9IFtdCiAgICB3ZWlnaHRzID0gW10KICAgIGlmIHdlaWdodGVkOgogICAgICAgIGZvciBsIGluIHJhbmdlKExfbWluLCBMX21heCArIDEpOgogICAgICAgICAgICB3ZWlnaHRzLmFwcGVuZChucC5leHAoLXRlbXBlcmF0dXJlKmwpKQogICAgICAgIHByb2JzID0gbnAuYXJyYXkoW3cgLyBzdW0od2VpZ2h0cykgZm9yIHcgaW4gd2VpZ2h0c10pCiAgICBlbHNlOgogICAgICAgIHdlaWdodHMgPSBbMV0gKiAoTF9tYXggLSBMX21pbiArIDEpCgogICAgZGVmIGhhc19jeWNsZShzdGFydCwgdmlzaXRlZCwgc3RhY2spOgogICAgICAgICIiIkRldGVjdHMgaWYgYWRkaW5nIGEgbmV3IGRlcGVuZGVuY3kgaW50cm9kdWNlcyBhIGN5Y2xlLiIiIgogICAgICAgIGlmIHN0YXJ0IG5vdCBpbiB2aXNpdGVkOgogICAgICAgICAgICB2aXNpdGVkLmFkZChzdGFydCkKICAgICAgICAgICAgc3RhY2suYWRkKHN0YXJ0KQogICAgICAgICAgICBwcmludCgndmlzaXRlZDogJywgdmlzaXRlZCkKICAgICAgICAgICAgcHJpbnQoJ3N0YWNrOiAnLCBzdGFjaykKICAgICAgICAgICAgZm9yIG5laWdoYm9yIGluIGRlcGVuZGVuY3lfZ3JhcGhbc3RhcnRdOgogICAgICAgICAgICAgICAgaWYgbmVpZ2hib3IgaW4gc3RhY2s6CiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFRydWUKICAgICAgICAgICAgICAgIGVsaWYgaGFzX2N5Y2xlKG5laWdoYm9yLCB2aXNpdGVkLCBzdGFjayk6CiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFRydWUKICAgICAgICBpZiBzdGFydCBpbiBzdGFjazoKICAgICAgICAgICAgc3RhY2sucmVtb3ZlKHN0YXJ0KQogICAgICAgIHJldHVybiBGYWxzZQoKICAgIGZvciBfIGluIHJhbmdlKG51bV9ydWxlcyk6CiAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgaWYgd2VpZ2h0ZWQ6CiAgICAgICAgICAgICAgICBsZW5ndGggPSByYW5kb20uY2hvaWNlcyhyYW5nZShMX21pbiwgTF9tYXggKyAxKSwgd2VpZ2h0cz13ZWlnaHRzKVswXQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgbGVuZ3RoID0gcmFuZG9tLnJhbmRpbnQoTF9taW4sIExfbWF4KQogICAgICAgICAgICBydWxlX3JlbGF0aW9ucyA9IHJhbmRvbS5jaG9pY2VzKHJlbGF0aW9ucywgayA9IGxlbmd0aCArIDEpICMgdGhlIGZpcnN0IGVsZW1lbnQgaXMgdGhlIGltcGxpZWQgcmVsYXRpb24KICAgICAgICAgICAgdmFsaWRfcnVsZSA9IFRydWUKICAgICAgICAgICAgZm9yIGkgaW4gcmFuZ2UoMSwgbGVuKHJ1bGVfcmVsYXRpb25zKSk6CiAgICAgICAgICAgICAgICBkZXBlbmRlbmN5X2dyYXBoW3J1bGVfcmVsYXRpb25zWzBdXS5hZGQocnVsZV9yZWxhdGlvbnNbaV0pCgogICAgICAgICAgICAgICAgIyBDaGVjayBmb3IgY3ljbGVzCiAgICAgICAgICAgICAgICBpZiBoYXNfY3ljbGUocnVsZV9yZWxhdGlvbnNbaV0sIHNldCgpLCBzZXQoKSk6CiAgICAgICAgICAgICAgICAgICAgdmFsaWRfcnVsZSA9IEZhbHNlCiAgICAgICAgICAgICAgICAgICAgZm9yIGogaW4gcmFuZ2UoMSwgaSArIDEpOgogICAgICAgICAgICAgICAgICAgICAgICBkZXBlbmRlbmN5X2dyYXBoW3J1bGVfcmVsYXRpb25zWzBdXS5yZW1vdmUocnVsZV9yZWxhdGlvbnNbal0pCiAgICAgICAgICAgICAgICAgICAgYnJlYWsKCiAgICAgICAgICAgIGlmIHZhbGlkX3J1bGU6CiAgICAgICAgICAgICAgICBydWxlcy5hcHBlbmQodHVwbGUocnVsZV9yZWxhdGlvbnMpKQogICAgICAgICAgICAgICAgYnJlYWsKICAgIAogICAgcHJpbnQoJ3J1bGVzOiAnLCBydWxlcykKICAgIHJldHVybiBydWxlcwoKZGVmIGdldF9ub2RlX3R5cGVzKHJ1bGVzLCBtYXhfbnVtX3JlbGF0aW9uc19wZXJfbm9kZT0zKToKICAgICMgbWFwIG5vZGUgdHlwZXMgdG8gb3V0IHJlbGF0aW9ucwogICAgbm9kZV90eXBlcyA9IHt9CiAgICAjIG1hcCBvdXQgcmVsYXRpb25zIHRvIG5vZGUgdHlwZXMKICAgIHIybm9kZV90eXBlcyA9IGRlZmF1bHRkaWN0KGxpc3QpCiAgICBmb3IgcnVsZSBpbiBydWxlczoKICAgICAgICBmb3IgaSBpbiByYW5nZShsZW4ocnVsZSkpOgogICAgICAgICAgICBub2RlX3R5cGUgPSBsZW4obm9kZV90eXBlcykKICAgICAgICAgICAgaWYgaSA9PSAwOgogICAgICAgICAgICAgICAgbm9kZV90eXBlc1tub2RlX3R5cGVdID0gW3J1bGVbaV0sIHJ1bGVbMV1dCiAgICAgICAgICAgICAgICByMm5vZGVfdHlwZXNbcnVsZVtpXV0uYXBwZW5kKG5vZGVfdHlwZSkKICAgICAgICAgICAgICAgIHIybm9kZV90eXBlc1tydWxlWzFdXS5hcHBlbmQobm9kZV90eXBlKQogICAgICAgICAgICBlbGlmIGkgPT0gbGVuKHJ1bGUpIC0gMToKICAgICAgICAgICAgICAgIG5vZGVfdHlwZXNbbm9kZV90eXBlXSA9IFsnLScgKyBydWxlW2ldLCAnLScgKyBydWxlWzBdXQogICAgICAgICAgICAgICAgcjJub2RlX3R5cGVzWyctJyArIHJ1bGVbaV1dLmFwcGVuZChub2RlX3R5cGUpCiAgICAgICAgICAgICAgICByMm5vZGVfdHlwZXNbJy0nICsgcnVsZVswXV0uYXBwZW5kKG5vZGVfdHlwZSkKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIG5vZGVfdHlwZXNbbm9kZV90eXBlXSA9IFsnLScgKyBydWxlW2ldLCBydWxlW2krMV1dCiAgICAgICAgICAgICAgICByMm5vZGVfdHlwZXNbJy0nICsgcnVsZVtpXV0uYXBwZW5kKG5vZGVfdHlwZSkKICAgICAgICAgICAgICAgIHIybm9kZV90eXBlc1tydWxlW2krMV1dLmFwcGVuZChub2RlX3R5cGUpCiAgICAgICAgICAgICAgICAKICAgIHByaW50KG5vZGVfdHlwZXMpCiAgICBwcmludChyMm5vZGVfdHlwZXMpCiAgICAKICAgIGZvciBudW1fcnMgaW4gcmFuZ2UoMiwgbWF4X251bV9yZWxhdGlvbnNfcGVyX25vZGUpOiAKICAgICAgICBwb3NzaWJsZV9uZXdfbm9kZV90eXBlcyA9IFtdCiAgICAgICAgZm9yIHIgaW4gcjJub2RlX3R5cGVzOgogICAgICAgICAgICBhbHRfcnMgPSBbXQogICAgICAgICAgICBmb3Igbm9kZV90eXBlIGluIHIybm9kZV90eXBlc1tyXToKICAgICAgICAgICAgICAgIGZvciBfciBpbiBub2RlX3R5cGVzW25vZGVfdHlwZV06CiAgICAgICAgICAgICAgICAgICAgaWYgX3IgIT0gcjoKICAgICAgICAgICAgICAgICAgICAgICAgYWx0X3JzLmFwcGVuZChfcikKICAgICAgICAgICAgYWx0X3JzID0gbGlzdChzZXQoYWx0X3JzKSkKICAgICAgICAgICAgZm9yIG5vZGVfdHlwZSBpbiByMm5vZGVfdHlwZXNbcl06CiAgICAgICAgICAgICAgICBpZiBsZW4obm9kZV90eXBlc1tub2RlX3R5cGVdKSA9PSBudW1fcnM6CiAgICAgICAgICAgICAgICAgICAgZm9yIF9yIGluIGFsdF9yczoKICAgICAgICAgICAgICAgICAgICAgICAgaWYgX3Igbm90IGluIG5vZGVfdHlwZXNbbm9kZV90eXBlXToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBvc3NpYmxlX25ld19ub2RlX3R5cGVzLmFwcGVuZCh0dXBsZShzb3J0ZWQoW19yXSArIGxpc3Qobm9kZV90eXBlc1tub2RlX3R5cGVdKSkpKQogICAgICAgICAgICBwcmludChwb3NzaWJsZV9uZXdfbm9kZV90eXBlcykKICAgICAgICAgICAgcG9zc2libGVfbmV3X25vZGVfdHlwZXMgKz0gbGlzdChzZXQocG9zc2libGVfbmV3X25vZGVfdHlwZXMpKQogICAgICAgIHBvc3NpYmxlX25ld19ub2RlX3R5cGVzID0gbGlzdChzZXQocG9zc2libGVfbmV3X25vZGVfdHlwZXMpKQogICAgICAgIHByaW50KHBvc3NpYmxlX25ld19ub2RlX3R5cGVzKQogICAgICAgICAgICAKICAgICAgICBmb3IgcnMgaW4gcG9zc2libGVfbmV3X25vZGVfdHlwZXM6CiAgICAgICAgICAgIG5ld19ub2RlX3R5cGUgPSBsZW4obm9kZV90eXBlcykKICAgICAgICAgICAgbm9kZV90eXBlc1tuZXdfbm9kZV90eXBlXSA9IGxpc3QocnMpCiAgICAgICAgICAgIGZvciBfciBpbiByczoKICAgICAgICAgICAgICAgIHIybm9kZV90eXBlc1tfcl0uYXBwZW5kKG5ld19ub2RlX3R5cGUpCiAgICAgICAgICAgICAgICAKICAgIHJldHVybiBub2RlX3R5cGVzCgpkZWYgZ2V0X2Fkal9vdXRfcmVsYXRpb25zKHJ1bGVzKToKICAgIGFkaiA9IGRlZmF1bHRkaWN0KGxpc3QpCiAgICBmb3IgcnVsZSBpbiBydWxlczoKICAgICAgICBmb3IgaSBpbiByYW5nZShsZW4ocnVsZSkpOgogICAgICAgICAgICBpZiBpID09IDA6CiAgICAgICAgICAgICAgICBhZGpbcnVsZVtpXV0uYXBwZW5kKHJ1bGVbMV0pCiAgICAgICAgICAgICAgICBhZGpbcnVsZVsxXV0uYXBwZW5kKHJ1bGVbaV0pCiAgICAgICAgICAgIGVsaWYgaSA9PSBsZW4ocnVsZSkgLSAxOgogICAgICAgICAgICAgICAgYWRqWyctJyArIHJ1bGVbaV1dLmFwcGVuZCgnLScgKyBydWxlWzBdKQogICAgICAgICAgICAgICAgYWRqWyctJyArIHJ1bGVbMF1dLmFwcGVuZCgnLScgKyBydWxlW2ldKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgYWRqWyctJyArIHJ1bGVbaV1dLmFwcGVuZChydWxlW2krMV0pCiAgICAgICAgICAgICAgICBhZGpbcnVsZVtpKzFdXS5hcHBlbmQoJy0nICsgcnVsZVtpXSkKICAgIHJldHVybiBhZGoKICAgICAgICAgICAgICAgICAgICAKZGVmIGxhdGVudF9ydWxlX2dyYXBoKG51bV9ydWxlcz01MCwgTF9taW49MiwgTF9tYXg9NCwgbj0xMDAwMCwgbT0xMCwgbl9yPTIwMCwgCiAgICAgICAgICAgICAgICAgICAgICBudW1fdGVzdD0xMDAwLCBudW1fdHJhaW49MTUwMDAwLCBjaGVja19mcmVxdWVuY3k9MTAwLAogICAgICAgICAgICAgICAgICAgICAgcG93ZXJfbGF3PUZhbHNlLCBpbml0aWFsX2dyYXBoPU5vbmUsCiAgICAgICAgICAgICAgICAgICAgICBsZW5ndGhfd2VpZ2h0ZWQ9RmFsc2UsIG1jbWM9MC4yLCB0ZW1wZXJhdHVyZT0wLjI1LCAKICAgICAgICAgICAgICAgICAgICAgIGRlZHVjdGlibGVfcmF0aW89MC41KToKICAgICMgR2VuZXJhdGUgcmVsYXRpb25zIGFuZCBlbnRpdGllcwogICAgcHJpbnQoIm1jbWM6ICIsIG1jbWMpCiAgICByZWxhdGlvbnMgPSBbJ1AnICsgc3RyKGkpIGZvciBpIGluIHJhbmdlKG5fcildCiAgICBhbGxfcnVsZXMgPSBnZW5lcmF0ZV9ydWxlcyhyZWxhdGlvbnMsIG1heChuX3IvL0xfbWluLCBudW1fcnVsZXMpLCBMX21pbiwgTF9tYXgpCiAgICByMnJ1bGVzID0ge30KICAgIGZvciBydWxlIGluIGFsbF9ydWxlczoKICAgICAgICBpZiBydWxlWzBdIG5vdCBpbiByMnJ1bGVzOgogICAgICAgICAgICByMnJ1bGVzW3J1bGVbMF1dID0gW10gIAogICAgICAgIHIycnVsZXNbcnVsZVswXV0uYXBwZW5kKHJ1bGVbMTpdKQogICAgbnVtX3RyaXBsZXMgPSAwCiAgICByZXBlYXRlZF9lbnRpdGllcyA9IGRlZmF1bHRkaWN0KGxpc3QpICMgbWFwIGluIHJlbGF0aW9uIHRvIGVudGl0aWVzCiAgICBjaGlsZF9yZWxhdGlvbnMgPSBbXQogICAgZm9yIHJ1bGUgaW4gYWxsX3J1bGVzOgogICAgICAgIGNoaWxkX3JlbGF0aW9ucyArPSBydWxlWzE6XQogICAgY2hpbGRfcmVsYXRpb25zID0gbGlzdChzZXQoY2hpbGRfcmVsYXRpb25zKSkKICAgIGNoaWxkX3JlbGF0aW9ucyArPSBbJy0nICsgciBmb3IgciBpbiBjaGlsZF9yZWxhdGlvbnNdCiAgICBkZWR1Y3RpYmxlX3J1bGVzID0gcmFuZG9tLnNhbXBsZShhbGxfcnVsZXMsIG51bV9ydWxlcykKICAgIGlmIGxlbmd0aF93ZWlnaHRlZDoKICAgICAgICB3ZWlnaHRzID0gW2ludCgxMDAqbnAuZXhwKC10ZW1wZXJhdHVyZSpsZW4ocnVsZSkpKSBmb3IgcnVsZSBpbiBhbGxfcnVsZXNdCiAgICBlbHNlOgogICAgICAgIHdlaWdodHMgPSBbMSBmb3IgXyBpbiBhbGxfcnVsZXNdCiAgICByZXBlYXRlZF9ydWxlcyA9IFtdCiAgICBmb3IgcnVsZSwgd2VpZ2h0IGluIHppcChhbGxfcnVsZXMsIHdlaWdodHMpOgogICAgICAgIGZvciBfIGluIHJhbmdlKHdlaWdodCk6CiAgICAgICAgICAgIHJlcGVhdGVkX3J1bGVzLmFwcGVuZChydWxlKQogICAgcmFuZG9tLnNodWZmbGUocmVwZWF0ZWRfcnVsZXMpCiAgICBhZGogPSBnZXRfYWRqX291dF9yZWxhdGlvbnMocmVwZWF0ZWRfcnVsZXMpCiAgICBhbGxfZGVkdWN0aWJsZXMgPSB7fQogICAgCiAgICBpZiBpbml0aWFsX2dyYXBoIGlzIE5vbmU6CiAgICAgICAgIyBEZWZhdWx0IGluaXRpYWwgZ3JhcGgKICAgICAgICBHID0gbnguRGlHcmFwaCgpCiAgICAgICAgbm9kZV9pZCA9IDAKICAgICAgICBtaW5fcmVwZWF0ZWRfZW50aXRpZXMgPSAwCiAgICAgICAgd2hpbGUgbWluX3JlcGVhdGVkX2VudGl0aWVzIDwgbToKICAgICAgICAgICAgZm9yIHJ1bGUgaW4gYWxsX3J1bGVzOgogICAgICAgICAgICAgICAgc291cmNlID0gJ1EnICsgc3RyKG5vZGVfaWQpCiAgICAgICAgICAgICAgICBub2RlX2lkICs9IDEKICAgICAgICAgICAgICAgIGggPSBzb3VyY2UKICAgICAgICAgICAgICAgIGZvciByIGluIHJ1bGVbMTpdOgogICAgICAgICAgICAgICAgICAgIHQgPSAnUScgKyBzdHIobm9kZV9pZCkKICAgICAgICAgICAgICAgICAgICBub2RlX2lkICs9IDEKICAgICAgICAgICAgICAgICAgICBudW1fdHJpcGxlcyArPSBhZGRfZWRnZShHLCBoLCB0LCByKQogICAgICAgICAgICAgICAgICAgIHJlcGVhdGVkX2VudGl0aWVzW3JdLmFwcGVuZCh0KQogICAgICAgICAgICAgICAgICAgIHJlcGVhdGVkX2VudGl0aWVzWyctJyArIHJdLmFwcGVuZChoKQogICAgICAgICAgICAgICAgICAgIGggPSB0CiAgICAgICAgICAgICAgICBudW1fdHJpcGxlcyArPSBhZGRfZWRnZShHLCBzb3VyY2UsIHQsIHJ1bGVbMF0pCiAgICAgICAgICAgICAgICByZXBlYXRlZF9lbnRpdGllc1tydWxlWzBdXS5hcHBlbmQodCkKICAgICAgICAgICAgICAgIHJlcGVhdGVkX2VudGl0aWVzWyctJyArIHJ1bGVbMF1dLmFwcGVuZChzb3VyY2UpCgogICAgICAgICAgICBtaW5fcmVwZWF0ZWRfZW50aXRpZXMgPSBtaW4oW2xlbihzZXQocmVwZWF0ZWRfZW50aXRpZXNbcl0pKSBmb3IgciBpbiBjaGlsZF9yZWxhdGlvbnNdKQogICAgZWxzZToKICAgICAgICBpZiBsZW4oaW5pdGlhbF9ncmFwaCkgPCBtIG9yIGxlbihpbml0aWFsX2dyYXBoKSA+IG46CiAgICAgICAgICAgIHJhaXNlIG54Lk5ldHdvcmtYRXJyb3IoCiAgICAgICAgICAgICAgICBmIkluaXRpYWwgZ3JhcGggbmVlZHMgYmV0d2VlbiBtPXttfSBhbmQgbj17bn0gbm9kZXMiCiAgICAgICAgICAgICkKICAgICAgICBHID0gaW5pdGlhbF9ncmFwaC5jb3B5KCkKICAgICAgICBub2RlX2lkID0gbGVuKEcpCgogICAgaWYgbm90IHBvd2VyX2xhdzoKICAgICAgICByZXBlYXRlZF9lbnRpdGllcyA9IHtyOiBsaXN0KHNldChyZXBlYXRlZF9lbnRpdGllc1tyXSkpIGZvciByIGluIHJlcGVhdGVkX2VudGl0aWVzfQogICAgCiAgICAjIFN0YXJ0IGFkZGluZyB0aGUgb3RoZXIgbm9kZXMuCiAgICB3aGlsZSBub2RlX2lkIDwgbjoKICAgICAgICBzb3VyY2UgPSAnUScgKyBzdHIobm9kZV9pZCkKICAgICAgICBub2RlX2lkICs9IDEKICAgICAgICBwb3NzaWJsZV9yZWxhdGlvbnMgPSBbX3IgZm9yIF9yIGluIGFkaiBpZiBfciBpbiBjaGlsZF9yZWxhdGlvbnNdCiAgICAgICAgaWYgbGVuKHBvc3NpYmxlX3JlbGF0aW9ucykgPT0gMDoKICAgICAgICAgICAgcHJpbnQoJ25vIGFkaiByZWxhdGlvbnMnKQogICAgICAgICAgICBicmVhawogICAgICAgIHByaW50KCdhZGQgY2hpbGQgZWRnZScpCiAgICAgICAgY2hvc2VuX2VkZ2VzID0gW10KICAgICAgICBzdG9wID0gRmFsc2UKICAgICAgICBmb3IgXyBpbiByYW5nZShtKToKICAgICAgICAgICAgaXQgPSAwCiAgICAgICAgICAgIHdoaWxlIChyLCB0KSBpbiBjaG9zZW5fZWRnZXM6CiAgICAgICAgICAgICAgICByID0gcmFuZG9tLmNob2ljZShwb3NzaWJsZV9yZWxhdGlvbnMpCiAgICAgICAgICAgICAgICB0ID0gcmFuZG9tLmNob2ljZShyZXBlYXRlZF9lbnRpdGllc1tyXSkKICAgICAgICAgICAgICAgIGl0ICs9IDEKICAgICAgICAgICAgICAgIGlmIGl0ID4gMTAwOgogICAgICAgICAgICAgICAgICAgIHByaW50KCdmYWlsZWQgdG8gZmluZCBlZGdlJykKICAgICAgICAgICAgICAgICAgICBzdG9wID0gVHJ1ZQogICAgICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgIGlmIHN0b3Agb3IgbGVuKHBvc3NpYmxlX3JlbGF0aW9ucykgPT0gMDoKICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgIAogICAgICAgICAgICBwb3NzaWJsZV9yZWxhdGlvbnMgPSBbX3IgZm9yIF9yIGluIGFkaltyXSBpZiBfciBpbiBjaGlsZF9yZWxhdGlvbnNdCiAgICAgICAgICAgIGNob3Nlbl9lZGdlcy5hcHBlbmQoKHIsIHQpKQogICAgICAgICAgICBpZiByWzBdID09ICctJzoKICAgICAgICAgICAgICAgIG51bV90cmlwbGVzICs9IGFkZF9lZGdlKEcsIHQsIHNvdXJjZSwgclsxOl0pCiAgICAgICAgICAgICAgICByZXBlYXRlZF9lbnRpdGllc1tyWzE6XV0uYXBwZW5kKHNvdXJjZSkKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIG51bV90cmlwbGVzICs9IGFkZF9lZGdlKEcsIHNvdXJjZSwgdCwgcikKICAgICAgICAgICAgICAgIHJlcGVhdGVkX2VudGl0aWVzWyctJyArIHJdLmFwcGVuZChzb3VyY2UpCiAgICAgICAgICAgIHJlcGVhdGVkX2VudGl0aWVzW3JdLmFwcGVuZCh0KQogICAgICAgICAgICBpZiBsZW4ocG9zc2libGVfcmVsYXRpb25zKSA9PSAwOgogICAgICAgICAgICAgICAgcHJpbnQoJ25vIGFkaiByZWxhdGlvbnMnKQogICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgICAgICAgICAKICAgICAgICBpZiBub3QgcG93ZXJfbGF3OgogICAgICAgICAgICByZXBlYXRlZF9lbnRpdGllcyA9IHtyOiBsaXN0KHNldChyZXBlYXRlZF9lbnRpdGllc1tyXSkpIGZvciByIGluIHJlcGVhdGVkX2VudGl0aWVzfQogICAgICAgICAgICAKICAgICAgICBpZiBub2RlX2lkICUgY2hlY2tfZnJlcXVlbmN5ID09IDAgb3Igbm9kZV9pZCA9PSBuLTE6CiAgICAgICAgICAgICMgYWRkIGRlZHVjdGlibGVzCiAgICAgICAgICAgIGFsbF9ub2RlcyA9IGxpc3QoRy5ub2RlcykKICAgICAgICAgICAgcmFuZG9tLnNodWZmbGUoYWxsX25vZGVzKQogICAgICAgICAgICBmb3IgaCBpbiBhbGxfbm9kZXM6CiAgICAgICAgICAgICAgICBmb3IgcnVsZSBpbiBkZWR1Y3RpYmxlX3J1bGVzOgogICAgICAgICAgICAgICAgICAgIGhlYWRfbGlzdCA9IFtoXQogICAgICAgICAgICAgICAgICAgIHIgPSBydWxlWzBdCiAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgZm9yIF9yIGluIHJ1bGVbMTpdOgogICAgICAgICAgICAgICAgICAgICAgICBuZXh0X2hlYWRfbGlzdCA9IFtdCiAgICAgICAgICAgICAgICAgICAgICAgIGZvciBlX2ggaW4gaGVhZF9saXN0OgogICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgZV9oIG5vdCBpbiBHLm5vZGVzOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmb3IgZV90IGluIEdbZV9oXToKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiBfciBpbiBHW2VfaF1bZV90XVsnaWQnXTogCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIHJhbmRvbS5yYW5kb20oKSA8IG1jbWM6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXh0X2hlYWRfbGlzdC5hcHBlbmQoZV90KQogICAgICAgICAgICAgICAgICAgICAgICBoZWFkX2xpc3QgPSBuZXh0X2hlYWRfbGlzdAogICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICBmb3IgdCBpbiBoZWFkX2xpc3Q6CiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChoLCByLCB0KSBub3QgaW4gYWxsX2RlZHVjdGlibGVzOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgYWxsX2RlZHVjdGlibGVzWyhoLCByLCB0KV0gPSBbcnVsZV0KICAgICAgICAgICAgICAgICAgICAgICAgZWxpZiBydWxlIG5vdCBpbiBhbGxfZGVkdWN0aWJsZXNbKGgsIHIsIHQpXToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFsbF9kZWR1Y3RpYmxlc1soaCwgciwgdCldLmFwcGVuZChydWxlKQogICAgICAgICAgICAgICAgICAgICAgICBpZiBub3QgRy5oYXNfZWRnZShoLCB0KSBvciByIG5vdCBpbiBHW2hdW3RdWydpZCddOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgcHJpbnQoJ2FkZCBkZWR1Y3RpYmxlIGVkZ2UnKQogICAgICAgICAgICAgICAgICAgICAgICAgICAgYWRkX2VkZ2UoRywgaCwgdCwgcikKICAgICAgICAgICAgICAgICAgICAgICAgICAgIG51bV90cmlwbGVzICs9IDEKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlcGVhdGVkX2VudGl0aWVzW3JdLmFwcGVuZCh0KQogICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVwZWF0ZWRfZW50aXRpZXNbJy0nICsgcl0uYXBwZW5kKGgpCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgIGF0b21pY190cmlwbGVzID0gW10KICAgIGRlZHVjdGlibGVfdHJpcGxlcyA9IFtdCiAgICBmb3IgaCwgdCBpbiBHLmVkZ2VzOgogICAgICAgIGZvciByIGluIEdbaF1bdF1bJ2lkJ106CiAgICAgICAgICAgIGlmIChoLCByLCB0KSBub3QgaW4gYWxsX2RlZHVjdGlibGVzOgogICAgICAgICAgICAgICAgYXRvbWljX3RyaXBsZXMuYXBwZW5kKChoLCByLCB0KSkKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIGRlZHVjdGlibGVfdHJpcGxlcy5hcHBlbmQoKGgsIHIsIHQpKQogICAgcmFuZG9tLnNodWZmbGUoYXRvbWljX3RyaXBsZXMpCiAgICByYW5kb20uc2h1ZmZsZShkZWR1Y3RpYmxlX3RyaXBsZXMpCiAgICBhc3NlcnQgbGVuKGF0b21pY190cmlwbGVzKSA+PSBpbnQobnVtX3RyYWluICogKDEtZGVkdWN0aWJsZV9yYXRpbykpCiAgICBhc3NlcnQgbGVuKGRlZHVjdGlibGVfdHJpcGxlcykgPj0gaW50KG51bV90cmFpbiAqIGRlZHVjdGlibGVfcmF0aW8pICsgMiAqIG51bV90ZXN0CiAgICAKICAgIHJlbW92ZV90cmlwbGVzID0gW10KICAgIHRyYWluX2F0b21pY190cmlwbGVzID0gYXRvbWljX3RyaXBsZXNbOmludChudW1fdHJhaW4gKiAoMS1kZWR1Y3RpYmxlX3JhdGlvKSldCiAgICByZW1vdmVfdHJpcGxlcyArPSBhdG9taWNfdHJpcGxlc1tpbnQobnVtX3RyYWluICogKDEtZGVkdWN0aWJsZV9yYXRpbykpOl0KICAgIHRyYWluX2RlZHVjdGlibGVfdHJpcGxlcyA9IGRlZHVjdGlibGVfdHJpcGxlc1s6aW50KG51bV90cmFpbiAqIGRlZHVjdGlibGVfcmF0aW8pXQogICAgcmVtb3ZlX3RyaXBsZXMgKz0gZGVkdWN0aWJsZV90cmlwbGVzW2ludChudW1fdHJhaW4gKiBkZWR1Y3RpYmxlX3JhdGlvKTpdCiAgICAKICAgIGZvciBoLCByLCB0IGluIHJlbW92ZV90cmlwbGVzOgogICAgICAgIF90ID0gdAogICAgICAgIHJzID0gR1toXVtfdF1bJ2lkJ10KICAgICAgICBpZiByIGluIHJzOgogICAgICAgICAgICBpZiBsZW4ocnMpID09IDE6CiAgICAgICAgICAgICAgICBHLnJlbW92ZV9lZGdlKGgsIF90KQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgR1toXVtfdF1bJ2lkJ10ucmVtb3ZlKHIpCiAgICAKICAgIHRyYWluX3RyaXBsZXMgPSB0cmFpbl9kZWR1Y3RpYmxlX3RyaXBsZXMgKyB0cmFpbl9hdG9taWNfdHJpcGxlcwogICAgcmFuZG9tLnNodWZmbGUodHJhaW5fdHJpcGxlcykKICAgIHByaW50KCJudW0gdHJhaW4gdHJpcGxlczogIiwgbGVuKHRyYWluX3RyaXBsZXMpKQogICAgCiAgICByMnJ1bGUgPSB7fQogICAgZm9yIHJ1bGUgaW4gZGVkdWN0aWJsZV9ydWxlczoKICAgICAgICBpZiBydWxlWzBdIGluIHIycnVsZToKICAgICAgICAgICAgcjJydWxlW3J1bGVbMF1dLmFwcGVuZChydWxlWzE6XSkKICAgICAgICBlbHNlOgogICAgICAgICAgICByMnJ1bGVbcnVsZVswXV0gPSBbcnVsZVsxOl1dCiAgICAKICAgIGRlZiBjaGVja19kZWR1Y3RpYmxlKHRyaXBsZSk6CiAgICAgICAgaCwgciwgdCA9IHRyaXBsZQogICAgICAgIGFsdF90cyA9IFtdCiAgICAgICAgZm9yIHJ1bGUgaW4gcjJydWxlW3JdOgogICAgICAgICAgICBoZWFkX2xpc3QgPSBbaF0KICAgICAgICAgICAgZm9yIF9yIGluIHJ1bGU6CiAgICAgICAgICAgICAgICBuZXh0X2hlYWRfbGlzdCA9IFtdCiAgICAgICAgICAgICAgICBmb3IgZV9oIGluIGhlYWRfbGlzdDoKICAgICAgICAgICAgICAgICAgICBmb3IgZV90IGluIEdbZV9oXToKICAgICAgICAgICAgICAgICAgICAgICAgaWYgX3IgaW4gR1tlX2hdW2VfdF1bJ2lkJ106CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXh0X2hlYWRfbGlzdC5hcHBlbmQoZV90KQogICAgICAgICAgICAgICAgaGVhZF9saXN0ID0gbmV4dF9oZWFkX2xpc3QKICAgICAgICAgICAgYWx0X3RzICs9IGhlYWRfbGlzdAogICAgICAgIGlmIHQgaW4gYWx0X3RzOgogICAgICAgICAgICByZXR1cm4gVHJ1ZQogICAgICAgIHJldHVybiBGYWxzZQogICAgCiAgICBpZF90ZXN0X3RyaXBsZXMgPSBbXQogICAgZm9yIGkgaW4gcmFuZ2UoaW50KG51bV90cmFpbiAqIGRlZHVjdGlibGVfcmF0aW8pLCBsZW4oZGVkdWN0aWJsZV90cmlwbGVzKSk6CiAgICAgICAgaWYgY2hlY2tfZGVkdWN0aWJsZShkZWR1Y3RpYmxlX3RyaXBsZXNbaV0pOgogICAgICAgICAgICBpZF90ZXN0X3RyaXBsZXMuYXBwZW5kKGRlZHVjdGlibGVfdHJpcGxlc1tpXSkKICAgICAgICBpZiBsZW4oaWRfdGVzdF90cmlwbGVzKSA9PSBudW1fdGVzdDoKICAgICAgICAgICAgYnJlYWsKICAgICAgICAKICAgIGlkX3Rlc3RfcnVsZXMgPSBbYWxsX2RlZHVjdGlibGVzW3RyaXBsZV0gZm9yIHRyaXBsZSBpbiBpZF90ZXN0X3RyaXBsZXNdCiAgICBwcmludCgibnVtIGlkIHRlc3QgdHJpcGxlczogIiwgbGVuKGlkX3Rlc3RfdHJpcGxlcykpCiAgICAKICAgIHJ1bGUydHJpcGxlcyA9IGRlZmF1bHRkaWN0KGxpc3QpCiAgICBmb3IgdHJpcGxlIGluIGRlZHVjdGlibGVfdHJpcGxlc1tpKzE6XToKICAgICAgICBmb3IgcnVsZSBpbiBhbGxfZGVkdWN0aWJsZXNbdHJpcGxlXToKICAgICAgICAgICAgcnVsZTJ0cmlwbGVzW3J1bGVdLmFwcGVuZCh0cmlwbGUpCiAgICAKICAgICMgdW5pZm9ybWx5IHNhbXBsZSB0ZXN0aW5nIHRyaXBsZXMgZnJvbSBlYWNoIHJ1bGUKICAgIHVuaWZvcm1fdGVzdF90cmlwbGVzID0gW10KICAgIGZvciBydWxlIGluIHJ1bGUydHJpcGxlczoKICAgICAgICB0cmlwbGVzID0gW10KICAgICAgICBmb3IgdHJpcGxlIGluIHJ1bGUydHJpcGxlc1tydWxlXToKICAgICAgICAgICAgaWYgY2hlY2tfZGVkdWN0aWJsZSh0cmlwbGUpOgogICAgICAgICAgICAgICAgdHJpcGxlcy5hcHBlbmQodHJpcGxlKQogICAgICAgICAgICAgICAgCiAgICAgICAgaWYgbGVuKHRyaXBsZXMpID4gbnVtX3Rlc3QvL2xlbihydWxlMnRyaXBsZXMpOgogICAgICAgICAgICB1bmlmb3JtX3Rlc3RfdHJpcGxlcyArPSByYW5kb20uc2FtcGxlKHRyaXBsZXMsIG51bV90ZXN0Ly9sZW4ocnVsZTJ0cmlwbGVzKSkKICAgICAgICBlbHNlOgogICAgICAgICAgICB1bmlmb3JtX3Rlc3RfdHJpcGxlcyArPSB0cmlwbGVzCgogICAgcmFuZG9tLnNodWZmbGUodW5pZm9ybV90ZXN0X3RyaXBsZXMpCiAgICB1bmlmb3JtX3Rlc3RfcnVsZXMgPSBbYWxsX2RlZHVjdGlibGVzW3RyaXBsZV0gZm9yIHRyaXBsZSBpbiB1bmlmb3JtX3Rlc3RfdHJpcGxlc10KICAgIHByaW50KCJudW0gdW5pZm9ybSB0ZXN0IHRyaXBsZXM6ICIsIGxlbih1bmlmb3JtX3Rlc3RfdHJpcGxlcykpCgogICAgcmV0dXJuIEcsIGRlZHVjdGlibGVfcnVsZXMsIHRyYWluX3RyaXBsZXMsIGlkX3Rlc3RfdHJpcGxlcywgaWRfdGVzdF9ydWxlcywgdW5pZm9ybV90ZXN0X3RyaXBsZXMsIHVuaWZvcm1fdGVzdF9ydWxlcw==)

import networkx as nx

import numpy as np

import random

from collections import defaultdict

def add\_edge(G, h, t, r):

num\_edges = 0

if G.has\_edge(h, t):

if r not in G[h][t][’id’]:

G[h][t][’id’].append(r)

num\_edges += 1

else:

print(’edge already exists’)

else:

G.add\_edge(h, t, id=[r])

num\_edges += 1

print(’add edge: ’, (h, r, t), ’num edges: ’, num\_edges)

return num\_edges

def generate\_rules(relations, num\_rules, L\_min, L\_max, weighted=False, temperature=0.25):

# Generate K acyclic logic rules with varying lengths

dependency\_graph = defaultdict(set)

rules = []

weights = []

if weighted:

for l in range(L\_min, L\_max + 1):

weights.append(np.exp(-temperature\*l))

probs = np.array([w / sum(weights) for w in weights])

else:

weights = [1] \* (L\_max - L\_min + 1)

def has\_cycle(start, visited, stack):

”””Detects if adding a new dependency introduces a cycle.”””

if start not in visited:

visited.add(start)

stack.add(start)

print(’visited: ’, visited)

print(’stack: ’, stack)

for neighbor in dependency\_graph[start]:

if neighbor in stack:

return True

elif has\_cycle(neighbor, visited, stack):

return True

if start in stack:

stack.remove(start)

return False

for \_ in range(num\_rules):

while True:

if weighted:

length = random.choices(range(L\_min, L\_max + 1), weights=weights)[0]

else:

length = random.randint(L\_min, L\_max)

rule\_relations = random.choices(relations, k = length + 1) # the first element is the implied relation

valid\_rule = True

for i in range(1, len(rule\_relations)):

dependency\_graph[rule\_relations[0]].add(rule\_relations[i])

# Check for cycles

if has\_cycle(rule\_relations[i], set(), set()):

valid\_rule = False

for j in range(1, i + 1):

dependency\_graph[rule\_relations[0]].remove(rule\_relations[j])

break

if valid\_rule:

rules.append(tuple(rule\_relations))

break

print(’rules: ’, rules)

return rules

def get\_node\_types(rules, max\_num\_relations\_per\_node=3):

# map node types to out relations

node\_types = {}

# map out relations to node types

r2node\_types = defaultdict(list)

for rule in rules:

for i in range(len(rule)):

node\_type = len(node\_types)

if i == 0:

node\_types[node\_type] = [rule[i], rule[1]]

r2node\_types[rule[i]].append(node\_type)

r2node\_types[rule[1]].append(node\_type)

elif i == len(rule) - 1:

node\_types[node\_type] = [’-’ + rule[i], ’-’ + rule[0]]

r2node\_types[’-’ + rule[i]].append(node\_type)

r2node\_types[’-’ + rule[0]].append(node\_type)

else:

node\_types[node\_type] = [’-’ + rule[i], rule[i+1]]

r2node\_types[’-’ + rule[i]].append(node\_type)

r2node\_types[rule[i+1]].append(node\_type)

print(node\_types)

print(r2node\_types)

for num\_rs in range(2, max\_num\_relations\_per\_node):

possible\_new\_node\_types = []

for r in r2node\_types:

alt\_rs = []

for node\_type in r2node\_types[r]:

for \_r in node\_types[node\_type]:

if \_r != r:

alt\_rs.append(\_r)

alt\_rs = list(set(alt\_rs))

for node\_type in r2node\_types[r]:

if len(node\_types[node\_type]) == num\_rs:

for \_r in alt\_rs:

if \_r not in node\_types[node\_type]:

possible\_new\_node\_types.append(tuple(sorted([\_r] + list(node\_types[node\_type]))))

print(possible\_new\_node\_types)

possible\_new\_node\_types += list(set(possible\_new\_node\_types))

possible\_new\_node\_types = list(set(possible\_new\_node\_types))

print(possible\_new\_node\_types)

for rs in possible\_new\_node\_types:

new\_node\_type = len(node\_types)

node\_types[new\_node\_type] = list(rs)

for \_r in rs:

r2node\_types[\_r].append(new\_node\_type)

return node\_types

def get\_adj\_out\_relations(rules):

adj = defaultdict(list)

for rule in rules:

for i in range(len(rule)):

if i == 0:

adj[rule[i]].append(rule[1])

adj[rule[1]].append(rule[i])

elif i == len(rule) - 1:

adj[’-’ + rule[i]].append(’-’ + rule[0])

adj[’-’ + rule[0]].append(’-’ + rule[i])

else:

adj[’-’ + rule[i]].append(rule[i+1])

adj[rule[i+1]].append(’-’ + rule[i])

return adj

def latent\_rule\_graph(num\_rules=50, L\_min=2, L\_max=4, n=10000, m=10, n\_r=200,

num\_test=1000, num\_train=150000, check\_frequency=100,

power\_law=False, initial\_graph=None,

length\_weighted=False, mcmc=0.2, temperature=0.25,

deductible\_ratio=0.5):

# Generate relations and entities

print(”mcmc: ”, mcmc)

relations = [’P’ + str(i) for i in range(n\_r)]

all\_rules = generate\_rules(relations, max(n\_r//L\_min, num\_rules), L\_min, L\_max)

r2rules = {}

for rule in all\_rules:

if rule[0] not in r2rules:

r2rules[rule[0]] = []

r2rules[rule[0]].append(rule[1:])

num\_triples = 0

repeated\_entities = defaultdict(list) # map in relation to entities

child\_relations = []

for rule in all\_rules:

child\_relations += rule[1:]

child\_relations = list(set(child\_relations))

child\_relations += [’-’ + r for r in child\_relations]

deductible\_rules = random.sample(all\_rules, num\_rules)

if length\_weighted:

weights = [int(100\*np.exp(-temperature\*len(rule))) for rule in all\_rules]

else:

weights = [1 for \_ in all\_rules]

repeated\_rules = []

for rule, weight in zip(all\_rules, weights):

for \_ in range(weight):

repeated\_rules.append(rule)

random.shuffle(repeated\_rules)

adj = get\_adj\_out\_relations(repeated\_rules)

all\_deductibles = {}

if initial\_graph is None:

# Default initial graph

G = nx.DiGraph()

node\_id = 0

min\_repeated\_entities = 0

while min\_repeated\_entities < m:

for rule in all\_rules:

source = ’Q’ + str(node\_id)

node\_id += 1

h = source

for r in rule[1:]:

t = ’Q’ + str(node\_id)

node\_id += 1

num\_triples += add\_edge(G, h, t, r)

repeated\_entities[r].append(t)

repeated\_entities[’-’ + r].append(h)

h = t

num\_triples += add\_edge(G, source, t, rule[0])

repeated\_entities[rule[0]].append(t)

repeated\_entities[’-’ + rule[0]].append(source)

min\_repeated\_entities = min([len(set(repeated\_entities[r])) for r in child\_relations])

else:

if len(initial\_graph) < m or len(initial\_graph) > n:

raise nx.NetworkXError(

f”Initial graph needs between m={m} and n={n} nodes”

)

G = initial\_graph.copy()

node\_id = len(G)

if not power\_law:

repeated\_entities = {r: list(set(repeated\_entities[r])) for r in repeated\_entities}

# Start adding the other nodes.

while node\_id < n:

source = ’Q’ + str(node\_id)

node\_id += 1

possible\_relations = [\_r for \_r in adj if \_r in child\_relations]

if len(possible\_relations) == 0:

print(’no adj relations’)

break

print(’add child edge’)

chosen\_edges = []

stop = False

for \_ in range(m):

it = 0

while (r, t) in chosen\_edges:

r = random.choice(possible\_relations)

t = random.choice(repeated\_entities[r])

it += 1

if it > 100:

print(’failed to find edge’)

stop = True

break

if stop or len(possible\_relations) == 0:

break

possible\_relations = [\_r for \_r in adj[r] if \_r in child\_relations]

chosen\_edges.append((r, t))

if r[0] == ’-’:

num\_triples += add\_edge(G, t, source, r[1:])

repeated\_entities[r[1:]].append(source)

else:

num\_triples += add\_edge(G, source, t, r)

repeated\_entities[’-’ + r].append(source)

repeated\_entities[r].append(t)

if len(possible\_relations) == 0:

print(’no adj relations’)

break

if not power\_law:

repeated\_entities = {r: list(set(repeated\_entities[r])) for r in repeated\_entities}

if node\_id % check\_frequency == 0 or node\_id == n-1:

# add deductibles

all\_nodes = list(G.nodes)

random.shuffle(all\_nodes)

for h in all\_nodes:

for rule in deductible\_rules:

head\_list = [h]

r = rule[0]

for \_r in rule[1:]:

next\_head\_list = []

for e\_h in head\_list:

if e\_h not in G.nodes:

continue

for e\_t in G[e\_h]:

if \_r in G[e\_h][e\_t][’id’]:

if random.random() < mcmc:

next\_head\_list.append(e\_t)

head\_list = next\_head\_list

for t in head\_list:

if (h, r, t) not in all\_deductibles:

all\_deductibles[(h, r, t)] = [rule]

elif rule not in all\_deductibles[(h, r, t)]:

all\_deductibles[(h, r, t)].append(rule)

if not G.has\_edge(h, t) or r not in G[h][t][’id’]:

print(’add deductible edge’)

add\_edge(G, h, t, r)

num\_triples += 1

repeated\_entities[r].append(t)

repeated\_entities[’-’ + r].append(h)

atomic\_triples = []

deductible\_triples = []

for h, t in G.edges:

for r in G[h][t][’id’]:

if (h, r, t) not in all\_deductibles:

atomic\_triples.append((h, r, t))

else:

deductible\_triples.append((h, r, t))

random.shuffle(atomic\_triples)

random.shuffle(deductible\_triples)

assert len(atomic\_triples) >= int(num\_train \* (1-deductible\_ratio))

assert len(deductible\_triples) >= int(num\_train \* deductible\_ratio) + 2 \* num\_test

remove\_triples = []

train\_atomic\_triples = atomic\_triples[:int(num\_train \* (1-deductible\_ratio))]

remove\_triples += atomic\_triples[int(num\_train \* (1-deductible\_ratio)):]

train\_deductible\_triples = deductible\_triples[:int(num\_train \* deductible\_ratio)]

remove\_triples += deductible\_triples[int(num\_train \* deductible\_ratio):]

for h, r, t in remove\_triples:

\_t = t

rs = G[h][\_t][’id’]

if r in rs:

if len(rs) == 1:

G.remove\_edge(h, \_t)

else:

G[h][\_t][’id’].remove(r)

train\_triples = train\_deductible\_triples + train\_atomic\_triples

random.shuffle(train\_triples)

print(”num train triples: ”, len(train\_triples))

r2rule = {}

for rule in deductible\_rules:

if rule[0] in r2rule:

r2rule[rule[0]].append(rule[1:])

else:

r2rule[rule[0]] = [rule[1:]]

def check\_deductible(triple):

h, r, t = triple

alt\_ts = []

for rule in r2rule[r]:

head\_list = [h]

for \_r in rule:

next\_head\_list = []

for e\_h in head\_list:

for e\_t in G[e\_h]:

if \_r in G[e\_h][e\_t][’id’]:

next\_head\_list.append(e\_t)

head\_list = next\_head\_list

alt\_ts += head\_list

if t in alt\_ts:

return True

return False

id\_test\_triples = []

for i in range(int(num\_train \* deductible\_ratio), len(deductible\_triples)):

if check\_deductible(deductible\_triples[i]):

id\_test\_triples.append(deductible\_triples[i])

if len(id\_test\_triples) == num\_test:

break

id\_test\_rules = [all\_deductibles[triple] for triple in id\_test\_triples]

print(”num id test triples: ”, len(id\_test\_triples))

rule2triples = defaultdict(list)

for triple in deductible\_triples[i+1:]:

for rule in all\_deductibles[triple]:

rule2triples[rule].append(triple)

# uniformly sample testing triples from each rule

uniform\_test\_triples = []

for rule in rule2triples:

triples = []

for triple in rule2triples[rule]:

if check\_deductible(triple):

triples.append(triple)

if len(triples) > num\_test//len(rule2triples):

uniform\_test\_triples += random.sample(triples, num\_test//len(rule2triples))

else:

uniform\_test\_triples += triples

random.shuffle(uniform\_test\_triples)

uniform\_test\_rules = [all\_deductibles[triple] for triple in uniform\_test\_triples]

print(”num uniform test triples: ”, len(uniform\_test\_triples))

return G, deductible\_rules, train\_triples, id\_test\_triples, id\_test\_rules, uniform\_test\_triples, uniform\_test\_rules

[◄](/html/2504.03634)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2504.03635)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2504.03635)
[View original  
on arXiv](https://arxiv.org/abs/2504.03635)[►](/html/2504.03636)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon May 5 16:58:47 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
