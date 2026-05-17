---
arxiv: '2305.18446'
authors:
- Kuan-Yu Chen
- Ping-Han Chiang
- Hsin-Rung Chou
- Ting-Wei Chen
- Tien-Hao Chang
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'Trompt: Towards a Better Deep Neural Network for Tabular Data'
url: http://arxiv.org/abs/2305.18446v2
year: 2023
---

# Trompt: Towards a Better Deep Neural Network for Tabular Data

Kuan-Yu Chen
  
Ping-Han Chiang
  
Hsin-Rung Chou
  
Ting-Wei Chen
  
Darby Tien-Hao Chang

###### Abstract

Tabular data is arguably one of the most commonly used data structures in various practical domains, including finance, healthcare and e-commerce.
However, based on a recently published tabular benchmark, we can see deep neural networks still fall behind tree-based models on tabular datasets (Grinsztajn et al., [2022](#bib.bib17)).
In this paper, we propose *Trompt*–which stands for Tabular Prompt–a novel architecture inspired by prompt learning of language models.
The essence of prompt learning is to adjust a large pre-trained model through a set of prompts outside the model without directly modifying the model.
Based on this idea, Trompt separates the learning strategy of tabular data into two parts for the intrinsic information of a table and the varied information among samples.
Trompt is evaluated with the benchmark mentioned above.
The experimental results demonstrate that Trompt outperforms state-of-the-art deep neural networks and is comparable to tree-based models ([Figure 1](#S0.F1 "In Trompt: Towards a Better Deep Neural Network for Tabular Data")).

Tabular Data, Prompt Learning, Model Architecture Design

!(/html/2305.18446/assets/figures/benchmark_classif_medium.jpg)

(a) Medium-sized classification task.

!(/html/2305.18446/assets/figures/benchmark_regression_medium.jpg)

(b) Medium-sized regression task.

!(/html/2305.18446/assets/figures/benchmark_classif_large.jpg)

(c) Large-sized classification task.

!(/html/2305.18446/assets/figures/benchmark_regression_large.jpg)

(d) Large-sized regression task.

Figure 1: Benchmark results.

## 1 Introduction

Tabular data plays a vital role in many real world applications, such as financial statements for banks to evaluate the credibility of a company, diagnostic reports for doctors to identify the aetiology of a patient, and customer records for e-commerce platforms to discover the potential interest of a customer.
In general, tabular data can be used to record activities consisting of heterogeneous features and has many practical usages.

On the other hand, deep learning has achieved a great success in various domains, including computer vision, natural language processing (NLP) and robotics (He et al., [2016](#bib.bib19); Redmon et al., [2016](#bib.bib30); Gu et al., [2017](#bib.bib18); Devlin et al., [2018](#bib.bib11)).
Besides extraordinary performance, there are numerous benefits of the end-to-end optimization nature of deep learning, including (i) online learning with streaming data (Sahoo et al., [2017](#bib.bib32)), (ii) multi-model integration that incorporates different types of input, e.g., image and text (Ramachandram & Taylor, [2017](#bib.bib29)) and (iii) representation learning that realizes semi-supervised learning and generative modeling (Van Engelen & Hoos, [2020](#bib.bib37); Goodfellow et al., [2020](#bib.bib15)).

Consequently, researchers have been dedicated to apply deep learning on tabular data, either through (i) transformer (Huang et al., [2020](#bib.bib20); Somepalli et al., [2021](#bib.bib36); Gorishniy et al., [2021](#bib.bib16)) or (ii) inductive bias investigation (Katzir et al., [2020](#bib.bib21); Arik & Pfister, [2021](#bib.bib1)).

Though many of the previous publications claimed that they have achieved the state of the art, further researches pointed that previous works were evaluated on favorable datasets and tree-based models still show superior performances in the realm of tabular data (Borisov et al., [2021](#bib.bib4); Gorishniy et al., [2021](#bib.bib16); Shwartz-Ziv & Armon, [2022](#bib.bib35)).
For a fair comparison between different algorithms, a standard benchmark for tabular data was proposed by (Grinsztajn et al., [2022](#bib.bib17)).
The benchmark, denoted as *Grinsztajn45* in this work, consists of 45 curated datasets from various domains.

In this paper, we propose a novel prompt-inspired architecture, *Trompt*, which abbreviates Tabular Prompt.
Prompt learning has played an important role in the recent development of language models.
For example, GPT-3 can well handle a wide range of tasks with an appropriate prompt engineering (Radford et al., [2018](#bib.bib28); Brown et al., [2020](#bib.bib6)).
In Trompt, prompt is utilized to derive feature importances that vary in different samples.
Trompt consists of multiple *Trompt Cell*s and a shared *Trompt Downstream* as [Figure 2](#S2.F2 "In 2.2 Tabular Neural Network ‣ 2 Related Work ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Each Trompt Cell is responsible for feature extraction, while the Trompt Downstream is for prediction.

The performance of Trompt is evaluated on the Grinsztajn45 benchmark and compared with three deep learning models and five tree-based models.
[Figure 1](#S0.F1 "In Trompt: Towards a Better Deep Neural Network for Tabular Data") illustrates the overall evaluation results on Grinsztajn45.
The x-axis is the number of hyperparameter search iterations and y-axis is the normalized performance.
In [Figure 1](#S0.F1 "In Trompt: Towards a Better Deep Neural Network for Tabular Data"), Trompt is consistently better than state-of-the-art deep learning models (SAINT and FT-Transformer) and the gap between deep learning models and tree-based models is narrowed.

Our key contributions are summarized as follows:

* •

  The experiments are conducted on a recognized tabular benchmark, Grinsztajn45.
  Additionally, we add two well-performed tree-based models, LightGBM (Ke et al., [2017](#bib.bib22)) and CatBoost (Prokhorenkova et al., [2018](#bib.bib27)) to baselines.
* •

  Trompt achieves state-of-the-art performance among deep learning models and narrows the performance gap between deep learning models and tree-based models.
* •

  Thorough empirical studies and ablation tests were conducted to verify the design of Trompt.
  The results further shed light on future research directions of the architecture design of tabular neural network.

## 2 Related Work

In this section, we first discuss the prompt learning of language models.
Secondly, we discuss two research branches of tabular neural networks, transformer and inductive bias investigation.
Lastly, we discuss the differences between Trompt and the related works and highlight the uniqueness of our work.

### 2.1 Prompt Learning

The purpose of prompt learning is to transform the input and output of downstream tasks to the original task used to build a pre-trained model.
Unlike fine-tuning that changes the task and usually involves updating model weights, a pre-train model with prompts can dedicate itself to one task.
With prompt learning, a small amount of data or even zero-shot can achieve good results (Radford et al., [2018](#bib.bib28); Brown et al., [2020](#bib.bib6)).
The emergence of prompt learning substantially improves the application versatility of pre-trained models that are too large for common users to fine-tune.

To prompt a language model, one can insert a task-specific prompt before a sentence and hint the model to adjust its responses for different tasks (Brown et al., [2020](#bib.bib6)).
Prompts can either be discrete or soft.
The former are composed of discrete tokens from the vocabulary of natural languages (Radford et al., [2018](#bib.bib28); Brown et al., [2020](#bib.bib6)), while the latter are learned representations (Li & Liang, [2021](#bib.bib26); Lester et al., [2021](#bib.bib25)).

### 2.2 Tabular Neural Network

Transformer. Self-attention has revolutionized NLP since 2017 (Vaswani et al., [2017](#bib.bib39)), and soon been adopted by other domains, such as computer vision, reinforcement learning and speech recognition (Dosovitskiy et al., [2020](#bib.bib12); Chen et al., [2021](#bib.bib8); Zhang et al., [2020](#bib.bib40)).
The intention of transformer blocks is to capture the relationships among features, which can be applied on tabular data as well.

TabTransformer (Huang et al., [2020](#bib.bib20)) is the first transformer-based tabular neural network.
However, TabTransformer only fed categorical features to transformer blocks and ignored the potential relationships among categorical and numerical features.
FT-Transformer (Gorishniy et al., [2021](#bib.bib16)) fixed this issue through feeding both categorical and numerical features to transformer blocks.
SAINT (Somepalli et al., [2021](#bib.bib36)) further improved FT-Transformer through applying attentions on not only the feature dimensions but also the sample dimensions.

!(/html/2305.18446/assets/x1.png)

Figure 2: Overall architecture of the proposed Trompt.

Inductive Bias Investigation. Deep neural networks perform well on tasks with clear inductive bias.
For example, Convolutional Neural Network (CNN) works well on images. The kernel of CNN is designed to capture local patterns since neighboring pixels usually relate to each other (LeCun et al., [1995](#bib.bib23)).
Recurrent Neural Networks (RNN) is widely used in language understanding because the causal relationship among words is well encapsulated through recurrent units (Rumelhart et al., [1986](#bib.bib31)).
However, unlike other popular tasks, the inductive bias of tabular data has not been well discovered.

Given the fact that tree-based model has been the solid state of the art for tabular data (Borisov et al., [2021](#bib.bib4); Gorishniy et al., [2021](#bib.bib16); Shwartz-Ziv & Armon, [2022](#bib.bib35)), Net-DNF (Katzir et al., [2020](#bib.bib21)) and TabNet (Arik & Pfister, [2021](#bib.bib1)) hypothesized that the inductive bias for tabular data might be the learning strategy of tree-based model.
The strategy is to find the optimal root-to-leaf decision paths by selecting a portion of the features and deriving the optimal split from the selected features in non-leaf nodes.
To emulate the learning strategy, TabNet utilized sequential attention and sparsity regularization.
On the other hand, Net-DNF theoretically proved that decision tree is equivalent to some disjunctive normal form (DNF) and proposed disjunctive neural normal form to emulate a DNF formula.

### 2.3 The Uniqueness of Trompt

We argue that the column importances of tabular data are not invariant for all samples and can be grouped into multiple modalities.
Since prompt learning is born to adapt a model to multiple tasks, the concept is used in Trompt to handle multiple modalities.
To this end, Trompt separates the learning strategy of tabular data into two parts.
The first part, analogous to pre-trained models, focus on learning the intrinsic column information of a table.
The second part, analogous to prompts, focus on diversifying the feature importances of different samples.

As far as our understanding, Trompt is the first prompt-inspired tabular neural network.
Compared to transformer-based models, Trompt learns separated column importances instead of focusing on the interactions among columns.
Compared to TabNet and Net-DNF, Trompt handle multiple modalities by emulating prompt learning instead of the branch split of decision tree.

!(/html/2305.18446/assets/x2.png)

Figure 3: Architecture of a Trompt Cell.

## 3 Trompt

In this section, we elaborate on the architecture design of Trompt.
As [Figure 2](#S2.F2 "In 2.2 Tabular Neural Network ‣ 2 Related Work ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") shows, Trompt consists of multiple Trompt Cells and a shared Trompt Downstream.
Each Trompt Cell is responsible for feature extraction and providing diverse representations, while the Trompt Downstream is for prediction.
The details of Trompt Cell and Trompt Downstream are discussed in [Section 3.1](#S3.SS1 "3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [Section 3.2](#S3.SS2 "3.2 Trompt Downstream ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), respectively.
In [Section 3.3](#S3.SS3 "3.3 Prompt Learning of Trompt ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), we further discuss the prompt learning of Trompt.

### 3.1 Trompt Cell

[Figure 3](#S2.F3 "In 2.3 The Uniqueness of Trompt ‣ 2 Related Work ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") illustrates the architecture of a Trompt Cell, which can be divided into three parts.
The first part derives feature importances (𝐌importancesubscript𝐌importance\mathbf{M}\_{\text{importance}}) based on column embeddings (𝐄columnsubscript𝐄column\mathbf{E}\_{\text{column}}), the previous cell’s output (𝐎prevsubscript𝐎prev\mathbf{O}\_{\text{prev}}) and prompt embeddings (𝐄promptsubscript𝐄prompt\mathbf{E}\_{\text{prompt}}).
The second part transforms the input into feature embeddings (𝐄featuresubscript𝐄feature\mathbf{E}\_{\text{feature}}) with two paths for categorical and numerical columns, respectively.
The third part expands 𝐄featuresubscript𝐄feature\mathbf{E}\_{\text{feature}} for the later multiplication.

The details of the first part are illustrated in [Section 3.1.1](#S3.SS1.SSS1 "3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and the details of the second and third parts are illustrated in [Section 3.1.2](#S3.SS1.SSS2 "3.1.2 Construct and Expand Feature Embeddings ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Lastly, the generation of the output of a Trompt Cell is illustrated in [Section 3.1.3](#S3.SS1.SSS3 "3.1.3 Generate Output ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

#### 3.1.1 Derive Feature Importances

Let 𝐄column∈ℝC×dsubscript𝐄columnsuperscriptℝ𝐶𝑑\mathbf{E}\_{\text{column}}\in\mathbb{R}^{C\times{d}} be column embeddings and 𝐄prompt∈ℝP×dsubscript𝐄promptsuperscriptℝ𝑃𝑑\mathbf{E}\_{\text{prompt}}\in\mathbb{R}^{P\times{d}} be prompt embeddings.
C𝐶C is the number of columns of a table defined by the dataset, while P𝑃P and d𝑑d are hyperparameters for the number of prompts and the hidden dimension, respectively.
Both 𝐄columnsubscript𝐄column\mathbf{E}\_{\text{column}} and 𝐄promptsubscript𝐄prompt\mathbf{E}\_{\text{prompt}} are input independent and trainable.
Let 𝐎prev∈ℝB×P×dsubscript𝐎prevsuperscriptℝ𝐵𝑃𝑑\mathbf{O}\_{\text{prev}}\in\mathbb{R}^{B\times{P}\times{d}} be the previous cell’s output and B𝐵B be the batch size.

𝐎prevsubscript𝐎prev\mathbf{O}\_{\text{prev}} is fused with the prompt embeddings as [Equations 1](#S3.E1 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [2](#S3.E2 "Equation 2 ‣ 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Since 𝐄promptsubscript𝐄prompt\mathbf{E}\_{\text{prompt}} is input independent and lack a batch dimension, 𝐄promptsubscript𝐄prompt\mathbf{E}\_{\text{prompt}} is expanded to 𝐒𝐄promptsubscript𝐒𝐄prompt\mathbf{SE}\_{\text{prompt}} through the stack operation as [Equation 1](#S3.E1 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Later, we concatenate 𝐒𝐄promptsubscript𝐒𝐄prompt\mathbf{SE}\_{\text{prompt}} and 𝐎prevsubscript𝐎prev\mathbf{O}\_{\text{prev}} and then reduce the dimension of the concatenated tensor back to ℝB×P×dsuperscriptℝ𝐵𝑃𝑑\mathbb{R}^{B\times{P}\times{d}} for the final addition as [Equation 2](#S3.E2 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

For the same reason as 𝐄promptsubscript𝐄prompt\mathbf{E}\_{\text{prompt}}, the 𝐄columnsubscript𝐄column\mathbf{E}\_{\text{column}} is expanded to 𝐒𝐄columnsubscript𝐒𝐄column\mathbf{SE}\_{\text{column}} as [Equation 3](#S3.E3 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Subsequently, feature importances are derived through [Equation 4](#S3.E4 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), where ⊗tensor-product\otimes is the batch matrix multiplication, ⊺⊺\intercal is the batch transpose, and the 𝚜𝚘𝚏𝚝𝚖𝚊𝚡𝚜𝚘𝚏𝚝𝚖𝚊𝚡\mathtt{softmax} is applied to the column axis.

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐒𝐄prompt=𝚜𝚝𝚊𝚌𝚔​(𝐄prompt)∈ℝB×P×dsubscript𝐒𝐄prompt𝚜𝚝𝚊𝚌𝚔subscript𝐄promptsuperscriptℝ𝐵𝑃𝑑\mathbf{SE}\_{\text{prompt}}=\mathtt{stack}(\mathbf{E}\_{\text{prompt}})\in\mathbb{R}^{B\times{P}\times{d}} |  | (1) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐒𝐄^prompt=𝚍𝚎𝚗𝚜𝚎​(𝚌𝚘𝚗𝚌𝚊𝚝​(𝐒𝐄prompt,𝐎prev))+𝐒𝐄prompt+𝐎prev∈ℝB×P×dsubscript^𝐒𝐄prompt𝚍𝚎𝚗𝚜𝚎𝚌𝚘𝚗𝚌𝚊𝚝subscript𝐒𝐄promptsubscript𝐎prevsubscript𝐒𝐄promptsubscript𝐎prevsuperscriptℝ𝐵𝑃𝑑\begin{split}\mathbf{\hat{SE}}\_{\text{prompt}}=&\mathtt{dense}(\mathtt{concat}(\mathbf{SE}\_{\text{prompt}},\mathbf{O}\_{\text{prev}}))\\ &+\mathbf{SE}\_{\text{prompt}}\\ &+\mathbf{O}\_{\text{prev}}\\ &\in\mathbb{R}^{B\times{P}\times{d}}\end{split} |  | (2) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐒𝐄column=𝚜𝚝𝚊𝚌𝚔​(𝐄column)∈ℝB×C×dsubscript𝐒𝐄column𝚜𝚝𝚊𝚌𝚔subscript𝐄columnsuperscriptℝ𝐵𝐶𝑑\mathbf{SE}\_{\text{column}}=\mathtt{stack}(\mathbf{E}\_{\text{column}})\in\mathbb{R}^{B\times{C}\times{d}} |  | (3) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐌importance=𝚜𝚘𝚏𝚝𝚖𝚊𝚡​(𝐒𝐄^prompt⊗𝐒𝐄column⊺)∈ℝB×P×Csubscript𝐌importance𝚜𝚘𝚏𝚝𝚖𝚊𝚡tensor-productsubscript^𝐒𝐄promptsuperscriptsubscript𝐒𝐄column⊺superscriptℝ𝐵𝑃𝐶\mathbf{M}\_{\text{importance}}=\mathtt{softmax}(\mathbf{\hat{SE}}\_{\text{prompt}}\otimes\mathbf{SE}\_{\text{column}}^{\intercal})\in\mathbb{R}^{B\times{P}\times{C}} |  | (4) |

The output of the first part is 𝐌importance∈ℝB×P×Csubscript𝐌importancesuperscriptℝ𝐵𝑃𝐶\mathbf{M}\_{\text{importance}}\in\mathbb{R}^{B\times{P}\times{C}}, which accommodates the feature importances yielded by P𝑃P prompts.
Notice that the column embeddings are not connected to the input and the prompt embeddings are fused with the previous cell’s output.
In [Section 3.3](#S3.SS3 "3.3 Prompt Learning of Trompt ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), we further discuss these designs and their connections to the prompt learning of NLP.

#### 3.1.2 Construct and Expand Feature Embeddings

In Trompt, categorical features are embedded through a embedding layer and numerical features are embedded through a dense layer as previous works (Somepalli et al., [2021](#bib.bib36); Gorishniy et al., [2021](#bib.bib16)).
The embedding construction procedure is illustrated in part two of [Figure 3](#S2.F3 "In 2.3 The Uniqueness of Trompt ‣ 2 Related Work ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), where 𝐄feature∈ℝB×C×dsubscript𝐄featuresuperscriptℝ𝐵𝐶𝑑\mathbf{E}\_{\text{feature}}\in\mathbb{R}^{B\times{C}\times{d}} is the feature embeddings of the batch.

The shapes of 𝐌importancesubscript𝐌importance\mathbf{M}\_{\text{importance}} and 𝐄featuresubscript𝐄feature\mathbf{E}\_{\text{feature}} are ℝB×P×Csuperscriptℝ𝐵𝑃𝐶\mathbb{R}^{B\times{P}\times{C}} and ℝB×C×dsuperscriptℝ𝐵𝐶𝑑\mathbb{R}^{B\times{C}\times{d}}, respectively.
Since 𝐄featuresubscript𝐄feature\mathbf{E}\_{\text{feature}} lacks the prompt dimension, Trompt expands 𝐄featuresubscript𝐄feature\mathbf{E}\_{\text{feature}} into 𝐄^feature∈ℝB×P×C×dsubscript^𝐄featuresuperscriptℝ𝐵𝑃𝐶𝑑\mathbf{\hat{E}}\_{\text{feature}}\in\mathbb{R}^{B\times{P}\times{C}\times{d}} to accommodate the P𝑃P prompts by a dense layer in part three of [Figure 3](#S2.F3 "In 2.3 The Uniqueness of Trompt ‣ 2 Related Work ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

#### 3.1.3 Generate Output

The output of Trompt Cell is the column-wise sum of the element-wise multiplication of 𝐄^featuresubscript^𝐄feature\mathbf{\hat{E}}\_{\text{feature}} and 𝐌importancesubscript𝐌importance\mathbf{M}\_{\text{importance}} as [Equation 5](#S3.E5 "In 3.1.3 Generate Output ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), where ⊙direct-product\odot is element-wise multiplication.
Notice that, during element-wise multiplication, the shape of 𝐌importancesubscript𝐌importance\mathbf{M}\_{\text{importance}} is considered ℝB×P×C×1superscriptℝ𝐵𝑃𝐶1\mathbb{R}^{B\times{P}\times{C}\times{1}}.
In addition, since column is the third axis, the shape is reduced from ℝB×P×C×dsuperscriptℝ𝐵𝑃𝐶𝑑\mathbb{R}^{B\times{P}\times{C}\times{d}} to ℝB×P×dsuperscriptℝ𝐵𝑃𝑑\mathbb{R}^{B\times{P}\times{d}} after column-wise summation.

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐎=∑i=1C(𝐄^feature⊙𝐌importance):,:,i,:∈ℝB×P×d𝐎superscriptsubscript𝑖1𝐶subscriptdirect-productsubscript^𝐄featuresubscript𝐌importance  ::𝑖:superscriptℝ𝐵𝑃𝑑\mathbf{O}=\sum\_{i=1}^{C}({\mathbf{\hat{E}}\_{\text{feature}}\odot\mathbf{M}\_{\text{importance}})\_{:,:,i,:}}\in\mathbb{R}^{B\times{P}\times{d}} |  | (5) |

!(/html/2305.18446/assets/x3.png)

Figure 4: Architecture of a Trompt Downstream.

### 3.2 Trompt Downstream

A Trompt Downstream makes a prediction based on a Trompt Cell’s output, which contains representations corresponding to P𝑃P prompt embeddings.
To aggregate these representations, the weight for each prompt is first derived through a dense layer and a softmax activation function as [Equation 6](#S3.E6 "In 3.2 Trompt Downstream ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Afterwards, the weighted sum is calculated as [Equation 7](#S3.E7 "In 3.2 Trompt Downstream ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

The prediction is subsequently made through two dense layers as [Equation 8](#S3.E8 "In 3.2 Trompt Downstream ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), where T𝑇T is the target dimension.
For classification tasks, T𝑇T is the number of target classes.
For regression tasks, T𝑇T is set to 1.
As [Figure 2](#S2.F2 "In 2.2 Tabular Neural Network ‣ 2 Related Work ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") shows, a sample gets a prediction through a Trompt Cell and thus multiple predictions through all cells.
During training, the loss of each prediction is separately calculated and the loss is summed up to update model weights.
During inference, on the other hand, predictions through all cells are simply averaged as the final prediction as [Equation 9](#S3.E9 "In 3.2 Trompt Downstream ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), where L𝐿L is the number of Trompt Cells.

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐖𝚙𝚛𝚘𝚖𝚙𝚝=𝚜𝚘𝚏𝚝𝚖𝚊𝚡​(𝚍𝚎𝚗𝚜𝚎​(𝐎))∈ℝB×Psubscript𝐖𝚙𝚛𝚘𝚖𝚙𝚝𝚜𝚘𝚏𝚝𝚖𝚊𝚡𝚍𝚎𝚗𝚜𝚎𝐎superscriptℝ𝐵𝑃\mathbf{W}\_{\mathtt{prompt}}=\mathtt{softmax}(\mathtt{dense}(\mathbf{O}))\in\mathbb{R}^{B\times{P}} |  | (6) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐎^=∑i=1P(𝐖prompt⊙𝐎):,i,:∈ℝB×d^𝐎superscriptsubscript𝑖1𝑃subscriptdirect-productsubscript𝐖prompt𝐎  :𝑖:superscriptℝ𝐵𝑑\mathbf{\hat{O}}=\sum\_{i=1}^{P}(\mathbf{W}\_{\text{prompt}}\odot\mathbf{O})\_{:,i,:}\in\mathbb{R}^{B\times{d}} |  | (7) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐏=𝚍𝚎𝚗𝚜𝚎​(𝚛𝚎𝚕𝚞​(𝚍𝚎𝚗𝚜𝚎​(𝐎^)))∈ℝB×T𝐏𝚍𝚎𝚗𝚜𝚎𝚛𝚎𝚕𝚞𝚍𝚎𝚗𝚜𝚎^𝐎superscriptℝ𝐵𝑇\mathbf{P}=\mathtt{dense}(\mathtt{relu}(\mathtt{dense}(\mathbf{\hat{O}})))\in\mathbb{R}^{B\times{T}} |  | (8) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | l​o​s​s=∑i=1L𝚕𝚘𝚜𝚜​\_​𝚏𝚗​(𝐏i,y)p​r​e​d=∑i=1L𝐏i/L𝑙𝑜𝑠𝑠superscriptsubscript𝑖1𝐿𝚕𝚘𝚜𝚜\_𝚏𝚗subscript𝐏𝑖𝑦𝑝𝑟𝑒𝑑superscriptsubscript𝑖1𝐿subscript𝐏𝑖𝐿\begin{split}&loss=\sum\_{i=1}^{L}{\mathtt{loss\char 95\relax fn}(\mathbf{P}\_{i}},y)\\ &pred=\sum\_{i=1}^{L}{\mathbf{P}\_{i}}/L\end{split} |  | (9) |

Table 1: Analogy of the prompt learning of Trompt to that of NLP.

  

| Problem  Identification | Implemented by | Inspired by |
| --- | --- | --- |
| Sample-invariant  Intrinsic Properties | 𝐄columnsubscript𝐄column\mathbf{E}\_{\text{column}} | Fixed Large  Language Model |
| Sample-specific  Feature Importances | 𝐌importancesubscript𝐌importance\mathbf{M}\_{\text{importance}} | Task-specific  Predictions |

### 3.3 Prompt Learning of Trompt

Trompt’s architecture is specifically designed for tabular data, taking into account the unique characteristics of this type of data and the impressive performance of tree-based models.
Unlike conventional operations, the design may appear unconventional and detached from tabular data features.
In this section, we explain the rationale behind Trompt’s network design and how we adapted prompt learning to a tabular neural network.

Tabular data is structured, with each column representing a specific dataset property that remains constant across individual samples.
The success of tree-based models relies on assigning feature importances to individual samples.
This concept has been explored in models such as TabNet (Arik & Pfister, [2021](#bib.bib1)) and Net-DNF (Katzir et al., [2020](#bib.bib21)).
However, tree-based algorithms do not explicitly assign feature importances to individual samples.
Instead, importances vary implicitly along the path from the root to a leaf node.
Only the columns involved in this path are considered important features for the samples reaching the corresponding leaf node, representing sample-specific feature importances.

Given the fundamental characteristic of tabular data and the learning strategy of tree-based models, Trompt aims to combine the intrinsic properties of columns with sample-specific feature importances using a prompt learning-inspired architecture from NLP (Radford et al., [2018](#bib.bib28); Brown et al., [2020](#bib.bib6)).
Trompt employs column embeddings to represent the intrinsic properties of each column and prompt embeddings to prompt column embeddings, generating feature importances for given prompts.
Both column embeddings and prompt embeddings are invariant across samples.
However, before prompting column embeddings with prompt embeddings, the prompt embeddings are fused with the output of the previous Trompt Cell as shown in [Equation 2](#S3.E2 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), enabling input-related representations to flow through and derive sample-specific feature importances.
The ”prompt” mechanism in Trompt is implemented as a matrix multiplication in [Equation 4](#S3.E4 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

A conceptual analogy of Trompt’s prompt learning approach to NLP is presented in [Table 1](#S3.T1 "In 3.2 Trompt Downstream ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
It’s important to note that the implementation details of prompt learning differ substantially between tabular data and NLP tasks due to the fundamental differences between the two domains.
Therefore, appropriate adjustments must be made to bridge these two domains.

## 4 Experiments

In this section, the experimental results and analyses are presented.
First, we elaborate on the settings of experiments and the configurations of Trompt in [Section 4.1](#S4.SS1 "4.1 Setup ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Second, the performance of Trompt on Grinsztajn45 is reported in [Section 4.2](#S4.SS2 "4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Third, ablation studies regarding the hyperparameters and the architecture of Trompt are studied in [Section 4.3](#S4.SS3 "4.3 Ablation Study ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Lastly, the interpretability of Trompt is investigated using synthetic and real-world datasets in [Section 4.4](#S4.SS4 "4.4 Interpretability ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

### 4.1 Setup

The performance and ablation study of Trompt primarily focus on the Grinsztajn45 benchmark (Grinsztajn et al., [2022](#bib.bib17)) 111<https://github.com/LeoGrin/tabular-benchmark>. This benchmark comprises datasets from various domains and follows a unified methodology for evaluating different models, providing a fair and comprehensive assessment.
Furthermore, we evaluate the performance of Trompt on datasets selected by FT-Transformer and SAINT to compare it with state-of-the-art tabular neural networks.

For interpretability analysis, we follow the experimental settings of TabNet (Arik & Pfister, [2021](#bib.bib1)).
This involves using two synthetic datasets (Syn2 and Syn4) and a real-world dataset (mushroom) to visualize attention masks.

The settings of Grinsztajn45 are presented in [Section 4.1.1](#S4.SS1.SSS1 "4.1.1 Settings of Grinsztajn45 ‣ 4.1 Setup ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and the implementation details of Trompt are presented in [Section 4.1.2](#S4.SS1.SSS2 "4.1.2 Implementation Details ‣ 4.1 Setup ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Furthermore, the settings of datasets chosen by FT-Transformer and SAINT are provided in [Section B.2](#A2.SS2 "B.2 Datasets chosen by FT-Transformer ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [Section B.3](#A2.SS3 "B.3 Datasets chosen by SAINT ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), respectively.

#### 4.1.1 Settings of Grinsztajn45

To fairly evaluate the performance, we follow the configurations of Grinsztajn45, including train test data split, data preprocessing and evaluation metric.
Grinsztajn45 comprises two kinds of tasks, classification tasks and regression tasks.
Please see [Section A.1](#A1.SS1 "A.1 Dataset Selection Criteria ‣ Appendix A Settings of Grinsztajn45 ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [Section A.2](#A1.SS2 "A.2 Dataset Normalization ‣ Appendix A Settings of Grinsztajn45 ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for the dataset selection criteria and dataset normalization process of Grinsztajn45.
The tasks are further grouped according to (i) the size of datasets (medium-sized and large-sized) and (ii) the inclusion of categorical features (numerical only and heterogeneous).

In addition, we make the following adjustments: (i) models with incomplete experimental results in (Grinsztajn et al., [2022](#bib.bib17)) are omitted, (ii) two well-performed tree-based models are added for comparison, and (iii) Trompt used a hyperparameter search space smaller than its opponents.
The details of the adjustments are described in [Section A.3](#A1.SS3 "A.3 Baseline Models ‣ Appendix A Settings of Grinsztajn45 ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [Section A.4](#A1.SS4 "A.4 Hyperparameter Search Mechanism ‣ Appendix A Settings of Grinsztajn45 ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

#### 4.1.2 Implementation Details

Trompt is implemented using PyTorch.
The default hyperparameters are shown in [Table 2](#S4.T2 "In 4.1.2 Implementation Details ‣ 4.1 Setup ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
The size of embeddings and the hidden dimension of dense layers are configured d𝑑d.
Note that only the size of column and prompt embeddings must be the same by the architecture design.
The hidden dimension of dense layers is set as d𝑑d to reduce hyperparameters and save computing resources.
On the other hand, the number of prompts and the number of Trompt Cells are set to P𝑃P and L𝐿L.
Please refer to [Appendix F](#A6 "Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for the hyperparameter search spaces for all baselines and Trompt.

Table 2: Default hyperparameters of Trompt.

  

| Hyperparameter | Symbol | Value |
| --- | --- | --- |
| Feature Embeddings  Prompt/Column Embeddings  Hidden Dimension | d𝑑d | 128 |
| Prompts | P𝑃P | 128 |
| Layer | L𝐿L | 6 |

### 4.2 Evaluation Results

The results of classification tasks are discussed in [Section 4.2.1](#S4.SS2.SSS1 "4.2.1 Classification ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and the results of regression tasks are discussed in [Section 4.2.2](#S4.SS2.SSS2 "4.2.2 Regression ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
The evaluation metrics are accuracy and r2-score for classification and regression tasks, respectively.
In this section, we report an overall result and leave results of individual datasets in [Section B.1](#A2.SS1 "B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
In addition, the evaluation results on datasets chosen by FT-Transformer and SAINT are provided in [Section B.2](#A2.SS2 "B.2 Datasets chosen by FT-Transformer ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [Section B.3](#A2.SS3 "B.3 Datasets chosen by SAINT ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), respectively.

#### 4.2.1 Classification

On the medium-sized classification tasks, [Figure 5](#S4.F5 "In 4.2.1 Classification ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") shows that Trompt outperforms DNN models.
The curve of Trompt is consistently above deep neural networks (SAINT, FT-Transformer and ResNet) on tasks with and without categorical features.
Additionally, Trompt narrows the gap between deep neural networks and tree-based models, especially on the tasks with heterogeneous features.
In [Figure 5(b)](#S4.F5.sf2 "In Figure 5 ‣ 4.2.1 Classification ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), Trompt seems to be a member of the leading cluster with four tree-based models. The GradientBoostingTree starts slow but catches up the leading cluster in the end of search. The other deep neural networks forms the second cluster and have a gap to the leading one.

On the large-sized classification tasks, tree-based models remain the leading positions but the gap to deep neural networks is obscure.
This echoes that deep neural networks requires more samples for training (LeCun et al., [2015](#bib.bib24)).
[Figure 6(a)](#S4.F6.sf1 "In Figure 6 ‣ 4.2.1 Classification ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") shows that Trompt outperforms ALL models on the task with numerical features and [Figure 6(b)](#S4.F6.sf2 "In Figure 6 ‣ 4.2.1 Classification ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") shows that Trompt achieves a comparable performance to FT-Transformer on tasks with heterogeneous features.

With the small hyperparameter search space, the curve of Trompt is relatively flat.
The flat curve also suggests that Trompt performs well with its default hyperparameters.
Its performance after an exhausted search is worthy of future exploring.

!(/html/2305.18446/assets/figures/numerical_classif.jpg)

(a) Numerical features only.

!(/html/2305.18446/assets/figures/categorical_classif.jpg)

(b) Heterogeneous features.

Figure 5: Benchmark on medium-sized classification datasets.

!(/html/2305.18446/assets/figures/numerical_classif-large.jpg)

(a) Numerical features only.

!(/html/2305.18446/assets/figures/categorical_classif-large.jpg)

(b) Heterogeneous features.

Figure 6: Benchmark on large-sized classification datasets.

#### 4.2.2 Regression

On the medium-sized regression tasks, [Figure 7](#S4.F7 "In 4.2.2 Regression ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") shows that Trompt outperforms deep neural networks as the curves of Trompt are consistently higher than SAINT, FT-Transformer and ResNet on tasks with and without categorical features.
The gap between deep neural networks and tree-based models is less obvious in [Figure 7(a)](#S4.F7.sf1 "In Figure 7 ‣ 4.2.2 Regression ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") than that in [Figure 7(b)](#S4.F7.sf2 "In Figure 7 ‣ 4.2.2 Regression ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
On the tasks with numerical features only, Trompt achieves a comparable performance with random forest.
On the tasks with heterogeneous features, Trompt narrows the gap but is below all the tree-based models.

On the large-sized regression tasks with numerical features only, [Figure 8(a)](#S4.F8.sf1 "In Figure 8 ‣ 4.2.2 Regression ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") shows that Trompt is slightly worse than SAINT and FT-Transformer in the end of search.
On the large-sized regression tasks with heterogeneous features, [Figure 8(b)](#S4.F8.sf2 "In Figure 8 ‣ 4.2.2 Regression ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") shows that Trompt outperforms deep neural networks with a large margin.

In general, deep learning models are not good at handling categorical features.
Trompt alleviates this weakness as shown in all tasks with heterogeneous features in [Figure 5](#S4.F5 "In 4.2.1 Classification ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")–[Figure 8](#S4.F8 "In 4.2.2 Regression ‣ 4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
Trompt achieves superior performance over state-of-the-art deep neural networks except on the large-sized regression tasks with numerical features only.

!(/html/2305.18446/assets/figures/numerical_regression_quantile.jpg)

(a) Numerical features only.

!(/html/2305.18446/assets/figures/categorical_regression_quantile.jpg)

(b) Heterogeneous features.

Figure 7: Benchmark on medium-sized regression datasets.

!(/html/2305.18446/assets/figures/numerical_regression_quantile-large.jpg)

(a) Numerical features only.

!(/html/2305.18446/assets/figures/categorical_regression_quantile-large.jpg)

(b) Heterogeneous features.

Figure 8: Benchmark on large-sized regression datasets.

### 4.3 Ablation Study

In this subsection, we discuss the ablation study results of Trompt regarding hyperparameters and architecture design.
Please refer to [Appendix C](#A3 "Appendix C Settings of Ablation Study ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for the settings of the ablation study.
In the main article, we report two major ablations on (i) the number of prompts and (ii) the necessity of expanding feature embeddings by a dense layer.
Other ablations can be found in [Appendix D](#A4 "Appendix D More Ablation Studies ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

Ablations on the number of prompts.
Prompt embeddings (𝐄promptsubscript𝐄prompt\mathbf{E}\_{\text{prompt}}) stand a vital role to derive the feature importances.
Here we discuss the effectiveness of adjusting the number of prompts.

As shown in [Table 3](#S4.T3 "In 4.3 Ablation Study ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), setting the number of prompts to one results in the worse results.
However, halving and doubling the default number (128) do not have much effect on the performance.
The results demonstrate that Trompt is not sensitive to the number of prompts, as long as the number of prompts is enough to accommodate the modalities of the dataset.

Table 3: The performance of different number of prompts.

  

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1 | 64 | 128 (default) | 256 |
| Classification | 79.74%percent79.7479.74\% | 81.76%percent81.7681.76\% | 81.81%percent81.8181.81\% | 81.85%percent81.8581.85\% |
| Regression | 72.07%percent72.0772.07\% | 74.11%percent74.1174.11\% | 74.15%percent74.1574.15\% | 74.14%percent74.1474.14\% |

Ablations on expanding feature embeddings by a dense layer.
Part three of [Figure 3](#S2.F3 "In 2.3 The Uniqueness of Trompt ‣ 2 Related Work ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") uses a dense layer to expand feature embeddings to accommodate P𝑃P prompts.
Here we discuss the necessity of the dense layer.

As you can see in [Table 4](#S4.T4 "In 4.3 Ablation Study ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), adding a dense layer really leads to better results and is a one of the key architecture designs of Trompt.
By design, adding the dense layer enables Trompt to generate different feature embeddings for each prompt.
Without the dense layer, Trompt is degraded to a simplified situation where each prompt uses the same feature embeddings.
The results of [Table 3](#S4.T3 "In 4.3 Ablation Study ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [Table 4](#S4.T4 "In 4.3 Ablation Study ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") suggest that the variation of feature importances, which comes from both the prompt embedding and the expansion dense layer, is the key to the excellent performance of Trompt.

Table 4: The performance of with and without applying feature transformation on Input Transform.

  

|  |  |  |
| --- | --- | --- |
|  | w (default) | w/o |
| Classification | 81.81%percent81.8181.81\% | 80.76%percent80.7680.76\% |
| Regression | 74.15%percent74.1574.15\% | 73.73%percent73.7373.73\% |

Table 5: The top-3 importance score ratio on the mushroom dataset.

  

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1st | 2nd | 3rd |
| RandomForest | odor (15.11%percent15.1115.11\%) | gill-size (12.37%percent12.3712.37\%) | gill-color (10.42%percent10.4210.42\%) |
| XGBoost | spore-print-color (29.43%percent29.4329.43\%) | odor (22.71%percent22.7122.71\%) | cap-color (14.07%percent14.0714.07\%) |
| LightGBM | spore-print-color (22.08%percent22.0822.08\%) | gill-color (14.95%percent14.9514.95\%) | odor (12.96%percent12.9612.96\%) |
| CatBoost | odor (72.43%percent72.4372.43\%) | spore-print-color (10.57%percent10.5710.57\%) | gill-size (2.71%percent2.712.71\%) |
| GradientBoostingTree | gill-color (31.08%percent31.0831.08\%) | spore-print-color (19.89%percent19.8919.89\%) | odor (17.44%percent17.4417.44\%) |
| Trompt (ours) | odor (24.93%percent24.9324.93\%) | gill-size (8.13%percent8.138.13\%) | gill-color (5.73%percent5.735.73\%) |

### 4.4 Interpretability

Besides outstanding performance, tree-based models are well-known for their interpretability.
Here we explore whether Trompt can also provide concise feature importances that highlighted salient features.
To investigate this, we conduct experiments on both synthetic datasets and real-world datasets, following the experimental design of TabNet (Arik & Pfister, [2021](#bib.bib1)).
To derive the feature importances of Trompt for each sample, 𝐌importance∈ℝB×P×Csubscript𝐌importancesuperscriptℝ𝐵𝑃𝐶\mathbf{M}\_{\text{importance}}\in\mathbb{R}^{B\times{P}\times{C}} is reduced to 𝐌^importance∈ℝB×Csubscript^𝐌importancesuperscriptℝ𝐵𝐶\mathbf{\hat{M}}\_{\text{importance}}\in\mathbb{R}^{B\times{C}} as [Equation 10](#S4.E10 "In 4.4 Interpretability ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), where the weight of 𝐌importancesubscript𝐌importance\mathbf{M}\_{\text{importance}} is the 𝐖promptsubscript𝐖prompt\mathbf{W}\_{\text{prompt}} of [Equation 6](#S3.E6 "In 3.2 Trompt Downstream ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

Notice that all Trompt Cells derive separated feature importances.
We demonstrate the averaged results of all cells here and leave the results of each cell in [Section E.1](#A5.SS1 "E.1 Feature Importances of Each Layer ‣ Appendix E More Interpretability Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐌^importance=∑i=1P(𝐖prompt⊙𝐌importance):,i,:∈ℝB×Csubscript^𝐌importancesuperscriptsubscript𝑖1𝑃subscriptdirect-productsubscript𝐖promptsubscript𝐌importance  :𝑖:superscriptℝ𝐵𝐶\mathbf{\hat{M}}\_{\text{importance}}=\sum\_{i=1}^{P}(\mathbf{W}\_{\text{prompt}}\odot\mathbf{M}\_{\text{importance}})\_{:,i,:}\in\mathbb{R}^{B\times{C}} |  | (10) |

Synthetic datasets.
The Syn2 and Syn4 datasets are used to study the feature importances learned by each model (Chen et al., [2018](#bib.bib7)).
A model is trained on oversampled training set (10k to 100k) using default hyperparameters and evaluated on 20 randomly picked testing samples.
The configuration is identical to that in TabNet (Arik & Pfister, [2021](#bib.bib1)).

[Figure 9](#S4.F9 "In 4.4 Interpretability ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [Figure 10](#S4.F10 "In 4.4 Interpretability ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") compare the important features of the dataset and those learned by Trompt.
In the Syn2 dataset, features 2–5 are important ([Figure 9(a)](#S4.F9.sf1 "In Figure 9 ‣ 4.4 Interpretability ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")) and Trompt excellently focuses on them ([Figure 9(b)](#S4.F9.sf2 "In Figure 9 ‣ 4.4 Interpretability ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")).
In the Syn4 dataset, either features 0–1 or 2–5 could be important based on the value of feature 10 ([Figure 10(a)](#S4.F10.sf1 "In Figure 10 ‣ 4.4 Interpretability ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")).
As [Figure 10](#S4.F10 "In 4.4 Interpretability ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") shows, Trompt still properly focuses on features 0–5 and discovers the influence of feature 10.

!(/html/2305.18446/assets/figures/syn2-real.png)

(a) Important features.

!(/html/2305.18446/assets/figures/syn2-mask.png)

(b) Feature importances of Trompt.

Figure 9: Attention mask on Syn2 dataset (synthetic).

!(/html/2305.18446/assets/figures/syn4-real.png)

(a) Important features.

!(/html/2305.18446/assets/figures/syn4-mask.png)

(b) Feature importances of Trompt.

Figure 10: Attention mask on Syn4 dataset (synthetic).

Real-world datasets.
The mushroom dataset (Dua & Graff, [2017](#bib.bib13)) is used as the real-world dataset for visualization as TabNet (Arik & Pfister, [2021](#bib.bib1)).
With only the *Odor* feature, most machine learning models can achieve >95%absentpercent95>95\% test accuracy (Arik & Pfister, [2021](#bib.bib1)).
As a result, a high feature importance is expected on Odor.

[Table 5](#S4.T5 "In 4.3 Ablation Study ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") shows the three most important features of Trompt and five tree-based models.
As shown, all models place Odor in their top three.
The second and third places of Trompt, *gill-size* and *gill-color*, also appear in the top three of the other models.
Actually, *cap-color* is selected only by XGBoost.
If it is excluded, the union of the top important features of all models comes down to four features.
The one Trompt missed is *spore-print-color*, which is the fifth place of Trompt.
Overall speaking, the important features selected by Trompt are consistent with those by tree-based models, and can therefore be used in various analyses that are familiar in the field of machine learning.

To further demonstrate that the experimental results were not ad-hoc, we repeat the experiments on additional real-world datasets.
Please see [Section E.2](#A5.SS2 "E.2 Additional Real-world Datasets ‣ Appendix E More Interpretability Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for the details and experimental results.

## 5 Discussion

In this section, we further explore the “prompt” mechanism of Trompt.
[Section 5.1](#S5.SS1 "5.1 Further exploration of the ”prompt” mechanism in Trompt ‣ 5 Discussion ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") clarifies the underlying hypothesis of how the prompt learning of Trompt fits for tabular data.
In addition, as Trompt is partially inspired by the learning strategy of tree-based models, we further discussed the difference between Trompt and tree-based models in [Section 5.2](#S5.SS2 "5.2 The differences between Trompt and Tree-based Models ‣ 5 Discussion ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

### 5.1 Further exploration of the ”prompt” mechanism in Trompt

The ”prompt” mechanism in Trompt is realized as [Equation 4](#S3.E4 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
This equation involves a matrix multiplication of expanded prompt embeddings (𝐒𝐄^prompt∈ℝB×P×dsubscript^𝐒𝐄promptsuperscriptℝ𝐵𝑃𝑑\mathbf{\hat{SE}}\_{\text{prompt}}\in\mathbb{R}^{B\times{P}\times{d}}) and the transpose of expanded column embeddings (𝐒𝐄column∈ℝB×C×dsubscript𝐒𝐄columnsuperscriptℝ𝐵𝐶𝑑\mathbf{SE}\_{\text{column}}\in\mathbb{R}^{B\times{C}\times{d}}).
It results in 𝐌importance∈ℝP×Csubscript𝐌importancesuperscriptℝ𝑃𝐶\mathbf{M}\_{\text{importance}}\in\mathbb{R}^{P\times{C}}, which represents prompt-to-column feature importances.
The matrix multiplication calculates the cosine-based distance between 𝐒𝐄^promptsubscript^𝐒𝐄prompt\mathbf{\hat{SE}}\_{\text{prompt}} and 𝐒𝐄columnsubscript𝐒𝐄column\mathbf{SE}\_{\text{column}}, and favors high similarity between the sample-specific representations and sample-invariant intrinsic properties.

To make it clearer, 𝐒𝐄^promptsubscript^𝐒𝐄prompt\mathbf{\hat{SE}}\_{\text{prompt}} consists of P𝑃P embeddings that are specific to individual samples, except for the first Trompt Cell where 𝐎prevsubscript𝐎prev\mathbf{O}\_{\text{prev}} is a zero tensor since there is no previous Trompt Cell, as stated in [Equations 1](#S3.E1 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [2](#S3.E2 "Equation 2 ‣ 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
On the other hand, 𝐒𝐄columnsubscript𝐒𝐄column\mathbf{SE}\_{\text{column}} consists of C𝐶C embeddings that represent intrinsic properties specific to a tabular dataset as stated in [Equation 3](#S3.E3 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

Unlike self-attention, which calculates the distance between queries and keys and derives token-to-token similarity measures, Trompt calculates the distance between 𝐒𝐄^promptsubscript^𝐒𝐄prompt\mathbf{\hat{SE}}\_{\text{prompt}} and 𝐒𝐄columnsubscript𝐒𝐄column\mathbf{SE}\_{\text{column}} in LABEL:{eq:soft-select} to derive sample-to-intrinsic-property similarity measures.
The underlying idea of the calculation is to capture the distance between each sample and intrinsic property of a tabular dataset and we hypothesize that incorporating the intrinsic properties into the modeling of a tabular neural network might help making good predictions.

### 5.2 The differences between Trompt and Tree-based Models

As discussed in [Section 3.3](#S3.SS3 "3.3 Prompt Learning of Trompt ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), the idea of using prompt learning to derive feature importances, is inspired by the learning algorithm of tree-based models and the intrinsic properties of tabular data.
As a result, Trompt and tree-based models share a common characteristic in that they enable sample-dependent feature importances.
However, there are two main differences between them.
First, to incorporate the intrinsic properties of tabular data, Trompt uses column embeddings to share the column information across samples, while the learning strategy of tree-based models learn column information in their node-split nature.
Second, Trompt and tree-based models use different techniques to learn feature importance.
Trompt derives feature importances explicitly through prompt learning, while tree-based models vary the feature importances implicitly in the root-to-leaf path.

## 6 Conclusion

In this study, we introduce Trompt, a novel network architecture for tabular data analysis.
Trompt utilizes prompt learning to determine varying feature importances in individual samples.
Our evaluation shows that Trompt outperforms state-of-the-art deep neural networks (SAINT and FT-Transformer) and closes the performance gap between deep neural networks and tree-based models.

The emergence of prompt learning in deep learning is promising.
While the design of Trompt may not be intuitive or perfect for language model prompts, it demonstrates the potential of leveraging prompts in tabular data analysis.
This work introduces a new strategy for deep neural networks to challenge tree-based models and future research in this direction can explore more prompt-inspired architectures.

## References

* Arik & Pfister (2021)

  Arik, S. Ö. and Pfister, T.
  Tabnet: Attentive interpretable tabular learning.
  In *Proceedings of the AAAI Conference on Artificial
  Intelligence*, volume 35, pp.  6679–6687, 2021.
* Averagemn (2019)

  Averagemn.
  Lgbm with hyperopt tuning, 2019.
  URL
  <https://www.kaggle.com/code/donkeys/lgbm-with-hyperopt-tuning/notebook>.
  [Online; accessed 5-January-2023].
* Bahmani (2022)

  Bahmani, M.
  Understanding lightgbm parameters (and how to tune them), 2022.
  URL <https://neptune.ai/blog/lightgbm-parameters-guide>.
  [Online; accessed 5-January-2023].
* Borisov et al. (2021)

  Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., and Kasneci,
  G.
  Deep neural networks and tabular data: A survey.
  *arXiv preprint arXiv:2110.01889*, 2021.
* Breiman (2001)

  Breiman, L.
  Random forests.
  *Machine learning*, 45(1):5–32, 2001.
* Brown et al. (2020)

  Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P.,
  Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.
  Language models are few-shot learners.
  *Advances in neural information processing systems*,
  33:1877–1901, 2020.
* Chen et al. (2018)

  Chen, J., Song, L., Wainwright, M., and Jordan, M.
  Learning to explain: An information-theoretic perspective on model
  interpretation.
  In *International Conference on Machine Learning*, pp. 883–892. PMLR, 2018.
* Chen et al. (2021)

  Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P.,
  Srinivas, A., and Mordatch, I.
  Decision transformer: Reinforcement learning via sequence modeling.
  *Advances in neural information processing systems*,
  34:15084–15097, 2021.
* Chen et al. (2015)

  Chen, T., He, T., Benesty, M., Khotilovich, V., Tang, Y., Cho, H., Chen, K.,
  et al.
  Xgboost: extreme gradient boosting.
  *R package version 0.4-2*, 1(4):1–4, 2015.
* Cortez et al. (2009)

  Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J.
  Modeling wine preferences by data mining from physicochemical
  properties.
  *Decision support systems*, 47(4):547–553,
  2009.
* Devlin et al. (2018)

  Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K.
  Bert: Pre-training of deep bidirectional transformers for language
  understanding.
  *arXiv preprint arXiv:1810.04805*, 2018.
* Dosovitskiy et al. (2020)

  Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
  Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.
  An image is worth 16x16 words: Transformers for image recognition at
  scale.
  *arXiv preprint arXiv:2010.11929*, 2020.
* Dua & Graff (2017)

  Dua, D. and Graff, C.
  UCI machine learning repository, 2017.
  URL <http://archive.ics.uci.edu/ml>.
* Friedman (2001)

  Friedman, J. H.
  Greedy function approximation: a gradient boosting machine.
  *Annals of statistics*, pp.  1189–1232, 2001.
* Goodfellow et al. (2020)

  Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair,
  S., Courville, A., and Bengio, Y.
  Generative adversarial networks.
  *Communications of the ACM*, 63(11):139–144, 2020.
* Gorishniy et al. (2021)

  Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A.
  Revisiting deep learning models for tabular data.
  *Advances in Neural Information Processing Systems*,
  34:18932–18943, 2021.
* Grinsztajn et al. (2022)

  Grinsztajn, L., Oyallon, E., and Varoquaux, G.
  Why do tree-based models still outperform deep learning on typical
  tabular data?
  In *Thirty-sixth Conference on Neural Information Processing
  Systems Datasets and Benchmarks Track*, 2022.
  URL <https://openreview.net/forum?id=Fp7__phQszn>.
* Gu et al. (2017)

  Gu, S., Holly, E., Lillicrap, T., and Levine, S.
  Deep reinforcement learning for robotic manipulation with
  asynchronous off-policy updates.
  In *2017 IEEE international conference on robotics and
  automation (ICRA)*, pp.  3389–3396. IEEE, 2017.
* He et al. (2016)

  He, K., Zhang, X., Ren, S., and Sun, J.
  Deep residual learning for image recognition.
  In *Proceedings of the IEEE conference on computer vision and
  pattern recognition*, pp.  770–778, 2016.
* Huang et al. (2020)

  Huang, X., Khetan, A., Cvitkovic, M., and Karnin, Z.
  Tabtransformer: Tabular data modeling using contextual embeddings.
  *arXiv preprint arXiv:2012.06678*, 2020.
* Katzir et al. (2020)

  Katzir, L., Elidan, G., and El-Yaniv, R.
  Net-dnf: Effective deep modeling of tabular data.
  In *International Conference on Learning Representations*, 2020.
* Ke et al. (2017)

  Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu,
  T.-Y.
  Lightgbm: A highly efficient gradient boosting decision tree.
  *Advances in neural information processing systems*, 30, 2017.
* LeCun et al. (1995)

  LeCun, Y., Bengio, Y., et al.
  Convolutional networks for images, speech, and time series.
  *The handbook of brain theory and neural networks*,
  3361(10):1995, 1995.
* LeCun et al. (2015)

  LeCun, Y., Bengio, Y., and Hinton, G.
  Deep learning.
  *nature*, 521(7553):436–444, 2015.
* Lester et al. (2021)

  Lester, B., Al-Rfou, R., and Constant, N.
  The power of scale for parameter-efficient prompt tuning.
  *arXiv preprint arXiv:2104.08691*, 2021.
* Li & Liang (2021)

  Li, X. L. and Liang, P.
  Prefix-tuning: Optimizing continuous prompts for generation.
  *arXiv preprint arXiv:2101.00190*, 2021.
* Prokhorenkova et al. (2018)

  Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A.
  Catboost: unbiased boosting with categorical features.
  *Advances in neural information processing systems*, 31, 2018.
* Radford et al. (2018)

  Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al.
  Improving language understanding by generative pre-training.
  2018.
* Ramachandram & Taylor (2017)

  Ramachandram, D. and Taylor, G. W.
  Deep multimodal learning: A survey on recent advances and trends.
  *IEEE signal processing magazine*, 34(6):96–108, 2017.
* Redmon et al. (2016)

  Redmon, J., Divvala, S., Girshick, R., and Farhadi, A.
  You only look once: Unified, real-time object detection.
  In *Proceedings of the IEEE conference on computer vision and
  pattern recognition*, pp.  779–788, 2016.
* Rumelhart et al. (1986)

  Rumelhart, D. E., Hinton, G. E., and Williams, R. J.
  Learning representations by back-propagating errors.
  *nature*, 323(6088):533–536, 1986.
* Sahoo et al. (2017)

  Sahoo, D., Pham, Q., Lu, J., and Hoi, S. C.
  Online deep learning: Learning deep neural networks on the fly.
  *arXiv preprint arXiv:1711.03705*, 2017.
* scikit learn (2023a)

  scikit learn.
  sklearn.ensemble.histgradientboostingclassifier, 2023a.
  URL
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>.
  [Online; accessed 21-January-2023].
* scikit learn (2023b)

  scikit learn.
  sklearn.preprocessing.quantiletransformer, 2023b.
  URL
  <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html>.
  [Online; accessed 26-January-2023].
* Shwartz-Ziv & Armon (2022)

  Shwartz-Ziv, R. and Armon, A.
  Tabular data: Deep learning is not all you need.
  *Information Fusion*, 81:84–90, 2022.
  ISSN 1566-2535.
  doi: https://doi.org/10.1016/j.inffus.2021.11.011.
  URL
  <https://www.sciencedirect.com/science/article/pii/S1566253521002360>.
* Somepalli et al. (2021)

  Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., and Goldstein, T.
  Saint: Improved neural networks for tabular data via row attention
  and contrastive pre-training.
  *arXiv preprint arXiv:2106.01342*, 2021.
* Van Engelen & Hoos (2020)

  Van Engelen, J. E. and Hoos, H. H.
  A survey on semi-supervised learning.
  *Machine Learning*, 109(2):373–440, 2020.
* Vanschoren et al. (2013)

  Vanschoren, J., van Rijn, J. N., Bischl, B., and Torgo, L.
  Openml: networked science in machine learning.
  *SIGKDD Explorations*, 15(2):49–60, 2013.
  doi: 10.1145/2641190.2641198.
  URL <http://doi.acm.org/10.1145/2641190.264119>.
* Vaswani et al. (2017)

  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, Ł., and Polosukhin, I.
  Attention is all you need.
  *Advances in neural information processing systems*, 30, 2017.
* Zhang et al. (2020)

  Zhang, Q., Lu, H., Sak, H., Tripathi, A., McDermott, E., Koo, S., and Kumar, S.
  Transformer transducer: A streamable speech recognition model with
  transformer encoders and rnn-t loss.
  In *ICASSP 2020-2020 IEEE International Conference on Acoustics,
  Speech and Signal Processing (ICASSP)*, pp.  7829–7833. IEEE, 2020.

## Appendix A Settings of Grinsztajn45

In this section, we provide brief summaries with regard to dataset selection criteria in [Section A.1](#A1.SS1 "A.1 Dataset Selection Criteria ‣ Appendix A Settings of Grinsztajn45 ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), dataset normalization in [Section A.2](#A1.SS2 "A.2 Dataset Normalization ‣ Appendix A Settings of Grinsztajn45 ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), baseline models in [Section A.3](#A1.SS3 "A.3 Baseline Models ‣ Appendix A Settings of Grinsztajn45 ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and hyperparameter search mechanism in [Section A.4](#A1.SS4 "A.4 Hyperparameter Search Mechanism ‣ Appendix A Settings of Grinsztajn45 ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

### A.1 Dataset Selection Criteria

Grinsztajn45 (Grinsztajn et al., [2022](#bib.bib17)) selects 45 tabular datasets from various domains mainly provided by OpenML (Vanschoren et al., [2013](#bib.bib38)), which is listed in section A.1 of their paper.

The dataset selection criteria are summarized below.
Please refer to section 3.1 of the original paper for detailed selection criteria.

* •

  The datasets contain heterogeneous features.
* •

  They are not high dimensional.
* •

  They contain I.I.D. data.
* •

  They contain real-world data.
* •

  They are not too small.
* •

  They are not too easy.
* •

  They are not deterministic.

### A.2 Dataset Normalization

To ensure the homogeneity of the datasets and focus on challenges specific to tabular data, Grinsztajn45 did some modifications to the datasets to make sure that the datasets in the benchmark conform to the following criteria.
Please refer to section 3.2 of the original paper for detailed modification.

* •

  The training sets are truncated to medium-sized (10,000) or large-sized (50,000).
* •

  All missing data were removed from the datasets.
* •

  The classes are balanced.
* •

  Categorical features with more than 20 items were removed
* •

  Numerical features with less than 10 unique values were removed.
* •

  Numerical features with 2 unique values are converted to categorical features.

### A.3 Baseline Models

The paper by Grinsztajn45 presents the performance of four DNN models and four tree-based models.
The DNN models include MLP (Gorishniy et al., [2021](#bib.bib16)), ResNet (Gorishniy et al., [2021](#bib.bib16)), FT-Transformer (Gorishniy et al., [2021](#bib.bib16)), and SAINT (Somepalli et al., [2021](#bib.bib36)).
The tree-based models consist of RandomForest (Breiman, [2001](#bib.bib5)), GradientBoostingTree (Friedman, [2001](#bib.bib14)), XGBoost (Chen et al., [2015](#bib.bib9)), and HistGradientBoostingTree (scikit learn, [2023a](#bib.bib33)).

However, two models, namely MLP (Gorishniy et al., [2021](#bib.bib16)) and HistGradientBoostingTree (scikit learn, [2023a](#bib.bib33)), were omitted from the evaluation due to incomplete experimental results in Grinsztajn45 (Grinsztajn et al., [2022](#bib.bib17)).
To provide a comprehensive comparison, we have included LightGBM (Ke et al., [2017](#bib.bib22)) and CatBoost (Prokhorenkova et al., [2018](#bib.bib27)) as additional models.
These models were selected based on their excellent performance and popularity.

### A.4 Hyperparameter Search Mechanism

Grinsztajn45 evaluates models based on the results of a random search that consumes 20,000 compute-hours, as mentioned in Section 3.3 of the paper (Grinsztajn et al., [2022](#bib.bib17)).
Since different models have varying inference and update times, the number of random search iterations completed within the same compute-hour differs for each model.
For instance, Model A may perform around two hundred iterations, while Model B may perform around three hundred iterations within 20,000 hours.
To ensure a fair evaluation, the iterations are truncated based on the minimum iteration count among all the compared models.

Due to limited computing resources, we have chosen a small search space ([Table 30](#A6.T30 "In Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")) consisting of 40 parameter combinations.
To avoid unfairly truncating random search results of other models, and compromising the low search iterations of Trompt, we duplicated the grid search results of Trompt to exceed the lowest search iteration count among the models provided by Grinsztajn45.
For instance, if the lowest search iteration of a model was three hundreds, the search results of Trompt will be oversampled to surpass three hundreds and avoid being the lower bound, so other models can retain same search iterations as provided by Grinsztajn45.
As a result, the other models can retain the same search iterations as provided by Grinsztajn45.

Grinsztajn45’s suggested evaluation procedure involves an extensive hyperparameter search that explores hundreds of parameter combinations.
However, due to limited computing resources, we have selected a smaller search space of 40 parameter combinations ([Table 30](#A6.T30 "In Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") in [Appendix F](#A6 "Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")) for Trompt.
Please refer to [Appendix F](#A6 "Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for the hyperparameter search spaces of all models.

## Appendix B More Evaluation Results

In [Section B.1](#A2.SS1 "B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), we present additional evaluation results for Grinsztajn45, which expand upon the findings and analysis presented in the original paper (Grinsztajn et al., [2022](#bib.bib17)).
These additional results provide further insights and contribute to a more comprehensive understanding of the evaluated models.

Furthermore, we include evaluation results on different datasets using the datasets selected by FT-Transformer (Gorishniy et al., [2021](#bib.bib16)) and SAINT (Somepalli et al., [2021](#bib.bib36)) in [Section B.2](#A2.SS2 "B.2 Datasets chosen by FT-Transformer ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [Section B.3](#A2.SS3 "B.3 Datasets chosen by SAINT ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), respectively.
By applying these datasets to the models, we aim to assess the performance of Trompt in different scenarios and gain a deeper understanding of its capabilities and generalizability.

### B.1 Grinsztajn45

In main paper, we have discussed the overall performance of Trompt using the learning curves during hyperparameter optimization.
In this section, we present quantitative evaluation results of both default and optimized hyperparameters.
In addition, we provide the figures of individual datasets for reference.

The quantitative evaluation results of classification and regression tasks are discussed in [Section B.1.1](#A2.SS1.SSS1 "B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [Section B.1.2](#A2.SS1.SSS2 "B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") respectively.
For classification datasets, we use accuracy as the evaluation metric.
For regression datasets, we use r2-score as the evaluation metric.
As a result, in both categories, the higher the number, the better the result.
Besides evaluation metrics, the ranking of each model is also provided.
To derive ranking, we calculate the mean and standard deviation of all rankings on datasets of a task.
Notice that since the names of some datasets are long, we first denote each dataset a notation in [Tables 6](#A2.T6 "In B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [7](#A2.T7 "Table 7 ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [8](#A2.T8 "Table 8 ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and use them in following tables.

Table 6: Notation of medium-sized datasets (1).

  

| Notation | Dataset |
| --- | --- |
| A1 | KDDCup09\_upselling |
| A2 | compass |
| A3 | covertype |
| A4 | electricity |
| A5 | eye\_movements |
| A6 | rl |
| A7 | road-safety |
| B1 | Higgs |
| B2 | MagicTelescope |
| B3 | MiniBooNE |
| B4 | bank-marketing |
| B5 | california |
| B6 | covertype |
| B7 | credit |
| B8 | electricity |
| B9 | eye\_movements |
| B10 | house\_16H |
| B11 | jannis |
| B12 | kdd\_ipums\_la\_97-small |
| B13 | phoneme |
| B14 | pol |
| B15 | wine |

Table 7: Notation of medium-sized datasets (2).

  

| Notation | Dataset |
| --- | --- |
| C1 | Bike\_Sharing\_Demand |
| C2 | Brazilian\_houses |
| C3 | Mercedes\_Benz\_Greener\_Manufacturing |
| C4 | OnlineNewsPopularity |
| C5 | SGEMM\_GPU\_kernel\_performance |
| C6 | analcatdata\_supreme |
| C7 | black\_friday |
| C8 | diamonds |
| C9 | house\_sales |
| C10 | nyc-taxi-green-dec-2016 |
| C11 | particulate-matter-ukair-2017 |
| C12 | visualizing\_soil |
| C13 | yprop\_4\_1 |
| D1 | Ailerons |
| D2 | Bike\_Sharing\_Demand |
| D3 | Brazilian\_houses |
| D4 | MiamiHousing2016 |
| D5 | california |
| D6 | cpu\_act |
| D7 | diamonds |
| D8 | elevators |
| D9 | fifa |
| D10 | house\_16H |
| D11 | house\_sales |
| D12 | houses |
| D13 | medical\_charges |
| D14 | nyc-taxi-green-dec-2016 |
| D15 | pol |
| D16 | sulfur |
| D17 | superconduct |
| D18 | wine\_quality |
| D19 | year |

Table 8: Notation of large-sized datasets.

  

| Notation | Dataset |
| --- | --- |
| 𝔸𝔸\mathbb{A}1 | covertype |
| 𝔸𝔸\mathbb{A}2 | road-safety |
| 𝔹𝔹\mathbb{B}1 | covertype |
| 𝔹𝔹\mathbb{B}2 | Higgs |
| 𝔹𝔹\mathbb{B}3 | MiniBooNE |
| 𝔹𝔹\mathbb{B}4 | jannis |
| ℂℂ\mathbb{C}1 | black\_friday |
| ℂℂ\mathbb{C}2 | diamonds |
| ℂℂ\mathbb{C}3 | nyc-taxi-green-dec-2016 |
| ℂℂ\mathbb{C}4 | particulate-matter-ukair-2017 |
| ℂℂ\mathbb{C}5 | SGEMM\_GPU\_kernel\_performance |
| 𝔻𝔻\mathbb{D}1 | diamonds |
| 𝔻𝔻\mathbb{D}2 | nyc-taxi-green-dec-2016 |
| 𝔻𝔻\mathbb{D}3 | year |

#### B.1.1 Classification

The evaluation results for medium-sized classification tasks are presented in [Table 9](#A2.T9 "In B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for heterogeneous features, and in [Tables 10](#A2.T10 "In B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [11](#A2.T11 "Table 11 ‣ B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for numerical features only.

For large-sized classification tasks, the results can be found in [Table 12](#A2.T12 "In B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for heterogeneous features, and in [Table 13](#A2.T13 "In B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for numerical features only.

Furthermore, individual figures illustrating the performance of Trompt on medium-sized tasks are provided in [Figure 11](#A2.F11 "In B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for heterogeneous features, and in [Figure 12](#A2.F12 "In B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for numerical features only.
The individual figures for large-sized tasks can be found in [Figure 13](#A2.F13 "In B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for heterogeneous features, and in [Figure 14](#A2.F14 "In B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for numerical features only.

The evaluation results consistently demonstrate that Trompt outperforms state-of-the-art deep neural networks (FT-Transformer and SAINT) across all classification tasks (refer to [Tables 9](#A2.T9 "In B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [10](#A2.T10 "Table 10 ‣ B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [11](#A2.T11 "Table 11 ‣ B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [12](#A2.T12 "Table 12 ‣ B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [13](#A2.T13 "Table 13 ‣ B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")).
Moreover, Trompt’s default rankings consistently yield better performance than the searched rankings, indicating its strength in default configurations without tuning.
Remarkably, in a large-sized task with numerical features only, Trompt even surpasses tree-based models (refer to [Table 13](#A2.T13 "In B.1.1 Classification ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")).

Table 9: The performance of medium-sized classification task (*heterogeneous features*).

  

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | A1 | A2 | A3 | A4 | A5 | A6 | A7 | Ranking |
| Default | | | | | | | | |
| Trompt (ours) | 78.91%percent78.9178.91\% | 78.59%percent78.5978.59\% | 87.29%percent87.2987.29\% | 84.50%percent84.5084.50\% | 64.25%percent64.2564.25\% | 75.13%percent75.1375.13\% | 75.80%percent75.8075.80\% | 3.71±1.78plus-or-minus3.711.783.71\pm 1.78 |
| FT-Transformer | 78.56%percent78.5678.56\% | 73.43%percent73.4373.43\% | 85.57%percent85.5785.57\% | 82.71%percent82.7182.71\% | 58.79%percent58.7958.79\% | 71.52%percent71.5271.52\% | 73.90%percent73.9073.90\% | 6.29±2.00plus-or-minus6.292.006.29\pm 2.00 |
| ResNet | 74.24%percent74.2474.24\% | 73.78%percent73.7873.78\% | 82.49%percent82.4982.49\% | 81.99%percent81.9981.99\% | 57.14%percent57.1457.14\% | 66.51%percent66.5166.51\% | 73.45%percent73.4573.45\% | 8.29±2.92plus-or-minus8.292.928.29\pm 2.92 |
| SAINT | 79.00%percent79.0079.00\% | 70.09%percent70.0970.09\% | 83.04%percent83.0483.04\% | 82.42%percent82.4282.42\% | 58.62%percent58.6258.62\% | 67.69%percent67.6967.69\% | 75.89%percent75.8975.89\% | 6.86±2.55plus-or-minus6.862.556.86\pm 2.55 |
| CatBoost | 79.90%percent79.9079.90\% | 74.22%percent74.2274.22\% | 83.69%percent83.6983.69\% | 85.01%percent85.0185.01\% | 64.62%percent64.6264.62\% | 75.29%percent75.2975.29\% | 76.80%percent76.8076.80\% | 2.93±2.55plus-or-minus2.932.552.93\pm 2.55 |
| LightGBM | 78.70%percent78.7078.70\% | 73.63%percent73.6373.63\% | 83.23%percent83.2383.23\% | 86.37%percent86.3786.37\% | 64.48%percent64.4864.48\% | 77.04%percent77.0477.04\% | 76.43%percent76.4376.43\% | 3.86±1.93plus-or-minus3.861.933.86\pm 1.93 |
| XGBoost | 78.39%percent78.3978.39\% | 74.46%percent74.4674.46\% | 84.13%percent84.1384.13\% | 87.86%percent87.8687.86\% | 64.77%percent64.7764.77\% | 78.42%percent78.4278.42\% | 75.94%percent75.9475.94\% | 3.00±2.92plus-or-minus3.002.923.00\pm 2.92 |
| RandomForest | 79.38%percent79.3879.38\% | 79.28%percent79.2879.28\% | 84.75%percent84.7584.75\% | 86.24%percent86.2486.24\% | 63.62%percent63.6263.62\% | 73.82%percent73.8273.82\% | 75.45%percent75.4575.45\% | 3.71±1.86plus-or-minus3.711.863.71\pm 1.86 |
| GradientBoostingTree | 79.90%percent79.9079.90\% | 72.01%percent72.0172.01\% | 78.92%percent78.9278.92\% | 82.94%percent82.9482.94\% | 61.81%percent61.8161.81\% | 69.60%percent69.6069.60\% | 75.00%percent75.0075.00\% | 6.36±2.51plus-or-minus6.362.516.36\pm 2.51 |
| Searched | | | | | | | | |
| Trompt (ours) | 79.00%percent79.0079.00\% | 79.55%percent79.5579.55\% | 88.29%percent88.2988.29\% | 85.13%percent85.1385.13\% | 64.29%percent64.2964.29\% | 76.02%percent76.0276.02\% | 76.38%percent76.3876.38\% | 4.43±2.20plus-or-minus4.432.204.43\pm 2.20 |
| FT-Transformer | 78.00%percent78.0078.00\% | 75.30%percent75.3075.30\% | 86.64%percent86.6486.64\% | 84.01%percent84.0184.01\% | 59.85%percent59.8559.85\% | 70.38%percent70.3870.38\% | 76.86%percent76.8676.86\% | 5.57±2.19plus-or-minus5.572.195.57\pm 2.19 |
| ResNet | 76.87%percent76.8776.87\% | 74.35%percent74.3574.35\% | 85.17%percent85.1785.17\% | 82.68%percent82.6882.68\% | 57.82%percent57.8257.82\% | 69.59%percent69.5969.59\% | 75.85%percent75.8575.85\% | 8.43±2.73plus-or-minus8.432.738.43\pm 2.73 |
| SAINT | 77.80%percent77.8077.80\% | 71.87%percent71.8771.87\% | 84.95%percent84.9584.95\% | 83.32%percent83.3283.32\% | 58.54%percent58.5458.54\% | 68.20%percent68.2068.20\% | 76.43%percent76.4376.43\% | 7.86±2.64plus-or-minus7.862.647.86\pm 2.64 |
| CatBoost | 80.50%percent80.5080.50\% | 76.87%percent76.8776.87\% | 87.48%percent87.4887.48\% | 87.73%percent87.7387.73\% | 66.48%percent66.4866.48\% | 78.67%percent78.6778.67\% | 77.16%percent77.1677.16\% | 2.43±2.71plus-or-minus2.432.712.43\pm 2.71 |
| LightGBM | 79.81%percent79.8179.81\% | 78.15%percent78.1578.15\% | 86.62%percent86.6286.62\% | 88.64%percent88.6488.64\% | 66.14%percent66.1466.14\% | 77.69%percent77.6977.69\% | 76.43%percent76.4376.43\% | 3.14±2.05plus-or-minus3.142.053.14\pm 2.05 |
| XGBoost | 79.69%percent79.6979.69\% | 76.83%percent76.8376.83\% | 86.25%percent86.2586.25\% | 88.52%percent88.5288.52\% | 66.57%percent66.5766.57\% | 77.18%percent77.1877.18\% | 76.69%percent76.6976.69\% | 3.57±1.93plus-or-minus3.571.933.57\pm 1.93 |
| RandomForest | 79.38%percent79.3879.38\% | 79.28%percent79.2879.28\% | 85.89%percent85.8985.89\% | 87.76%percent87.7687.76\% | 65.70%percent65.7065.70\% | 79.79%percent79.7979.79\% | 75.88%percent75.8875.88\% | 4.29±2.27plus-or-minus4.292.274.29\pm 2.27 |
| GradientBoostingTree | 80.01%percent80.0180.01\% | 73.77%percent73.7773.77\% | 85.55%percent85.5585.55\% | 87.85%percent87.8587.85\% | 63.30%percent63.3063.30\% | 77.58%percent77.5877.58\% | 76.23%percent76.2376.23\% | 5.29±2.17plus-or-minus5.292.175.29\pm 2.17 |

Table 10: The performance of medium-sized classification task (*numerical features only*) (1).

  

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
| Default | | | | | | | | |
| Trompt (ours) | 69.26%percent69.2669.26\% | 86.30%percent86.3086.30\% | 93.82%percent93.8293.82\% | 79.36%percent79.3679.36\% | 89.09%percent89.0989.09\% | 82.68%percent82.6882.68\% | 75.84%percent75.8475.84\% | 82.89%percent82.8982.89\% |
| FT-Transformer | 66.94%percent66.9466.94\% | 84.42%percent84.4284.42\% | 92.80%percent92.8092.80\% | 80.09%percent80.0980.09\% | 87.40%percent87.4087.40\% | 80.42%percent80.4280.42\% | 74.32%percent74.3274.32\% | 81.24%percent81.2481.24\% |
| ResNet | 65.39%percent65.3965.39\% | 85.11%percent85.1185.11\% | 93.10%percent93.1093.10\% | 78.68%percent78.6878.68\% | 86.90%percent86.9086.90\% | 79.09%percent79.0979.09\% | 74.99%percent74.9974.99\% | 80.91%percent80.9180.91\% |
| SAINT | 69.29%percent69.2969.29\% | 85.16%percent85.1685.16\% | 93.18%percent93.1893.18\% | 79.18%percent79.1879.18\% | 87.69%percent87.6987.69\% | 78.05%percent78.0578.05\% | 76.49%percent76.4976.49\% | 81.25%percent81.2581.25\% |
| CatBoost | 71.30%percent71.3071.30\% | 86.14%percent86.1486.14\% | 93.64%percent93.6493.64\% | 80.45%percent80.4580.45\% | 90.21%percent90.2190.21\% | 80.16%percent80.1680.16\% | 76.95%percent76.9576.95\% | 84.48%percent84.4884.48\% |
| LightGBM | 70.79%percent70.7970.79\% | 85.47%percent85.4785.47\% | 93.16%percent93.1693.16\% | 80.33%percent80.3380.33\% | 90.06%percent90.0690.06\% | 79.50%percent79.5079.50\% | 77.17%percent77.1777.17\% | 84.34%percent84.3484.34\% |
| XGBoost | 69.25%percent69.2569.25\% | 85.31%percent85.3185.31\% | 93.29%percent93.2993.29\% | 79.81%percent79.8179.81\% | 90.30%percent90.3090.30\% | 79.87%percent79.8779.87\% | 75.91%percent75.9175.91\% | 86.11%percent86.1186.11\% |
| RandomForest | 70.12%percent70.1270.12\% | 85.56%percent85.5685.56\% | 92.09%percent92.0992.09\% | 79.46%percent79.4679.46\% | 88.80%percent88.8088.80\% | 81.35%percent81.3581.35\% | 76.64%percent76.6476.64\% | 84.79%percent84.7984.79\% |
| GradientBoostingTree | 70.49%percent70.4970.49\% | 84.44%percent84.4484.44\% | 92.16%percent92.1692.16\% | 80.27%percent80.2780.27\% | 88.00%percent88.0088.00\% | 76.85%percent76.8576.85\% | 77.52%percent77.5277.52\% | 82.16%percent82.1682.16\% |
| Searched | | | | | | | | |
| Trompt (ours) | 69.60%percent69.6069.60\% | 86.35%percent86.3586.35\% | 93.74%percent93.7493.74\% | 79.30%percent79.3079.30\% | 89.28%percent89.2889.28\% | 83.73%percent83.7383.73\% | 76.52%percent76.5276.52\% | 83.12%percent83.1283.12\% |
| FT-Transformer | 70.67%percent70.6770.67\% | 85.26%percent85.2685.26\% | 93.59%percent93.5993.59\% | 80.22%percent80.2280.22\% | 88.61%percent88.6188.61\% | 81.22%percent81.2281.22\% | 76.50%percent76.5076.50\% | 81.94%percent81.9481.94\% |
| ResNet | 69.02%percent69.0269.02\% | 85.62%percent85.6285.62\% | 93.69%percent93.6993.69\% | 79.13%percent79.1379.13\% | 87.28%percent87.2887.28\% | 80.21%percent80.2180.21\% | 76.28%percent76.2876.28\% | 80.98%percent80.9880.98\% |
| SAINT | 70.73%percent70.7370.73\% | 84.85%percent84.8584.85\% | 93.54%percent93.5493.54\% | 79.29%percent79.2979.29\% | 88.92%percent88.9288.92\% | 80.27%percent80.2780.27\% | 76.24%percent76.2476.24\% | 81.84%percent81.8481.84\% |
| CatBoost | 71.46%percent71.4671.46\% | 85.92%percent85.9285.92\% | 93.84%percent93.8493.84\% | 80.39%percent80.3980.39\% | 90.32%percent90.3290.32\% | 82.98%percent82.9882.98\% | 77.59%percent77.5977.59\% | 86.33%percent86.3386.33\% |
| LightGBM | 71.01%percent71.0171.01\% | 85.70%percent85.7085.70\% | 93.71%percent93.7193.71\% | 80.15%percent80.1580.15\% | 90.13%percent90.1390.13\% | 81.81%percent81.8181.81\% | 77.13%percent77.1377.13\% | 85.94%percent85.9485.94\% |
| XGBoost | 71.36%percent71.3671.36\% | 86.05%percent86.0586.05\% | 93.66%percent93.6693.66\% | 80.34%percent80.3480.34\% | 90.12%percent90.1290.12\% | 81.75%percent81.7581.75\% | 77.26%percent77.2677.26\% | 86.94%percent86.9486.94\% |
| RandomForest | 70.76%percent70.7670.76\% | 85.41%percent85.4185.41\% | 92.65%percent92.6592.65\% | 79.82%percent79.8279.82\% | 89.21%percent89.2189.21\% | 82.73%percent82.7382.73\% | 77.25%percent77.2577.25\% | 86.14%percent86.1486.14\% |
| GradientBoostingTree | 71.00%percent71.0071.00\% | 85.57%percent85.5785.57\% | 93.22%percent93.2293.22\% | 80.26%percent80.2680.26\% | 89.68%percent89.6889.68\% | 81.72%percent81.7281.72\% | 77.27%percent77.2777.27\% | 86.24%percent86.2486.24\% |

Table 11: The performance of medium-sized classification task (*numerical features only*) (2).

  

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | B9 | B10 | B11 | B12 | B13 | B14 | B15 | Ranking |
| Default | | | | | | | | |
| Trompt (ours) | 61.60%percent61.6061.60\% | 88.05%percent88.0588.05\% | 76.89%percent76.8976.89\% | 86.61%percent86.6186.61\% | 88.67%percent88.6788.67\% | 98.49%percent98.4998.49\% | 79.07%percent79.0779.07\% | 4.07±2.61plus-or-minus4.072.614.07\pm 2.61 |
| FT-Transformer | 58.62%percent58.6258.62\% | 87.16%percent87.1687.16\% | 72.94%percent72.9472.94\% | 87.16%percent87.1687.16\% | 85.67%percent85.6785.67\% | 98.08%percent98.0898.08\% | 77.21%percent77.2177.21\% | 6.93±2.06plus-or-minus6.932.066.93\pm 2.06 |
| ResNet | 56.06%percent56.0656.06\% | 86.48%percent86.4886.48\% | 70.70%percent70.7070.70\% | 86.94%percent86.9486.94\% | 85.37%percent85.3785.37\% | 94.87%percent94.8794.87\% | 77.06%percent77.0677.06\% | 8.20±2.05plus-or-minus8.202.058.20\pm 2.05 |
| SAINT | 57.18%percent57.1857.18\% | 88.19%percent88.1988.19\% | 76.04%percent76.0476.04\% | 88.32%percent88.3288.32\% | 85.28%percent85.2885.28\% | 97.04%percent97.0497.04\% | 75.90%percent75.9075.90\% | 6.20±2.13plus-or-minus6.202.136.20\pm 2.13 |
| CatBoost | 63.87%percent63.8763.87\% | 88.59%percent88.5988.59\% | 77.85%percent77.8577.85\% | 87.98%percent87.9887.98\% | 87.44%percent87.4487.44\% | 98.46%percent98.4698.46\% | 78.58%percent78.5878.58\% | 2.47±2.03plus-or-minus2.472.032.47\pm 2.03 |
| LightGBM | 64.39%percent64.3964.39\% | 88.43%percent88.4388.43\% | 77.27%percent77.2777.27\% | 87.43%percent87.4387.43\% | 86.90%percent86.9086.90\% | 98.38%percent98.3898.38\% | 79.81%percent79.8179.81\% | 3.27±1.82plus-or-minus3.271.823.27\pm 1.82 |
| XGBoost | 64.75%percent64.7564.75\% | 88.16%percent88.1688.16\% | 76.00%percent76.0076.00\% | 87.31%percent87.3187.31\% | 87.05%percent87.0587.05\% | 98.35%percent98.3598.35\% | 79.78%percent79.7879.78\% | 4.13±1.97plus-or-minus4.131.974.13\pm 1.97 |
| RandomForest | 63.16%percent63.1663.16\% | 87.92%percent87.9287.92\% | 76.34%percent76.3476.34\% | 88.32%percent88.3288.32\% | 88.01%percent88.0188.01\% | 98.10%percent98.1098.10\% | 80.30%percent80.3080.30\% | 3.93±2.16plus-or-minus3.932.163.93\pm 2.16 |
| GradientBoostingTree | 62.33%percent62.3362.33\% | 87.68%percent87.6887.68\% | 76.17%percent76.1776.17\% | 88.32%percent88.3288.32\% | 84.26%percent84.2684.26\% | 96.71%percent96.7196.71\% | 77.09%percent77.0977.09\% | 5.80±2.52plus-or-minus5.802.525.80\pm 2.52 |
| Searched | | | | | | | | |
| Trompt (ours) | 62.71%percent62.7162.71\% | 88.46%percent88.4688.46\% | 76.99%percent76.9976.99\% | 87.25%percent87.2587.25\% | 88.67%percent88.6788.67\% | 98.38%percent98.3898.38\% | 78.58%percent78.5878.58\% | 4.80±2.47plus-or-minus4.802.474.80\pm 2.47 |
| FT-Transformer | 58.30%percent58.3058.30\% | 88.15%percent88.1588.15\% | 76.43%percent76.4376.43\% | 89.12%percent89.1289.12\% | 85.66%percent85.6685.66\% | 98.45%percent98.4598.45\% | 76.74%percent76.7476.74\% | 6.47±2.41plus-or-minus6.472.416.47\pm 2.41 |
| ResNet | 57.03%percent57.0357.03\% | 87.54%percent87.5487.54\% | 74.63%percent74.6374.63\% | 88.23%percent88.2388.23\% | 85.87%percent85.8785.87\% | 94.86%percent94.8694.86\% | 77.41%percent77.4177.41\% | 7.73±2.50plus-or-minus7.732.507.73\pm 2.50 |
| SAINT | 58.90%percent58.9058.90\% | 88.27%percent88.2788.27\% | 77.22%percent77.2277.22\% | 89.05%percent89.0589.05\% | 85.39%percent85.3985.39\% | 98.12%percent98.1298.12\% | 76.87%percent76.8776.87\% | 6.93±2.22plus-or-minus6.932.226.93\pm 2.22 |
| CatBoost | 65.07%percent65.0765.07\% | 88.54%percent88.5488.54\% | 77.95%percent77.9577.95\% | 88.02%percent88.0288.02\% | 88.83%percent88.8388.83\% | 98.47%percent98.4798.47\% | 79.89%percent79.8979.89\% | 1.93±2.36plus-or-minus1.932.361.93\pm 2.36 |
| LightGBM | 65.43%percent65.4365.43\% | 88.62%percent88.6288.62\% | 77.70%percent77.7077.70\% | 88.18%percent88.1888.18\% | 87.60%percent87.6087.60\% | 98.21%percent98.2198.21\% | 79.55%percent79.5579.55\% | 3.53±1.44plus-or-minus3.531.443.53\pm 1.44 |
| XGBoost | 65.83%percent65.8365.83\% | 88.83%percent88.8388.83\% | 77.83%percent77.8377.83\% | 88.12%percent88.1288.12\% | 86.81%percent86.8186.81\% | 98.09%percent98.0998.09\% | 79.46%percent79.4679.46\% | 3.20±2.22plus-or-minus3.202.223.20\pm 2.22 |
| RandomForest | 65.04%percent65.0465.04\% | 87.80%percent87.8087.80\% | 77.27%percent77.2777.27\% | 87.95%percent87.9587.95\% | 88.45%percent88.4588.45\% | 98.20%percent98.2098.20\% | 78.96%percent78.9678.96\% | 5.33±1.88plus-or-minus5.331.885.33\pm 1.88 |
| GradientBoostingTree | 63.04%percent63.0463.04\% | 88.22%percent88.2288.22\% | 77.17%percent77.1777.17\% | 88.32%percent88.3288.32\% | 86.68%percent86.6886.68\% | 98.06%percent98.0698.06\% | 78.56%percent78.5678.56\% | 5.07±1.77plus-or-minus5.071.775.07\pm 1.77 |

Table 12: The performance of large-sized classification task (*heterogeneous features*).

  

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔸𝔸\mathbb{A}1 | 𝔸𝔸\mathbb{A}2 | Ranking |
| Default | | | |
| Trompt (ours) | 92.76%percent92.7692.76\% | 78.36%percent78.3678.36\% | 1.50±4.36plus-or-minus1.504.361.50\pm 4.36 |
| FT-Transformer | 93.17%percent93.1793.17\% | 76.09%percent76.0976.09\% | 4.50±3.61plus-or-minus4.503.614.50\pm 3.61 |
| ResNet | 89.45%percent89.4589.45\% | 76.53%percent76.5376.53\% | 6.00±2.25plus-or-minus6.002.256.00\pm 2.25 |
| SAINT | 91.23%percent91.2391.23\% | 77.31%percent77.3177.31\% | 4.50±1.73plus-or-minus4.501.734.50\pm 1.73 |
| CatBoost | 88.27%percent88.2788.27\% | 78.21%percent78.2178.21\% | 4.50±1.73plus-or-minus4.501.734.50\pm 1.73 |
| LightGBM | 84.76%percent84.7684.76\% | 77.97%percent77.9777.97\% | 6.00±2.84plus-or-minus6.002.846.00\pm 2.84 |
| XGBoost | 87.81%percent87.8187.81\% | 78.22%percent78.2278.22\% | 4.50±2.65plus-or-minus4.502.654.50\pm 2.65 |
| RandomForest | 90.66%percent90.6690.66\% | 77.67%percent77.6777.67\% | 4.50±1.00plus-or-minus4.501.004.50\pm 1.00 |
| GradientBoostingTree | 79.46%percent79.4679.46\% | 75.19%percent75.1975.19\% | 9.00±4.62plus-or-minus9.004.629.00\pm 4.62 |
| Searched | | | |
| Trompt (ours) | 93.95%percent93.9593.95\% | 78.44%percent78.4478.44\% | 3.50±3.40plus-or-minus3.503.403.50\pm 3.40 |
| FT-Transformer | 93.61%percent93.6193.61\% | 78.92%percent78.9278.92\% | 3.50±2.36plus-or-minus3.502.363.50\pm 2.36 |
| ResNet | 92.27%percent92.2792.27\% | 78.40%percent78.4078.40\% | 8.00±3.61plus-or-minus8.003.618.00\pm 3.61 |
| SAINT | 92.54%percent92.5492.54\% | 77.96%percent77.9677.96\% | 8.50±4.36plus-or-minus8.504.368.50\pm 4.36 |
| CatBoost | 93.70%percent93.7093.70\% | 80.15%percent80.1580.15\% | 1.50±4.36plus-or-minus1.504.361.50\pm 4.36 |
| LightGBM | 93.25%percent93.2593.25\% | 79.75%percent79.7579.75\% | 4.00±1.32plus-or-minus4.001.324.00\pm 1.32 |
| XGBoost | 93.07%percent93.0793.07\% | 79.91%percent79.9179.91\% | 4.00±2.18plus-or-minus4.002.184.00\pm 2.18 |
| RandomForest | 93.30%percent93.3093.30\% | 78.13%percent78.1378.13\% | 6.00±2.47plus-or-minus6.002.476.00\pm 2.47 |
| GradientBoostingTree | 92.99%percent92.9992.99\% | 78.59%percent78.5978.59\% | 6.00±1.76plus-or-minus6.001.766.00\pm 1.76 |

Table 13: The performance of large-sized classification task (*numerical features only*).

  

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | 𝔹𝔹\mathbb{B}1 | 𝔹𝔹\mathbb{B}2 | 𝔹𝔹\mathbb{B}3 | 𝔹𝔹\mathbb{B}4 | Ranking |
| Default | | | | | |
| Trompt (ours) | 72.13%percent72.1372.13\% | 94.68%percent94.6894.68\% | 90.04%percent90.0490.04\% | 79.54%percent79.5479.54\% | 1.38±3.44plus-or-minus1.383.441.38\pm 3.44 |
| FT-Transformer | 69.60%percent69.6069.60\% | 94.03%percent94.0394.03\% | 89.83%percent89.8389.83\% | 75.86%percent75.8675.86\% | 6.00±2.96plus-or-minus6.002.966.00\pm 2.96 |
| ResNet | 69.88%percent69.8869.88\% | 94.09%percent94.0994.09\% | 88.01%percent88.0188.01\% | 73.58%percent73.5873.58\% | 6.00±2.78plus-or-minus6.002.786.00\pm 2.78 |
| SAINT | 71.81%percent71.8171.81\% | 94.36%percent94.3694.36\% | 86.94%percent86.9486.94\% | 78.60%percent78.6078.60\% | 3.75±1.82plus-or-minus3.751.823.75\pm 1.82 |
| CatBoost | 72.61%percent72.6172.61\% | 94.32%percent94.3294.32\% | 83.77%percent83.7783.77\% | 79.54%percent79.5479.54\% | 2.88±3.01plus-or-minus2.883.012.88\pm 3.01 |
| LightGBM | 72.12%percent72.1272.12\% | 93.71%percent93.7193.71\% | 80.71%percent80.7180.71\% | 78.70%percent78.7078.70\% | 5.00±2.17plus-or-minus5.002.175.00\pm 2.17 |
| XGBoost | 71.64%percent71.6471.64\% | 93.67%percent93.6793.67\% | 83.61%percent83.6183.61\% | 78.28%percent78.2878.28\% | 6.00±1.50plus-or-minus6.001.506.00\pm 1.50 |
| RandomForest | 71.58%percent71.5871.58\% | 93.08%percent93.0893.08\% | 87.67%percent87.6787.67\% | 77.97%percent77.9777.97\% | 6.00±1.80plus-or-minus6.001.806.00\pm 1.80 |
| GradientBoostingTree | 71.03%percent71.0371.03\% | 92.25%percent92.2592.25\% | 76.98%percent76.9876.98\% | 77.18%percent77.1877.18\% | 8.00±3.29plus-or-minus8.003.298.00\pm 3.29 |
| Searched | | | | | |
| Trompt (ours) | 72.86%percent72.8672.86\% | 94.36%percent94.3694.36\% | 91.27%percent91.2791.27\% | 79.88%percent79.8879.88\% | 3.25±2.97plus-or-minus3.252.973.25\pm 2.97 |
| FT-Transformer | 72.86%percent72.8672.86\% | 94.42%percent94.4294.42\% | 90.57%percent90.5790.57\% | 79.59%percent79.5979.59\% | 3.25±2.07plus-or-minus3.252.073.25\pm 2.07 |
| ResNet | 72.29%percent72.2972.29\% | 94.46%percent94.4694.46\% | 89.36%percent89.3689.36\% | 78.11%percent78.1178.11\% | 6.75±3.49plus-or-minus6.753.496.75\pm 3.49 |
| SAINT | 72.65%percent72.6572.65\% | 94.45%percent94.4594.45\% | 89.53%percent89.5389.53\% | 79.30%percent79.3079.30\% | 5.50±1.67plus-or-minus5.501.675.50\pm 1.67 |
| CatBoost | 72.99%percent72.9972.99\% | 94.55%percent94.5594.55\% | 90.19%percent90.1990.19\% | 79.89%percent79.8979.89\% | 1.75±3.49plus-or-minus1.753.491.75\pm 3.49 |
| LightGBM | 72.55%percent72.5572.55\% | 94.39%percent94.3994.39\% | 89.71%percent89.7189.71\% | 79.32%percent79.3279.32\% | 6.00±0.89plus-or-minus6.000.896.00\pm 0.89 |
| XGBoost | 72.81%percent72.8172.81\% | 94.40%percent94.4094.40\% | 89.32%percent89.3289.32\% | 79.67%percent79.6779.67\% | 5.25±2.30plus-or-minus5.252.305.25\pm 2.30 |
| RandomForest | 71.98%percent71.9871.98\% | 93.53%percent93.5393.53\% | 90.59%percent90.5990.59\% | 78.85%percent78.8578.85\% | 7.00±3.96plus-or-minus7.003.967.00\pm 3.96 |
| GradientBoostingTree | 72.49%percent72.4972.49\% | 94.07%percent94.0794.07\% | 89.79%percent89.7989.79\% | 79.34%percent79.3479.34\% | 6.25±1.95plus-or-minus6.251.956.25\pm 1.95 |

!(/html/2305.18446/assets/figures/categorical_classif-individual.jpg)

Figure 11: Benchmark on *every* medium-sized classification dataset with heterogeneous features.

!(/html/2305.18446/assets/figures/numerical_classif-individual.jpg)

Figure 12: Benchmark on *every* medium-sized classification dataset with numerical features only.

!(/html/2305.18446/assets/figures/categorical_classif-individual-large.jpg)

Figure 13: Benchmark on *every* large-sized classification dataset with heterogeneous features.

!(/html/2305.18446/assets/figures/numerical_classif-individual-large.jpg)

Figure 14: Benchmark on *every* large-sized classification dataset with numerical features only.

#### B.1.2 Regression

The evaluation results for medium-sized regression datasets are presented in [Tables 14](#A2.T14 "In B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [15](#A2.T15 "Table 15 ‣ B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for heterogeneous features, and in [Tables 16](#A2.T16 "In B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [17](#A2.T17 "Table 17 ‣ B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [18](#A2.T18 "Table 18 ‣ B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for numerical features only.

For large-sized regression datasets, the results can be found in [Table 19](#A2.T19 "In B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for heterogeneous features, and in [Table 20](#A2.T20 "In B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for numerical features only.

Furthermore, individual figures illustrating the performance of Trompt on medium-sized regression tasks are provided in [Figure 15](#A2.F15 "In B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for heterogeneous features, and in [Figure 16](#A2.F16 "In B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for numerical features only.
The individual figures for large-sized tasks can be found in [Figure 17](#A2.F17 "In B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for heterogeneous features, and in [Figure 18](#A2.F18 "In B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") for numerical features only.

The evaluation results consistently demonstrate that Trompt outperforms state-of-the-art deep neural networks (SAINT and FT-Transformer) on medium-sized regression tasks (refer to [Tables 14](#A2.T14 "In B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [15](#A2.T15 "Table 15 ‣ B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [16](#A2.T16 "Table 16 ‣ B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [17](#A2.T17 "Table 17 ‣ B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [18](#A2.T18 "Table 18 ‣ B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")).
However, Trompt’s performance is slightly inferior to other deep neural networks on large-sized datasets (refer to [Tables 19](#A2.T19 "In B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [20](#A2.T20 "Table 20 ‣ B.1.2 Regression ‣ B.1 Grinsztajn45 ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")).
Nevertheless, it is worth noting that the performance of Trompt remains consistently competitive when considering all benchmark results.

Table 14: The performance of medium-sized regression task (*heterogeneous features*) (1).

  

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| Default | | | | | | | | |
| Trompt (ours) | 93.93%percent93.9393.93\% | 99.63%percent99.6399.63\% | 54.09%percent54.0954.09\% | 8.71%percent8.718.71\% | 99.96%percent99.9699.96\% | 94.70%percent94.7094.70\% | 57.94%percent57.9457.94\% | 98.88%percent98.8898.88\% |
| FT-Transformer | 93.21%percent93.2193.21\% | 88.00%percent88.0088.00\% | 54.24%percent54.2454.24\% | 0.00%percent0.000.00\% | 99.96%percent99.9699.96\% | 93.99%percent93.9993.99\% | 31.46%percent31.4631.46\% | 98.84%percent98.8498.84\% |
| ResNet | 89.90%percent89.9089.90\% | 87.47%percent87.4787.47\% | 51.99%percent51.9951.99\% | 0.00%percent0.000.00\% | 99.72%percent99.7299.72\% | 91.09%percent91.0991.09\% | 10.79%percent10.7910.79\% | 98.46%percent98.4698.46\% |
| SAINT | 92.50%percent92.5092.50\% | 99.20%percent99.2099.20\% | 54.25%percent54.2554.25\% | 11.23%percent11.2311.23\% | 99.96%percent99.9699.96\% | 95.10%percent95.1095.10\% | 40.72%percent40.7240.72\% | 98.47%percent98.4798.47\% |
| CatBoost | 94.21%percent94.2194.21\% | 99.59%percent99.5999.59\% | 56.33%percent56.3356.33\% | 15.16%percent15.1615.16\% | 99.97%percent99.9799.97\% | 98.01%percent98.0198.01\% | 61.70%percent61.7061.70\% | 99.11%percent99.1199.11\% |
| LightGBM | 94.02%percent94.0294.02\% | 99.38%percent99.3899.38\% | 54.77%percent54.7754.77\% | 14.41%percent14.4114.41\% | 99.97%percent99.9799.97\% | 98.23%percent98.2398.23\% | 61.68%percent61.6861.68\% | 99.01%percent99.0199.01\% |
| XGBoost | 93.93%percent93.9393.93\% | 99.76%percent99.7699.76\% | 49.71%percent49.7149.71\% | 6.64%percent6.646.64\% | 99.97%percent99.9799.97\% | 97.59%percent97.5997.59\% | 58.93%percent58.9358.93\% | 98.96%percent98.9698.96\% |
| RandomForest | 93.61%percent93.6193.61\% | 99.30%percent99.3099.30\% | 50.78%percent50.7850.78\% | 13.16%percent13.1613.16\% | 99.98%percent99.9899.98\% | 98.00%percent98.0098.00\% | 55.85%percent55.8555.85\% | 98.79%percent98.7998.79\% |
| GradientBoostingTree | 84.15%percent84.1584.15\% | 99.62%percent99.6299.62\% | 57.17%percent57.1757.17\% | 15.30%percent15.3015.30\% | 99.97%percent99.9799.97\% | 98.27%percent98.2798.27\% | 61.34%percent61.3461.34\% | 98.42%percent98.4298.42\% |
| Searched | | | | | | | | |
| Trompt (ours) | 94.50%percent94.5094.50\% | 99.75%percent99.7599.75\% | 56.87%percent56.8756.87\% | 13.05%percent13.0513.05\% | 99.96%percent99.9699.96\% | 97.93%percent97.9397.93\% | 60.17%percent60.1760.17\% | 98.99%percent98.9998.99\% |
| FT-Transformer | 93.58%percent93.5893.58\% | 88.12%percent88.1288.12\% | 54.90%percent54.9054.90\% | 14.05%percent14.0514.05\% | 99.97%percent99.9799.97\% | 97.63%percent97.6397.63\% | 37.93%percent37.9337.93\% | 98.96%percent98.9698.96\% |
| ResNet | 93.65%percent93.6593.65\% | 87.83%percent87.8387.83\% | 54.47%percent54.4754.47\% | 12.95%percent12.9512.95\% | 99.96%percent99.9699.96\% | 97.83%percent97.8397.83\% | 35.56%percent35.5635.56\% | 98.79%percent98.7998.79\% |
| SAINT | 93.89%percent93.8993.89\% | 99.51%percent99.5199.51\% | 55.14%percent55.1455.14\% | 13.90%percent13.9013.90\% | 99.97%percent99.9799.97\% | 94.59%percent94.5994.59\% | 58.72%percent58.7258.72\% | 98.72%percent98.7298.72\% |
| CatBoost | 94.87%percent94.8794.87\% | 99.60%percent99.6099.60\% | 57.74%percent57.7457.74\% | 16.54%percent16.5416.54\% | 99.97%percent99.9799.97\% | 98.33%percent98.3398.33\% | 61.79%percent61.7961.79\% | 99.18%percent99.1899.18\% |
| LightGBM | 94.37%percent94.3794.37\% | 99.42%percent99.4299.42\% | 55.58%percent55.5855.58\% | 14.41%percent14.4114.41\% | 99.97%percent99.9799.97\% | 98.23%percent98.2398.23\% | 61.68%percent61.6861.68\% | 99.07%percent99.0799.07\% |
| XGBoost | 94.62%percent94.6294.62\% | 99.76%percent99.7699.76\% | 56.87%percent56.8756.87\% | 16.21%percent16.2116.21\% | 99.98%percent99.9899.98\% | 98.30%percent98.3098.30\% | 61.88%percent61.8861.88\% | 99.12%percent99.1299.12\% |
| RandomForest | 93.79%percent93.7993.79\% | 99.34%percent99.3499.34\% | 57.55%percent57.5557.55\% | 14.94%percent14.9414.94\% | 99.98%percent99.9899.98\% | 98.07%percent98.0798.07\% | 60.91%percent60.9160.91\% | 98.79%percent98.7998.79\% |
| GradientBoostingTree | 94.07%percent94.0794.07\% | 99.46%percent99.4699.46\% | 57.53%percent57.5357.53\% | 15.27%percent15.2715.27\% | 99.98%percent99.9899.98\% | 98.13%percent98.1398.13\% | 61.54%percent61.5461.54\% | 98.98%percent98.9898.98\% |

Table 15: The performance of medium-sized regression task (*heterogeneous features*) (2).

  

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | C9 | C10 | C11 | C12 | C13 | Ranking |
| Default | | | | | | |
| Trompt (ours) | 89.02%percent89.0289.02\% | 9.61%percent9.619.61\% | 64.94%percent64.9464.94\% | 99.95%percent99.9599.95\% | 0.64%percent0.640.64\% | 5.38±2.02plus-or-minus5.382.025.38\pm 2.02 |
| FT-Transformer | 87.38%percent87.3887.38\% | 12.38%percent12.3812.38\% | 65.43%percent65.4365.43\% | 99.94%percent99.9499.94\% | 0.00%percent0.000.00\% | 6.88±1.74plus-or-minus6.881.746.88\pm 1.74 |
| ResNet | 86.45%percent86.4586.45\% | 0.00%percent0.000.00\% | 65.23%percent65.2365.23\% | 98.70%percent98.7098.70\% | 0.00%percent0.000.00\% | 8.35±2.13plus-or-minus8.352.138.35\pm 2.13 |
| SAINT | 88.01%percent88.0188.01\% | 17.48%percent17.4817.48\% | 64.80%percent64.8064.80\% | 99.98%percent99.9899.98\% | 0.00%percent0.000.00\% | 6.31±1.54plus-or-minus6.311.546.31\pm 1.54 |
| CatBoost | 89.75%percent89.7589.75\% | 54.63%percent54.6354.63\% | 69.16%percent69.1669.16\% | 99.99%percent99.9999.99\% | 4.97%percent4.974.97\% | 2.15±2.13plus-or-minus2.152.132.15\pm 2.13 |
| LightGBM | 89.05%percent89.0589.05\% | 54.48%percent54.4854.48\% | 68.74%percent68.7468.74\% | 99.99%percent99.9999.99\% | 4.91%percent4.914.91\% | 3.00±1.74plus-or-minus3.001.743.00\pm 1.74 |
| XGBoost | 88.34%percent88.3488.34\% | 56.99%percent56.9956.99\% | 66.16%percent66.1666.16\% | 100.00%percent100.00100.00\% | 0.00%percent0.000.00\% | 4.08±2.46plus-or-minus4.082.464.08\pm 2.46 |
| RandomForest | 87.44%percent87.4487.44\% | 56.18%percent56.1856.18\% | 65.44%percent65.4465.44\% | 100.00%percent100.00100.00\% | 5.92%percent5.925.92\% | 4.23±2.27plus-or-minus4.232.274.23\pm 2.27 |
| GradientBoostingTree | 86.93%percent86.9386.93\% | 46.90%percent46.9046.90\% | 67.17%percent67.1767.17\% | 99.94%percent99.9499.94\% | 0.00%percent0.000.00\% | 4.62±2.92plus-or-minus4.622.924.62\pm 2.92 |
| Searched | | | | | | |
| Trompt (ours) | 89.16%percent89.1689.16\% | 48.04%percent48.0448.04\% | 66.33%percent66.3366.33\% | 99.99%percent99.9999.99\% | 3.59%percent3.593.59\% | 5.77±1.98plus-or-minus5.771.985.77\pm 1.98 |
| FT-Transformer | 88.85%percent88.8588.85\% | 50.44%percent50.4450.44\% | 67.18%percent67.1867.18\% | 99.90%percent99.9099.90\% | 3.18%percent3.183.18\% | 7.23±1.70plus-or-minus7.231.707.23\pm 1.70 |
| ResNet | 88.10%percent88.1088.10\% | 42.42%percent42.4242.42\% | 65.50%percent65.5065.50\% | 99.76%percent99.7699.76\% | 2.11%percent2.112.11\% | 8.31±2.08plus-or-minus8.312.088.31\pm 2.08 |
| SAINT | 89.18%percent89.1889.18\% | 36.42%percent36.4236.42\% | 66.93%percent66.9366.93\% | 99.99%percent99.9999.99\% | 1.21%percent1.211.21\% | 7.00±1.98plus-or-minus7.001.987.00\pm 1.98 |
| CatBoost | 89.84%percent89.8489.84\% | 56.79%percent56.7956.79\% | 69.33%percent69.3369.33\% | 100.00%percent100.00100.00\% | 9.08%percent9.089.08\% | 2.00±2.24plus-or-minus2.002.242.00\pm 2.24 |
| LightGBM | 89.33%percent89.3389.33\% | 54.48%percent54.4854.48\% | 68.74%percent68.7468.74\% | 100.00%percent100.00100.00\% | 4.91%percent4.914.91\% | 4.31±1.18plus-or-minus4.311.184.31\pm 1.18 |
| XGBoost | 89.65%percent89.6589.65\% | 57.82%percent57.8257.82\% | 69.08%percent69.0869.08\% | 100.00%percent100.00100.00\% | 8.01%percent8.018.01\% | 2.15±1.74plus-or-minus2.151.742.15\pm 1.74 |
| RandomForest | 87.50%percent87.5087.50\% | 58.48%percent58.4858.48\% | 67.44%percent67.4467.44\% | 100.00%percent100.00100.00\% | 9.52%percent9.529.52\% | 4.31±2.80plus-or-minus4.312.804.31\pm 2.80 |
| GradientBoostingTree | 89.05%percent89.0589.05\% | 57.29%percent57.2957.29\% | 68.30%percent68.3068.30\% | 100.00%percent100.00100.00\% | 5.54%percent5.545.54\% | 3.92±1.35plus-or-minus3.921.353.92\pm 1.35 |

Table 16: The performance of medium-sized regression task (*numerical features only*) (1).

  

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | D1 | D2 | D3 | D4 | D5 | D6 | D7 |
| Default | | | | | | | |
| Trompt (ours) | 84.80%percent84.8084.80\% | 68.29%percent68.2968.29\% | 99.70%percent99.7099.70\% | 92.75%percent92.7592.75\% | 81.17%percent81.1781.17\% | 97.23%percent97.2397.23\% | 94.15%percent94.1594.15\% |
| FT-Transformer | 83.80%percent83.8083.80\% | 66.92%percent66.9266.92\% | 99.71%percent99.7199.71\% | 91.87%percent91.8791.87\% | 79.20%percent79.2079.20\% | 96.85%percent96.8596.85\% | 93.85%percent93.8593.85\% |
| ResNet | 82.54%percent82.5482.54\% | 64.52%percent64.5264.52\% | 99.57%percent99.5799.57\% | 91.41%percent91.4191.41\% | 75.06%percent75.0675.06\% | 96.75%percent96.7596.75\% | n​a​n𝑛𝑎𝑛nan |
| SAINT | 0.00%percent0.000.00\% | 67.85%percent67.8567.85\% | 99.39%percent99.3999.39\% | 91.46%percent91.4691.46\% | 82.04%percent82.0482.04\% | 98.33%percent98.3398.33\% | 94.24%percent94.2494.24\% |
| CatBoost | 85.76%percent85.7685.76\% | 69.93%percent69.9369.93\% | 99.60%percent99.6099.60\% | 93.56%percent93.5693.56\% | 86.16%percent86.1686.16\% | 98.56%percent98.5698.56\% | 94.57%percent94.5794.57\% |
| LightGBM | 84.68%percent84.6884.68\% | 69.28%percent69.2869.28\% | 99.38%percent99.3899.38\% | 92.25%percent92.2592.25\% | 84.33%percent84.3384.33\% | 98.46%percent98.4698.46\% | 94.49%percent94.4994.49\% |
| XGBoost | 82.58%percent82.5882.58\% | 67.93%percent67.9367.93\% | 99.76%percent99.7699.76\% | 92.03%percent92.0392.03\% | 84.04%percent84.0484.04\% | 98.25%percent98.2598.25\% | 94.09%percent94.0994.09\% |
| RandomForest | 83.71%percent83.7183.71\% | 67.32%percent67.3267.32\% | 99.29%percent99.2999.29\% | 91.41%percent91.4191.41\% | 81.54%percent81.5481.54\% | 98.23%percent98.2398.23\% | 93.96%percent93.9693.96\% |
| GradientBoostingTree | 83.95%percent83.9583.95\% | 67.58%percent67.5867.58\% | 99.62%percent99.6299.62\% | 89.42%percent89.4289.42\% | 80.46%percent80.4680.46\% | 98.34%percent98.3498.34\% | 94.41%percent94.4194.41\% |
| Searched | | | | | | | |
| Trompt (ours) | 85.08%percent85.0885.08\% | 68.57%percent68.5768.57\% | 99.62%percent99.6299.62\% | 92.80%percent92.8092.80\% | 84.53%percent84.5384.53\% | 98.61%percent98.6198.61\% | 94.31%percent94.3194.31\% |
| FT-Transformer | 83.90%percent83.9083.90\% | 67.17%percent67.1767.17\% | 99.77%percent99.7799.77\% | 91.87%percent91.8791.87\% | 83.00%percent83.0083.00\% | 97.87%percent97.8797.87\% | 94.34%percent94.3494.34\% |
| ResNet | 83.21%percent83.2183.21\% | 66.71%percent66.7166.71\% | 99.69%percent99.6999.69\% | 91.36%percent91.3691.36\% | 82.03%percent82.0382.03\% | 98.07%percent98.0798.07\% | n​a​n𝑛𝑎𝑛nan |
| SAINT | 78.31%percent78.3178.31\% | 68.44%percent68.4468.44\% | 99.41%percent99.4199.41\% | 92.10%percent92.1092.10\% | 83.67%percent83.6783.67\% | 98.39%percent98.3998.39\% | 94.42%percent94.4294.42\% |
| CatBoost | 85.92%percent85.9285.92\% | 70.31%percent70.3170.31\% | 99.62%percent99.6299.62\% | 93.78%percent93.7893.78\% | 86.90%percent86.9086.90\% | 98.67%percent98.6798.67\% | 94.59%percent94.5994.59\% |
| LightGBM | 84.68%percent84.6884.68\% | 69.28%percent69.2869.28\% | 99.28%percent99.2899.28\% | 93.33%percent93.3393.33\% | 84.80%percent84.8084.80\% | 98.31%percent98.3198.31\% | 94.49%percent94.4994.49\% |
| XGBoost | 84.58%percent84.5884.58\% | 69.43%percent69.4369.43\% | 99.76%percent99.7699.76\% | 93.59%percent93.5993.59\% | 85.64%percent85.6485.64\% | 98.61%percent98.6198.61\% | 94.55%percent94.5594.55\% |
| RandomForest | 83.75%percent83.7583.75\% | 68.69%percent68.6968.69\% | 99.33%percent99.3399.33\% | 92.42%percent92.4292.42\% | 83.02%percent83.0283.02\% | 98.28%percent98.2898.28\% | 94.53%percent94.5394.53\% |
| GradientBoostingTree | 84.25%percent84.2584.25\% | 68.94%percent68.9468.94\% | 99.60%percent99.6099.60\% | 92.43%percent92.4392.43\% | 84.48%percent84.4884.48\% | 98.51%percent98.5198.51\% | 94.47%percent94.4794.47\% |

Table 17: The performance of medium-sized regression task (*numerical features only*) (2).

  

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | D8 | D9 | D10 | D11 | D12 | D13 | D14 |
| Default | | | | | | | |
| Trompt (ours) | 89.69%percent89.6989.69\% | 62.96%percent62.9662.96\% | 54.53%percent54.5354.53\% | 88.04%percent88.0488.04\% | 83.52%percent83.5283.52\% | 97.88%percent97.8897.88\% | 16.99%percent16.9916.99\% |
| FT-Transformer | 91.01%percent91.0191.01\% | 63.03%percent63.0363.03\% | 48.90%percent48.9048.90\% | 87.42%percent87.4287.42\% | 81.10%percent81.1081.10\% | 97.82%percent97.8297.82\% | 5.86%percent5.865.86\% |
| ResNet | 88.77%percent88.7788.77\% | 62.01%percent62.0162.01\% | 47.62%percent47.6247.62\% | 84.71%percent84.7184.71\% | 75.92%percent75.9275.92\% | 97.80%percent97.8097.80\% | 22.34%percent22.3422.34\% |
| SAINT | 87.30%percent87.3087.30\% | 64.59%percent64.5964.59\% | 50.30%percent50.3050.30\% | 87.34%percent87.3487.34\% | 81.59%percent81.5981.59\% | 97.81%percent97.8197.81\% | 46.65%percent46.6546.65\% |
| CatBoost | 91.17%percent91.1791.17\% | 66.18%percent66.1866.18\% | 51.01%percent51.0151.01\% | 88.73%percent88.7388.73\% | 84.72%percent84.7284.72\% | 97.82%percent97.8297.82\% | 52.91%percent52.9152.91\% |
| LightGBM | 88.59%percent88.5988.59\% | 66.49%percent66.4966.49\% | 51.95%percent51.9551.95\% | 88.12%percent88.1288.12\% | 83.51%percent83.5183.51\% | 97.85%percent97.8597.85\% | 53.06%percent53.0653.06\% |
| XGBoost | 88.48%percent88.4888.48\% | 64.75%percent64.7564.75\% | 48.14%percent48.1448.14\% | 87.43%percent87.4387.43\% | 83.74%percent83.7483.74\% | 97.73%percent97.7397.73\% | 54.87%percent54.8754.87\% |
| RandomForest | 83.37%percent83.3783.37\% | 63.58%percent63.5863.58\% | 51.12%percent51.1251.12\% | 86.87%percent86.8786.87\% | 82.99%percent82.9982.99\% | 97.67%percent97.6797.67\% | 54.54%percent54.5454.54\% |
| GradientBoostingTree | 80.22%percent80.2280.22\% | 66.31%percent66.3166.31\% | 47.33%percent47.3347.33\% | 86.16%percent86.1686.16\% | 78.74%percent78.7478.74\% | 97.94%percent97.9497.94\% | 45.15%percent45.1545.15\% |
| Searched | | | | | | | |
| Trompt (ours) | 90.69%percent90.6990.69\% | 65.13%percent65.1365.13\% | 46.50%percent46.5046.50\% | 88.27%percent88.2788.27\% | 83.57%percent83.5783.57\% | 97.92%percent97.9297.92\% | 45.57%percent45.5745.57\% |
| FT-Transformer | 91.37%percent91.3791.37\% | 64.69%percent64.6964.69\% | 48.67%percent48.6748.67\% | 87.56%percent87.5687.56\% | 83.05%percent83.0583.05\% | 97.92%percent97.9297.92\% | 47.43%percent47.4347.43\% |
| ResNet | 90.82%percent90.8290.82\% | 64.19%percent64.1964.19\% | 48.16%percent48.1648.16\% | 86.72%percent86.7286.72\% | 82.08%percent82.0882.08\% | 97.91%percent97.9197.91\% | 46.78%percent46.7846.78\% |
| SAINT | 92.27%percent92.2792.27\% | 65.06%percent65.0665.06\% | 49.40%percent49.4049.40\% | 87.87%percent87.8787.87\% | 82.03%percent82.0382.03\% | 97.94%percent97.9497.94\% | 49.58%percent49.5849.58\% |
| CatBoost | 91.56%percent91.5691.56\% | 66.39%percent66.3966.39\% | 41.22%percent41.2241.22\% | 88.89%percent88.8988.89\% | 85.53%percent85.5385.53\% | 97.93%percent97.9397.93\% | 54.06%percent54.0654.06\% |
| LightGBM | 88.59%percent88.5988.59\% | 66.49%percent66.4966.49\% | 51.60%percent51.6051.60\% | 88.45%percent88.4588.45\% | 85.33%percent85.3385.33\% | 97.85%percent97.8597.85\% | 53.06%percent53.0653.06\% |
| XGBoost | 90.67%percent90.6790.67\% | 66.79%percent66.7966.79\% | 54.63%percent54.6354.63\% | 88.76%percent88.7688.76\% | 84.95%percent84.9584.95\% | 97.87%percent97.8797.87\% | 55.23%percent55.2355.23\% |
| RandomForest | 83.82%percent83.8283.82\% | 65.47%percent65.4765.47\% | 49.15%percent49.1549.15\% | 87.10%percent87.1087.10\% | 82.77%percent82.7782.77\% | 97.89%percent97.8997.89\% | 56.04%percent56.0456.04\% |
| GradientBoostingTree | 85.84%percent85.8485.84\% | 66.32%percent66.3266.32\% | 52.49%percent52.4952.49\% | 88.32%percent88.3288.32\% | 84.07%percent84.0784.07\% | 97.94%percent97.9497.94\% | 55.21%percent55.2155.21\% |

Table 18: The performance of medium-sized regression task (*numerical features only*) (3).

  

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | D15 | D16 | D17 | D18 | D19 | Ranking |
| Default | | | | | | |
| Trompt (ours) | 95.13%percent95.1395.13\% | 80.96%percent80.9680.96\% | 87.91%percent87.9187.91\% | 31.68%percent31.6831.68\% | 18.41%percent18.4118.41\% | 4.68±2.29plus-or-minus4.682.294.68\pm 2.29 |
| FT-Transformer | 94.16%percent94.1694.16\% | 82.70%percent82.7082.70\% | 88.01%percent88.0188.01\% | 26.98%percent26.9826.98\% | 0.00%percent0.000.00\% | 6.21±2.29plus-or-minus6.212.296.21\pm 2.29 |
| ResNet | 84.68%percent84.6884.68\% | 74.54%percent74.5474.54\% | 87.14%percent87.1487.14\% | 26.86%percent26.8626.86\% | 8.13%percent8.138.13\% | 8.06±2.08plus-or-minus8.062.088.06\pm 2.08 |
| SAINT | 99.04%percent99.0499.04\% | 80.52%percent80.5280.52\% | 89.22%percent89.2289.22\% | 36.25%percent36.2536.25\% | 25.92%percent25.9225.92\% | 5.32±1.86plus-or-minus5.321.865.32\pm 1.86 |
| CatBoost | 98.63%percent98.6398.63\% | 86.85%percent86.8586.85\% | 90.51%percent90.5190.51\% | 45.00%percent45.0045.00\% | 27.34%percent27.3427.34\% | 2.05±2.16plus-or-minus2.052.162.05\pm 2.16 |
| LightGBM | 98.70%percent98.7098.70\% | 81.43%percent81.4381.43\% | 89.79%percent89.7989.79\% | 42.86%percent42.8642.86\% | 25.50%percent25.5025.50\% | 3.05±1.89plus-or-minus3.051.893.05\pm 1.89 |
| XGBoost | 98.50%percent98.5098.50\% | 83.49%percent83.4983.49\% | 89.55%percent89.5589.55\% | 42.37%percent42.3742.37\% | 16.33%percent16.3316.33\% | 4.47±2.04plus-or-minus4.472.044.47\pm 2.04 |
| RandomForest | 98.67%percent98.6798.67\% | 84.47%percent84.4784.47\% | 90.20%percent90.2090.20\% | 48.28%percent48.2848.28\% | 20.69%percent20.6920.69\% | 5.26±2.40plus-or-minus5.262.405.26\pm 2.40 |
| GradientBoostingTree | 93.49%percent93.4993.49\% | 81.04%percent81.0481.04\% | 85.62%percent85.6285.62\% | 37.57%percent37.5737.57\% | 24.21%percent24.2124.21\% | 5.84±2.60plus-or-minus5.842.605.84\pm 2.60 |
| Searched | | | | | | |
| Trompt (ours) | 99.58%percent99.5899.58\% | 84.15%percent84.1584.15\% | 89.49%percent89.4989.49\% | 40.91%percent40.9140.91\% | 26.03%percent26.0326.03\% | 5.11±1.97plus-or-minus5.111.975.11\pm 1.97 |
| FT-Transformer | 99.44%percent99.4499.44\% | 84.26%percent84.2684.26\% | 88.26%percent88.2688.26\% | 36.07%percent36.0736.07\% | 23.96%percent23.9623.96\% | 6.37±2.48plus-or-minus6.372.486.37\pm 2.48 |
| ResNet | 94.99%percent94.9994.99\% | 81.45%percent81.4581.45\% | 89.22%percent89.2289.22\% | 36.11%percent36.1136.11\% | 21.73%percent21.7321.73\% | 7.61±2.31plus-or-minus7.612.317.61\pm 2.31 |
| SAINT | 99.56%percent99.5699.56\% | 78.81%percent78.8178.81\% | 89.37%percent89.3789.37\% | 37.38%percent37.3837.38\% | 26.45%percent26.4526.45\% | 5.79±2.46plus-or-minus5.792.465.79\pm 2.46 |
| CatBoost | 99.24%percent99.2499.24\% | 86.84%percent86.8486.84\% | 90.94%percent90.9490.94\% | 50.11%percent50.1150.11\% | 28.26%percent28.2628.26\% | 2.26±2.46plus-or-minus2.262.462.26\pm 2.46 |
| LightGBM | 98.70%percent98.7098.70\% | 81.31%percent81.3181.31\% | 90.48%percent90.4890.48\% | 42.86%percent42.8642.86\% | 25.50%percent25.5025.50\% | 4.84±2.25plus-or-minus4.842.254.84\pm 2.25 |
| XGBoost | 98.97%percent98.9798.97\% | 86.03%percent86.0386.03\% | 91.02%percent91.0291.02\% | 50.06%percent50.0650.06\% | 28.04%percent28.0428.04\% | 2.79±2.11plus-or-minus2.792.112.79\pm 2.11 |
| RandomForest | 98.87%percent98.8798.87\% | 85.64%percent85.6485.64\% | 90.89%percent90.8990.89\% | 50.43%percent50.4350.43\% | 24.09%percent24.0924.09\% | 5.58±2.33plus-or-minus5.582.335.58\pm 2.33 |
| GradientBoostingTree | 98.91%percent98.9198.91\% | 81.31%percent81.3181.31\% | 90.36%percent90.3690.36\% | 45.55%percent45.5545.55\% | 26.94%percent26.9426.94\% | 4.58±1.69plus-or-minus4.581.694.58\pm 1.69 |

Table 19: The performance of large-sized regression task (*heterogeneous features*).

  

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | ℂℂ\mathbb{C}1 | ℂℂ\mathbb{C}2 | ℂℂ\mathbb{C}3 | ℂℂ\mathbb{C}4 | ℂℂ\mathbb{C}5 | Ranking |
| Default | | | | | | |
| Trompt (ours) | 99.96%percent99.9699.96\% | 60.97%percent60.9760.97\% | 99.17%percent99.1799.17\% | 40.35%percent40.3540.35\% | 70.48%percent70.4870.48\% | 5.20±1.50plus-or-minus5.201.505.20\pm 1.50 |
| FT-Transformer | 99.94%percent99.9499.94\% | 35.14%percent35.1435.14\% | 99.23%percent99.2399.23\% | 40.61%percent40.6140.61\% | 67.61%percent67.6167.61\% | 5.80±2.48plus-or-minus5.802.485.80\pm 2.48 |
| ResNet | 98.95%percent98.9598.95\% | 33.70%percent33.7033.70\% | 98.16%percent98.1698.16\% | 39.71%percent39.7139.71\% | 66.60%percent66.6066.60\% | 8.00±2.86plus-or-minus8.002.868.00\pm 2.86 |
| SAINT | 99.97%percent99.9799.97\% | 38.91%percent38.9138.91\% | 99.18%percent99.1899.18\% | 54.80%percent54.8054.80\% | 68.74%percent68.7468.74\% | 4.80±0.75plus-or-minus4.800.754.80\pm 0.75 |
| CatBoost | 99.98%percent99.9899.98\% | 63.32%percent63.3263.32\% | 99.28%percent99.2899.28\% | 60.50%percent60.5060.50\% | 70.68%percent70.6870.68\% | 1.80±2.25plus-or-minus1.802.251.80\pm 2.25 |
| LightGBM | 99.98%percent99.9899.98\% | 63.24%percent63.2463.24\% | 99.16%percent99.1699.16\% | 57.69%percent57.6957.69\% | 70.37%percent70.3770.37\% | 3.60±1.67plus-or-minus3.601.673.60\pm 1.67 |
| XGBoost | 99.98%percent99.9899.98\% | 63.45%percent63.4563.45\% | 99.22%percent99.2299.22\% | 62.44%percent62.4462.44\% | 70.60%percent70.6070.60\% | 1.60±2.73plus-or-minus1.602.731.60\pm 2.73 |
| RandomForest | −- | −- | −- | −- | −- | −- |
| GradientBoostingTree | 99.98%percent99.9899.98\% | 61.65%percent61.6561.65\% | 98.57%percent98.5798.57\% | 48.09%percent48.0948.09\% | 67.73%percent67.7367.73\% | 5.20±1.36plus-or-minus5.201.365.20\pm 1.36 |
| Searched | | | | | | |
| Trompt (ours) | 99.98%percent99.9899.98\% | 62.86%percent62.8662.86\% | 99.18%percent99.1899.18\% | 54.79%percent54.7954.79\% | 70.73%percent70.7370.73\% | 6.20±2.26plus-or-minus6.202.266.20\pm 2.26 |
| FT-Transformer | 99.98%percent99.9899.98\% | 39.00%percent39.0039.00\% | 99.26%percent99.2699.26\% | 57.02%percent57.0257.02\% | 70.45%percent70.4570.45\% | 5.80±1.75plus-or-minus5.801.755.80\pm 1.75 |
| ResNet | 99.98%percent99.9899.98\% | 39.38%percent39.3839.38\% | 99.23%percent99.2399.23\% | 54.30%percent54.3054.30\% | 68.71%percent68.7168.71\% | 6.40±2.88plus-or-minus6.402.886.40\pm 2.88 |
| SAINT | 99.98%percent99.9899.98\% | 39.53%percent39.5339.53\% | 99.26%percent99.2699.26\% | 56.58%percent56.5856.58\% | 69.73%percent69.7369.73\% | 5.20±1.55plus-or-minus5.201.555.20\pm 1.55 |
| CatBoost | 99.98%percent99.9899.98\% | 63.62%percent63.6263.62\% | 99.33%percent99.3399.33\% | 62.64%percent62.6462.64\% | 71.17%percent71.1771.17\% | 2.60±2.25plus-or-minus2.602.252.60\pm 2.25 |
| LightGBM | 99.98%percent99.9899.98\% | 63.24%percent63.2463.24\% | 99.24%percent99.2499.24\% | 57.69%percent57.6957.69\% | 70.99%percent70.9970.99\% | 4.60±1.86plus-or-minus4.601.864.60\pm 1.86 |
| XGBoost | 99.98%percent99.9899.98\% | 63.90%percent63.9063.90\% | 99.32%percent99.3299.32\% | 64.79%percent64.7964.79\% | 71.22%percent71.2271.22\% | 1.20±2.80plus-or-minus1.202.801.20\pm 2.80 |
| RandomForest | −- | −- | −- | −- | −- | −- |
| GradientBoostingTree | 99.98%percent99.9899.98\% | 63.06%percent63.0663.06\% | 99.18%percent99.1899.18\% | 63.62%percent63.6263.62\% | 70.58%percent70.5870.58\% | 4.00±2.07plus-or-minus4.002.074.00\pm 2.07 |

Table 20: The performance of large-sized regression task (*numerical features only*).

  

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝔻𝔻\mathbb{D}1 | 𝔻𝔻\mathbb{D}2 | 𝔻𝔻\mathbb{D}3 | Ranking |
| Default | | | | |
| Trompt (ours) | 94.58%percent94.5894.58\% | 33.79%percent33.7933.79\% | 24.98%percent24.9824.98\% | 5.67±1.41plus-or-minus5.671.415.67\pm 1.41 |
| FT-Transformer | 94.52%percent94.5294.52\% | 11.98%percent11.9811.98\% | 11.72%percent11.7211.72\% | 7.33±3.07plus-or-minus7.333.077.33\pm 3.07 |
| ResNet | 94.10%percent94.1094.10\% | 24.69%percent24.6924.69\% | 11.88%percent11.8811.88\% | 7.33±2.95plus-or-minus7.332.957.33\pm 2.95 |
| SAINT | 94.45%percent94.4594.45\% | 53.44%percent53.4453.44\% | 28.87%percent28.8728.87\% | 4.33±2.06plus-or-minus4.332.064.33\pm 2.06 |
| CatBoost | 94.76%percent94.7694.76\% | 58.47%percent58.4758.47\% | 30.20%percent30.2030.20\% | 1.33±3.37plus-or-minus1.333.371.33\pm 3.37 |
| LightGBM | 94.75%percent94.7594.75\% | 56.07%percent56.0756.07\% | 28.10%percent28.1028.10\% | 2.67±2.22plus-or-minus2.672.222.67\pm 2.22 |
| XGBoost | 94.74%percent94.7494.74\% | 60.87%percent60.8760.87\% | 25.12%percent25.1225.12\% | 3.00±2.22plus-or-minus3.002.223.00\pm 2.22 |
| RandomForest | −- | −- | −- | −- |
| GradientBoostingTree | 94.59%percent94.5994.59\% | 46.35%percent46.3546.35\% | 25.74%percent25.7425.74\% | 4.33±0.48plus-or-minus4.330.484.33\pm 0.48 |
| Searched | | | | |
| Trompt (ours) | 94.61%percent94.6194.61\% | 52.42%percent52.4252.42\% | 29.71%percent29.7129.71\% | 7.33±3.30plus-or-minus7.333.307.33\pm 3.30 |
| FT-Transformer | 94.63%percent94.6394.63\% | 53.82%percent53.8253.82\% | 30.51%percent30.5130.51\% | 5.67±1.83plus-or-minus5.671.835.67\pm 1.83 |
| ResNet | 94.64%percent94.6494.64\% | 52.84%percent52.8452.84\% | 28.01%percent28.0128.01\% | 7.00±2.63plus-or-minus7.002.637.00\pm 2.63 |
| SAINT | 94.65%percent94.6594.65\% | 54.94%percent54.9454.94\% | 30.46%percent30.4630.46\% | 5.00±0.50plus-or-minus5.000.505.00\pm 0.50 |
| CatBoost | 94.80%percent94.8094.80\% | 59.97%percent59.9759.97\% | 31.30%percent31.3031.30\% | 2.00±2.63plus-or-minus2.002.632.00\pm 2.63 |
| LightGBM | 94.75%percent94.7594.75\% | 56.07%percent56.0756.07\% | 28.10%percent28.1028.10\% | 4.67±1.71plus-or-minus4.671.714.67\pm 1.71 |
| XGBoost | 94.80%percent94.8094.80\% | 62.36%percent62.3662.36\% | 30.75%percent30.7530.75\% | 1.33±3.37plus-or-minus1.333.371.33\pm 3.37 |
| RandomForest | −- | −- | −- | −- |
| GradientBoostingTree | 94.72%percent94.7294.72\% | 61.72%percent61.7261.72\% | 30.73%percent30.7330.73\% | 3.00±1.71plus-or-minus3.001.713.00\pm 1.71 |

!(/html/2305.18446/assets/figures/categorical_regression_quantile-individual.jpg)

Figure 15: Benchmark on *every* medium-sized regression dataset with heterogeneous features.

!(/html/2305.18446/assets/figures/numerical_regression_quantile-individual.jpg)

Figure 16: Benchmark on *every* medium-sized regression dataset with numerical features only.

!(/html/2305.18446/assets/figures/categorical_regression_quantile-individual-large.jpg)

Figure 17: Benchmark on *every* large-sized regression dataset with heterogeneous features.

!(/html/2305.18446/assets/figures/numerical_regression_quantile-individual-large.jpg)

Figure 18: Benchmark on *every* large-sized regression dataset with numerical features only.

### B.2 Datasets chosen by FT-Transformer

In this section, we further investigate the performance of Trompt on datasets selected by FT-Transformer (Gorishniy et al., [2021](#bib.bib16)), which encompass different domains, task types, and sizes.
To ensure a fair comparison, we adjust the model sizes of Trompt to match those of FT-Transformer by reducing the dimensions of its hidden layers.

It’s important to note that due to limited computing resources, Trompt did not undergo hyperparameter search.
Instead, we obtained the performance of FT-Transformer from its original paper.
In terms of the learning strategy, Trompt was trained for 100 epochs, and the performance was evaluated using the checkpoint at the 100th epoch.
This approach was adopted as we observed that the datasets chosen by FT-Transformer are often large, making overfitting less likely.

As shown in [Table 21](#A2.T21 "In B.2 Datasets chosen by FT-Transformer ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), Trompt generally achieves comparable or slightly inferior performance when compared to the default hyperparameter settings of FT-Transformer on the datasets specifically chosen by FT-Transformer.
It is important to note that the reported performance is an average result based on three random seeds.

Table 21: The performance on datasets chosen by FT-Transformer.

  

| Dataset | Metric | Trompt (ours) | FT (Default) | FT (Tune) | #Parameters (Trompt) | #Parameters (FT) |
| --- | --- | --- | --- | --- | --- | --- |
| CA | RMSE | 0.4740.4740.474 | 0.4690.4690.469 | 0.4590.4590.459 | 850,852  850852850,852 | 894,913  894913894,913 |
| AD | Acc. | 0.86290.86290.8629 | 0.8570.8570.857 | 0.8590.8590.859 | 863,509  863509863,509 | 915,458  915458915,458 |
| HE | Acc. | 0.36900.36900.3690 | 0.3810.3810.381 | 0.3910.3910.391 | 873,883  873883873,883 | 921,316  921316921,316 |
| JA | Acc. | 0.72690.72690.7269 | 0.7250.7250.725 | 0.7320.7320.732 | 876,079  876079876,079 | 913,156  913156913,156 |
| HI | Acc. | 0.72790.72790.7279 | 0.7250.7250.725 | 0.7290.7290.729 | 861,781  861781861,781 | 902,786  902786902,786 |
| AL | Acc. | 0.93170.93170.9317 | 0.9530.9530.953 | 0.960.960.96 | 1,044,523  10445231,044,523 | 1,133,800  11338001,133,800 |
| EP | Acc. | 0.89320.89320.8932 | 0.89590.89590.8959 | 0.89820.89820.8982 | 1,638,931  16389311,638,931 | 1,659,841  16598411,659,841 |
| YE | RMSE | 8.82188.82188.8218 | 8.8898.8898.889 | 8.8558.8558.855 | 895,132  895132895,132 | 926,401  926401926,401 |
| CO | Acc. | 0.90480.90480.9048 | 0.9670.9670.967 | 0.9700.9700.970 | 876,466  876466876,466 | 913,735  913735913,735 |
| YA | RMSE | 0.75370.75370.7537 | 0.7560.7560.756 | 0.7560.7560.756 | 1,223,992  12239921,223,992 | 1,160,257  11602571,160,257 |
| MI | RMSE | 0.74680.74680.7468 | 0.7470.7470.747 | 0.7460.7460.746 | 919,972  919972919,972 | 944,065  944065944,065 |

### B.3 Datasets chosen by SAINT

In this section, we conducted further evaluation of Trompt on datasets selected by SAINT (Somepalli et al., [2021](#bib.bib36)), which cover various domains, task types, and sizes.
To ensure fair comparison, we adjusted the model sizes of Trompt to match those of SAINT by reducing the dimensions of its hidden layers.

It is important to note that due to limited computing resources, Trompt did not undergo hyperparameter search.
Instead, we obtained the performances of SAINT from its original paper. In terms of the learning strategy, Trompt was trained for 100 epochs, and the performance was evaluated using the checkpoint with the lowest validation loss.
This approach was adopted as we observed that some datasets chosen by SAINT are often small, and models are more prone to overfitting.

As shown in [Table 22](#A2.T22 "In B.3 Datasets chosen by SAINT ‣ Appendix B More Evaluation Results ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), Trompt achieves comparable performance to SAINT on the datasets specifically chosen by SAINT.
It is worth mentioning that the reported performance is based on a single random seed.

Table 22: The performance on datasets chosen by SAINT.

  

| OpenML ID | Metric | Trompt (ours) | SAINT | #Parameters (Trompt) | #Parameters (SAINT) |
| --- | --- | --- | --- | --- | --- |
| 31 | AUC | 0.82650.82650.8265 | 0.79000.79000.7900 | 7,578,619  75786197,578,619 | 8,233,739  82337398,233,739 |
| 1017 | AUC | 0.89330.89330.8933 | 0.84300.84300.8430 | 39,521,539  3952153939,521,539 | 84,093,615  8409361584,093,615 |
| 44 | AUC | 0.98350.98350.9835 | 0.99100.99100.9910 | 38,675,971  3867597138,675,971 | 58,399,221  5839922158,399,221 |
| 1111 | AUC | 0.81140.81140.8114 | 0.80800.80800.8080 | 60,085,567  6008556760,085,567 | 61,716,420  6171642061,716,420 |
| 1487 | AUC | 0.92300.92300.9230 | 0.91900.91900.9190 | 38,733,571  3873357138,733,571 | 91,681,626  9168162691,681,626 |
| 1494 | AUC | 0.92580.92580.9258 | 0.93700.93700.9370 | 29,659,027  2965902729,659,027 | 31,136,311  3113631131,136,311 |
| 1590 | AUC | 0.91650.91650.9165 | 0.92100.92100.9210 | 3,945,643  39456433,945,643 | 4,420,452  44204524,420,452 |
| 4134 | AUC | 0.84190.84190.8419 | 0.85300.85300.8530 | 45,276,931  4527693145,276,931 | 3,296,373,186  32963731863,296,373,186 |
| 42178 | AUC | 0.84540.84540.8454 | 0.85700.85700.8570 | 65,51,239  655123965,51,239 | 7,500,881  75008817,500,881 |
| 42733 | AUC | 0.68200.68200.6820 | 0.67600.67600.6760 | 29,743,735  2974373529,743,735 | 30,585,898  3058589830,585,898 |
| 1596 | Acc. | 0.9602810.9602810.960281 | 0.94600.94600.9460 | 38,665,096  3866509638,665,096 | 52,507,599  5250759952,507,599 |
| 4541 | Acc. | 0.60710.60710.6071 | 0.60600.60600.6060 | 40,478,596  4047859640,478,596 | 44,131,471  4413147144,131,471 |
| 40664 | Acc. | 0.99130.99130.9913 | 1.00001.00001.0000 | 8,664,841  86648418,664,841 | 8,960,176  89601768,960,176 |
| 40685 | Acc. | 0.99970.99970.9997 | 0.99900.99900.9990 | 1,969,996  19699961,969,996 | 2,142,668  21426682,142,668 |
| 188 | Acc. | 0.66220.66220.6622 | 0.68000.68000.6800 | 6,569,098  65690986,569,098 | 7,547,934  75479347,547,934 |
| 40687 | Acc. | 0.74630.74630.7463 | 0.73500.73500.7350 | 3,203,035  32030353,203,035 | 3,381,200  33812003,381,200 |
| 40975 | Acc. | 0.98840.98840.9884 | 0.99700.99700.9970 | 1,037,761  10377611,037,761 | 1,147,867  11478671,147,867 |
| 41166 | Acc. | 0.70640.70640.7064 | 0.70100.70100.7010 | 34,490,755  3449075534,490,755 | 35,807,954  3580795435,807,954 |
| 41169 | Acc. | 0.38390.38390.3839 | 0.37700.37700.3770 | 13,802,953  1380295313,802,953 | 14,361,949  1436194914,361,949 |
| 42734 | Acc. | 0.74950.74950.7495 | 0.75200.75200.7520 | 8,922,568  89225688,922,568 | 9,205,592  92055929,205,592 |
| 422 | RMSE | 0.02720.02720.0272 | 0.02700.02700.0270 | 39,478,402  3947840239,478,402 | 76,649,015  7664901576,649,015 |
| 541 | RMSE | 7.91607.91607.9160 | 11.661011.661011.6610 | 684,082  684082684,082 | 897,840  897840897,840 |
| 42563 | RMSE | 23094.413023094.413023094.4130 | 33112.387033112.387033112.3870 | 38,900,098  3890009838,900,098 | 109,678,283  109678283109,678,283 |
| 42571 | RMSE | 1918.39821918.39821918.3982 | 1953.39101953.39101953.3910 | 17,456,806  1745680617,456,806 | 19,048,879  1904887919,048,879 |
| 42705 | RMSE | 8.93518.93518.9351 | 10.282010.282010.2820 | 38,840,962  3884096238,840,962 | 173,809,579  173809579173,809,579 |
| 42724 | RMSE | 12144.912112144.912112144.9121 | 11577.678011577.678011577.6780 | 38,683,522  3868352238,683,522 | 62,405,052  6240505262,405,052 |
| 42726 | RMSE | 2.07352.07352.0735 | 2.11302.11302.1130 | 1,466,218  14662181,466,218 | 1,775,189  17751891,775,189 |
| 42727 | RMSE | 0.15020.15020.1502 | 0.14500.14500.1450 | 35,502,610  3550261035,502,610 | 37,517,460  3751746037,517,460 |
| 42728 | RMSE | 16.378016.378016.3780 | 12.578012.578012.5780 | 2,049,022  20490222,049,022 | 2,234,102  22341022,234,102 |
| 42729 | RMSE | 1.94361.94361.9436 | 1.88201.88201.8820 | 6,682,150  66821506,682,150 | 6,922,958  69229586,922,958 |

## Appendix C Settings of Ablation Study

In the ablation study, we explored different approaches to normalize the regression targets for regression tasks.
Specifically, we compared standardization (mean subtraction and scaling) with the quantile transformation used in Grinsztajn45 (Grinsztajn et al., [2022](#bib.bib17)), which relies on the Scikit-learn library’s quantile transformation (scikit learn, [2023b](#bib.bib34)).

Based on our experiments, we found that standardization generally leads to better performance compared to quantile transformation, as demonstrated in [Table 23](#A3.T23 "In Appendix C Settings of Ablation Study ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"). To ensure a fair comparison, all results in [Section 4.2](#S4.SS2 "4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") were obtained using the configurations specified in Grinsztajn45.

In the ablation study, we simply selected the better normalization approach based on its performance.
We provide these details here to explain the performance differences observed in the regression tasks discussed in [Section 4.2](#S4.SS2 "4.2 Evaluation Results ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), as well as those in [Section 4.3](#S4.SS3 "4.3 Ablation Study ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [Appendix D](#A4 "Appendix D More Ablation Studies ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").

Table 23: Average r2-score of Trompt using different target normalizations on Grinsztajn45 regression tasks.

  

| Target Normalization | r2-score |
| --- | --- |
| Quantile Transformation | 70.55%percent70.5570.55\% |
| Standardization | 74.15%percent74.1574.15\% |

## Appendix D More Ablation Studies

In [Section D.1](#A4.SS1 "D.1 Hyperparameters ‣ Appendix D More Ablation Studies ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), we present additional ablation studies focusing on different values of various hyperparameters. We investigate the impact of varying these hyperparameters on the performance of Trompt.

Furthermore, in [Section D.2](#A4.SS2 "D.2 Architecture ‣ Appendix D More Ablation Studies ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), we delve into the necessity of key components in the architecture of Trompt. We conduct ablation experiments to examine the effect of removing or modifying these components on the overall performance of Trompt.

These additional ablation studies aim to provide further insights into the role and importance of different hyperparameters and architectural components in Trompt.

### D.1 Hyperparameters

Ablations on the size of hidden dimension.

The hidden dimension (d𝑑d) parameter in Trompt plays a crucial role in configuring various parts of the model, such as the size of dense layers and embeddings. To evaluate the impact of different values of d𝑑d, we conducted experiments using Trompt with six different values of d𝑑d.

The results presented in [Table 24](#A4.T24 "In D.1 Hyperparameters ‣ Appendix D More Ablation Studies ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") demonstrate that Trompt achieves good performance when an adequate amount of hidden dimension is used, particularly when d𝑑d is larger than 32. This suggests that a larger hidden dimension allows Trompt to capture and represent more complex patterns and relationships in the data, leading to improved performance.

Table 24: The performance of different number of hidden dimension.

  

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 8 | 16 | 32 | 64 | 128 (Default) | 256 |
| Classification | 79.53%percent79.5379.53\% | 80.49%percent80.4980.49\% | 81.16%percent81.1681.16\% | 81.62%percent81.6281.62\% | 81.81%percent81.8181.81\% | 81.69%percent81.6981.69\% |
| Regression | 72.63%percent72.6372.63\% | 73.61%percent73.6173.61\% | 74.22%percent74.2274.22\% | 74.30%percent74.3074.30\% | 74.15%percent74.1574.15\% | 74.47%percent74.4774.47\% |

Ablations on the number of Trompt Cells.

The number of Trompt Cells (L𝐿L) has a significant impact on the model capacity of Trompt. As shown in [Table 25](#A4.T25 "In D.1 Hyperparameters ‣ Appendix D More Ablation Studies ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), the evaluation results indicate that increasing the number of cells leads to better performance.

In particular, Trompt performs poorly when L=1𝐿1L=1. This can be attributed to the design of the Trompt Cell, as depicted in the first part of [Figure 3](#S2.F3 "In 2.3 The Uniqueness of Trompt ‣ 2 Related Work ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), which relies on the output from the previous cell (𝐎prevsubscript𝐎prev\mathbf{O}\_{\text{prev}}) to absorb input-dependent information.

When L=1𝐿1L=1, the first Trompt Cell lacks the previous cell’s output, resulting in feature importances that are irrelevant to the input and becoming deterministic feature importances for all samples. This degradation in performance can be observed in the evaluation results.

Therefore, it is evident that a larger number of Trompt Cells is necessary to effectively capture and leverage input-dependent information and achieve better performance in Trompt.

Table 25: The performance of different number of Trompt Cells.

  

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1 | 3 | 6 (default) | 12 |
| Classification | 79.70%percent79.7079.70\% | 81.36%percent81.3681.36\% | 81.81%percent81.8181.81\% | 82.10%percent82.1082.10\% |
| Regression | 70.47%percent70.4770.47\% | 73.57%percent73.5773.57\% | 74.15%percent74.1574.15\% | 74.61%percent74.6174.61\% |

### D.2 Architecture

Ablations on whether the output of previous Trompt Cell is connected to current Trompt Cell.

The connection between the output of the previous Trompt Cell and the current Trompt Cell is crucial, as it allows for the fusion of prompt embeddings with input-related representations.
This fusion results in sample-wise feature importances, providing valuable insights into the importance of each feature.
Without this connection, the feature importances of each Trompt Cell would become deterministic and lose their variability.
As illustrated in [Table 26](#A4.T26 "In D.2 Architecture ‣ Appendix D More Ablation Studies ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), connecting the output of the previous Trompt Cell yields improved performance in both regression and classification tasks.

Table 26: The performance of whether the output of previous Trompt Cell is connected to current Trompt Cell.

  

|  |  |  |
| --- | --- | --- |
|  | True (default) | False |
| Classification | 81.81%percent81.8181.81\% | 81.68%percent81.6881.68\% |
| Regression | 74.15%percent74.1574.15\% | 73.82%percent73.8273.82\% |

Ablations on whether column embeddings are input independent.

When constructing column embeddings, we deliberately design them to be independent of the input and to capture the intrinsic properties of the tabular dataset through end-to-end training.
In this particular experiment, we examined the impact of sharing the column embeddings (𝐄promptsubscript𝐄prompt\mathbf{E}\_{\text{prompt}}) and input embeddings (𝐄featuresubscript𝐄feature\mathbf{E}\_{\text{feature}}), which compromises the input-independent nature of column embeddings.
The results in [Table 27](#A4.T27 "In D.2 Architecture ‣ Appendix D More Ablation Studies ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") demonstrate that maintaining input-independent column embeddings leads to improved performance in both regression and classification tasks.

Table 27: The performance of whether column embeddings are input independent.

  

|  |  |  |
| --- | --- | --- |
|  | True | False (default) |
| Classification | 81.66%percent81.6681.66\% | 81.81%percent81.8181.81\% |
| Regression | 74.03%percent74.0374.03\% | 74.15%percent74.1574.15\% |

## Appendix E More Interpretability Experiments

In the main paper, we presented the average of 𝐌^importancesubscript^𝐌importance\mathbf{\hat{M}}\_{\text{importance}} for each Trompt Cell.
In [Section E.1](#A5.SS1 "E.1 Feature Importances of Each Layer ‣ Appendix E More Interpretability Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), we provide the individual 𝐌^importancesubscript^𝐌importance\mathbf{\hat{M}}\_{\text{importance}} values for each Trompt Cell.
Furthermore, in [Section E.2](#A5.SS2 "E.2 Additional Real-world Datasets ‣ Appendix E More Interpretability Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), we offer additional results on real-world datasets.

### E.1 Feature Importances of Each Layer

As evident from the attention visualization in [Figures 19](#A5.F19 "In E.1 Feature Importances of Each Layer ‣ Appendix E More Interpretability Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [20](#A5.F20 "Figure 20 ‣ E.1 Feature Importances of Each Layer ‣ Appendix E More Interpretability Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), Trompt effectively directs its attention towards important features in both the Syn2 and Syn4 datasets.
It is worth noting that in our experiments, we employed default hyperparameters, as outlined in [Table 2](#S4.T2 "In 4.1.2 Implementation Details ‣ 4.1 Setup ‣ 4 Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), resulting in Trompt being composed of six Trompt Cells.

!(/html/2305.18446/assets/figures/syn2-real.png)

(a) Important Features.

!(/html/2305.18446/assets/figures/syn2-mask-layer.png)

(b) Masks of Trompt.

Figure 19: Attention masks of each layer on Syn2 dataset.

!(/html/2305.18446/assets/figures/syn4-real.png)

(a) Important Features.

!(/html/2305.18446/assets/figures/syn4-mask-layer.png)

(b) Masks of Trompt.

Figure 20: Attention masks of each layer on Syn4 dataset.

### E.2 Additional Real-world Datasets

The additional interpretability experiments were conducted on the red wine quality dataset and white wine quality dataset (Cortez et al., [2009](#bib.bib10)).
According to the descriptions of dataset, feature selections are required since there are noisy columns in both datasets.
The experimental results are presented in [Tables 28](#A5.T28 "In E.2 Additional Real-world Datasets ‣ Appendix E More Interpretability Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [29](#A5.T29 "Table 29 ‣ E.2 Additional Real-world Datasets ‣ Appendix E More Interpretability Experiments ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
The results indicate that both Trompt and tree-based models yielded comparable feature importances.
Specifically, Trompt assigned higher scores to the alcohol and sulphates columns in the red wine quality dataset, and the volatile acidity column in the white wine quality dataset.

Table 28: The top-3 importance score ratio on the red wine quality dataset.

  

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1st | 2nd | 3rd |
| RandomForest | alcohol (27.17%percent27.1727.17\%) | sulphates (15.44%percent15.4415.44\%) | volatile acidity (10.92%percent10.9210.92\%) |
| XGBoost | alcohol (35.42%percent35.4235.42\%) | sulphates (15.44%percent15.4415.44\%) | volatile acidity (7.56%percent7.567.56\%) |
| LightGBM | alcohol (26.08%percent26.0826.08\%) | sulphates (15.75%percent15.7515.75\%) | volatile acidity (10.63%percent10.6310.63\%) |
| CatBoost | sulphates (16.29%percent16.2916.29\%) | alcohol (15.67%percent15.6715.67\%) | volatile acidity (10.40%percent10.4010.40\%) |
| GradientBoostingTree | alcohol (26.27%percent26.2726.27\%) | sulphates (16.24%percent16.2416.24\%) | volatile acidity (11.12%percent11.1211.12\%) |
| Trompt (ours) | alcohol (11.83%percent11.8311.83\%) | sulphates (10.94%percent10.9410.94\%) | total sulfur dioxide (9.78%percent9.789.78\%) |

Table 29: The top-3 importance score ratio on the white wine quality dataset.

  

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1st | 2nd | 3rd |
| RandomForest | alcohol (24.22%percent24.2224.22\%) | volatile acidity (12.44%percent12.4412.44\%) | free sulfur dioxide (11.78%percent11.7811.78\%) |
| XGBoost | alcohol (31.87%percent31.8731.87\%) | free sulfur dioxide (11.38%percent11.3811.38\%) | volatile acidity (10.05%percent10.0510.05\%) |
| LightGBM | alcohol (24.02%percent24.0224.02\%) | volatile acidity (12.47%percent12.4712.47\%) | free sulfur dioxide (11.45%percent11.4511.45\%) |
| CatBoost | alcohol (17.34%percent17.3417.34\%) | volatile acidity (12.07%percent12.0712.07\%) | free sulfur dioxide (11.47%percent11.4711.47\%) |
| GradientBoostingTree | alcohol (27.84%percent27.8427.84\%) | volatile acidity (13.59%percent13.5913.59\%) | free sulfur dioxide (12.87%percent12.8712.87\%) |
| Trompt (ours) | fixed acidity (10.91%percent10.9110.91\%) | volatile acidity (10.47%percent10.4710.47\%) | pH (10.37%percent10.3710.37\%) |

## Appendix F Hyperparameter Search Spaces

The hyperparameter search space of all models is defined in [Tables 30](#A6.T30 "In Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [31](#A6.T31 "Table 31 ‣ Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [32](#A6.T32 "Table 32 ‣ Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [33](#A6.T33 "Table 33 ‣ Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [34](#A6.T34 "Table 34 ‣ Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [35](#A6.T35 "Table 35 ‣ Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [36](#A6.T36 "Table 36 ‣ Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [37](#A6.T37 "Table 37 ‣ Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data"), [38](#A6.T38 "Table 38 ‣ Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") and [39](#A6.T39 "Table 39 ‣ Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data").
We use the same search spaces for the models tested in Grinsztajn45 and additionally define the search spaces for CatBoost, LightGBM, and Trompt since they are newly added.
For CatBoost, we followed the search spaces declared by FT-Transformer (Gorishniy et al., [2021](#bib.bib16)).
For LightGBM, we followed the search spaces suggested by practitioners (Averagemn, [2019](#bib.bib2); Bahmani, [2022](#bib.bib3)).

Notice that for the hyperparameter search space of Trompt, we focus on the variation of deriving feature importances (part one of [Figure 3](#S2.F3 "In 2.3 The Uniqueness of Trompt ‣ 2 Related Work ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")).
In the default design, we apply concatenation on 𝐒𝐄promptsubscript𝐒𝐄prompt\mathbf{SE}\_{\text{prompt}} and 𝐎prevsubscript𝐎prev\mathbf{O}\_{\text{prev}}.
Here, we explore the possibility of summation.
Additionally, if we applied summation, the following dense layer is not necessary.
Here, we explore the possibility of removing the dense layer.
As for dense, we explore the variation of sharing weight among all prompts.
Lastly, removing residual connections of Equation [Equation 2](#S3.E2 "In 3.1.1 Derive Feature Importances ‣ 3.1 Trompt Cell ‣ 3 Trompt ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") is also explored.
Besides the variation of deriving feature importances, we also explore removing the residual connection of expanding feature embeddings (part three of [Figure 3](#S2.F3 "In 2.3 The Uniqueness of Trompt ‣ 2 Related Work ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data")).
In addition, we adjust the minimal batch ratio so that Trompt can be trained using different batch sizes.

To clarify, since the dense layer must be applied if concatenation was applied, and sharing dense must be false if the dense layer was not applied, the effective parameter combinations of [Table 30](#A6.T30 "In Appendix F Hyperparameter Search Spaces ‣ Trompt: Towards a Better Deep Neural Network for Tabular Data") amount to 40.

Table 30: Hyperparameter space of Trompt.

  

| Parameter | Distribution |
| --- | --- |
| Feature Importances Type | [𝚌𝚘𝚗𝚌𝚊𝚝,𝚊𝚍𝚍]𝚌𝚘𝚗𝚌𝚊𝚝𝚊𝚍𝚍[\mathtt{concat},\mathtt{add}] |
| Feature Importances Dense | [𝚝𝚛𝚞𝚎,𝚏𝚊𝚕𝚜𝚎]𝚝𝚛𝚞𝚎𝚏𝚊𝚕𝚜𝚎[\mathtt{true},\mathtt{false}] |
| Feature Importances Residual Connection | [𝚝𝚛𝚞𝚎,𝚏𝚊𝚕𝚜𝚎]𝚝𝚛𝚞𝚎𝚏𝚊𝚕𝚜𝚎[\mathtt{true},\mathtt{false}] |
| Feature Importances Sharing Dense | [𝚝𝚛𝚞𝚎,𝚏𝚊𝚕𝚜𝚎]𝚝𝚛𝚞𝚎𝚏𝚊𝚕𝚜𝚎[\mathtt{true},\mathtt{false}] |
| Feature Embeddings Residual Connection | [𝚝𝚛𝚞𝚎,𝚏𝚊𝚕𝚜𝚎]𝚝𝚛𝚞𝚎𝚏𝚊𝚕𝚜𝚎[\mathtt{true},\mathtt{false}] |
| Minimal Batch Ratio | [0.1,0.01]0.10.01[0.1,0.01] |

Table 31: Hyperparameter space of FT-Transformer.

  

| Parameter | Distribution |
| --- | --- |
| Num Layers | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[1,6]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝16\mathtt{uniform\\_int}[1,6] |
| Feature Embedding Size | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[64,512]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝64512\mathtt{uniform\\_int}[64,512] |
| Residual Dropout | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0,0.5]𝚞𝚗𝚒𝚏𝚘𝚛𝚖00.5\mathtt{uniform}[0,0.5] |
| Attention Dropout | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0,0.5]𝚞𝚗𝚒𝚏𝚘𝚛𝚖00.5\mathtt{uniform}[0,0.5] |
| FFN Dropout | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0,0.5]𝚞𝚗𝚒𝚏𝚘𝚛𝚖00.5\mathtt{uniform}[0,0.5] |
| FFN Factor | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[2/3,8/3]𝚞𝚗𝚒𝚏𝚘𝚛𝚖2383\mathtt{uniform}[2/3,8/3] |
| Learning Rate | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1​e−5,1​e−3]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖1𝑒51𝑒3\mathtt{log\\_uniform}[1e-5,1e-3] |
| Weight Decay | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1​e−6,1​e−3]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖1𝑒61𝑒3\mathtt{log\\_uniform}[1e-6,1e-3] |
| KV Compression | [𝚝𝚛𝚞𝚎,𝚏𝚊𝚕𝚜𝚎]𝚝𝚛𝚞𝚎𝚏𝚊𝚕𝚜𝚎[\mathtt{true},\mathtt{false}] |
| LKV Compression Sharing | [𝚑𝚎𝚊𝚍𝚠𝚒𝚜𝚎,𝚔𝚎𝚢​\_​𝚟𝚊𝚕𝚞𝚎]𝚑𝚎𝚊𝚍𝚠𝚒𝚜𝚎𝚔𝚎𝚢\_𝚟𝚊𝚕𝚞𝚎[\mathtt{headwise},\mathtt{key\\_value}] |
| Learning Rate Scheduler | [𝚝𝚛𝚞𝚎,𝚏𝚊𝚕𝚜𝚎]𝚝𝚛𝚞𝚎𝚏𝚊𝚕𝚜𝚎[\mathtt{true},\mathtt{false}] |
| Batch Size | [256,512,1024]  2565121024[256,512,1024] |

Table 32: Hyperparameter space of ResNet.

  

| Parameter | Distribution |
| --- | --- |
| Num Layers | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[1,16]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝116\mathtt{uniform\\_int}[1,16] |
| Layers Size | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[64,1024]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝641024\mathtt{uniform\\_int}[64,1024] |
| Hidden Factor | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1,4]𝚞𝚗𝚒𝚏𝚘𝚛𝚖14\mathtt{uniform}[1,4] |
| Hidden Dropout | [0,0.5]00.5[0,0.5] |
| Residual Dropout | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0,0.5]𝚞𝚗𝚒𝚏𝚘𝚛𝚖00.5\mathtt{uniform}[0,0.5] |
| Learning Rate | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1​e−5,1​e−2]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖1𝑒51𝑒2\mathtt{log\\_uniform}[1e-5,1e-2] |
| Weight Decay | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1​e−8,1​e−3]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖1𝑒81𝑒3\mathtt{log\\_uniform}[1e-8,1e-3] |
| Category Embedding Size | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[64,512]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝64512\mathtt{uniform\\_int}[64,512] |
| Normalization | [𝚋𝚊𝚝𝚌𝚑​\_​𝚗𝚘𝚛𝚖,𝚕𝚊𝚢𝚎𝚛​\_​𝚗𝚘𝚛𝚖]𝚋𝚊𝚝𝚌𝚑\_𝚗𝚘𝚛𝚖𝚕𝚊𝚢𝚎𝚛\_𝚗𝚘𝚛𝚖[\mathtt{batch\\_norm},\mathtt{layer\\_norm}] |
| Learning Rate Scheduler | [𝚝𝚛𝚞𝚎,𝚏𝚊𝚕𝚜𝚎]𝚝𝚛𝚞𝚎𝚏𝚊𝚕𝚜𝚎[\mathtt{true},\mathtt{false}] |
| Batch Size | [256,512,1024]  2565121024[256,512,1024] |

Table 33: Hyperparameter space of MLP.

  

| Parameter | Distribution |
| --- | --- |
| Num Layers | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[1,8]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝18\mathtt{uniform\\_int}[1,8] |
| Layer Size | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[16,1024]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝161024\mathtt{uniform\\_int}[16,1024] |
| Dropout | [0,0.5]00.5[0,0.5] |
| Learning Rate | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1​e−5,1​e−2]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖1𝑒51𝑒2\mathtt{log\\_uniform}[1e-5,1e-2] |
| Category Embedding Size | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[64,512]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝64512\mathtt{uniform\\_int}[64,512] |
| Learning Rate Scheduler | [𝚝𝚛𝚞𝚎,𝚏𝚊𝚕𝚜𝚎]𝚝𝚛𝚞𝚎𝚏𝚊𝚕𝚜𝚎[\mathtt{true},\mathtt{false}] |
| Batch Size | [256,512,1024]  2565121024[256,512,1024] |

Table 34: Hyperparameter space of SAINT.

  

| Parameter | Distribution |
| --- | --- |
| Num Layers | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[1,2,3,6,12]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝  123612\mathtt{uniform\\_int}[1,2,3,6,12] |
| Num Heads | [2,4,8]  248[2,4,8] |
| Layer Size | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[32,64,128]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝  3264128\mathtt{uniform\\_int}[32,64,128] |
| Dropout | [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]  00.10.20.30.40.50.60.70.8[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8] |
| Learning Rate | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1​e−5,1​e−3]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖1𝑒51𝑒3\mathtt{log\\_uniform}[1e-5,1e-3] |
| Batch Size | [128,256]128256[128,256] |

Table 35: Hyperparameter space of CatBoost.

  

| Parameter | Distribution |
| --- | --- |
| Max Depth | [3,4,5,6,7,8,9,10]  345678910[3,4,5,6,7,8,9,10] |
| Learning Rate | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1​e−5,1]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖1𝑒51\mathtt{log\\_uniform}[1e-5,1] |
| Iterations | 𝚚𝚞𝚊𝚗𝚝𝚒𝚕𝚎​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[100,6000]𝚚𝚞𝚊𝚗𝚝𝚒𝚕𝚎\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖1006000\mathtt{quantile\\_uniform}[100,6000] |
| Bagging Temperature | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0,1]𝚞𝚗𝚒𝚏𝚘𝚛𝚖01\mathtt{uniform}[0,1] |
| L2 Leaf Reg | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1,10]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖110\mathtt{log\\_uniform}[1,10] |
| Leaf Estimation Iteration | [1,2,3,4,5,6,7,8,9,10]  12345678910[1,2,3,4,5,6,7,8,9,10] |

Table 36: Hyperparameter space of LightGBM.

  

| Parameter | Distribution |
| --- | --- |
| Learning Rate | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0.001,1]𝚞𝚗𝚒𝚏𝚘𝚛𝚖0.0011\mathtt{uniform}[0.001,1] |
| Max Depth | [1,2,3,4,5,6,7,8,9,10,11]  1234567891011[1,2,3,4,5,6,7,8,9,10,11] |
| Bagging Fraction | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0.1,1.0]𝚞𝚗𝚒𝚏𝚘𝚛𝚖0.11.0\mathtt{uniform}[0.1,1.0] |
| Bagging Frequency | [1,2,3,4,5]  12345[1,2,3,4,5] |
| Num Leaves | 𝚚𝚞𝚊𝚗𝚝𝚒𝚕𝚎​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[30,150]𝚚𝚞𝚊𝚗𝚝𝚒𝚕𝚎\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖30150\mathtt{quantile\\_uniform}[30,150] |
| Feature Fraction | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0.1,1.0]𝚞𝚗𝚒𝚏𝚘𝚛𝚖0.11.0\mathtt{uniform}[0.1,1.0] |
| Num Estimators | 100010001000 |
| Boosting | [𝚐𝚋𝚍𝚝,𝚛𝚏,𝚍𝚊𝚛𝚝]  𝚐𝚋𝚍𝚝𝚛𝚏𝚍𝚊𝚛𝚝[\mathtt{gbdt},\mathtt{rf},\mathtt{dart}] |

Table 37: Hyperparameter space of XGBoost.

  

| Parameter | Distribution |
| --- | --- |
| Max Depth | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[1,11]𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝111\mathtt{uniform\\_int}[1,11] |
| Num Estimators | 100010001000 |
| Min Child Weight | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[1,1​e​2]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝11𝑒2\mathtt{log\\_uniform\\_int}[1,1e2] |
| Subsample | 𝚞𝚗𝚒𝚏𝚛𝚘𝚖​[0.5,1]𝚞𝚗𝚒𝚏𝚛𝚘𝚖0.51\mathtt{unifrom}[0.5,1] |
| Learning Rate | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚛𝚘𝚖​[1​e−5,0.7]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚛𝚘𝚖1𝑒50.7\mathtt{log\\_unifrom}[1e-5,0.7] |
| Col Sample by Level | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0.5,1]𝚞𝚗𝚒𝚏𝚘𝚛𝚖0.51\mathtt{uniform}[0.5,1] |
| Col Sample by Tree | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0.5,1]𝚞𝚗𝚒𝚏𝚘𝚛𝚖0.51\mathtt{uniform}[0.5,1] |
| Gamma | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1​e−8,7]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖1𝑒87\mathtt{log\\_uniform}[1e-8,7] |
| Lambda | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1,4]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖14\mathtt{log\\_uniform}[1,4] |
| Alpha | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[1​e−8,1​e​2]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖1𝑒81𝑒2\mathtt{log\\_uniform}[1e-8,1e2] |

Table 38: Hyperparameter space of RandomForest.

  

| Parameter | Distribution |
| --- | --- |
| Max Depth | [𝚗𝚘𝚗𝚎,2,3,4]​([0.7,0.1,0.1,0.1])  𝚗𝚘𝚗𝚎234  0.70.10.10.1[\mathtt{none},2,3,4]([0.7,0.1,0.1,0.1]) |
| Num Estimators | 250250250 |
| Criterion | [𝚐𝚒𝚗𝚒,𝚎𝚗𝚝𝚛𝚘𝚙𝚢]​([𝚜𝚚𝚞𝚊𝚛𝚎𝚍​\_​𝚎𝚛𝚛𝚘𝚛,𝚊𝚋𝚜𝚘𝚕𝚞𝚝𝚎​\_​𝚎𝚛𝚛𝚘𝚛])𝚐𝚒𝚗𝚒𝚎𝚗𝚝𝚛𝚘𝚙𝚢𝚜𝚚𝚞𝚊𝚛𝚎𝚍\_𝚎𝚛𝚛𝚘𝚛𝚊𝚋𝚜𝚘𝚕𝚞𝚝𝚎\_𝚎𝚛𝚛𝚘𝚛[\mathtt{gini},\mathtt{entropy}]([\mathtt{squared\\_error},\mathtt{absolute\\_error}]) |
| Max Features | [𝚜𝚚𝚛𝚝,𝚕𝚘𝚐𝟸,𝚗𝚘𝚗𝚎,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  𝚜𝚚𝚛𝚝𝚕𝚘𝚐𝟸𝚗𝚘𝚗𝚎0.10.20.30.40.50.60.70.80.9[\mathtt{sqrt},\mathtt{log2},\mathtt{none},0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] |
| Min Samples Split | [2,3]​([0.95,0.05])230.950.05[2,3]([0.95,0.05]) |
| Min Samples Leaf | 𝚕𝚘𝚐​\_​𝚞𝚗𝚒𝚏𝚘𝚛𝚖​\_​𝚒𝚗𝚝​[1.5,50.5]𝚕𝚘𝚐\_𝚞𝚗𝚒𝚏𝚘𝚛𝚖\_𝚒𝚗𝚝1.550.5\mathtt{log\\_uniform\\_int}[1.5,50.5] |
| Bootstrap | [𝚝𝚛𝚞𝚎,𝚏𝚊𝚕𝚜𝚎]𝚝𝚛𝚞𝚎𝚏𝚊𝚕𝚜𝚎[\mathtt{true},\mathtt{false}] |
| Min Impurity Decrease | [0.0,0.01,0.02,0.05]​([0.85,0.05,0.05,0.05])  0.00.010.020.05  0.850.050.050.05[0.0,0.01,0.02,0.05]([0.85,0.05,0.05,0.05]) |

Table 39: Hyperparameter space of GradientBoostingTree.

  

| Parameter | Distribution |
| --- | --- |
| Loss | [𝚍𝚎𝚟𝚒𝚊𝚗𝚌𝚎,𝚎𝚡𝚙𝚘𝚗𝚎𝚗𝚝𝚒𝚊𝚕]​(c​l​a​s​s​i​f)​([𝚜𝚚𝚞𝚊𝚛𝚎𝚍​\_​𝚎𝚛𝚛𝚘𝚛,𝚊𝚋𝚜𝚘𝚕𝚞𝚝𝚎​\_​𝚎𝚛𝚛𝚘𝚛,𝚑𝚞𝚋𝚎𝚛])​(r​e​g​r​e​s​s​i​o​n)𝚍𝚎𝚟𝚒𝚊𝚗𝚌𝚎𝚎𝚡𝚙𝚘𝚗𝚎𝚗𝚝𝚒𝚊𝚕𝑐𝑙𝑎𝑠𝑠𝑖𝑓  𝚜𝚚𝚞𝚊𝚛𝚎𝚍\_𝚎𝚛𝚛𝚘𝚛𝚊𝚋𝚜𝚘𝚕𝚞𝚝𝚎\_𝚎𝚛𝚛𝚘𝚛𝚑𝚞𝚋𝚎𝚛 𝑟𝑒𝑔𝑟𝑒𝑠𝑠𝑖𝑜𝑛[\mathtt{deviance},\mathtt{exponential}](classif)([\mathtt{squared\\_error},\mathtt{absolute\\_error},\mathtt{huber}])(regression) |
| Learning Rate | 𝚕𝚘𝚐​\_​𝚗𝚘𝚛𝚖𝚊𝚕​[𝚕𝚘𝚐​(0.01),𝚕𝚘𝚐​(10)]𝚕𝚘𝚐\_𝚗𝚘𝚛𝚖𝚊𝚕𝚕𝚘𝚐0.01𝚕𝚘𝚐10\mathtt{log\\_normal}[\mathtt{log}(0.01),\mathtt{log}(10)] |
| Subsample | 𝚞𝚗𝚒𝚏𝚘𝚛𝚖​[0.5,1]𝚞𝚗𝚒𝚏𝚘𝚛𝚖0.51\mathtt{uniform}[0.5,1] |
| Num Estimators | 100010001000 |
| Criterion | [𝚏𝚛𝚒𝚎𝚍𝚖𝚊𝚗​\_​𝚖𝚜𝚎,𝚜𝚚𝚞𝚊𝚛𝚎𝚍​\_​𝚎𝚛𝚛𝚘𝚛]𝚏𝚛𝚒𝚎𝚍𝚖𝚊𝚗\_𝚖𝚜𝚎𝚜𝚚𝚞𝚊𝚛𝚎𝚍\_𝚎𝚛𝚛𝚘𝚛[\mathtt{friedman\\_mse},\mathtt{squared\\_error}] |
| Max Depth | [𝚗𝚘𝚗𝚎,2,3,4,5]​([0.1,0.1,0.5,0.1,0.1])  𝚗𝚘𝚗𝚎2345  0.10.10.50.10.1[\mathtt{none},2,3,4,5]([0.1,0.1,0.5,0.1,0.1]) |
| Min Samples Split | [2.3]​([0.95,0.05])delimited-[]2.30.950.05[2.3]([0.95,0.05]) |
| Min Impurity Decrease | [0.0,0.01,0.02,0.05]​([0.85,0.05])  0.00.010.020.05 0.850.05[0.0,0.01,0.02,0.05]([0.85,0.05]) |
| Max Leaf Nodes | [𝚗𝚘𝚗𝚎,5,10,15]​([0.85,0.5])  𝚗𝚘𝚗𝚎51015 0.850.5[\mathtt{none},5,10,15]([0.85,0.5]) |

Table 40: Hyperparameter space of HistGradientBoosting.

  

| Parameter | Distribution |
| --- | --- |
| Loss | [𝚜𝚚𝚞𝚊𝚛𝚎𝚍​\_​𝚎𝚛𝚛𝚘𝚛,𝚊𝚋𝚜𝚘𝚕𝚞𝚝𝚎​\_​𝚎𝚛𝚛𝚘𝚛,𝚑𝚞𝚋𝚎𝚛]​(r​e​g​r​e​s​s​i​o​n)  𝚜𝚚𝚞𝚊𝚛𝚎𝚍\_𝚎𝚛𝚛𝚘𝚛𝚊𝚋𝚜𝚘𝚕𝚞𝚝𝚎\_𝚎𝚛𝚛𝚘𝚛𝚑𝚞𝚋𝚎𝚛 𝑟𝑒𝑔𝑟𝑒𝑠𝑠𝑖𝑜𝑛[\mathtt{squared\\_error},\mathtt{absolute\\_error},\mathtt{huber}](regression) |
| Learning Rate | 𝚕𝚘𝚐​\_​𝚗𝚘𝚛𝚖𝚊𝚕​[𝚕𝚘𝚐​(0.01),𝚕𝚘𝚐​(10)]𝚕𝚘𝚐\_𝚗𝚘𝚛𝚖𝚊𝚕𝚕𝚘𝚐0.01𝚕𝚘𝚐10\mathtt{log\\_normal}[\mathtt{log}(0.01),\mathtt{log}(10)] |
| Max Iteration | 100010001000 |
| Min Depth | [𝚗𝚘𝚗𝚎,2,3,4]  𝚗𝚘𝚗𝚎234[\mathtt{none},2,3,4] |
| Min Samples Leaf | 𝚗𝚘𝚛𝚖𝚊𝚕​\_​𝚒𝚗𝚝​[20,2]𝚗𝚘𝚛𝚖𝚊𝚕\_𝚒𝚗𝚝202\mathtt{normal\\_int}[20,2] |
| Max Leaf Nodes | 𝚗𝚘𝚛𝚖𝚊𝚕​\_​𝚒𝚗𝚝​[31,5]𝚗𝚘𝚛𝚖𝚊𝚕\_𝚒𝚗𝚝315\mathtt{normal\\_int}[31,5] |
