---
arxiv: '2403.00194'
authors:
- Benjamin Cohen-Wang
- Joshua Vendrow
- Aleksander Madry
parser: ar5iv
retrieved: '2026-05-22'
source: paper
title: Ask Your Distribution Shift if Pre-Training is Right for You
url: https://arxiv.org/abs/2403.00194
year: 2024
---

[2403.00194] Ask Your Distribution Shift if Pre-Training is Right for You



\doparttoc\faketableofcontents

# Ask Your Distribution Shift if Pre-Training is Right for You

Benjamin Cohen-Wang
  
MIT
  
bencw@mit.edu
  
  
Joshua Vendrow
  
MIT
  
jvendrow@mit.edu
  
  
Aleksander Mądry
  
MIT
  
madry@mit.edu

###### Abstract

Pre-training is a widely used approach to develop models that are robust to distribution shifts.
However, in practice, its effectiveness varies: fine-tuning a pre-trained model improves robustness significantly in some cases but *not at all* in others (compared to training from scratch).
In this work, we seek to characterize the failure modes that pre-training *can* and *cannot* address.
In particular, we focus on two possible failure modes of models under distribution shift: poor extrapolation (e.g., they cannot generalize to a different domain) and biases in the training data (e.g., they rely on spurious features).
Our study suggests that, as a rule of thumb, pre-training can help mitigate poor extrapolation but not dataset biases.
After providing theoretical motivation and empirical evidence for this finding, we explore two of its implications for developing robust models:
(1) pre-training and interventions designed to prevent exploiting biases have complementary robustness benefits, and
(2) fine-tuning on a (very) small, non-diverse but *de-biased* dataset can result in significantly more robust models than fine-tuning on a large and diverse but biased dataset.111Code is available at <https://github.com/MadryLab/pretraining-distribution-shift-robustness>

### 1 Introduction

A common paradigm for developing machine learning models is pre-training them on a large, diverse dataset (e.g., ImageNet [deng2009imagenet], JFT-300M [sun2017revisiting], LAION-5B [schuhmann2022laion]) and then fine-tuning them on task-specific data.
Indeed, compared to training from scratch, fine-tuning a pre-trained model often significantly improves performance and reduces computational costs [razavian2014cnn, sun2017revisiting, kornblith2019better].

Yet another benefit that pre-training may offer is *distribution shift robustness*.
Specifically, machine learning models tend to suffer from distribution shifts, i.e., changes between the *reference distribution* used to develop the model and the *shifted distribution* that the model actually encounters when deployed.
For example, a tumor identification model trained on tissue slide images from one hospital might perform poorly when deployed at another hospital [bandi2018detection, koh2020wilds].
Notably, different models (with different architectures, hyperparameters, etc.) tend to be similarly sensitive to a given distribution shift.
However, models pre-trained on auxiliary data and then fine-tuned on the reference distribution can break this trend, exhibiting substantially higher performance on the shifted distribution than models trained from scratch with the same performance on the reference distribution [taori2020when, miller2020effect, miller2021accuracy, andreassen2021evolution, wortsman2021robust].

These robustness benefits of pre-training are promising, but they are *not* universal. In particular, fine-tuning the same pre-trained model can yield significant robustness gains on some distribution shifts but not on others (Section [3](#S3 "3 The Robustness Benefits of Pre-Training Vary ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
Would a solution to attain robustness to the latter shifts then be to fine-tune a larger model pre-trained on more data?
Or are there fundamental limitations to the robustness that pre-training can provide?
To answer these questions, we would like to develop a more fine-grained understanding of when pre-training can improve robustness. Specifically, we ask:

*Can we identify and characterize the failure modes that pre-training can and cannot address?*

Recall that under distribution shift, models can fail in a number of ways.
One of them is their inability to *extrapolate* effectively outside of the reference distribution [gulrajani2020search, koh2020wilds].
If, for instance, a model is trained only on photos taken during the day, then it might fail when deployed on photos taken at night.

Models can also underperform even when the shifted distribution does not contain anything “new.”
In particular, they can fail due to *biases* in the reference distribution.
For example, if a certain feature is spuriously correlated with the label in the reference distribution, a model might learn to exploit this relationship and fail on examples encountered during deployment where it does not hold [arjovsky2019invariant, geirhos2020shortcut].

#### 1.1 Our contributions

To identify the failure modes that pre-training can address, we study the robustness benefits of pre-training under two types of distribution shifts:
(1) shifts where extrapolation is necessary and (2) shifts where extrapolation is not needed.
We start by analyzing a simple logistic regression setting and illustrate why pre-training might improve robustness to the former type of shift, but not the latter (Section [4](#S4 "4 Studying Pre-Training in a Logistic Regression Setting ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
We subsequently build on this intuition by measuring the robustness benefits of pre-training on synthetic and natural distribution shifts of each type (Section [5](#S5 "5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
Our results suggest the following rule of thumb: pre-training can help with extrapolation, but does not address other failures, for example, those stemming from dataset biases.

###### Implications for developing robust models.

Guided by this rule of thumb, we explore two related avenues for harnessing pre-training to develop robust models.

1. 1.

   Combining pre-training with interventions designed to handle bias (Section [6](#S6 "6 Combining Pre-Training with Interventions for Handling Bias ‣ Ask Your Distribution Shift if Pre-Training is Right for You")): There are a number of robustness interventions specifically designed to mitigate biases present in a training dataset [byrd2019effect, sagawa2019distributionally, liu2021just, kirichenko2022last, idrissi2022simple].
   Our findings suggest that pre-training and this kind of intervention address two different sources of failures (the former helping with extrapolating and the latter with avoiding dataset biases) and thus may be viewed as complementary.
   We indeed find that combining them can yield models with both sets of benefits.
2. 2.

   Curating datasets for fine-tuning (Section [7](#S7 "7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You")):
   One possible intervention that aims to address dataset biases is curating a de-biased dataset.
   In general, however, the de-biasing process might be prohibitively expensive.
   That said, we find that if we leverage pre-training to help with extrapolation, we might only need a small, non-diverse fine-tuning dataset; such a dataset might actually be feasible to de-bias.
   For example, we demonstrate that fine-tuning on a carefully de-biased hair color classification dataset with only 64 examples yields greater robustness than fine-tuning on the entire CelebA dataset [liu2015faceattributes].

### 2 Background

###### Fine-tuning a pre-trained model.

Methods for fine-tuning a pre-trained model vary: two common strategies are *full fine-tuning*, in which one continues training the entire model, and *linear probing*, in which one only fine-tunes the final layer.
Some recent pre-trained models with natural language supervision (e.g., CLIP [radford2021learning], ALIGN [jia2021scaling]) can also be adapted to a downstream task in a *zero-shot* context (i.e., without fine-tuning) by specifying the task through a text description.
In this work, we focus on the full fine-tuning strategy, which typically outperforms linear probing and zero-shot models on the reference distribution.
We also will sometimes consider linear probing or zero-shot adaptation followed by full fine-tuning;
this can in some cases improve over full fine-tuning alone in terms of robustness and performance [kumar2022fine].
We discuss other fine-tuning strategies in Appendix [D.1](#A4.SS1 "D.1 Alternative fine-tuning strategies ‣ Appendix D Additional Discussion ‣ Ask Your Distribution Shift if Pre-Training is Right for You").

###### Measuring robustness.

For many distribution shifts, different models trained from scratch on the reference distribution exhibit similar degrees of robustness to the shift.
Specifically, when varying architectures, hyperparameters and training methods there is often a strong *linear* relationship between the *reference accuracy* and *shifted accuracy*222For a linear relationship, accuracies are *probit-scaled* (transformed by the inverse of the Gaussian CDF). (i.e., the accuracies on the reference and shifted distributions, respectively) [taori2020when, miller2020effect, miller2021accuracy].
This relationship, dubbed *accuracy on the line*, can be visualized by plotting shifted accuracies against reference accuracies and finding a linear fit.
When this linear trend is strong (i.e., shifted accuracies are highly correlated with reference accuracies), one can predict the shifted accuracy of models trained from scratch from their reference accuracy.
Furthermore, to quantify the robustness of a model trained with a robustness intervention beyond the “baseline” of models trained from scratch, one can measure the amount by which its shifted accuracy exceeds the linear fit’s prediction, a metric known as *effective robustness* (ER) [taori2020when].
In this work, we choose to study distribution shifts for which accuracy on the line holds (i.e., the linear fit is strong) and quantify robustness by computing effective robustness (see, e.g., Figure [1](#S3.F1 "Figure 1 ‣ 3 The Robustness Benefits of Pre-Training Vary ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
See Appendix [B.1.2](#A2.SS1.SSS2 "B.1.2 Measuring effective robustness ‣ B.1 General ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for additional details.

### 3 The Robustness Benefits of Pre-Training Vary

Figure 1: 
The robustness benefits of pre-training vary.
On the ImageNet-V2 distribution shift (left), different pre-trained models all exhibit very little effective robustness (ER), i.e., little improvement over the linear trend of models trained from scratch (see Section [2](#S2 "2 Background ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
Meanwhile, on the ImageNet Sketch distribution shift (right), some of these pre-trained models exhibit substantial effective robustness.
We report average effective robustness with a 95% confidence interval in the top left of each plot.

Our investigation is motivated by the following observation:

Pre-training can significantly improve robustness to some distribution shifts but not others.

To illustrate this, we consider two distribution shifts of ImageNet [deng2009imagenet]: ImageNet-V2 [recht2018imagenet] and ImageNet Sketch [wang2019learning].
For each of these shifts, we measure the effective robustness (see Section [2](#S2 "2 Background ‣ Ask Your Distribution Shift if Pre-Training is Right for You")) of various pre-trained models.
Specifically, we first establish a baseline for robustness by evaluating 787878 models trained from scratch on ImageNet (from PyTorch Image Models [rw2019timm]).
We observe a strong linear relationship between their reference and shifted accuracies (see Figure [1](#S3.F1 "Figure 1 ‣ 3 The Robustness Benefits of Pre-Training Vary ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
Next, we evaluate 555555 pre-trained models that are fine-tuned on ImageNet (also from PyTorch Image Models) and measure the improvements in shifted accuracy over the linear trend.
See Appendix [B.2](#A2.SS2 "B.2 The robustness benefits of pre-training vary ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for the exact setup.

We find that while some of the pre-trained models exhibit substantial effective robustness on ImageNet Sketch, they all exhibit very little effective robustness on ImageNet-V2.
These pre-trained models represent a wide variety of model architectures, pre-training datasets and pre-training algorithms–the largest model has 1 billion parameters and is pre-trained on a dataset of 2 billion image-text pairs.
Yet, the highest effective robustness attained by any of these models on ImageNet-V2 is just 1.80%percent1.801.80\%.
This suggests that pre-training alone might not suffice to address certain types of failures that occur under distribution shift.
We would like to better understand this limitation; can we identify and characterize these types of failures?

### 4 Studying Pre-Training in a Logistic Regression Setting

Figure 2: Illustration of logistic regression setting. (a) Consider a reference dataset that lies within a subspace Wrefsubscript𝑊refW\_{\text{ref}} of ℝdsuperscriptℝ𝑑\mathbb{R}^{d}. (b) Models trained from different initializations all learn the same (optimal) decision boundary in Wrefsubscript𝑊refW\_{\text{ref}}, but may behave differently outside of Wrefsubscript𝑊refW\_{\text{ref}}. (c) Under shifts within Wrefsubscript𝑊refW\_{\text{ref}}, models with different initializations are equally robust. (d) Under shifts outside of Wrefsubscript𝑊refW\_{\text{ref}}, initialization can affect robustness.

Our central goal is to understand the failure modes that pre-training *can* and *cannot* address.
To this end, we first study the robustness benefits of pre-training in a simple logistic regression setting (see Figure [2](#S4.F2 "Figure 2 ‣ 4 Studying Pre-Training in a Logistic Regression Setting ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).

###### Setup.

Suppose that we are given access to a reference dataset Srefsubscript𝑆refS\_{\text{ref}} of input-label pairs, each consisting of a d𝑑d-dimensional input x∈ℝd𝑥superscriptℝ𝑑x\in\mathbb{R}^{d} and a binary label y∈{−1,1}𝑦11y\in\{-1,1\}.
We are concerned with finding weights w∈ℝd𝑤superscriptℝ𝑑w\in\mathbb{R}^{d} that minimize the (standard) logistic loss on Srefsubscript𝑆refS\_{\text{ref}}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lref​(w)=∑(x,y)∈Sreflog⁡(1+e−w⊤​x⋅y).subscript𝐿ref𝑤subscript𝑥𝑦subscript𝑆ref1superscript𝑒⋅superscript𝑤top𝑥𝑦L\_{\text{ref}}(w)=\sum\_{(x,y)\in S\_{\text{ref}}}\log(1+e^{-w^{\top}x\cdot y}). |  | (1) |

We assume that the reference dataset Srefsubscript𝑆refS\_{\text{ref}} satisfies the following conditions:

1. 1.

   Inputs in Srefsubscript𝑆refS\_{\text{ref}} lie within a k𝑘k-dimensional (with k<d𝑘𝑑k<d) subspace Wrefsubscript𝑊refW\_{\text{ref}} of ℝdsuperscriptℝ𝑑\mathbb{R}^{d}. Intuitively, this condition corresponds to features lacking certain variation in the reference dataset.
2. 2.

   The logistic loss Lrefsubscript𝐿refL\_{\text{ref}} has a minimum value. This condition ensures that minimizing Lrefsubscript𝐿refL\_{\text{ref}} is well-defined. Note that there may be multiple weights that attain this minimum value.

Starting with initial weights winitsubscript𝑤initw\_{\text{init}} (which, in our case, are either random or the result of pre-training), suppose that we use gradient descent to minimize Lref​(w)subscript𝐿ref𝑤L\_{\text{ref}}(w).
We would like to understand how well the resulting model performs under distribution shift.
In particular, what role does pre-training play through winitsubscript𝑤initw\_{\text{init}}?
To answer this question, we establish the following theorem (proof in Appendix [A](#A1 "Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")):

###### Theorem 4.1.

Suppose that we start with initial weights winit∈ℝdsubscript𝑤initsuperscriptℝ𝑑w\_{\text{init}}\in\mathbb{R}^{d} and run gradient descent to minimize Lref​(w)subscript𝐿ref𝑤L\_{\text{ref}}(w). With an appropriately chosen learning rate, gradient descent converges to weights w^^𝑤\hat{w} that minimize Lrefsubscript𝐿refL\_{\text{ref}}. Furthermore, w^^𝑤\hat{w} can be written as

|  |  |  |  |
| --- | --- | --- | --- |
|  | w^=wref∗+projWref⊥​winit.^𝑤subscriptsuperscript𝑤refsubscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤init\hat{w}=w^{\*}\_{\text{ref}}+\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}. |  | (2) |

Here, wref∗subscriptsuperscript𝑤refw^{\*}\_{\text{ref}} is a property of the reference dataset Srefsubscript𝑆refS\_{\text{ref}} and lies within the reference subspace Wrefsubscript𝑊refW\_{\text{ref}}.
Meanwhile, projWref⊥​winitsubscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤init\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}} is the component of winitsubscript𝑤initw\_{\text{init}} that is orthogonal to Wrefsubscript𝑊refW\_{\text{ref}}.

Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ Setup. ‣ 4 Studying Pre-Training in a Logistic Regression Setting ‣ Ask Your Distribution Shift if Pre-Training is Right for You") implies that there are multiple weights that attain the minimum value of Lrefsubscript𝐿refL\_{\text{ref}}, and that the initial weights winitsubscript𝑤initw\_{\text{init}} determine which of them we learn.
Specifically, we can decompose the learned weights w^^𝑤\hat{w} into two terms: wref∗subscriptsuperscript𝑤refw^{\*}\_{\text{ref}} and projWref⊥​winitsubscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤init\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}.
Notice that the first term is just a property of the reference dataset and is in the reference subspace Wrefsubscript𝑊refW\_{\text{ref}}, while the second term depends on winitsubscript𝑤initw\_{\text{init}} and is *orthogonal* to Wrefsubscript𝑊refW\_{\text{ref}}.
As a result, the reference dataset itself fully specifies the model’s behavior on inputs in Wrefsubscript𝑊refW\_{\text{ref}}, while the initialization determines how the model extends outside of Wrefsubscript𝑊refW\_{\text{ref}}.
Consequently, changing a model’s initialization (e.g., with pre-training) can affect performance outside of Wrefsubscript𝑊refW\_{\text{ref}}, but not within Wrefsubscript𝑊refW\_{\text{ref}}.

This observation gives rise to an intuition that will guide our investigations in the remainder of this work:
pre-training can improve robustness to a distribution shift *only* when the shifted distribution contains “out-of-support” inputs, that is, inputs that could not be reasonably sampled from the reference distribution.
In other words, pre-training helps specifically with extrapolation outside of the reference distribution.

### 5 Exploring the Empirical Robustness Benefits of Pre-Training

In Section [4](#S4 "4 Studying Pre-Training in a Logistic Regression Setting ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we found that in a simple logistic regression setting, pre-training helps *specifically* with extrapolation.
We now want to assess whether this principle holds more broadly.
To do so, we measure the robustness benefits of pre-training under two types of shifts: *in-support shifts*, where models *cannot* fail due to poor extrapolation (but might fail for other reasons, e.g., dataset biases), and *out-of-support shifts*, where models *can* fail due to poor extrapolation (see Figure [3](#S5.F3 "Figure 3 ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
We begin by describing these two types of shifts in more detail and providing intuitions for why pre-training might improve robustness to out-of-support shifts, but not in-support shifts.

Figure 3: 
Examples of in-support and out-of-support shifts.
One example of an *in-support shift* (left) is a shift in which the indoor/outdoor frequencies of animal appearances change, but the possible combinations of animal and setting remain the same.
An example of an *out-of-support shift* (right) is a shift from day to night: the nighttime setting is entirely novel.

###### In-support shift.

A distribution shift is *in-support* if any input that could be sampled from the shifted distribution could also be reasonably sampled from the reference distribution.
In other words, the shifted distribution does not contain anything “new”; however, an in-support shift can still cause failures if, for example, the reference distribution is *biased*.
To illustrate this failure mode, consider a cat vs. dog image classification task in which photos are either taken indoors or outdoors.
Suppose that in the reference distribution 90% of cats appear indoors and 90% of dogs appear outdoors (i.e., the setting is spuriously correlated with the animal).
A model trained on this distribution would likely rely (at least in part) on indoor vs. outdoor features [xiao2020noise, geirhos2020shortcut].
Thus, under a shift in which the setting/animal correlation is reversed (which would be in-support but out-of-distribution), the model would likely underperform.
If pre-training helps specifically with extrapolation, then it would not address this failure mode and, more generally, could not improve robustness to in-support shifts.

###### Out-of-support shift.

A distribution shift is *out-of-support* if there exists an input that could be sampled from the shifted distribution but could not be reasonably sampled from the reference distribution.
For example, consider a cat vs. dog image classification task in which photos from the reference distribution are taken during the day and photos from the shifted distribution are taken at night.
In this case, the shifted distribution contains images with previously unseen lighting conditions.
Here, a model trained from scratch might learn features that are sensitive to lighting and thus fail under the shift.
Meanwhile, pre-training could provide priors for extrapolating, e.g., by producing features that are agnostic to lighting conditions as a starting point, leading to greater robustness.

#### 5.1 Constructing synthetic in-support and out-of-support shifts

Figure 4: 
Robustness of pre-trained models to synthetic in-support and out-of-support shifts.
For each of two in-support shifts (left) and two out-of-support shifts (right) constructed by modifying ImageNet, the reference and shifted accuracies of models trained from scratch (in blue) are linearly correlated.
Pre-trained models exhibit little effective robustness (ER), i.e., little improvement over the linear trend (see Section [2](#S2 "2 Background ‣ Ask Your Distribution Shift if Pre-Training is Right for You")), on the in-support shifts, but have significant effective robustness on the out-of-support shifts (averages with 95% confidence intervals in the top left of each plot).
Error bars denote 95% confidence intervals over 4 random trials.

We now want to measure the robustness gains that pre-training provides on in-support and out-of-support shifts.
To this end, we explicity construct two shifts of each type by modifying ImageNet [deng2009imagenet]:
(1) a “spurious tint shift” in which we add a tint that is spuriously correlated with the label in the reference dataset, but not in the shifted dataset,
(2) a “label shift” in which the relative frequencies of classes change between the reference and shifted datasets,
(3) an “unseen tint shift” in add a random tint in the shifted dataset, and
(4) a “flip shift” in which we vertically flip images in the shifted dataset (see the top of Figure [4](#S5.F4 "Figure 4 ‣ 5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for visualizations).

For each shift, as a baseline, we train a ViT-B/32 [dosovitskiy2020image] model from scratch on the reference dataset.
We evaluate this model at different epochs and find a strong linear relationship between reference and shifted accuracy, i.e., the *accuracy on the line* phenomenon occurs333Typically, one evaluates different models to find this relationship. We use different epochs due to computational constraints. (see Figure [4](#S5.F4 "Figure 4 ‣ 5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
Next, we fine-tune pre-trained ViT-B/32 models and measure their effective robustness above this baseline.
We consider two pre-trained models: CLIP [radford2021learning] and AugReg [steiner2021train], and three (full) fine-tuning strategies: standard full fine-tuning (FT), linear probing followed by full fine-tuning (LP-FT) and zero-shot initialization followed by full fine-tuning (ZS-FT).
We select fine-tuning hyperparameters that maximize accuracy on the reference distribution (in Appendix [C.1.1](#A3.SS1.SSS1 "C.1.1 How does the choice of fine-tuning hyperparameters affect robustness? ‣ C.1 Constructing synthetic in-support and out-of-support shifts ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we find that other reasonable hyperparameter choices yield similar robustness).

We observe that pre-trained models exhibit substantial effective robustness on out-of-support shifts, but have close to zero effective robustness on in-support shifts (see Figure [4](#S5.F4 "Figure 4 ‣ 5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
In Appendix [C.1.2](#A3.SS1.SSS2 "C.1.2 How does the strength of the bias affect robustness to in-support shifts? ‣ C.1 Constructing synthetic in-support and out-of-support shifts ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we vary the strength of the biases in the in-support shifts and find that the effective robustness of pre-trained models remains close to zero.
See Appendix [B.3](#A2.SS3 "B.3 Constructing synthetic in-support and out-of-support shifts ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for a description of the exact setup.

#### 5.2 Dividing natural shifts into in-support and out-of-support splits

(a) 
Random samples from: ImageNet (top), the in-support split of ImageNet Sketch (middle) and out-of-support split of ImageNet Sketch (bottom).

(b) 
Average effective robustness of 555555 pre-trained models on each split of each of the three shifts.
Error bars denote 95% confidence intervals.

Figure 5: 
Dividing shifts of ImageNet into in-support and out-of-support splits.
We divide each of the ImageNet-V2, ImageNet Sketch and ImageNet-R datasets into an in-support split containing examples that look like ImageNet examples and an out-of-support split containing examples that look unlike ImageNet examples (see Appendix [B.4](#A2.SS4 "B.4 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for a description of the splitting method).
We display samples from each split of ImageNet Sketch in Figure [5(a)](#S5.F5.sf1 "In Figure 5 ‣ 5.2 Dividing natural shifts into in-support and out-of-support splits ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You") and report the average effective robustnesses of pre-trained models in Figure [5(b)](#S5.F5.sf2 "In Figure 5 ‣ 5.2 Dividing natural shifts into in-support and out-of-support splits ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You").
See Appendix [C.2.3](#A3.SS2.SSS3 "C.2.3 Scatter plots of reference vs. shifted accuracy ‣ C.2 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for scatterplots of reference vs. shifted accuracy.

So far, we have constructed synthetic in-support and out-of-support shifts and observed that pre-training can significantly improve robustness to the latter but not the former.
Now, we demonstrate that this principle seems to extend to natural shifts as well.
Note that it is hard to find natural shifts that are “purely” in-support.
After all, under natural shifts the shifted dataset may contain some inputs that are similar to those in the reference dataset and some that are not.
For example, in a shift from photos to sketches, some sketches may look more photorealistic but most would probably be clearly distinguishable from photos.
To be able to measure robustness to each type of shift, we thus *divide* several natural shifted datasets each into an “in-support split” containing inputs that look like they could have come from the reference dataset and an “out-of-support split” containing the remaining inputs.
We do so by training a classifier to distinguish between the reference and shifted datasets and using this classifier to approximate the probability of sampling a given shifted example from the reference distribution (see Appendix [B.4.1](#A2.SS4.SSS1 "B.4.1 Splitting a Shifted Dataset ‣ B.4 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for details).

Specifically, we consider three natural shifts of the ImageNet dataset: ImageNet-V2 [recht2018imagenet], which closely resembles ImageNet, ImageNet Sketch [wang2019learning], which consists of sketches of ImageNet classes, and ImageNet-R [hendrycks2020faces], which consists of “renditions” (e.g, paintings, sculptures, cartoons) of a subset of ImageNet classes.
We choose these shifted datasets because they include many inputs that look like they could have come from ImageNet and many that do not (according to our splitting method)444We also explored ObjectNet [barbu2019objectnet] and ImageNet-Vid-Robust [shankar2019image] but our splitting method marks fewer than 505050 examples from these shifted datasets as “in-support,” and thus we cannot reliably measure in-support accuracy..
In Figure [5(a)](#S5.F5.sf1 "In Figure 5 ‣ 5.2 Dividing natural shifts into in-support and out-of-support splits ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we visualize examples from the in-support and out-of-support splits of ImageNet Sketch.

Consistent with our hypothesis that pre-training helps specifically with extrapolation, on the out-of-support splits of ImageNet Sketch and ImageNet-R pre-trained models have substantially higher effective robustness than on the respective in-support splits (see Figure [5(b)](#S5.F5.sf2 "In Figure 5 ‣ 5.2 Dividing natural shifts into in-support and out-of-support splits ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
On both ImageNet-V2 splits, however, pre-trained models have very little effective robustness.
This may be because ImageNet-V2 is visually similar to ImageNet, so poor extrapolation might not be a significant failure mode (instead, the performance drop may be due to an increased presence of “harder” examples, as [recht2018imagenet] suggest).
Thus, if pre-training helps only with extrapolation, it would not be able to substantially improve robustness on the ImageNet-V2 out-of-support examples.
See Appendix [B.4.2](#A2.SS4.SSS2 "B.4.2 Specifications of ImageNet models ‣ B.4 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for a description of the exact setup.

### 6 Combining Pre-Training with Interventions for Handling Bias

Figure 6: Combining pre-training and *Deep Feature Reweighting* (DFR) on the WILDS-FMoW shift.
Pre-training and DFR (an intervention designed to handle dataset biases [kirichenko2022last]) each yield some effective robustness (ER) and combining these two interventions yields the most effective robustness (left).
The examples corrected by applying pre-training and DFR have little overlap (right), indicating that they largely improve performance on *different* subpopulations.
Meanwhile, the examples corrected by combining pre-training with DFR include most of the examples corrected by the individual interventions (right), suggesting that combining pre-training with DFR improves performance on *both* of these subpopulations.
Error bars denote 95% confidence intervals over 64 random trials.

Our observations in Section [5](#S5 "5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You") suggest that pre-training indeed can help prevent failures caused by poor extrapolation but not those stemming from biases in the reference dataset.
How, then, can we develop models that avoid *both* failure modes?
In this section, we explore one possible strategy: combining pre-training with interventions specifically designed to handle dataset biases.

In particular, we investigate the effectiveness of this strategy on WILDS-FMoW [christie2018functional, koh2020wilds], a distribution shift benchmark for classifying satellite images (in Appendix [C.3.1](#A3.SS3.SSS1 "C.3.1 Studying a synthetic shift ‣ C.3 Combining pre-training with interventions for handling bias ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we provide a similar analysis for a synthetic distribution shift).
In WILDS-FMoW, the reference dataset consists of satellite images taken between 2002 and 2012, while the shifted dataset consists of satellite images taken between 2016 and 2017.
Additionally, the images depict different regions and models typically underperform on underrepresented regions.
Following [koh2020wilds], we evaluate the *worst-group accuracy* (the minimum accuracy across groups—in our case, regions) on the shifted dataset.
Hence, robustness to this shift requires being able to both extrapolate to later years *and* perform consistently across regions (e.g., by avoiding biases that are harmful to performance on some regions).

Aiming to overcome these two challenges, we leverage two types of interventions.
To extrapolate better to later years, we initialize the model via pre-training; specifically, we obtain our model by fine-tuning a CLIP ResNet-50 model.
To handle potential biases in the reference dataset, we employ *Deep Feature Reweighting* (DFR) [kirichenko2022last], an intervention intended to de-bias a model by re-training just the final layer on group-balanced data.
We measure the effective robustness of each intervention over a baseline of ResNet-50 models trained from scratch.
We find that pre-training and DFR each yield some effective robustness and that combining the two yields greater effective robustness than applying either individually (see the left side of Figure [6](#S6.F6 "Figure 6 ‣ 6 Combining Pre-Training with Interventions for Handling Bias ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
See Appendix [B.5](#A2.SS5 "B.5 Combining pre-training with interventions for handling bias ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for a description of the exact setup.

###### Understanding robustness benefits.

We observe that combining pre-training and DFR can be effective for developing robust models, but is this actually because they address different failure modes, as we suggest?
To answer this question, we consider the *corrected examples* of each intervention, i.e., the set of test examples that are often classified incorrectly by a baseline model but correctly by model with the intervention (on average over 646464 trials).
We observe that the corrected examples of pre-training and DFR have little overlap (see the right side of Figure [6](#S6.F6 "Figure 6 ‣ 6 Combining Pre-Training with Interventions for Handling Bias ‣ Ask Your Distribution Shift if Pre-Training is Right for You")), suggesting that their benefits are indeed complementary.
Meanwhile, the corrected examples of combining pre-training with DFR include most of the corrected examples of the individual interventions.
This suggests that combining pre-training with DFR not only yields high effective robustness but in fact leads to models with both sets of benefits.

### 7 Curating Datasets for Fine-Tuning

(a) CelebA vs. our curated hair color classification dataset.
In the CelebA dataset (top), attributes such as gender are spuriously correlated with the class (blond vs. non-blond).
In our much smaller curated dataset (bottom), every real image is paired with a synthesized “counterfactual example” of the other class.
As a result, the primary difference between the blond and non-blond populations is hair color; other attributes such as gender, age and hair style are not predictive.
We include only females in our dataset to illustrate that diversity might not be necessary for robustness when fine-tuning.

(b) 
Fine-tuning on our curated dataset.
Fine-tuning a pre-trained model on the CelebA dataset (orange) yields little effective robustness over a baseline of models trained from scratch (blue).
However, fine-tuning the same pre-trained model on just 646464 examples from our curated dataset (red) yields a model with both high effective robustness and high accuracy.
Training from scratch on our curated dataset (green) also yields high effective robustness,
but results in substantially lower accuracy than pre-trained models, even with many more examples.
Error bars denote 95% confidence intervals over 646464 random trials.

Figure 7: Fine-tuning a pre-trained model on a small, non-diverse but de-biased dataset (see Figure [7(a)](#S7.F7.sf1 "In Figure 7 ‣ 7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You")) yields a robust and performant model for hair color classification in CelebA (see Figure [7(b)](#S7.F7.sf2 "In Figure 7 ‣ 7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).

In Section [6](#S6 "6 Combining Pre-Training with Interventions for Handling Bias ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we explored pairing pre-training with interventions specifically designed to address dataset biases.
We observed that this strategy can be effective for developing models that both extrapolate effectively *and* avoid undesirable biases present in the reference distribution.

In this section, we highlight one such intervention: training on a carefully curated (and, in particular, de-biased) dataset *instead* of the original reference dataset.
In general, de-biasing a large and diverse dataset may be prohibitively expensive.
However, if we can rely on pre-training for extrapolation (as suggested in Section [5](#S5 "5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You")), we might only need a small, non-diverse fine-tuning dataset, which would be more feasible to de-bias.
Thus, curating such a dataset and then fine-tuning a large pre-trained model on it might be a relatively inexpensive method for developing robust and performant models.

As a case study, we consider the task of predicting hair color (blond vs. non-blond) in the CelebA dataset [liu2015faceattributes].
In this dataset, hair color is spuriously correlated with other attributes (especially gender).
For example, 24%percent2424\% of females are blond, while only 2%percent22\% of males are blond.
Following works studying *group robustness* [sagawa2019distributionally, liu2021just, kirichenko2022last], we measure worst-group accuracy to assess robustness rather than measuring accuracy on an explicit shifted dataset.
In this case, the four groups are blond females, non-blond females, blond males and non-blond males.
A model exploiting the spurious correlation between gender and hair color would likely perform poorly on the underrepresented group of blond males.

###### Curating a de-biased dataset.

To curate a de-biased dataset for hair color classification with n𝑛n examples, we construct a “counterfactual example” for each of n/2𝑛2n/2 CelebA examples by changing the person’s hair to a color corresponding to the opposite class (i.e., blond to non-blond and vice versa).
We ensure that attributes besides hair color remain unchanged and include both the original and edited images in our dataset.
Hence, attributes that are spuriously correlated with hair color in the CelebA dataset (e.g., gender, age) are equally represented in the blond and non-blond populations of our curated dataset.
To illustrate that this dataset does *not* need to be diverse to yield high robustness and performance when fine-tuning, we restrict the dataset to include *only* females.
See Figure [7(a)](#S7.F7.sf1 "In Figure 7 ‣ 7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for a visualization of the dataset and Appendix [B.6](#A2.SS6 "B.6 Curating datasets for fine-tuning ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for the image editing process.
In Appendix [C.4.2](#A3.SS4.SSS2 "C.4.2 Exploring balancing instead of counterfactual editing ‣ C.4 Curating datasets for fine-tuning ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we consider the simpler curation strategy of balancing the number of samples from each group [idrissi2022simple] and find that counterfactual image editing is more effective.

###### Fine-tuning on a de-biased dataset.

As expected, models trained from scratch on the CelebA dataset exhibit high accuracy but very low worst-group accuracy, likely because they rely on gender to predict hair color (see Figure [7(b)](#S7.F7.sf2 "In Figure 7 ‣ 7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
Furthermore, a pre-trained CLIP ViT-B/32 model fine-tuned on the CelebA dataset exhibits very little effective robustness above these models trained from scratch, consistent with our hypothesis that pre-training does not mitigate dataset biases.
However, we observe that fine-tuning the same pre-trained model on *just* 646464 examples from our curated dataset yields a model with both high accuracy *and* effective robustness.
Finally, we also train models from scratch on our curated dataset and find that they exhibit substantial effective robustness, but require many more examples to attain a comparable accuracy.
This suggests that the extrapolation benefits of pre-training are key to make effective use of our small, non-diverse curated dataset.
In particular, as we illustrate in Appendix [C.4.1](#A3.SS4.SSS1 "C.4.1 Understanding the robustness benefits of pre-training when fine-tuning on a curated dataset ‣ C.4 Curating datasets for fine-tuning ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), pre-trained models extrapolate from the female-only curated dataset to males better than models trained from scratch.

### 8 Related Work

###### Characterizing distribution shifts.

There exists a plethora of definitions for characterizing distribution shifts, many of which are aligned with the in-support and out-of-support chracterizations that we discuss in this work.
For example, *domain generalization* involves shifts in which the reference and shifted distributions are from different domains [koh2020wilds, gulrajani2020search].
In a *subpopulation shift*, subpopulations appear with different frequencies in the reference and shifted distributions [santurkar2020breeds, koh2020wilds, yang2023change].
In shifts with *spurious correlations*, certain features are predictive in the reference distribution but not in the shifted distribution [arjovsky2019invariant, sagawa2020investigation].
Two more formal characterizations are *covariate shift* [shimodaira2000improving], under which p​(y|x)𝑝conditional𝑦𝑥p(y|x) is fixed, and *label shift* [lipton2018detecting], under which the label distribution may change but p​(x|y)𝑝conditional𝑥𝑦p(x|y) is fixed.
We relate these definitons to in-support and out-of-support shifts in Appendix [D.4](#A4.SS4 "D.4 Relating in-support and out-of-support shifts to existing characterizations ‣ Appendix D Additional Discussion ‣ Ask Your Distribution Shift if Pre-Training is Right for You").

###### Robustness benefits of pre-training.

Several works have suggested that pre-training can be an effective strategy for improving robustness to distribution shifts [hendrycks2019using, hendrycks2020faces, hendrycks2020pretrained, tu2020empirical, wiles2021fine, andreassen2021evolution, bommasani2021opportunities, liu2022empirical, ramanujan2023connection].
In particular, [wiles2021fine] define different types of distribution shifts and find that pre-training frequently improves performance under these shifts, while most other interventions primarily help in specific settings.
In the natural language processing setting, [tu2020empirical] argue that when pre-training helps with spurious correlations, it is because pre-trained models can generalize better from the small number of counterexamples to these correlations; as we discuss in Appendix [D.5](#A4.SS5 "D.5 Understanding the robustness of pre-trained language models to spurious correlations ‣ Appendix D Additional Discussion ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), this is consistent with our intuition that pre-training helps specifically with extrapolation.
Lastly, [bommasani2021opportunities] discuss failure modes that pre-training is unlikely to address including spurious correlations (both in pre-training and fine-tuning datasets) and extrapolation across time.

### 9 Conclusion

In this work, we study the failure modes that pre-training alone can and cannot address. Our findings suggest that pre-training can help mitigate failures caused by poor extrapolation (e.g., inability to generalize to a new domain) but might not address other failures, such as those stemming from dataset biases.
In light of this observation, dataset biases present a fundamental limitation that cannot be overcome by simply leveraging additional pre-training data or larger models.
We thus encourage practitioners not to treat pre-training as a panacea for robustness. Instead, they should consider the specific failures modes they might encounter to determine if pre-training can help.

### Acknowledgements

The authors would like to thank Sharut Gupta, Alaa Khaddaj and Harshay Shah and for helpful feedback on the draft.

Work supported in part by the NSF grant DMS-2134108 and Open Philanthropy.
This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0015.

## 

\parttoc

### Appendix A Theoretical Results

#### A.1 Proof of Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ Setup. ‣ 4 Studying Pre-Training in a Logistic Regression Setting ‣ Ask Your Distribution Shift if Pre-Training is Right for You")

###### Setup.

Suppose that we are given access to a reference dataset Srefsubscript𝑆refS\_{\text{ref}} of input-label pairs (x,y)𝑥𝑦(x,y), with x∈ℝd𝑥superscriptℝ𝑑x\in\mathbb{R}^{d} and y∈{−1,1}𝑦11y\in\{-1,1\}.
We decide to learn a linear classifier for this task by finding a weight w𝑤w that minimizes the (standard) logistic loss on Srefsubscript𝑆refS\_{\text{ref}}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lref​(w)=∑(x,y)∈Sreflog⁡(1+e−w⊤​x⋅y).subscript𝐿ref𝑤subscript𝑥𝑦subscript𝑆ref1superscript𝑒⋅superscript𝑤top𝑥𝑦L\_{\text{ref}}(w)=\sum\_{(x,y)\in S\_{\text{ref}}}\log(1+e^{-w^{\top}x\cdot y}). |  | (1) |

We assume that the reference dataset Srefsubscript𝑆refS\_{\text{ref}} satisfies the following conditions:

1. 1.

   Inputs in Srefsubscript𝑆refS\_{\text{ref}} lie within a k𝑘k-dimensional (with k<d𝑘𝑑k<d) subspace Wrefsubscript𝑊refW\_{\text{ref}} of ℝdsuperscriptℝ𝑑\mathbb{R}^{d}. Intuitively, this condition represents a lack of variation in certain features in the reference dataset.
2. 2.

   The logistic loss Lrefsubscript𝐿refL\_{\text{ref}} has a minimum value. This condition ensures that minimizing Lrefsubscript𝐿refL\_{\text{ref}} is well-defined. Note that there may be multiple weights that attain this minimum value.

See [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ Setup. ‣ 4 Studying Pre-Training in a Logistic Regression Setting ‣ Ask Your Distribution Shift if Pre-Training is Right for You")

To prove Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ Setup. ‣ 4 Studying Pre-Training in a Logistic Regression Setting ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we will first show that running gradient descent starting from an initialization within Wrefsubscript𝑊refW\_{\text{ref}} always converges to the same weights wref∗superscriptsubscript𝑤refw\_{\text{ref}}^{\*}.
We will then show that running gradient descent starting from an arbitrary initialization has the same convergence behavior except for an “offset” term projWref⊥​winitsubscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤init\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}} representing the component of the initialization that is orthogonal to Wrefsubscript𝑊refW\_{\text{ref}}.

##### A.1.1 Convexity and smoothness of the loss

We begin by providing the gradient and hessian of Lrefsubscript𝐿ref{L\_{\text{ref}}} and using these to establish convexity (Lemma [A.1](#A1.Thmtheorem1 "Lemma A.1. ‣ A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")) and smoothness (Lemma [A.2](#A1.Thmtheorem2 "Lemma A.2. ‣ A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")) properties of Lrefsubscript𝐿ref{L\_{\text{ref}}}.
The gradient of Lrefsubscript𝐿ref{L\_{\text{ref}}} is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇Lref​(w)=∑(x,y)∈Srefx⋅y⋅11+ew⊤​x⋅y.∇subscript𝐿ref𝑤subscript𝑥𝑦subscript𝑆ref⋅𝑥𝑦11superscript𝑒⋅superscript𝑤top𝑥𝑦\nabla{L\_{\text{ref}}}(w)=\sum\_{(x,y)\in S\_{\text{ref}}}x\cdot y\cdot\frac{1}{1+e^{w^{\top}x\cdot y}}. |  | (3) |

The Hessian of Lrefsubscript𝐿ref{L\_{\text{ref}}} is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇2Lref​(w)=∑(x,y)∈Srefx​x⊤⋅12+e−w⊤​x⋅y+ew⊤​x⋅y=X⊤​D​(w)​Xsuperscript∇2subscript𝐿ref𝑤subscript𝑥𝑦subscript𝑆ref⋅𝑥superscript𝑥top12superscript𝑒⋅superscript𝑤top𝑥𝑦superscript𝑒⋅superscript𝑤top𝑥𝑦superscript𝑋top𝐷𝑤𝑋\nabla^{2}{L\_{\text{ref}}}(w)=\sum\_{(x,y)\in S\_{\text{ref}}}xx^{\top}\cdot\frac{1}{2+e^{-w^{\top}x\cdot y}+e^{w^{\top}x\cdot y}}=X^{\top}D(w)X |  | (4) |

where X∈ℝ|Sref|×d𝑋superscriptℝsubscript𝑆ref𝑑X\in\mathbb{R}^{|S\_{\text{ref}}|\times d} is the matrix of inputs in Srefsubscript𝑆refS\_{\text{ref}} and D​(w)∈ℝ|Sref|×|Sref|𝐷𝑤superscriptℝsubscript𝑆refsubscript𝑆refD(w)\in\mathbb{R}^{|S\_{\text{ref}}|\times|S\_{\text{ref}}|} is the diagonal matrix with D​(w)i​i=12+e−w⊤​x⋅y+ew⊤​x⋅y𝐷subscript𝑤𝑖𝑖12superscript𝑒⋅superscript𝑤top𝑥𝑦superscript𝑒⋅superscript𝑤top𝑥𝑦D(w)\_{ii}=\frac{1}{2+e^{-w^{\top}x\cdot y}+e^{w^{\top}x\cdot y}}. Note in particular that the non-zero elements of D​(w)𝐷𝑤D(w) are in (0,1/4)014(0,1/4).

###### Lemma A.1.

The loss Lrefsubscript𝐿ref{L\_{\text{ref}}} is (1) convex on ℝdsuperscriptℝ𝑑{\mathbb{R}^{d}}, (2) strictly convex on Wrefsubscript𝑊refW\_{\text{ref}}, and (3) strongly convex on any closed convex subset of Wrefsubscript𝑊refW\_{\text{ref}}.

###### Proof.

According to Taylor’s Theorem, for any u,v∈ℝd

𝑢𝑣
superscriptℝ𝑑u,v\in{\mathbb{R}^{d}}, there exists a α∈[0,1]𝛼01\alpha\in[0,1] such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lref​(v)=Lref​(u)+∇Lref​(u)⊤​(v−u)+12⋅(v−u)⊤​∇2Lref​(v+α⋅(v−u))​(v−u).subscript𝐿ref𝑣subscript𝐿ref𝑢∇subscript𝐿refsuperscript𝑢top𝑣𝑢⋅12superscript𝑣𝑢topsuperscript∇2subscript𝐿ref𝑣⋅𝛼𝑣𝑢𝑣𝑢{L\_{\text{ref}}}(v)={L\_{\text{ref}}}(u)+\nabla{L\_{\text{ref}}}(u)^{\top}(v-u)+\frac{1}{2}\cdot(v-u)^{\top}\nabla^{2}{L\_{\text{ref}}}(v+\alpha\cdot(v-u))(v-u). |  | (5) |

1. 1.

   Convexity on ℝdsuperscriptℝ𝑑{\mathbb{R}^{d}}. To show that Lrefsubscript𝐿ref{L\_{\text{ref}}} is convex on ℝdsuperscriptℝ𝑑{\mathbb{R}^{d}}, we need to show that

   |  |  |  |
   | --- | --- | --- |
   |  | Lref​(v)≥Lref​(u)+∇Lref​(u)⊤​(v−u)subscript𝐿ref𝑣subscript𝐿ref𝑢∇subscript𝐿refsuperscript𝑢top𝑣𝑢{L\_{\text{ref}}}(v)\geq{L\_{\text{ref}}}(u)+\nabla{L\_{\text{ref}}}(u)^{\top}(v-u) |  |

   for any u,v∈ℝd
   𝑢𝑣superscriptℝ𝑑u,v\in{\mathbb{R}^{d}}. Using ([5](#A1.E5 "In Proof. ‣ Lemma A.1. ‣ A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")), it suffices to show that a⊤​∇2Lref​(w)​a≥0superscript𝑎topsuperscript∇2subscript𝐿ref𝑤𝑎0a^{\top}\nabla^{2}{L\_{\text{ref}}}(w)a\geq 0 for any a∈ℝd𝑎superscriptℝ𝑑a\in{\mathbb{R}^{d}} and w∈ℝd𝑤superscriptℝ𝑑w\in{\mathbb{R}^{d}}. Recall from ([4](#A1.E4 "In A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")) that ∇2Lref​(w)=X⊤​D​(w)​Xsuperscript∇2subscript𝐿ref𝑤superscript𝑋top𝐷𝑤𝑋\nabla^{2}{L\_{\text{ref}}}(w)=X^{\top}D(w)X. Thus, we have

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | a⊤​∇2Lref​(w)​asuperscript𝑎topsuperscript∇2subscript𝐿ref𝑤𝑎\displaystyle a^{\top}\nabla^{2}{L\_{\text{ref}}}(w)a | =a⊤​X⊤​D​(w)​X​aabsentsuperscript𝑎topsuperscript𝑋top𝐷𝑤𝑋𝑎\displaystyle=a^{\top}X^{\top}D(w)Xa |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | =‖D​(w)1/2​X​a‖22absentsubscriptsuperscriptnorm𝐷superscript𝑤12𝑋𝑎22\displaystyle=\|D(w)^{1/2}Xa\|^{2}\_{2} |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | ≥0absent0\displaystyle\geq 0 |  |
2. 2.

   Strict convexity on Wrefsubscript𝑊refW\_{\text{ref}}. Next, to show that Lrefsubscript𝐿ref{L\_{\text{ref}}} is strictly convex on Wrefsubscript𝑊refW\_{\text{ref}}, we need to show that

   |  |  |  |
   | --- | --- | --- |
   |  | Lref​(v)>Lref​(u)+∇Lref​(u)⊤​(v−u)subscript𝐿ref𝑣subscript𝐿ref𝑢∇subscript𝐿refsuperscript𝑢top𝑣𝑢{L\_{\text{ref}}}(v)>{L\_{\text{ref}}}(u)+\nabla{L\_{\text{ref}}}(u)^{\top}(v-u) |  |

   for any u,v∈Wref
   𝑢𝑣subscript𝑊refu,v\in W\_{\text{ref}}. Using ([5](#A1.E5 "In Proof. ‣ Lemma A.1. ‣ A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")), it suffices to show that a⊤​∇2Lref​(w)​a>0superscript𝑎topsuperscript∇2subscript𝐿ref𝑤𝑎0a^{\top}\nabla^{2}{L\_{\text{ref}}}(w)a>0 for any non-zero a∈Wref𝑎subscript𝑊refa\in W\_{\text{ref}} and w∈Wref𝑤subscript𝑊refw\in W\_{\text{ref}}. We know that a⊤​∇2Lref​(w)​a=‖D​(w)1/2​X​a‖22superscript𝑎topsuperscript∇2subscript𝐿ref𝑤𝑎subscriptsuperscriptnorm𝐷superscript𝑤12𝑋𝑎22a^{\top}\nabla^{2}{L\_{\text{ref}}}(w)a=\|D(w)^{1/2}Xa\|^{2}\_{2}. Since D​(w)𝐷𝑤D(w) is diagonal with positive entries along the diagonal, ‖D​(w)1/2​X​a‖22>0subscriptsuperscriptnorm𝐷superscript𝑤12𝑋𝑎220\|D(w)^{1/2}Xa\|^{2}\_{2}>0 if and only if X​a≠0𝑋𝑎0Xa\neq 0. Recall that Wrefsubscript𝑊refW\_{\text{ref}} is the subspace spanning the rows of X𝑋X. Hence, since a𝑎a is non-zero and is in Wrefsubscript𝑊refW\_{\text{ref}}, we know that X​a≠0𝑋𝑎0Xa\neq 0.
3. 3.

   Strong convexity on any closed convex subset of Wrefsubscript𝑊refW\_{\text{ref}}. Finally, to show that Lrefsubscript𝐿ref{L\_{\text{ref}}} is strongly convex on any closed convex subset T𝑇T of Wrefsubscript𝑊refW\_{\text{ref}}, we need to show that there exists an m>0𝑚0m>0 such that

   |  |  |  |
   | --- | --- | --- |
   |  | Lref​(v)≥Lref​(u)+∇Lref​(u)⊤​(v−u)+m2​‖v−u‖22subscript𝐿ref𝑣subscript𝐿ref𝑢∇subscript𝐿refsuperscript𝑢top𝑣𝑢𝑚2superscriptsubscriptnorm𝑣𝑢22{L\_{\text{ref}}}(v)\geq{L\_{\text{ref}}}(u)+\nabla{L\_{\text{ref}}}(u)^{\top}(v-u)+\frac{m}{2}\|v-u\|\_{2}^{2} |  |

   for any u,v∈T
   𝑢𝑣𝑇u,v\in T. Using ([5](#A1.E5 "In Proof. ‣ Lemma A.1. ‣ A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")), it suffices to show that there exists an m>0𝑚0m>0 such that a⊤​∇2Lref​(w)​a>m2⋅‖a‖22superscript𝑎topsuperscript∇2subscript𝐿ref𝑤𝑎⋅𝑚2superscriptsubscriptnorm𝑎22a^{\top}\nabla^{2}{L\_{\text{ref}}}(w)a>\frac{m}{2}\cdot\|a\|\_{2}^{2} for any a∈Wref𝑎subscript𝑊refa\in W\_{\text{ref}} and w∈T𝑤𝑇w\in T.
   Making use of the fact that T𝑇T is closed, let λminsubscript𝜆min\lambda\_{\text{min}} be the minimum diagonal entry of D​(w)𝐷𝑤D(w) for w∈T𝑤𝑇w\in T, that is,

   |  |  |  |
   | --- | --- | --- |
   |  | λmin=minw∈T⁡mini∈{1,…,|Sref|}⁡D​(w)i​i.subscript𝜆minsubscript𝑤𝑇subscript𝑖1…subscript𝑆ref𝐷subscript𝑤𝑖𝑖\lambda\_{\text{min}}=\min\_{w\in T}\min\_{i\in\{1,\ldots,|S\_{\text{ref}}|\}}D(w)\_{ii}. |  |

   Next, let cminsubscript𝑐minc\_{\text{min}} be the minimum value of ‖X​a‖22superscriptsubscriptnorm𝑋𝑎22\|Xa\|\_{2}^{2} over unit vectors a𝑎a in Wrefsubscript𝑊refW\_{\text{ref}}, that is,

   |  |  |  |
   | --- | --- | --- |
   |  | cmin=mina∈Wref,‖a‖2=1⁡‖X​a‖22.subscript𝑐minsubscriptformulae-sequence𝑎subscript𝑊refsubscriptnorm𝑎21superscriptsubscriptnorm𝑋𝑎22c\_{\text{min}}=\min\_{a\in W\_{\text{ref}},\|a\|\_{2}=1}\|Xa\|\_{2}^{2}. |  |

   We previously established that X​a≠0𝑋𝑎0Xa\neq 0 for any non-zero a∈Wref𝑎subscript𝑊refa\in W\_{\text{ref}}, which means that cmin>0subscript𝑐min0c\_{\text{min}}>0. Finally, we conclude that for m=2⋅λmin⋅cmin𝑚⋅2subscript𝜆minsubscript𝑐minm=2\cdot\lambda\_{\text{min}}\cdot c\_{\text{min}}, a⊤​∇2Lref​(w)​a=‖D​(w)1/2​X​a‖22≥λmin⋅cmin⋅‖a‖22=m2⋅‖a‖22superscript𝑎topsuperscript∇2subscript𝐿ref𝑤𝑎subscriptsuperscriptnorm𝐷superscript𝑤12𝑋𝑎22⋅subscript𝜆minsubscript𝑐minsubscriptsuperscriptnorm𝑎22⋅𝑚2superscriptsubscriptnorm𝑎22a^{\top}\nabla^{2}{L\_{\text{ref}}}(w)a=\|D(w)^{1/2}Xa\|^{2}\_{2}\geq\lambda\_{\text{min}}\cdot c\_{\text{min}}\cdot\|a\|^{2}\_{2}=\frac{m}{2}\cdot\|a\|\_{2}^{2}.

∎

###### Lemma A.2.

The gradient of the loss function ∇Lref∇subscript𝐿ref\nabla{L\_{\text{ref}}} is K𝐾K-Lipschitz with K=‖X‖op2/4𝐾subscriptsuperscriptnorm𝑋2op4K=\|X\|^{2}\_{\text{op}}/4.

###### Proof.

To show that ∇Lref∇subscript𝐿ref\nabla{L\_{\text{ref}}} is K𝐾K-Lipschitz, we need to show that ∇2Lref​(w)⪯K​Iprecedes-or-equalssuperscript∇2subscript𝐿ref𝑤𝐾𝐼\nabla^{2}{L\_{\text{ref}}}(w)\preceq KI.
Recall from ([4](#A1.E4 "In A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")) that ∇2Lref​(w)=X⊤​D​(w)​Xsuperscript∇2subscript𝐿ref𝑤superscript𝑋top𝐷𝑤𝑋\nabla^{2}{L\_{\text{ref}}}(w)=X^{\top}D(w)X. Thus, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | a⊤​∇2Lref​(w)​asuperscript𝑎topsuperscript∇2subscript𝐿ref𝑤𝑎\displaystyle a^{\top}\nabla^{2}{L\_{\text{ref}}}(w)a | =a⊤​X⊤​D​(w)​X​aabsentsuperscript𝑎topsuperscript𝑋top𝐷𝑤𝑋𝑎\displaystyle=a^{\top}X^{\top}D(w)Xa |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =‖D​(w)1/2​X​a‖22absentsubscriptsuperscriptnorm𝐷superscript𝑤12𝑋𝑎22\displaystyle=\|D(w)^{1/2}Xa\|^{2}\_{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤‖D​(w)1/2‖op2⋅‖X‖op2⋅‖a‖22absent⋅subscriptsuperscriptnorm𝐷superscript𝑤122opsubscriptsuperscriptnorm𝑋2opsubscriptsuperscriptnorm𝑎22\displaystyle\leq\|D(w)^{1/2}\|^{2}\_{\text{op}}\cdot\|X\|^{2}\_{\text{op}}\cdot\|a\|^{2}\_{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤(‖X‖op2/4)⋅‖a‖22.absent⋅subscriptsuperscriptnorm𝑋2op4subscriptsuperscriptnorm𝑎22\displaystyle\leq(\|X\|^{2}\_{\text{op}}/4)\cdot\|a\|^{2}\_{2}. |  |

In the final step, we use the fact that D​(w)𝐷𝑤D(w) is diagonal with non-zero elements in (0,1/4)014(0,1/4) to conclude that ‖D​(w)1/2‖op2≤1/4subscriptsuperscriptnorm𝐷superscript𝑤122op14\|D(w)^{1/2}\|^{2}\_{\text{op}}\leq 1/4.
∎

##### A.1.2 Convergence of gradient descent within the reference subspace

Next, we establish that there exists a unique minimumizer of Lrefsubscript𝐿ref{L\_{\text{ref}}} within the reference subspace Wrefsubscript𝑊refW\_{\text{ref}} (Lemma [A.3](#A1.Thmtheorem3 "Lemma A.3. ‣ A.1.2 Convergence of gradient descent within the reference subspace ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")) and that gradient descent converges to these weights (Lemma [A.4](#A1.Thmtheorem4 "Lemma A.4. ‣ A.1.2 Convergence of gradient descent within the reference subspace ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).

###### Lemma A.3.

There exists a unique wref∗∈Wrefsubscriptsuperscript𝑤refsubscript𝑊refw^{\*}\_{\text{ref}}\in W\_{\text{ref}} such that wref∗∈arg⁡minw⁡L​(w)subscriptsuperscript𝑤refsubscript𝑤𝐿𝑤w^{\*}\_{\text{ref}}\in\arg\min\_{w}L(w).

###### Proof.

We will first show that there exists a wref∗∈Wrefsubscriptsuperscript𝑤refsubscript𝑊refw^{\*}\_{\text{ref}}\in W\_{\text{ref}} such that wref∗∈arg⁡minw⁡Lref​(w)subscriptsuperscript𝑤refsubscript𝑤subscript𝐿ref𝑤w^{\*}\_{\text{ref}}\in\arg\min\_{w}{L\_{\text{ref}}}(w).
Let w∗∈arg⁡minw⁡Lref​(w)superscript𝑤subscript𝑤subscript𝐿ref𝑤w^{\*}\in\arg\min\_{w}{L\_{\text{ref}}}(w) be an arbitrary minimimum point of Lrefsubscript𝐿ref{L\_{\text{ref}}}.
By definition, for every (x,y)∈Sref𝑥𝑦subscript𝑆ref(x,y)\in S\_{\text{ref}}, x∈Wref𝑥subscript𝑊refx\in W\_{\text{ref}}. Hence, for every such x𝑥x, w⊤​x=projWref​w⊤​xsuperscript𝑤top𝑥subscriptprojsubscript𝑊refsuperscript𝑤top𝑥w^{\top}x={\text{proj}\_{W\_{\text{ref}}}}w^{\top}x.
This means that Lref​(w∗)=Lref​(projWref​w∗)subscript𝐿refsuperscript𝑤subscript𝐿refsubscriptprojsubscript𝑊refsuperscript𝑤{L\_{\text{ref}}}(w^{\*})={L\_{\text{ref}}}({\text{proj}\_{W\_{\text{ref}}}}w^{\*}), which implies that wref∗:=projWref​w∗∈arg⁡minw⁡Lref​(w)assignsubscriptsuperscript𝑤refsubscriptprojsubscript𝑊refsuperscript𝑤subscript𝑤subscript𝐿ref𝑤w^{\*}\_{\text{ref}}:={\text{proj}\_{W\_{\text{ref}}}}w^{\*}\in\arg\min\_{w}{L\_{\text{ref}}}(w), as desired.
Next, because Lrefsubscript𝐿ref{L\_{\text{ref}}} is strictly convex on Wrefsubscript𝑊refW\_{\text{ref}} (Lemma [A.1](#A1.Thmtheorem1 "Lemma A.1. ‣ A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")), wref∗subscriptsuperscript𝑤refw^{\*}\_{\text{ref}} is the only minimum point of Lrefsubscript𝐿ref{L\_{\text{ref}}} in Wrefsubscript𝑊refW\_{\text{ref}}.
∎

###### Lemma A.4.

If we start with winit∈Wrefsubscript𝑤initsubscript𝑊refw\_{\text{init}}\in W\_{\text{ref}} and run gradient descent with η=4/‖X‖op2𝜂4superscriptsubscriptnorm𝑋op2\eta=4/\|X\|\_{\text{op}}^{2} to minimize Lref​(w)subscript𝐿ref𝑤{L\_{\text{ref}}}(w), the weights will converge to wref∗subscriptsuperscript𝑤refw^{\*}\_{\text{ref}}.

###### Proof.

Suppose that we start with initial weights winit∈Wrefsubscript𝑤initsubscript𝑊ref{w\_{\text{init}}}\in W\_{\text{ref}} and run gradient descent to minimize Lrefsubscript𝐿ref{L\_{\text{ref}}} with learning rate η𝜂\eta.
In particular, let w(0)=winitsuperscript𝑤0subscript𝑤initw^{(0)}={w\_{\text{init}}} and w(t+1)=w(t)+η⋅∇Lref​(w(t))superscript𝑤𝑡1superscript𝑤𝑡⋅𝜂∇subscript𝐿refsuperscript𝑤𝑡w^{(t+1)}=w^{(t)}+\eta\cdot\nabla{L\_{\text{ref}}}(w^{(t)}).
Because Lrefsubscript𝐿ref{L\_{\text{ref}}} is convex (Lemma [A.1](#A1.Thmtheorem1 "Lemma A.1. ‣ A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")), ∇Lref∇subscript𝐿ref\nabla{L\_{\text{ref}}} is K𝐾K-Lipschitz with K=‖X‖op2/4𝐾superscriptsubscriptnorm𝑋op24K=\|X\|\_{\text{op}}^{2}/4 (Lemma [A.2](#A1.Thmtheorem2 "Lemma A.2. ‣ A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")), and η=4/‖X‖op2≤1/K𝜂4superscriptsubscriptnorm𝑋op21𝐾\eta=4/\|X\|\_{\text{op}}^{2}\leq 1/K, we know from Theorem 3.2 of [bubeck2014theory] that

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lref​(w(t))−Lref​(wref∗)≤K⋅‖winit−wref∗‖t−1.subscript𝐿refsuperscript𝑤𝑡subscript𝐿refsubscriptsuperscript𝑤ref⋅𝐾normsubscript𝑤initsubscriptsuperscript𝑤ref𝑡1{L\_{\text{ref}}}(w^{(t)})-{L\_{\text{ref}}}(w^{\*}\_{\text{ref}})\leq\frac{K\cdot\|w\_{\text{init}}-w^{\*}\_{\text{ref}}\|}{t-1}. |  | (6) |

Hence, the loss attained by w(t)superscript𝑤𝑡w^{(t)} converges to the optimal loss attained by wref∗subscriptsuperscript𝑤refw^{\*}\_{\text{ref}}.
To show that w(t)superscript𝑤𝑡w^{(t)} converges to wref∗subscriptsuperscript𝑤refw^{\*}\_{\text{ref}}, we will show that Lrefsubscript𝐿ref{L\_{\text{ref}}} is strongly convex on a set containing every w(t)superscript𝑤𝑡w^{(t)} for t≥0𝑡0t\geq 0.
In particular, consider the set WGD={w∈Wref∣‖w−wref∗‖2≤‖winit−wref∗‖2}subscript𝑊GDconditional-set𝑤subscript𝑊refsubscriptnorm𝑤subscriptsuperscript𝑤ref2subscriptnormsubscript𝑤initsubscriptsuperscript𝑤ref2W\_{\text{GD}}=\{w\in W\_{\text{ref}}\mid\|w-w^{\*}\_{\text{ref}}\|\_{2}\leq\|w\_{\text{init}}-w^{\*}\_{\text{ref}}\|\_{2}\} containing weights in Wrefsubscript𝑊refW\_{\text{ref}} at least as close to wref∗subscriptsuperscript𝑤refw^{\*}\_{\text{ref}} as winitsubscript𝑤init{w\_{\text{init}}}. Clearly, WGDsubscript𝑊GDW\_{\text{GD}} contains w(0)=winitsuperscript𝑤0subscript𝑤initw^{(0)}=w\_{\text{init}}. We know from Theorem 3.2 of [bubeck2014theory] that with each iteration of gradient descent we get closer to a minimum point, that is, ‖w(t+1)−wref∗‖≤‖w(t)−wref∗‖normsuperscript𝑤𝑡1subscriptsuperscript𝑤refnormsuperscript𝑤𝑡subscriptsuperscript𝑤ref\|w^{(t+1)}-w^{\*}\_{\text{ref}}\|\leq\|w^{(t)}-w^{\*}\_{\text{ref}}\|.
Additionally, because winitsubscript𝑤init{w\_{\text{init}}} and ∇Lref∇subscript𝐿ref\nabla{L\_{\text{ref}}} are in Wrefsubscript𝑊refW\_{\text{ref}}, every w(t)superscript𝑤𝑡w^{(t)} is in Wrefsubscript𝑊refW\_{\text{ref}}.
Hence, every w(t)superscript𝑤𝑡w^{(t)} is in WGDsubscript𝑊GDW\_{\text{GD}}.
Because WGDsubscript𝑊GDW\_{\text{GD}} is closed and convex, from Lemma [A.1](#A1.Thmtheorem1 "Lemma A.1. ‣ A.1.1 Convexity and smoothness of the loss ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You") we know that Lrefsubscript𝐿ref{L\_{\text{ref}}} is strongly convex on WGDsubscript𝑊GDW\_{\text{GD}}. This means that there exists an m>0𝑚0m>0 such that

|  |  |  |
| --- | --- | --- |
|  | Lref​(w(t))≥Lref​(wref∗)+∇Lref​(wref∗)⊤​(w(t)−wref∗)+m2⋅‖w(t)−wref∗‖22.subscript𝐿refsuperscript𝑤𝑡subscript𝐿refsubscriptsuperscript𝑤ref∇subscript𝐿refsuperscriptsubscriptsuperscript𝑤reftopsuperscript𝑤𝑡subscriptsuperscript𝑤ref⋅𝑚2superscriptsubscriptnormsuperscript𝑤𝑡subscriptsuperscript𝑤ref22{L\_{\text{ref}}}(w^{(t)})\geq{L\_{\text{ref}}}(w^{\*}\_{\text{ref}})+\nabla{L\_{\text{ref}}}(w^{\*}\_{\text{ref}})^{\top}(w^{(t)}-w^{\*}\_{\text{ref}})+\frac{m}{2}\cdot\|w^{(t)}-w^{\*}\_{\text{ref}}\|\_{2}^{2}. |  |

Plugging in ∇Lref​(wref∗)=0∇subscript𝐿refsubscriptsuperscript𝑤ref0\nabla{L\_{\text{ref}}}(w^{\*}\_{\text{ref}})=0 and rearranging, we get

|  |  |  |
| --- | --- | --- |
|  | ‖w(t)−wref∗‖22≤2m⋅(Lref​(w(t))−Lref​(wref∗)).superscriptsubscriptnormsuperscript𝑤𝑡subscriptsuperscript𝑤ref22⋅2𝑚subscript𝐿refsuperscript𝑤𝑡subscript𝐿refsubscriptsuperscript𝑤ref\|w^{(t)}-w^{\*}\_{\text{ref}}\|\_{2}^{2}\leq\frac{2}{m}\cdot({L\_{\text{ref}}}(w^{(t)})-{L\_{\text{ref}}}(w^{\*}\_{\text{ref}})). |  |

Finally, combining with ([6](#A1.E6 "In Proof. ‣ Lemma A.4. ‣ A.1.2 Convergence of gradient descent within the reference subspace ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")) yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖w(t)−wref∗‖22≤2⋅K⋅‖winit−wref∗‖m⋅(t−1)superscriptsubscriptnormsuperscript𝑤𝑡subscriptsuperscript𝑤ref22⋅2𝐾normsubscript𝑤initsubscriptsuperscript𝑤ref⋅𝑚𝑡1\|w^{(t)}-w^{\*}\_{\text{ref}}\|\_{2}^{2}\leq\frac{2\cdot K\cdot\|{w\_{\text{init}}}-w^{\*}\_{\text{ref}}\|}{m\cdot(t-1)} |  | (7) |

which completes our proof.
∎

##### A.1.3 Proof of Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ Setup. ‣ 4 Studying Pre-Training in a Logistic Regression Setting ‣ Ask Your Distribution Shift if Pre-Training is Right for You")

We are now ready to prove Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ Setup. ‣ 4 Studying Pre-Training in a Logistic Regression Setting ‣ Ask Your Distribution Shift if Pre-Training is Right for You"). Suppose that we start with initial weights winitsubscript𝑤initw\_{\text{init}} and run gradient descent to minimize Lrefsubscript𝐿refL\_{\text{ref}} with learning rate η=4/‖X‖op2𝜂4superscriptsubscriptnorm𝑋op2\eta=4/\|X\|\_{\text{op}}^{2}.
In particular, let w(0)=winitsuperscript𝑤0subscript𝑤initw^{(0)}=w\_{\text{init}} and w(t+1)=w(t)+η⋅∇Lref​(w(t))superscript𝑤𝑡1superscript𝑤𝑡⋅𝜂∇subscript𝐿refsuperscript𝑤𝑡w^{(t+1)}=w^{(t)}+\eta\cdot\nabla L\_{\text{ref}}(w^{(t)}) for t≥0𝑡0t\geq 0.
We will show that running gradient descent starting with an arbitrary winitsubscript𝑤initw\_{\text{init}} has the same behavior as running gradient descent with winitsubscript𝑤initw\_{\text{init}} projected onto Wrefsubscript𝑊refW\_{\text{ref}}.
To be more precise, suppose that we instead start with initial weights projWref​winitsubscriptprojsubscript𝑊refsubscript𝑤init{\text{proj}\_{W\_{\text{ref}}}}{w\_{\text{init}}} when running gradient descent.
In particular, with projW​usubscriptproj𝑊𝑢\text{proj}\_{W}u denoting the projection of u𝑢u onto a subspace W𝑊W, let wproj(0)=projWref​winitsuperscriptsubscript𝑤proj0subscriptprojsubscript𝑊refsubscript𝑤init{w\_{\text{proj}}}^{(0)}={\text{proj}\_{W\_{\text{ref}}}}{w\_{\text{init}}} and wproj(t+1)=wproj(t)+η⋅∇Lref​(wproj(t))superscriptsubscript𝑤proj𝑡1superscriptsubscript𝑤proj𝑡⋅𝜂∇subscript𝐿refsuperscriptsubscript𝑤proj𝑡{w\_{\text{proj}}}^{(t+1)}={w\_{\text{proj}}}^{(t)}+\eta\cdot\nabla L\_{\text{ref}}({w\_{\text{proj}}}^{(t)}) for t≥0𝑡0t\geq 0. Then the trajectory of w(t)superscript𝑤𝑡w^{(t)} is the same as that of wproj(t)superscriptsubscript𝑤proj𝑡{w\_{\text{proj}}}^{(t)} but with an additional component projWref⊥​winit=(winit−projWref​winit)subscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤initsubscript𝑤initsubscriptprojsubscript𝑊refsubscript𝑤init\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}=(w\_{\text{init}}-{\text{proj}\_{W\_{\text{ref}}}}{w\_{\text{init}}}).
That is,

|  |  |  |
| --- | --- | --- |
|  | w(t)=projWref⊥​winit+wproj(t).superscript𝑤𝑡subscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤initsuperscriptsubscript𝑤proj𝑡w^{(t)}=\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}+{w\_{\text{proj}}}^{(t)}. |  |

To show that this is the case, we will proceed by induction. As a base case,

|  |  |  |  |
| --- | --- | --- | --- |
|  | w(0)superscript𝑤0\displaystyle w^{(0)} | =winitabsentsubscript𝑤init\displaystyle={w\_{\text{init}}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(winit−projWref​winit)+projWref​winitabsentsubscript𝑤initsubscriptprojsubscript𝑊refsubscript𝑤initsubscriptprojsubscript𝑊refsubscript𝑤init\displaystyle=({w\_{\text{init}}}-{\text{proj}\_{W\_{\text{ref}}}}{w\_{\text{init}}})+{\text{proj}\_{W\_{\text{ref}}}}{w\_{\text{init}}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =projWref⊥​winit+wproj(0).absentsubscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤initsuperscriptsubscript𝑤proj0\displaystyle=\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}+{w\_{\text{proj}}}^{(0)}. |  |

For the inductive step, assume that the statement holds for t=k𝑡𝑘t=k. Then,

|  |  |  |  |
| --- | --- | --- | --- |
|  | w(k+1)superscript𝑤𝑘1\displaystyle w^{(k+1)} | =w(k)+η⋅∇Lref​(w(k))absentsuperscript𝑤𝑘⋅𝜂∇subscript𝐿refsuperscript𝑤𝑘\displaystyle=w^{(k)}+\eta\cdot\nabla L\_{\text{ref}}(w^{(k)}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =projWref⊥​winit+wproj(k)+η⋅∇Lref​(projWref⊥​winit+wproj(k))absentsubscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤initsuperscriptsubscript𝑤proj𝑘⋅𝜂∇subscript𝐿refsubscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤initsuperscriptsubscript𝑤proj𝑘\displaystyle=\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}+{w\_{\text{proj}}}^{(k)}+\eta\cdot\nabla L\_{\text{ref}}(\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}+{w\_{\text{proj}}}^{(k)}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =projWref⊥​winit+wproj(k)+η⋅∇Lref​(wproj(k))absentsubscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤initsuperscriptsubscript𝑤proj𝑘⋅𝜂∇subscript𝐿refsuperscriptsubscript𝑤proj𝑘\displaystyle=\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}+{w\_{\text{proj}}}^{(k)}+\eta\cdot\nabla L\_{\text{ref}}({w\_{\text{proj}}}^{(k)}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =projWref⊥​winit+wproj(k+1)absentsubscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤initsuperscriptsubscript𝑤proj𝑘1\displaystyle=\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}+{w\_{\text{proj}}}^{(k+1)} |  |

where in the third step we use the fact that ∇Lref​(u+v)=∇Lref​(u)∇subscript𝐿ref𝑢𝑣∇subscript𝐿ref𝑢\nabla L\_{\text{ref}}(u+v)=\nabla L\_{\text{ref}}(u) if v∈Wref⊥𝑣superscriptsubscript𝑊refbottomv\in W\_{\text{ref}}^{\bot}.
This completes the induction.
Because wproj(0)=projWref​winit∈Wrefsuperscriptsubscript𝑤proj0subscriptprojsubscript𝑊refsubscript𝑤initsubscript𝑊ref{w\_{\text{proj}}}^{(0)}={\text{proj}\_{W\_{\text{ref}}}}{w\_{\text{init}}}\in W\_{\text{ref}}, from Lemma [A.4](#A1.Thmtheorem4 "Lemma A.4. ‣ A.1.2 Convergence of gradient descent within the reference subspace ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You") (in particular, from ([7](#A1.E7 "In Proof. ‣ Lemma A.4. ‣ A.1.2 Convergence of gradient descent within the reference subspace ‣ A.1 Proof of Theorem 4.1 ‣ Appendix A Theoretical Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"))), we know that

|  |  |  |
| --- | --- | --- |
|  | ‖wproj(t)−wref∗‖22≤2⋅K⋅‖projWref​winit−wref∗‖m⋅(t−1).subscriptsuperscriptnormsuperscriptsubscript𝑤proj𝑡subscriptsuperscript𝑤ref22⋅2𝐾normsubscriptprojsubscript𝑊refsubscript𝑤initsubscriptsuperscript𝑤ref⋅𝑚𝑡1\|{w\_{\text{proj}}}^{(t)}-w^{\*}\_{\text{ref}}\|^{2}\_{2}\leq\frac{2\cdot K\cdot\|{\text{proj}\_{W\_{\text{ref}}}}{w\_{\text{init}}}-w^{\*}\_{\text{ref}}\|}{m\cdot(t-1)}. |  |

where K𝐾K and m𝑚m are positive constants. Finally, we conclude that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖w(t)−w^‖22subscriptsuperscriptnormsuperscript𝑤𝑡^𝑤22\displaystyle\|w^{(t)}-\hat{w}\|^{2}\_{2} | =‖(projWref⊥​winit+wproj(t))−(projWref⊥​winit+wref∗)‖22absentsubscriptsuperscriptnormsubscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤initsuperscriptsubscript𝑤proj𝑡subscriptprojsuperscriptsubscript𝑊refbottomsubscript𝑤initsubscriptsuperscript𝑤ref22\displaystyle=\|(\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}+{w\_{\text{proj}}}^{(t)})-(\text{proj}\_{W\_{\text{ref}}^{\bot}}w\_{\text{init}}+w^{\*}\_{\text{ref}})\|^{2}\_{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =‖wproj(t)−wref∗‖22absentsubscriptsuperscriptnormsuperscriptsubscript𝑤proj𝑡subscriptsuperscript𝑤ref22\displaystyle=\|{w\_{\text{proj}}}^{(t)}-w^{\*}\_{\text{ref}}\|^{2}\_{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤2⋅K⋅‖projWref​winit−wref∗‖m⋅(t−1)absent⋅2𝐾normsubscriptprojsubscript𝑊refsubscript𝑤initsubscriptsuperscript𝑤ref⋅𝑚𝑡1\displaystyle\leq\frac{2\cdot K\cdot\|{\text{proj}\_{W\_{\text{ref}}}}{w\_{\text{init}}}-w^{\*}\_{\text{ref}}\|}{m\cdot(t-1)} |  |

Hence, w(t)superscript𝑤𝑡w^{(t)} converges to w^^𝑤\hat{w}, completing our proof.

### Appendix B Experiment Details

#### B.1 General

##### B.1.1 Model training

All models are trained using the FFCV data-loading library [leclerc2022ffcv] on a cluster of A100 GPUs.

##### B.1.2 Measuring effective robustness

###### Effective robustness.

In this work, we quantify the robustness of pre-trained models using *effective robustness* (ER), a measure of the robustness a model above the “baseline” of models trained from scratch [taori2020when].
Computing this metric first involves establishing a relationship between the accuracies of baseline models (in our case, models trained from scratch on a reference dataset).
In particular, let Accref​(M)subscriptAccref𝑀\text{Acc}\_{\text{ref}}(M) and Accshift​(M)subscriptAccshift𝑀\text{Acc}\_{\text{shift}}(M) denote the accuracies of a model M𝑀M on test datasets drawn from the reference and shifted distributions, respectively.
Given a set ℳbaselinesubscriptℳbaseline\mathcal{M}\_{\text{baseline}} of baseline models, we compute a linear fit relating Φ−1​(Accref​(M))superscriptΦ1subscriptAccref𝑀\Phi^{-1}(\text{Acc}\_{\text{ref}}(M)) and Φ−1​(Accshift​(M))superscriptΦ1subscriptAccshift𝑀\Phi^{-1}(\text{Acc}\_{\text{shift}}(M)), where Φ−1superscriptΦ1\Phi^{-1} is the probit function (i.e., the inverse cumulative distribution function of the standard normal distribution).
We compute a linear fit relating probit-scaled accuracies (instead of the accuracies themselves) because this has been empirically observed to improve the strength of the linear relationship [miller2021accuracy, taori2020when].
Formally, we compute parameters a^^𝑎\hat{a} and b^^𝑏\hat{b} such that

|  |  |  |
| --- | --- | --- |
|  | a^,b^=arg⁡mina,b​∑M∈ℳbaseline‖(a⋅Φ−1​(Accref​(M))+b)−Φ−1​(Accshift​(M))‖22.  ^𝑎^𝑏 subscript  𝑎𝑏subscript𝑀subscriptℳbaselinesubscriptsuperscriptnorm⋅𝑎superscriptΦ1subscriptAccref𝑀𝑏superscriptΦ1subscriptAccshift𝑀22\hat{a},\hat{b}=\arg\min\_{a,b}\sum\_{M\in\mathcal{M}\_{\text{baseline}}}\|(a\cdot\Phi^{-1}(\text{Acc}\_{\text{ref}}(M))+b)-\Phi^{-1}(\text{Acc}\_{\text{shift}}(M))\|^{2}\_{2}. |  |

Let Acc^shift​(M)subscript^Accshift𝑀\widehat{\text{Acc}}\_{\text{shift}}(M) be the resulting function estimating shifted accuracy given reference accuracy, that is

|  |  |  |
| --- | --- | --- |
|  | Acc^shift​(M)=Φ​(a^⋅Φ−1​(Accref​(M))+b^).subscript^Accshift𝑀Φ⋅^𝑎superscriptΦ1subscriptAccref𝑀^𝑏\widehat{\text{Acc}}\_{\text{shift}}(M)=\Phi(\hat{a}\cdot\Phi^{-1}(\text{Acc}\_{\text{ref}}(M))+\hat{b}). |  |

Then the effective robustness of a model M𝑀M is

|  |  |  |
| --- | --- | --- |
|  | ER​(M)=Accshift​(M)−Acc^shift​(M)ER𝑀subscriptAccshift𝑀subscript^Accshift𝑀\text{ER}(M)=\text{Acc}\_{\text{shift}}(M)-\widehat{\text{Acc}}\_{\text{shift}}(M) |  |

Intuitively, effective robustness is the extent to which a model’s accuracy on the shifted distribution exceeds the accuracy of a baseline model with the same accuracy on the reference distribution (see Figure [B.1](#A2.F1 "Figure B.1 ‣ Effective robustness. ‣ B.1.2 Measuring effective robustness ‣ B.1 General ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).

Figure B.1: 
Visualization of effective robustness.
To compute effective robustness (ER), we first establishes a linear relationship between the (probit-scaled) accuracies of baseline models (blue) on the reference and shifted datasets.
The effective robustness (green) of a pre-trained model (orange) is the amount by which its actual accuracy on the shifted dataset exceeds the prediction of the linear trend.

###### Establishing a baseline for effective robustness.

To establish a baseline with respect to which we can measure effective robustness, we need a set of baseline models trained from scratch on the reference dataset.
The set of baseline models we consider varies by experiment.
In each of the experiments in which we measure effective robustness, we confirm that a strong linear relationship exists between the probit-scaled accuracies of our baseline models on the reference and shifted datasets (see, e.g., Figure [4](#S5.F4 "Figure 4 ‣ 5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).

#### B.2 The robustness benefits of pre-training vary

In Section [3](#S3 "3 The Robustness Benefits of Pre-Training Vary ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we illustrate that pre-trained models exhibit substantial effective robustness on the ImageNet Sketch distribution shift but very little effective robustness on the ImageNet-V2 distribution shift.
We consider 787878 models trained from scratch on ImageNet and 555555 pre-trained models fine-tuned on ImageNet, all taken from PyTorch Image Models [rw2019timm].
The pre-trained models represent a variety of model architectures (e.g., ResNet [he2015residual], ConvNeXt [liu2022convnet], ViT [dosovitskiy2020image]), pre-training datasets (e.g., IG-1B [mahajan2018exploring], LAION-2B [schuhmann2022laion], OpenAI’s WIT [radford2021learning]), and pre-training algorithms (e.g., supervised learning, CLIP [radford2021learning]).
The complete list of models used is available with our code at <https://github.com/MadryLab/pretraining-distribution-shift-robustness>.

#### B.3 Constructing synthetic in-support and out-of-support shifts

In Section [5.1](#S5.SS1 "5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we measure the effective robustness of various pre-trained and fine-tuned models on two in-support and two out-of-support shifts synthetically constructed by modifying ImageNet.

###### Specifications of synthetic shifts.

Here, we provide detailed descriptions of the four synthetic distribution shifts (see Figure [4](#S5.F4 "Figure 4 ‣ 5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for visualizations).

1. 1.

   Spurious tint shift (in-support): We tint images (i.e., replace each pixel with a mix of the original value, with weight 0.750.750.75 and a specific color, with weight 0.250.250.25) such that the tint is correlated with the label in the reference distribution but not in the shifted distribution (i.e., tint is a spurious feature). Specifically, in the reference distribution we apply tint with a class-specific color to pspurious=0.5subscript𝑝spurious0.5p\_{\text{spurious}}=0.5 of examples and a tint with a random color to the remaining 1−pspurious=0.51subscript𝑝spurious0.51-p\_{\text{spurious}}=0.5 of examples. Meanwhile, in the shifted distribution we apply a tint with a random color universally.
2. 2.

   Label shift (in-support): Label shift is a commonly studied type of distribution shift in which the relative frequencies of classes change, but p​(x|y)𝑝conditional𝑥𝑦p(x|y) is fixed.
   To construct a label shift, we sub-sample ImageNet such that in the reference distribution, a randomly selected 500500500 classes are less likely to appear than the remaining 500500500 classes.
   In particular, the selected classes appear with probability pminority=0.2subscript𝑝minority0.2p\_{\text{minority}}=0.2, while the remaining classes appear with probability 1−pminority=0.81subscript𝑝minority0.81-p\_{\text{minority}}=0.8.
   In the shifted distribution, these relative frequencies are reversed.
3. 3.

   Unseen tint shift (out-of-support): We randomly tint images in the shifted distribution (with the same protocol as in the spurious tint shift).
4. 4.

   Flip shift (out-of-support): We vertically flip images in the shifted distribution.

###### Shared model specifications.

When training, we use the FFCV implementation of *RandomResizedCropRGBImageDecoder*, resizing image crops to a resolution of 224×224224224224\times 224.
For data augmentation, we use the FFCV implementations of *RandomHorizontalFlip*.
When evaluating, we use the FFCV implementation of *CenterCropRGBImageDecoder* with a ratio of 224/256224256224/256, resizing image crops to a resolution of 224×224224224224\times 224.

###### Specifications of baseline models.

As a baseline, we train a ViT-B/32 model (the implementation of [open\_clip]) from scratch on ImageNet.
We run AdamW for 100100100 epochs, using a cosine learning rate schedule with a peak learning rate of 0.0030.0030.003 and 101010 warmup epochs, a batch size of 512512512, a weight decay of 0.10.10.1, label smoothing of 0.10.10.1 and gradient clipping at global norm 111.
To establish a baseline for effective robustness, we evaluate this model at epochs 505050 through 858585 (we stop at 858585 because the model’s accuracy at later epochs becomes highly correlated).
[miller2021accuracy] observe that evaluating a model trained from scratch at different epochs in this way often exhibit a strong linear relationship between their accuracies on the reference and shifted distributions (and the same relationship holds for models with different architectures, hyperparameters, etc.).

###### Specifications of pre-trained models and fine-tuning strategies.

We consider two different pre-trained models: a CLIP [radford2021learning] ViT-B/32 (the implementation of [open\_clip]) and AugReg [steiner2021train] (the implementation of [rw2019timm]).
For the AugReg model, we consider full fine-tuning (FT) and linear probing followed by full fine-tuning (LP-FT) [kumar2022fine].
We perform linear probing by running AdamW for 444 epochs, using a cosine learning rate schedule, a peak learning rate of 0.0010.0010.001, a batch size of 512512512, and without weight decay or gradient clipping.
For the CLIP model, we consider zero-shot initialization followed by full fine-tuning (ZS-FT) in addition to these two strategies.
We perform zero-shot initialization following [wortsman2021robust].

We fully fine-tune models by running AdamW for 888 epochs, using a cosine learning rate schedule with 111 warmup epoch.
We select the best peak learning rate (in terms of reference accuracy) among
3×10−4,1×10−4,3×10−5,1×10−5,3×10−6,1×10−6

3superscript1041superscript1043superscript1051superscript1053superscript1061superscript1063\times 10^{-4},1\times 10^{-4},3\times 10^{-5},1\times 10^{-5},3\times 10^{-6},1\times 10^{-6}.
We use a batch size of 512512512, a weight decay of 0.10.10.1, and gradient clipping at global norm 111.

#### B.4 Dividing natural shifts into in-support and out-of-support splits

##### B.4.1 Splitting a Shifted Dataset

To split a shifted dataset into an “in-support split” and an “out-of-support split”, we would ideally measure the reference distribution probability density prefsubscript𝑝refp\_{\text{ref}} of inputs in the shifted dataset and assign inputs with small prefsubscript𝑝refp\_{\text{ref}} to the out-of-support split.
Unfortunately, it is difficult to estimate prefsubscript𝑝refp\_{\text{ref}} directly when dealing with high-dimensional inputs (in this case, images).
Instead, we estimate the probability density *ratio* pref/pshiftsubscript𝑝refsubscript𝑝shiftp\_{\text{ref}}/p\_{\text{shift}}, that is, how much more likely an input is under the reference distribution than under the shifted distribution.
We then assign examples in the shifted dataset with pref/pshift<0.2subscript𝑝refsubscript𝑝shift0.2p\_{\text{ref}}/p\_{\text{shift}}<0.2 to the out-of-support split and examples with pref/pshift≥0.2subscript𝑝refsubscript𝑝shift0.2p\_{\text{ref}}/p\_{\text{shift}}\geq 0.2 to the in-support split.
We visualize examples in Figure [C.4](#A3.F4 "Figure C.4 ‣ C.2.2 Examples from in-support and out-of-support splits ‣ C.2 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You").

###### Estimating pref/pshiftsubscript𝑝refsubscript𝑝shiftp\_{\text{ref}}/p\_{\text{shift}}.

To estimate pref/pshiftsubscript𝑝refsubscript𝑝shiftp\_{\text{ref}}/p\_{\text{shift}}, we use a classifier trained to distinguish between examples from the reference and shifted datasets.
Specifically, let p𝑝p be a probability mass/density function over examples that can either be drawn from 𝒟refsubscript𝒟ref\mathcal{D}\_{\text{ref}} or 𝒟shiftsubscript𝒟shift\mathcal{D}\_{\text{shift}} (i.e., p𝑝p represents the distribution of a dataset created by joining a reference dataset and a shifted dataset).
Next, let yrefsubscript𝑦refy\_{\text{ref}} be the event that an example is drawn from 𝒟refsubscript𝒟ref\mathcal{D}\_{\text{ref}} and yshiftsubscript𝑦shifty\_{\text{shift}} be the event that an example is drawn from 𝒟shiftsubscript𝒟shift\mathcal{D}\_{\text{shift}}.
We can express the ratio pref/pshiftsubscript𝑝refsubscript𝑝shiftp\_{\text{ref}}/p\_{\text{shift}} as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | pref​(x)pshift​(x)subscript𝑝ref𝑥subscript𝑝shift𝑥\displaystyle\frac{p\_{\text{ref}}(x)}{p\_{\text{shift}}(x)} | =p​(x|yref)p​(x|yshift)absent𝑝conditional𝑥subscript𝑦ref𝑝conditional𝑥subscript𝑦shift\displaystyle=\frac{p(x|y\_{\text{ref}})}{p(x|y\_{\text{shift}})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =p​(yref|x)⋅p​(x)p​(yref)⋅p​(yshift)p​(yshift|x)⋅p​(x)absent⋅⋅𝑝conditionalsubscript𝑦ref𝑥𝑝𝑥𝑝subscript𝑦ref𝑝subscript𝑦shift⋅𝑝conditionalsubscript𝑦shift𝑥𝑝𝑥\displaystyle=\frac{p(y\_{\text{ref}}|x)\cdot p(x)}{p(y\_{\text{ref}})}\cdot\frac{p(y\_{\text{shift}})}{p(y\_{\text{shift}}|x)\cdot p(x)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =p​(yref|x)p​(yshift|x)⋅p​(yshift)p​(yref).absent⋅𝑝conditionalsubscript𝑦ref𝑥𝑝conditionalsubscript𝑦shift𝑥𝑝subscript𝑦shift𝑝subscript𝑦ref\displaystyle=\frac{p(y\_{\text{ref}}|x)}{p(y\_{\text{shift}}|x)}\cdot\frac{p(y\_{\text{shift}})}{p(y\_{\text{ref}})}. |  |

The terms p​(yref)𝑝subscript𝑦refp(y\_{\text{ref}}) and p​(yshift)𝑝subscript𝑦shiftp(y\_{\text{shift}}) are easy to estimate since they are simply the proportions of reference and shifted examples in p𝑝p.
Hence, to estimate pref/pshiftsubscript𝑝refsubscript𝑝shiftp\_{\text{ref}}/p\_{\text{shift}} we just need to estimate p​(yref|x)𝑝conditionalsubscript𝑦ref𝑥p(y\_{\text{ref}}|x) and p​(yshift|x)𝑝conditionalsubscript𝑦shift𝑥p(y\_{\text{shift}}|x).

To do so, we train a classifier to distinguish between reference and shifted examples on a dataset drawn from p𝑝p. We construct such a dataset by combining 100​K100𝐾100K samples from ImageNet with each of the shifted datasets (for ImageNet-R, which contains a subset of the classes of ImageNet, we restrict the 100​K100𝐾100K samples to these classes). Next, we fine-tune a CLIP ViT-L/14 pre-trained on LAION-2B from OpenCLIP [open\_clip] to distinguish between reference and shifted examples. We first fine-tune just the final layer with a learning rate of 0.10.10.1 and then fine-tune the entire model with the best learning rate selected from 2×10−4,1×10−4,5×10−5,2×10−5,1×10−5,5×10−6,2×10−6

2superscript1041superscript1045superscript1052superscript1051superscript1055superscript1062superscript1062\times 10^{-4},1\times 10^{-4},5\times 10^{-5},2\times 10^{-5},1\times 10^{-5},5\times 10^{-6},2\times 10^{-6} and 1×10−61superscript1061\times 10^{-6}. After training the classifier, we calibrate it through temperature scaling [guo2017calibration]. We then estimate p​(yref|x)𝑝conditionalsubscript𝑦ref𝑥p(y\_{\text{ref}}|x) and p​(yshift|x)𝑝conditionalsubscript𝑦shift𝑥p(y\_{\text{shift}}|x) by applying a sigmoid to its output, from which we can estimate pref/pshiftsubscript𝑝refsubscript𝑝shiftp\_{\text{ref}}/p\_{\text{shift}}. To estimate this ratio for the entire shifted dataset, we split the dataset into 101010 folds and train a classifier to estimate pref/pshiftsubscript𝑝refsubscript𝑝shiftp\_{\text{ref}}/p\_{\text{shift}} on each fold using the remaining 999 folds.

###### Calibrating the classifiers used for splitting

As discussed in Section [B.4](#A2.SS4 "B.4 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), our method for dividing a shifted dataset into an in-support split and an out-of-support split requires a *calibrated* classifier to distinguish between examples from the reference and shifted datasets. Recall that to distinguish between examples from the reference and shifted datasets, we fine-tune a CLIP [radford2021learning] ViT-L/14 [dosovitskiy2020image] pre-trained on LAION-2B from OpenCLIP [open\_clip]. Such over-parameterized models can be overconfident in their predictions (and thus uncalibrated), so we calibrate the classifier by rescaling its (logit) output, a method known as temperature scaling [guo2017calibration].

In particular, let f𝑓f be a (potentially uncalibrated) classifier trained to distinguish between examples from the reference and shifted datasets (where the output of f𝑓f is a logit). We find the scaling parameter α𝛼\alpha that minimizes the standard logistic loss of f𝑓f on a calibration set Scalsubscript𝑆calS\_{\text{cal}}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | α=arg⁡minα′​∑(x,y)∈Scallog⁡(1+e−α′⋅f​(x)⋅y).𝛼subscriptsuperscript𝛼′subscript𝑥𝑦subscript𝑆cal1superscript𝑒⋅⋅superscript𝛼′𝑓𝑥𝑦\alpha=\arg\min\_{\alpha^{\prime}}\sum\_{(x,y)\in S\_{\text{cal}}}\log(1+e^{-\alpha^{\prime}\cdot f(x)\cdot y}). |  | (8) |

We then define a rescaled classifier fcal​(x)=α⋅f​(x)subscript𝑓cal𝑥⋅𝛼𝑓𝑥f\_{\text{cal}}(x)=\alpha\cdot f(x) (which is used to estimate the ratio pref/pshiftsubscript𝑝refsubscript𝑝shiftp\_{\text{ref}}/p\_{\text{shift}}). We produce calibration curves of the rescaled classifiers for each of the shifted datasets we split (see Figure [B.2](#A2.F2 "Figure B.2 ‣ Calibrating the classifiers used for splitting ‣ B.4.1 Splitting a Shifted Dataset ‣ B.4 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You")) and observe that they are indeed well-calibrated.

Figure B.2: Calibration curves of classifiers used for splitting. We display calibration curves for the classifiers used to divide ImageNet-V2, ImageNet-Sketch and ImageNet-R into in-support and out-of-support splits. Specifically, we sort the outputs of each classifier on a combined dataset of reference and shifted examples into 100100100 bins (where bin edges are quantiles). For each bin, we compute the actual positive rate (i.e., the proportion of examples from the shifted dataset) and the average predicted probability of an example being from the shifted dataset. When we plot the actual positive rates against average predicted probabilities, they are close to equal (close to y=x𝑦𝑥y=x), suggesting that the classifiers are well-calibrated. Error bars denote 95% Clopper-Pearson confidence intervals.

##### B.4.2 Specifications of ImageNet models

To measure the robustness benefits of pre-training on in-support and out-of-support splits of ImageNet distribution shifts, we use the same suite of ImageNet models from PyTorch Image Models [rw2019timm] as detailed in Appendix [B.2](#A2.SS2 "B.2 The robustness benefits of pre-training vary ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You").

#### B.5 Combining pre-training with interventions for handling bias

###### Shared model specifications.

When training on the WILDS-FMoW dataset, we use the FFCV implementation of *RandomHorizontalFlip*.

###### Specifications of models trained from scratch.

We train models from scratch by running SGD for 646464 epochs, using a triangular learning rate schedule with a peak learning rate of 0.20.20.2 and 888 warmup epochs, a batch size of 128128128, a weight decay of 5×10−45superscript1045\times 10^{-4} and a momentum of 0.90.90.9.

###### Baseline specifications.

To establish a baseline, we train 100100100 ResNet-50 models from scratch on random subsets ranging from 25% of the reference dataset to the entire dataset.
We increase the number of epochs and warmup epochs inversely with the size of the subset.
[miller2021accuracy] observe that models trained from scratch in this way often exhibit a strong linear relationship between their accuracies on the reference and shifted distributions (and the same relationship holds for models with different architectures, hyperparameters, etc.).

###### Specifications of pre-trained models.

The pre-trained model in this experiment is a CLIP ResNet-50 model (the implementation of [open\_clip]), adapted using linear probing followed by full fine-tuning.
Note that the CLIP ResNet-50 architecture [radford2021learning] deviates from the standard ResNet-50 architecture of [he2015residual].
We perform linear probing by running AdamW for 888 epochs, using a cosine learning rate schedule, a peak learning rate of 0.0010.0010.001, a batch size of 512512512, and without weight decay or gradient clipping.
We fine-tune models by running AdamW for 161616 epochs, using a cosine learning rate schedule with a peak learning rate of 1×10−41superscript1041\times 10^{-4} and 222 warmup epochs, a batch size of 512512512, a weight decay of 0.10.10.1, and gradient clipping at global norm 111.

###### Our implementation of *Deep Feature Reweighting*.

The *Deep Feature Reweighting* (DFR) intervention proposed by [kirichenko2022last] aims to improve the robustness of a model on difficult subpopulations by using a validation dataset with group labels.
The algorithm consists of two steps: (1) train a standard model on the original training dataset, and (2) re-train only the final layer of the model (i.e., “re-weight” the features of the model) on the validation dataset to be more favorable to minority groups.
To re-train the final layer, [kirichenko2022last] repeatedly sample group-balanced subsets of the validation dataset, re-train the final layer on each subset, and then average the resulting re-trained final layers.
Our implementation differs slightly in that we assign sample weights to the validation dataset such that each group has equal total weight and re-train the final layer on the weighted validation dataset.
When applying *Deep Feature Reweighting* to WILDS-FMoW, we use the out-of-distribution validation set following [kirichenko2022last].

#### B.6 Curating datasets for fine-tuning

###### Image editing to synthesize “counterfactual examples”

In order to curate a “de-biased” dataset for hair color classification, we edit images from CelebA-HQ [karras2018progressive], a subset of the CelebA dataset with segmentation masks for each attribute provided by CelebAMask-HQ [lee2020maskgan].
To change the hair color in a given image, we use InstructPix2Pix [brooks2023instructpix2pix], a recent image editing model fine-tuned from Stable Diffusion [rombach2022high].
This model accepts an input image to be edited along with a prompt describing the desired change (e.g., “change the hair color to blond”).
We find that InstructPix2Pix is able to successfully edit the hair color; however, this model often makes undesired changes to attributes such as skin tone and eye color (see, e.g., the left side of Figure [B.3](#A2.F3 "Figure B.3 ‣ Image editing to synthesize “counterfactual examples” ‣ B.6 Curating datasets for fine-tuning ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
To ensure that we only edit hair color, we use the attribute masks to isolate the pixels in a given image corresponding to the hair region, and ignore any changes made outside of this area. When using a binary mask, this procedure could cause unnatural “edges” along the border of the mask.
Thus, we apply a Gaussian blur to the hair mask to smooth the transition when “merging” the original and edited images.

To edit an image from non-blond to blond, we use the prompt “change the hair color to blond.” When editing from blond to non-blond, however, we find that the prompt “change the hair color to non-blond” gives inconsistent results, likely because the instruction is vague. We observe that most non-blond people in the CelebA dataset have brown or black hair, so as a simple heuristic we randomly edit each image with either the prompt “change the hair color to brown” or the prompt “change the hair color to black.” See Figure [B.3](#A2.F3 "Figure B.3 ‣ Image editing to synthesize “counterfactual examples” ‣ B.6 Curating datasets for fine-tuning ‣ Appendix B Experiment Details ‣ Ask Your Distribution Shift if Pre-Training is Right for You") for a visualization of the image editing process.

Figure B.3: 
Synthesizing counterfactual examples. We edit hair color in CelebA-HQ images using InstructPix2Pix [brooks2023instructpix2pix]. However, this model can also make unwanted changes to attribute other than hair color, e.g., changing eye color (left). To avoid such issues, in the final image we incorporate only changes within the hair region of the image.

###### Shared model specifications.

Accuracy and worst-group accuracy on the CelebA dataset are sensitive to hyperparameter choices.
As a result, we conduct a grid search to select hyperparameters for each type of model.
We use class-balanced accuracy as the metric for hyperparameter selection, which empirically better correlates with worst-group accuracy than standard accuracy.

When selecting hyperparameters for a curated dataset of a given size, we randomly sample 323232 datasets of that size from a pool of 16,000

1600016,000 images (i.e., 8,000

80008,000 CelebA images and their corresponding counterfactual synthesized images) and average the class-balanced accuracies of models trained on each dataset.
When evaluating the accuracy and worst-group accuracy of models trained on a curated dataset of a given size, we similarly randomly sample 646464 datasets of that size and report average metrics.

For all models, we use the FFCV implementation of *RandomHorizontalFlip* for data augmentation.

###### Specifications of models trained from scratch.

We train ResNet-18 models from scratch by running SGD for 323232 epochs, using a triangular learning rate schedule with 444 warmup epochs.
We use a batch size of 128128128, a weight decay of 5×10−45superscript1045\times 10^{-4} and a momentum of 0.90.90.9.
We select the best combination of batch size and learning rate from batch sizes of 64,128,256,512

6412825651264,128,256,512 and learning rates of 0.5,0.2,0.1,0.05,0.02,0.01

0.50.20.10.050.020.010.5,0.2,0.1,0.05,0.02,0.01.

When training models from scratch on our curated dataset, we run SGD for 512512512 epochs and use a triangular learning rate schedule with 646464 warmup epochs.
We use a batch size equal to the total number of examples when it is less than 512512512 and a batch size of 512512512 otherwise.
We use a weight decay of 5×10−45superscript1045\times 10^{-4} and a momentum of 0.90.90.9.
We select the best learning rate from 0.5,0.2,0.1,0.05,0.02,0.01

0.50.20.10.050.020.010.5,0.2,0.1,0.05,0.02,0.01.

###### Baseline specifications.

To establish a baseline, we train 100100100 ResNet-50 models from scratch on random subsets ranging from 5% of the reference dataset to the entire dataset.
We increase the number of epochs and warmup epochs inversely with the size of the subset.
[miller2021accuracy] observe that models trained from scratch in this way often exhibit a strong linear relationship between their accuracies on the reference and shifted distributions (and the same relationship holds for models with different architectures, hyperparameters, etc.).

###### Specifications of pre-trained models.

The pre-trained model in this experiment is a CLIP ViT-B/32 model initialized as a zero-shot classifier with “blond” and “non-blond” as the class names.
We fine-tune models by running AdamW for 161616 epochs, using a cosine learning rate schedule with 222 warmup epochs, and a weight decay of 0.10.10.1.
We select the best combination of batch size and learning rate from batch sizes of 64,128,256,512

6412825651264,128,256,512 and learning rates of 3×10−5,1×10−5,3×10−6,1×10−6

3superscript1051superscript1053superscript1061superscript1063\times 10^{-5},1\times 10^{-5},3\times 10^{-6},1\times 10^{-6}.

When training on our curated dataset, we use a batch size of 646464 (the size of the dataset) and select the best learning rate from 3×10−5,1×10−5,3×10−6,1×10−6

3superscript1051superscript1053superscript1061superscript1063\times 10^{-5},1\times 10^{-5},3\times 10^{-6},1\times 10^{-6}.

### Appendix C Additional Results

#### C.1 Constructing synthetic in-support and out-of-support shifts

##### C.1.1 How does the choice of fine-tuning hyperparameters affect robustness?

In Section [5.1](#S5.SS1 "5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we select hyperparameters (in particular, learning rate) for fine-tuning that maximize accuracy on the reference distribution.
This reasonably simulates hyperparameter selection in practice because typically only samples from the reference distribution are available.

In this section, we investigate how the choice of hyperparameters affects the robustness of pre-trained models.
In particular, we would like to understand if pre-training yields little effective robustness to in-support shifts and substantial effective robustness to out-of-support shifts across a wider range of hyperparameter choices.
We study the spurious tint shift (an in-support shift) and the flip shift (an out-of-support shift) from Section [5.1](#S5.SS1 "5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You") and vary the learning rate, weight decay, number of epochs, and batch size of a CLIP ViT-B/32 initialized with zero-shot weights (Figure [C.1](#A3.F1 "Figure C.1 ‣ C.1.1 How does the choice of fine-tuning hyperparameters affect robustness? ‣ C.1 Constructing synthetic in-support and out-of-support shifts ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
With zero-shot initialization, the starting point of fine-tuning is a robust model that performs well on our task.
Hence, even under an in-support shift, hyperparameter choices that do not change the model substantially (e.g., low learning rate, small number of epochs) result in substantial effective robustness.
However, these hyperparameter choices generally result in lower absolute reference and shifted accuracies, and are thus unreasonable.
The hyperparameter choices that are relevant in practice are those with high reference accuracy, and these are the hyperparameters that we use in our experiments.

(a) In-support shift. The in-support shift we consider is the “spurious tint shift” in which we introduce a tint that is spuriously correlated with the label. On this in-support shift, learning rate and number of epochs influence effective robustness, but the best hyperparameter choices result in a model with little effective robustness.

(b) Out-of-support shift. The out-of-support shift we consider is the “flip shift” in which we pad images in the shifted distribution. On this out-of-support shift, batch size most significantly affects robustness, while learning rate and number of epochs affect overall performance.

Figure C.1: The effects of hyperparameter choices on robustness. We vary hyperparameters when fine-tuning a CLIP ViT-B/32 initialized with zero-shot weights on synthetic ImageNet shifts from Section [5.1](#S5.SS1 "5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You") (different shades of green). Varying certain hyperparameters (e.g., learning rate, number of epochs) can affect the effective robustness of pre-trained models even on an in-support shift. In our experiments, we choose hyperparameters which yield high reference accuracy (purple).

##### C.1.2 How does the strength of the bias affect robustness to in-support shifts?

In Section [5.1](#S5.SS1 "5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we consider two in-support shifts under which models might fail due to dataset biases.
In particular, in the spurious tint shift, we introduce a tint that is spuriously correlated with the label in the reference dataset, but not in the shifted dataset.
The probability that an example in the reference dataset has a class-specific tint (as opposed to a random tint) is determined by a parameter pspurioussubscript𝑝spuriousp\_{\text{spurious}} (set to 0.50.50.5 for the experiments in Figure [4](#S5.F4 "Figure 4 ‣ 5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
In the label shift, the relative frequencies of classes change between the reference and shifted datasets.
The classes are divided into “majority” and “minority” classes, with “minority” classes appearing with probability pminoritysubscript𝑝minorityp\_{\text{minority}} in the reference dataset (set to 0.20.20.2 for the experiments in Figure [4](#S5.F4 "Figure 4 ‣ 5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
In the shifted distribution, the relative frequencies of the classes are reversed.

In this section, we investigate how the strength of the bias, i.e., pspurioussubscript𝑝spuriousp\_{\text{spurious}} and pminoritysubscript𝑝minorityp\_{\text{minority}}, affects the robustness of pre-trained models to these in-support shifts.
We observe the average effective robustness of pre-trained models largely remains close to zero as we vary these parameters (see Figure [C.2](#A3.F2 "Figure C.2 ‣ C.1.2 How does the strength of the bias affect robustness to in-support shifts? ‣ C.1 Constructing synthetic in-support and out-of-support shifts ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).

(a) 
Spurious tint shift.
Across several different probabilities of the spurious class-specific tint (pspurioussubscript𝑝spuriousp\_{\text{spurious}}), the average effective robustness of pre-trained models (top left of each plot) on the in-support “spurious tint shift” is close to zero.
The one exception is the shift with pspurious=0.9subscript𝑝spurious0.9p\_{\text{spurious}}=0.9, where the effective robustness is higher.
This may be because this shift is “close” to an out-of-support shift, since the probability of observing an example with a random tint (as opposed to a class-specific tint) is low.
Hence, pre-training might help by extrapolating better from the small number of randomly tinted examples.

(b) 
Label shift.
Across several different probabilities of the minority classes (pminoritysubscript𝑝minorityp\_{\text{minority}}), the average effective robustness of pre-trained models (top left of each plot) on the in-support “label shift” is close to zero.
We note that in the shifts with pminority=0.1subscript𝑝minority0.1p\_{\text{minority}}=0.1 and pminority=0.15subscript𝑝minority0.15p\_{\text{minority}}=0.15, the effective robustness is slightly negative.
However, the linear correlation among baseline models is weak under these shifts, so these effective robustnesses are less meaningful.

Figure C.2: 
The effects of the strength of the bias on robustness to in-support shifts.
We vary the strength of the bias of the two synthetic ImageNet in-support shifts from Section [5.1](#S5.SS1 "5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You").
Broadly, the effective robustness of pre-trained models (top left of each plot) is close to zero across bias strengths.

#### C.2 Dividing natural shifts into in-support and out-of-support splits

##### C.2.1 Sizes of in-support and out-of-support splits

In Table [C.3](#A3.T3 "Table C.3 ‣ C.2.1 Sizes of in-support and out-of-support splits ‣ C.2 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we report the sizes of the in-support and out-of-support splits we compute for ImageNet-V2, ImageNet Sketch and ImageNet-R. The out-of-support splits are much larger than the in-support splits, perhaps because the large majority of the examples from these shifted datasets look unlike examples from ImageNet.

Table C.3: Sizes of in-support and out-of-support splits.

| Dataset | In-support split size | Out-of-support split size |
| --- | --- | --- |
| ImageNet-V2 | 1920 | 8080 |
| ImageNet Sketch | 162 | 50727 |
| ImageNet-R | 588 | 29412 |

##### C.2.2 Examples from in-support and out-of-support splits

In Figure [C.4](#A3.F4 "Figure C.4 ‣ C.2.2 Examples from in-support and out-of-support splits ‣ C.2 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we provide samples from the in-support and out-of-support splits we compute for ImageNet-V2, ImageNet-Sketch and ImageNet-R.

Figure C.4: Random samples from ImageNet and from the in-support and out-of-support splits of ImageNet-V2, ImageNet Sketch and ImageNet-R. In ImageNet-V2, it is difficult to distinguish between examples from the in-support and out-of-support splits. In ImageNet Sketch and ImageNet-R, examples from the in-support splits look more realistic (i.e., more like ImageNet examples) than examples from the out-of-support splits.

##### C.2.3 Scatter plots of reference vs. shifted accuracy

In Figure [C.5](#A3.F5 "Figure C.5 ‣ C.2.3 Scatter plots of reference vs. shifted accuracy ‣ C.2 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we provide scatter plots of accuracy on ImageNet vs. accuracy on the in-support and out-of-support splits of ImageNet-V2, ImageNet Sketch and ImageNet-R.

Figure C.5: 
Reference vs. shifted accuracy for in-support and out-of-support splits of ImageNet shifts.
On each of the three ImageNet shifts we consider, the average effective robustness (ER) of pre-trained models (orange) above the baseline of models trained from scratch (blue) on the in-support split (top) is small.
Meanwhile, their effective robustness can be very large on the out-of-support split (bottom).

##### C.2.4 Controlling for difficulty when measuring effective robustness

The significance of a given effective robustness depends on the “difficulty” of a distribution shift. For example, if a shift causes an accuracy drop of 5%percent55\%, an effective robustness of 4%percent44\% might be considered large, but if a shift that causes a drop of 25%percent2525\%, an effective robustness of 4%percent44\% would probably be considered small.
When we divide a shifted dataset into an in-support and out-of-support split, the out-of-support split is typically more difficult than the in-support split.
If we compare the effective robustness of pre-trained models on examples of similar difficulty in the in-support and out-of-support splits, do our findings from Section [5.2](#S5.SS2 "5.2 Dividing natural shifts into in-support and out-of-support splits ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You") still hold?
In particular, do pre-trained models still exhibit substantially higher robustness on out-of-support examples than on in-support examples?

To answer this question, we re-weight examples in out-of-support splits such that the difficulty distribution of the out-of-support split matches that of the in-support split.
Specifically, we quantify the difficulty of a given example in terms of the fraction of baseline models (of 777777 total baseline models) that classify it incorrectly.
Given an example of difficulty d𝑑d, we re-weight it by a factor of pin-support​(d)/pout-of-support​(d)subscript𝑝in-support𝑑subscript𝑝out-of-support𝑑p\_{\text{in-support}}(d)/p\_{\text{out-of-support}}(d) where pin-supportsubscript𝑝in-supportp\_{\text{in-support}} is the difficulty probability density function of the in-support split and pout-of-supportsubscript𝑝out-of-supportp\_{\text{out-of-support}} is the difficulty probability density function of the out-of-support split.
We then compute a “re-weighted” accuracy, which in turn yields a re-weighted effective robustness, on the out-of-support split.
Intuitively, this re-weighted effective robustness represents the effective robustness of pre-trained models on out-of-support examples of similar difficulty to in-support examples.

We report the re-weighted effective robustnesses in Figure [C.6](#A3.F6 "Figure C.6 ‣ C.2.4 Controlling for difficulty when measuring effective robustness ‣ C.2 Dividing natural shifts into in-support and out-of-support splits ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"). We observe that the re-weighted effective robustnesses of pre-trained models on out-of-support splits are indeed lower than the original effective robustnesses.
However, they are still substantially higher than the effective robustnesses on in-support splits.

Figure C.6: 
Re-weighed effective robustness of pre-trained models on in-support and out-of-support splits of ImageNet shifts.
When we re-weight examples in out-of-support splits to match the difficulty distributions of their corresponding in-support splits, the average effective robustnesses of pre-trained models (green) decrease relative to the original effective robustnesses (blue).
However, they are still very high on ImageNet Sketch and ImageNet-R.
Meanwhile, the average effective robustnesses of pre-trained models on in-support splits (orange) are consistently low.

#### C.3 Combining pre-training with interventions for handling bias

##### C.3.1 Studying a synthetic shift

In this section, we provide an additional experiment in a synthetic setting to further illustrate that pre-training and interventions designed to handle dataset biases can be complementary.
In Section [6](#S6 "6 Combining Pre-Training with Interventions for Handling Bias ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we discussed how robustness to the WILDS-FMoW distribution shift requires both extrapolating to later years and performing consistently across regions.
We construct a synthetic distribution shift using that similarly requires both extrapolating well and avoiding reliance on spurious features.
Specifically, we combine the tint and pad shifts from Section [5.1](#S5.SS1 "5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You").
We modify CIFAR-10 such that in the reference distribution, we add a tint that is spuriously correlated with the label: 80%percent8080\% of reference examples have a class-specific tint while the remaining 20%percent2020\% are randomly tinted.
Meanwhile, in the shifted distribution, examples are always randomly tinted and are also padded (we add 666 black pixels to each side of the original 32×32323232\times 32 CIFAR-10 images).

To extrapolate to padded examples, we initialize a CLIP ResNet-50 and perform linear probing followed by full fine-tuning on the reference distribution.
To handle the spurious correlation between tint and label, we consider the intervention of training on randomly tinted examples, which we refer to as *balancing*.
This is an “oracle” of sorts for handling dataset biases; it simply modifies the training distribution such that spurious features are not useful.

As with WILDS-FMoW, we find that pre-training and balancing each yield some effective robustness (see the left side of Figure [C.7](#A3.F7 "Figure C.7 ‣ C.3.1 Studying a synthetic shift ‣ C.3 Combining pre-training with interventions for handling bias ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
In this case, combining the two does not yield the greatest effective robustness, but does have the highest shifted accuracy.
We apply the same methodology as in Section [6](#S6 "6 Combining Pre-Training with Interventions for Handling Bias ‣ Ask Your Distribution Shift if Pre-Training is Right for You") to understand the robustness benefits of pre-training and balancing.
Here, we observe a greater overlap between the corrected examples of pre-training and balancing than we did for pre-training and DFR in the case of WILDS-FMoW (see the right side of Figure [C.7](#A3.F7 "Figure C.7 ‣ C.3.1 Studying a synthetic shift ‣ C.3 Combining pre-training with interventions for handling bias ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
This may be due to the fact that every example requires both extrapolation and avoiding reliance on the spurious bias.
In other words, the failure modes that pre-training and balancing are intended to address cooccur.
However, we note that there are still many examples that are corrected by one of pre-training and balancing, but not the other, suggesting complementary benefits.
Similarly to our observations with WILDS-FMoW, combining pre-training with balancing corrects most of the examples corrected by the individual interventions.
These results corroborate our finding that pre-training and interventions designed to handle dataset biases can be complementary.

Figure C.7: 
Combining pre-training and balancing on a synthetic CIFAR-10 distribution shift.
Pre-training and balancing (an “oracle” intervention for handling dataset biases) each yield some effective robustness (ER) and combining these two interventions yields a high effective robustness and the highest shifted accuracy (left).
A substantial number of examples are corrected by one of pre-training and balancing, but not the other (right), indicating that there are *different* subpopulations where they improve performance.
Meanwhile, the examples corrected by combining pre-training with balancing include most of the examples corrected by the individual interventions (right), suggesting that combining pre-training with balancing improves performance on *both* of these subpopulations.
Error bars denote 95% confidence intervals over 64 random trials.

#### C.4 Curating datasets for fine-tuning

##### C.4.1 Understanding the robustness benefits of pre-training when fine-tuning on a curated dataset

In Section [7](#S7 "7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we find that fine-tuning on a curated dataset with only 646464 examples can yield a performant and robust model for hair color classification.
We observe that pre-training is necessary for effective use of the small curated dataset;
in particular, training a model from scratch on a curated dataset yields robustness gains, but these gains are smaller and many more examples are required to attain comparable accuracy.

In this section, we shed additional light on how pre-training helps in this setting.
Based on our intuition from Sections [4](#S4 "4 Studying Pre-Training in a Logistic Regression Setting ‣ Ask Your Distribution Shift if Pre-Training is Right for You") and [5](#S5 "5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You") that pre-training helps specifically with extrapolation, we hypothesize that pre-training provides two benefits when training on a small curated dataset.
First, a pre-trained model may be able to extrapolate better from a small number of examples.
This would result in both higher accuracy on the original CelebA distribution and higher worst-group accuracy, which we observe in Figure [7(b)](#S7.F7.sf2 "In Figure 7 ‣ 7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You").
Second, recall that our curated dataset consists entirely of females, but hair color classification models are expected to perform well on males too.
To compare different model’s ability to extrapolate along this axis, we plot the balanced accuracy on males against the balanced accuracy on females.
In Figure [7(a)](#S7.F7.sf1 "In Figure 7 ‣ 7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we observe that the pre-trained model indeed generalizes better to males than models trained from scratch.

Figure C.8: 
Comparing extrapolation from females to males of pre-trained models and models trained from scratch.
We plot the balanced accuracy on males against the balanced accuracy of females of a pre-trained model fine-tuned on the curated dataset from Section [7](#S7 "7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You") (red) and models trained from scratch on this dataset (green).
Models trained from scratch establish a linear relationship between male and female balanced accuracy; however, the pre-trained model outperforms this trend, suggesting that it more effectively extrapolates to males from the female-only curated dataset.

##### C.4.2 Exploring balancing instead of counterfactual editing

In Section [7](#S7 "7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You"), we choose to curate a dataset by augmenting images from CelebA with “counterfactual examples” in which we edit the hair color to the opposite class.
We do so in order to *de-bias* this dataset as much as possible.
In this section, we explore a simpler approach to curating a dataset: balancing classes.
Similarly to our curated dataset, we constrain this balanced dataset to include only females.
As with our curated dataset, we observe that fine-tuning a pre-trained model on a class-balanced female-only dataset yields a robust and performant model for hair color classification (see Figure [9(a)](#A3.F9.sf1 "In Figure C.9 ‣ C.4.2 Exploring balancing instead of counterfactual editing ‣ C.4 Curating datasets for fine-tuning ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).
We also observe again that pre-training improves over training from scratch by helping with extrapolation from the female-only reference dataset to males (see Figure [9(b)](#A3.F9.sf2 "In Figure C.9 ‣ C.4.2 Exploring balancing instead of counterfactual editing ‣ C.4 Curating datasets for fine-tuning ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).

(a) 
Fine-tuning on a balanced female-only dataset.
Fine-tuning a pre-trained model on the CelebA dataset (orange) yields little effective robustness over a baseline of models trained from scratch (blue).
However, fine-tuning the same pre-trained model on just 646464 examples from a balanced female-only dataset (red) yields a model with both high effective robustness and high accuracy.
Training from scratch on a balanced female-only dataset (green) also yields high effective robustness,
but results in substantially lower accuracy than pre-trained models, even with many more examples.
Error bars denote 95% confidence intervals over 646464 random trials.

(b) 
Comparing extrapolation from females to males of pre-trained models and models trained from scratch.
We plot the balanced accuracy on males against the balanced accuracy of females of a pre-trained model fine-tuned on a balanced female-only dataset (red) and models trained from scratch on this dataset (green).
Models trained from scratch establish a linear relationship between male and female balanced accuracy; however, the pre-trained model outperforms this trend, suggesting that it more effectively extrapolates to males from the female-only reference dataset.

Figure C.9: Fine-tuning a pre-trained model on a small, non-diverse but de-biased dataset (in this case, a class-balanced female-only dataset) yields a robust and performant model for hair color classification in CelebA (see Figure [7(b)](#S7.F7.sf2 "In Figure 7 ‣ 7 Curating Datasets for Fine-Tuning ‣ Ask Your Distribution Shift if Pre-Training is Right for You")).

### Appendix D Additional Discussion

#### D.1 Alternative fine-tuning strategies

In this work, we focus on the common setting in which a pre-trained model is fully fine-tuned.
It is important to note that pre-trained models used in a zero-shot context (i.e., without fine-tuning) and partially fine-tuned models (e.g., only the final classification layer is updated) are frequently more robust than fully fine-tuned models [radford2021learning, miller2021accuracy, kumar2022fine].
Such models may have higher effective robustness than fully fine-tuned models or in some cases may even outperform fully fine-tuned models on the shifted distribution.
However, such models are typically less performant on the reference distribution than fully fine-tuned models.

Several works observe this tradeoff between performance on the reference distribution and robustness and devise methods for mitigating it, i.e., methods for *robust fine-tuning* [wortsman2021robust, hewitt2021ensembles, kumar2022fine].
For example, [kumar2022fine] argue that full fine-tuning “distorts” pre-trained features and propose linear probing *before* full fine-tuning (LP-FT) to prevent distortion.
They also suggest that fine-tuning a model initialized as a zero-shot classifier may have a similar effect.
In addition to full fine-tuning, in Section [5.1](#S5.SS1 "5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You") we thus consider LP-FT and zero-shot initialization for fine-tuning.
On in-support shifts, we observe that LP-FT and zero-shot initialization do not provide effective robustness benefits compared to full fine-tuning (see Figure [4](#S5.F4 "Figure 4 ‣ 5.1 Constructing synthetic in-support and out-of-support shifts ‣ 5 Exploring the Empirical Robustness Benefits of Pre-Training ‣ Ask Your Distribution Shift if Pre-Training is Right for You")), suggesting that these strategies do not help mitigate dataset biases.

Another strategy for robust fine-tuning is to ensemble a zero-shot model and a fully fine-tuned model.
Both weight-space ensembles [wortsman2021robust] and output-space ensembles [hewitt2021ensembles] have been shown to improve robustness, sometimes even without sacrificing performance on the reference distribution.
In fact, this strategy can yield robustness benefits even when dataset biases are a primary failure mode because the zero-shot model is independent of the biased reference dataset.
Our work seeks to complement such empirically effective strategies by providing an understanding of when they are necessary.
In particular, our findings suggest that ensembling is valuable precisely when dataset biases cause failures.

#### D.2 Can pre-training hurt extrapolation?

In this work, we discuss distribution shifts in which pre-training is beneficial to a model’s ability to extrapolation outside of the reference distribution.
A natural question to consider is whether pre-training can instead *hurt* it, yielding worse extrapolation than a model trained from scratch.
A recent work by [salman2022does] suggests that this is indeed possible.
Specifically, they show that biases of pre-trained models can persist during fine-tuning.
For example, a model pre-trained on ImageNet and fine-tuned on CIFAR-10 is highly sensitive to the presence of tennis balls (which are an ImageNet class but not a CIFAR-10 class).
Meanwhile, a model trained from scratch on CIFAR-10 is not particularly sensitive to tennis balls.
Thus, under a hypothetical “tennis ball shift” in which tennis balls appear in images in the shifted distribution, a pre-trained model would be less robust than a model trained from scratch.
In this instance, pre-training provides a *harmful* prior for how to extrapolate.

#### D.3 When does pre-training help with extrapolation?

In this work, we provide evidence that pre-training *can* help with extrapolation, but not with other failure modes.
A natural question to consider is whether a particular pre-trained model and fine-tuning strategy in fact *does* help with a given extrapolation task.
To this end, [ramanujan2023connection] explore how the composition of the pre-training dataset affects robustness on the WILDS-iWildCam distribution shift [koh2020wilds].
We consider further exploration of this question to be a valuable direction for future work.

#### D.4 Relating in-support and out-of-support shifts to existing characterizations

The characterizations relevant in this work, *in-support shift* and *out-of-support shift*, overlap with many existing definitions.
[ye2022ood] introduce notions of *correlation shift* and *diversity shift* (closely aligned with in-support and out-of-support shifts, respectively) and provide a method for measuring the “amount” of each type of shift in a given distribution shift (similar to our method for dividing a distribution shift into in-support and out-of-support splits).
Subpopulation shift (and its sub-types), shifts involving spurious correlations, covariate shift, and label shift are typically in-support.
However, there are exceptions; for example, some works consider subpopulation shifts in which a subpopulation does not appear in the reference distribution [santurkar2020breeds, yang2023change], which are out-of-support.
Domain generalization problems are nearly always out-of-support and extrapolating effectively outside of the reference distribution is often a key challenge of these tasks.

#### D.5 Understanding the robustness of pre-trained language models to spurious correlations

[tu2020empirical] study the robustness of pre-trained language models to distribution shifts with spurious correlations.
Their central finding is that pre-training *can* improve performance on shifted datasets in which spurious correlations do not hold.
They illustrate that this is because pre-trained models can generalize better from the small number of counterexamples to these correlations in the reference dataset.
This is a similar phenomenon to our observation from Figure [2(a)](#A3.F2.sf1 "In Figure C.2 ‣ C.1.2 How does the strength of the bias affect robustness to in-support shifts? ‣ C.1 Constructing synthetic in-support and out-of-support shifts ‣ Appendix C Additional Results ‣ Ask Your Distribution Shift if Pre-Training is Right for You"): pre-training can provide some effective robustness on in-support shifts that are “close” to an out-of-support shift.
In cases such as those discussed by [tu2020empirical], we hypothesize that pre-training can help to a limited extent by extrapolating better, but cannot mitigating the underlying failure mode of dataset biases.

#### D.6 Additional related work

###### Pre-training.

Pre-training a model (or taking an existing pre-trained model) and then fine-tuning it on a task-specific dataset is a common practice when developing machine learning models, often significantly improving performance over training a model from scratch [razavian2014cnn, sun2017revisiting, kornblith2019better, kolesnikov2019big]. Pre-training can be effective even when the downstream task is unrelated to the pre-training task, suggesting that pre-training yields useful general-purpose features; for example, object classification models trained on ImageNet [deng2009imagenet] are good initializations for remote sensing [xie2016transfer] and medical imaging [ke2021chextransfer] tasks. Although greatly effective, pre-training is not without limitations. In some settings, pre-training does not improve performance over a randomly initialized model trained for long enough [he2019rethinking]. Downstream performance can saturate as performance on the pre-training task improves [abnar2021exploring]. Finally, biases of pre-trained models can persist after fine-tuning [salman2022does].

###### Distribution shift robustness.

Machine learning models are often deployed in different environments from those in which they are trained. Such distribution shifts can cause models to significantly underperform [koh2020wilds, gulrajani2020search, hendrycks2020faces]. Numerous interventions have been proposed to improve the robustness of models, often targeting particular types of shifts. These include algorithmic interventions [arjovsky2019invariant, byrd2019effect, sagawa2019distributionally, liu2021just, kirichenko2022last, idrissi2022simple] (often requiring group information), data augmentations [hendrycks2020faces, goel2020model] and pre-training (discussed below). However, interventions proposed thus far have failed to provide consistent benefits across distribution shift benchmarks [koh2020wilds, gulrajani2020search, hendrycks2020faces, wiles2021fine, ye2022ood], rendering distribution shift robustness a persistent challenge.
