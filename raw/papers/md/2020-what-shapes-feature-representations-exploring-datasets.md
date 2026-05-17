---
arxiv: '2006.12433'
authors:
- Katherine L. Hermann
- Andrew K. Lampinen
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: What shapes feature representations? Exploring datasets, architectures, and
  training
url: http://arxiv.org/abs/2006.12433v2
year: 2020
---

# What shapes feature representations? Exploring datasets, architectures, and training

Katherine L. Hermann
  
Stanford University
  
hermannk@stanford.edu
  
&Andrew K. Lampinen11footnotemark: 1
  
Stanford University
  
andrewlampinen@gmail.com
  
Contributed equally

###### Abstract

In naturalistic learning problems, a model’s input contains a wide range of features, some useful for the task at hand, and others not. Of the useful features, which ones does the model use? Of the task-irrelevant features, which ones does the model represent? Answers to these questions are important for understanding the basis of models’ decisions, as well as for building models that learn versatile, adaptable representations useful beyond the original training task. We study these questions using synthetic datasets in which the task-relevance of input features can be controlled directly. We find that when two features redundantly predict the labels, the model preferentially represents one, and its preference reflects what was most linearly decodable from the untrained model. Over training, task-relevant features are enhanced, and task-irrelevant features are partially suppressed. Interestingly, in some cases, an easier, weakly predictive feature can suppress a more strongly predictive, but more difficult one. Additionally, models trained to recognize both easy and hard features learn representations most similar to models that use only the easy feature. Further, easy features lead to more consistent representations across model runs than do hard features. Finally, models have greater representational similarity to an untrained model than to models trained on a different task. Our results highlight the complex processes that determine which features a model represents.

## 1 Introduction

How does a deep neural model see the world at initialization, and what changes over the course of training? If there are many latent features in the training data — such as the colors, textures, and shapes of objects in visual datasets — how does the model separate task-relevant features from irrelevant ones? Does a model gain sensitivity to diagnostic features by enhancing its representation of them, or by suppressing its representations of other features? What does a model represent among task-irrelevant features, features that are task-relevant but unreliable, and features that are redundantly predictive, each of which are present in naturalistic datasets? How similar are feature representations across models as a function of training task? How does the similarity depend on the feature?

From an engineering perspective, these questions are important as the field tries to build models whose representations are versatile and general-purpose enough to support out-of-distribution generalization, and to transfer to new downstream tasks. For example, it is common practice to transfer weights or activations from ImageNet models to models used for other vision tasks [[15](#bib.bib15), e.g.], and to use BERT [[7](#bib.bib7)] as a starting point for language tasks. Yet untrained models can sometimes provide an initialization that is nearly as good as ImageNet pretraining [[12](#bib.bib12), [23](#bib.bib23)], raising the questions of what information is present in an untrained model, and how it is modified during training. Furthermore, models sometimes learn to use “shortcut features” [[10](#bib.bib10)] – an object’s texture, rather than shape [[11](#bib.bib11)], for example – that solve the training task, but fail to generalize robustly. This phenomenon highlights the importance of understanding which features are learned in a given setting, especially when features are correlated. Additionally, understanding how models select features is important for ensuring that models make equitable and non-biased decisions [[45](#bib.bib45), [47](#bib.bib47)].

From a scientific perspective, understanding which features a model learns, and how consistent feature representations are across model instances, is relevant to understanding how models compute [[2](#bib.bib2)], as well as how they should be compared to one another [[29](#bib.bib29)] or to neural data [[19](#bib.bib19), [6](#bib.bib6), [5](#bib.bib5), [39](#bib.bib39), [43](#bib.bib43)].

In this paper, we investigate the evolution of models’ feature preferences using synthetic datasets in which we can directly control the relationships between multiple features and task labels, which is not possible in naturalistic datasets like ImageNet. We create datasets from a data-generating process in which underlying latent variables give rise to input features like shape or texture (in our visual tasks), or linearly or non-linearly extractable features (in our binary features tasks). We train models on tasks that are either deterministically or probabilistically related to these input features. Leveraging the complementary approaches of decoding and representational similarity analysis [[24](#bib.bib24)], we probe feature representations within and between models across datasets, architectures, and training. Our results provide insight into how feature representations are affected by learning and how to analyze them. Our key contributions are:

* •

  We find that many input features are already partially represented in the higher layers of untrained models. Training on a task enhances task-relevant features (increases their decodability relative to an untrained model), and suppresses both task-irrelevant features *and* some task-relevant features.
* •

  We investigate what a model represents when multiple features predict the label. We find that, when a pair of features redundantly predicts the label, models prefer one of the features over the other, and the preference structure tracks untrained decodability. When only one feature is perfectly predictive, we find that the representations of a correlated feature remain roughly constant as a function of correlation, until the correlation becomes quite high.
* •

  We identify cases in which models are “lazy,” and suppress a more predictive feature in favor of an easier-to-extract, but less predictive, feature.
* •

  We find that easy features induce representations that are more consistent across model runs than do hard features, and that a multi-task model trained to report both easy and hard features produces representations that are very similar to those of a model trained only on the easy feature.

## 2 Related Work

Both theoretical and empirical work has investigated how models represent task-irrelevant features over training. Theoretical work by Shwartz and colleagues [[38](#bib.bib38)] suggests that deep models compress away task-irrelevant information as they learn a task, though the generality of this observation has been disputed [[34](#bib.bib34)]. Several papers have [[14](#bib.bib14), [16](#bib.bib16)] shown that it is possible to decode “category-orthogonal” features from CNNs trained on naturalistic datasets. However, catastrophic interference between new tasks and prior tasks shows that at least some features are suppressed or altered during training [[28](#bib.bib28), [21](#bib.bib21), [46](#bib.bib46), [33](#bib.bib33), [27](#bib.bib27)]. Thus, a tension exists in the literature about how features change over learning: is task-irrelevant information suppressed, or just ignored?

Theoretical work by Saxe and colleagues has found that supervised linear models learn the input features that account for the most variance in the data before learning features that account for less variance [[35](#bib.bib35)]. Given hierarchical data, models learn high-level semantic features before proceeding to lower-level ones [[36](#bib.bib36)]. This observation that models progressively differentiate task structure has been leveraged to explain the generalization benefits of optimal stopping: since stronger features have a higher signal-to-noise ratio, optimal stopping captures features to the extent that they are both important and not overly noisy [[25](#bib.bib25)]. Qualitatively connecting with this work, we ask the further question of how feature difficulty interacts with feature strength. For example, are features that are linearly extractable learned before features that require nonlinear extraction? How does a model trade off ease of extracting a feature and that feature’s reliability in predicting task labels?

Some work suggests that deep networks may have an inherent bias to favor simple functions. For example, there exist many more parameter choices which yield simple than complex functions [[41](#bib.bib41)]. The inductive biases of SGD in highly over-parameterized models might contribute to finding solutions that meet various notions of simplicity [[3](#bib.bib3), [4](#bib.bib4)]. Our empirical results help ground these theoretical predictions, and connect to prior explorations showing that deep networks sometimes exploit “shortcut features” – features which may be sufficient to solve a training task, but which may fail to generalize robustly and differ from the features preferred by people [[10](#bib.bib10)]. For example, in the vision domain, recent work has found that standard ImageNet models prefer to classify objects by texture rather than shape, whereas the reverse is true for people [[11](#bib.bib11)], though this bias can be ameliorated with data augmentation [[11](#bib.bib11), [13](#bib.bib13)]. Whereas Geirhos and colleagues studied the classification (output) behavior of models trained on ImageNet, a dataset whose joint image feature–label statistics are unknown and uncontrolled, our experiments investigate the decodability of visual features from intermediate model layers when trained on datasets with controlled statistics. Using analyses more similar to ours, [[13](#bib.bib13)] found that shape information is more decodable than texture in the convolutional layers of an ImageNet-trained CNN. Concurrent work by Shah and colleagues observes that models may prefer simple (e.g. linear) features even when difficult (e.g. non-linear) ones are more reliable [[37](#bib.bib37)], and Parascandolo et al. propose a new approach to optimization that could potentially ameliorate this affect [[31](#bib.bib31)].

Related to the question of which features models come to represent over training is the question of which features the model represents at initialization. Prior work has observed that the representations in untrained networks already transfer well to tasks like classification [[17](#bib.bib17), [32](#bib.bib32)], and denoising and inpainting ([[40](#bib.bib40)]). Evolved architectures with untrained weights have been successful in action-planning problems [[9](#bib.bib9)]. Further work has shown that minimal training – only adjusting BatchNorm parameters – of untrained models already yields surprisingly high classification performance [[8](#bib.bib8)].
In addition, untrained initializations often perform surprisingly similarly to pretrained ones after fine-tuning on a new task [[12](#bib.bib12), [23](#bib.bib23)]. Furthermore, features in untrained models can explain substantial amounts of neural data, although trained models generally perform slightly better [[44](#bib.bib44), [18](#bib.bib18), [39](#bib.bib39), [6](#bib.bib6), [5](#bib.bib5)]. Overall, this prior work observes that untrained feature representations capture a great deal of task-relevant structure; we explore how these initial representations change over training.

## 3 Does feature selection happen by enhancement or suppression?

Over training, as a model becomes sensitive to a target feature, does it build this sensitivity by enhancing the target feature, or by suppressing non-target features? How much enhancement and/or suppression occurs? In this set of experiments, we used two synthetic datasets, and trained vision models to classify images according to their shape, texture, or color. We then tested to what extent target and non-target features were decodable before and after training.

Datasets.
In a set of preliminary experiments, our datasets consisted of 224 ×\times 224 RGB images of objects with multiple features, each of which could be used as a label in a classification task. For each of these tasks, we created 5 cross-validation splits, creating validation sets by holding out a subset of values for the non-target features (3 classes per feature). For example, a validation set for a color classification task might include squares, triangles, and circles, while no training examples would have had these shapes. Thus, our classification tasks required generalizing the target feature (e.g. color) over the non-target feature(s) (e.g. shape and texture).

!(/html/2006.12433/assets/figures/stims.png)

Figure 1: Example vision dataset items. The Navon dataset is adapted from [[13](#bib.bib13)] and based on [[30](#bib.bib30)].

Navon dataset. As shown in Figure [1](#S3.F1 "Figure 1 ‣ 3 Does feature selection happen by enhancement or suppression? ‣ What shapes feature representations? Exploring datasets, architectures, and training"), this dual-feature dataset, modified from [[13](#bib.bib13)] based on [[30](#bib.bib30)], consisted of images in which a large letter (“shape”) is rendered in small copies of some other letter (“texture”). We rendered each shape-texture combination (26×\times26) at 5 positions, rotating the shape and texture independently at angles drawn from [−45,45]4545[-45,45] degrees, yielding 3250 items after excluding those with the same shape and texture.

Trifeature dataset. Images contained one of ten shapes, rendered in one of ten textures, in one of ten colors (see Figure [B.1](#A2.F1 "Figure B.1 ‣ B.3 Trifeature ‣ Appendix B Methods ‣ What shapes feature representations? Exploring datasets, architectures, and training")). Shapes were rendered within a 128×128128128128\times 128 square, which was rotated at an angle drawn from [−45,45]4545[-45,45] degrees and placed at a random position within the larger image, such that the shape was completely contained within the image, before an independently rotated texture and a color were applied. Textures were rendered so that they could be discriminated within at most a 10×10101010\times 10 square region. We rendered 100 versions of each color, shape, and texture combination, for a total pool of 100,000 images. For the experiments in this section, we created train sets of 3430 items and validation sets of 3570 images, in both of which the features were uncorrelated.

Model architecture.
We evaluated two model architectures: AlexNet and ResNet50 with output sizes modified to reflect the number of target classes (Navon: 23, Trifeature: 7). AlexNet’s other fully connected layers were also narrowed proportionally (Navon: 95 units each in fc6, fc7; Trifeature: 29 units each). An interesting question for future follow up is the effect of fully-connected layer width on feature representations. We used the torchvision implementation of the both models’ convolutional layers [[42](#bib.bib42)], and fixed weight initializations across experiments.

Training procedure.
Training and validation images were normalized by the mean and standard deviation of the training data. They were not cropped. All models were trained for 90 epochs using Adam optimization [[20](#bib.bib20)] with a learning rate of 3×10−43superscript1043\times 10^{-4}, weight decay of 10−4superscript10410^{-4}, and batch size of 64, using a modified version of the torchvision training script [[1](#bib.bib1)]. We selected the model corresponding to the highest validation accuracy over the training period.

Decoding.
To determine which visual features a trained model represents – that is, which features (of shape, color, and texture) can be extracted from layer activations of the trained model in response to a set of images – we trained decoders to map activations from some layer of a trained, frozen model to feature labels using a single linear layer followed by a softmax (multinomial logistic regression). Importantly, the labels on which the decoder was trained were not, in the general case, the same as the labels with which the network was trained. We decoded from upper layers of models (AlexNet: pool3, fc6, or fc7; ResNet-50: pre-pool, post-pool). See Appendix [B.1.1](#A2.SS1.SSS1 "B.1.1 Decoding from vision models (Sections 3 and 4) ‣ B.1 Decoding ‣ Appendix B Methods ‣ What shapes feature representations? Exploring datasets, architectures, and training") for additional details.

### 3.1 What’s already decodable from an untrained model?

Across two datasets (Navon and Trifeature) and two model architectures (AlexNet and ResNet-50), visual features were decodable significantly above chance from the upper layers of untrained models. As shown in Figures [2](#S3.F2 "Figure 2 ‣ 3.2 What’s decodable after training? ‣ 3 Does feature selection happen by enhancement or suppression? ‣ What shapes feature representations? Exploring datasets, architectures, and training") and [A.2](#A1.F2 "Figure A.2 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training"), the rank order of decoding accuracy by feature held across architectures and across the layers of AlexNet: color was most decodable, followed by shape and then texture. Color was perfectly decodable from the final convolutional layer of AlexNet as well as from ResNet-50. Interestingly, decoding accuracy for all features decreased through the fully connected layers of AlexNet.

In the Navon dataset (Figure [3](#S3.F3 "Figure 3 ‣ 3.2 What’s decodable after training? ‣ 3 Does feature selection happen by enhancement or suppression? ‣ What shapes feature representations? Exploring datasets, architectures, and training")), it was possible to decode shape and texture with accuracy >>60% (chance = 4.3%) from the final convolutional layer of AlexNet. Consistent with the Trifeature results, decoding accuracy fell off through the fully-connected layers, but was still above chance in the penultimate layer. Both features were decodable above chance from ResNet-50, at appx. 25% prior to the global average pool, dropping off slightly after the pool (see Figure [A.1](#A1.F1 "Figure A.1 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")).

One hypothesis is that the decodability of features from an untrained model reflects the model’s inductive biases, and might predict the extent to which a feature would be preserved after training the model on a different task.

### 3.2 What’s decodable after training?

In this section, we trained models to classify a target feature in the presence of one or more non-target, task-irrelevant (uncorrelated with the label) features.

!(/html/2006.12433/assets/figures/cst_experiments/cst_custom_alexnet_trained_vs_untrained_decoding.png)

Figure 2: Target features are enhanced, and non-target features are suppressed, relative to an untrained model in models with an AlexNet architecture trained on Trifeature tasks. Accuracy decoding features (shape, texture, color) from layers of an untrained model (far left) versus from models trained to classify shape, texture, or color (mean decoding accuracy across models trained on each of 5 cv-splits of the data; error bars indicate 95% CIs). Chance = 1717\frac{1}{7} = 14.3% (dashed gray line). Decoding accuracy is generally higher for target features in the trained than in the untrained model (enhanced) and lower for non-target features (suppressed).

!(/html/2006.12433/assets/figures/navon_experiments/navon_custom_alexnet_trained_vs_untrained_decoding.png)

Figure 3: Feature decodability in models with an AlexNet architecture trained on the Navon dataset. Accuracy decoding features (shape, texture) from an untrained model (left) versus from shape- (center) and texture-trained models (right). Results corresponding to trained models are mean across models trained on 5 cv splits. Chance = 123123\frac{1}{23} = 4.3% (dashed gray line). As for the Trifeature models (Figure [2](#S3.F2 "Figure 2 ‣ 3.2 What’s decodable after training? ‣ 3 Does feature selection happen by enhancement or suppression? ‣ What shapes feature representations? Exploring datasets, architectures, and training")), decoding accuracy in the untrained model decreases across layers, and the target features are enhanced, whereas the non-target features are suppressed.

Across architectures (AlexNet and ResNet-50), target features were enhanced: decoding accuracies were higher than in the untrained model, if they were not already at ceiling (Figures [3](#S3.F3 "Figure 3 ‣ 3.2 What’s decodable after training? ‣ 3 Does feature selection happen by enhancement or suppression? ‣ What shapes feature representations? Exploring datasets, architectures, and training"), [2](#S3.F2 "Figure 2 ‣ 3.2 What’s decodable after training? ‣ 3 Does feature selection happen by enhancement or suppression? ‣ What shapes feature representations? Exploring datasets, architectures, and training"), [A.1](#A1.F1 "Figure A.1 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training"), and [A.2](#A1.F2 "Figure A.2 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")). Target features were highly decodable from all layers. By contrast, non-target features were partially suppressed: decoding accuracies were lower than in the untrained model, but still above chance. Non-target features were most decodable from the the final convolutional layer of AlexNet and their decodability decreased layerwise, recapitulating the pattern found in the untrained models.

## 4 What if multiple features are predictive?

So far, we have considered which features a model represents when trained on datasets in which one feature is task-relevant and the others are not. However, in many real-world datasets, features exhibit some degree of correlation with one another. For example, in real-world objects, features like shape, texture, and color are often correlated, especially for natural objects. Here, we tested which features a model represents when multiple features predict the target to varying degrees.

Datasets.
In these experiments, we used two datasets that allowed us to control the predictivity of features as well as feature difficulty.

Trifeature (Correlated) datasets. We created versions of the Trifeature dataset with correlated features: we sampled train sets (4900 images) and validation sets (≥\geq 4900 images, varied with correlation) from the larger set of 100,000 images. We correlated a pair of features (e.g. shape and color) by choosing a conditional probability of one feature matching the other between 0.1 and 1. If this probability was 1, the features were perfectly correlated, and if it was 0.1, the features were uncorrelated. A pair of features was correlated across the set of images (not within individual images).

As an example, suppose that shape and color are correlated with conditional probability = 0.5. Then, in images containing triangles, half would be red and half would be a color sampled uniformly at random from the other color options (blue, white, etc.). Similarly, in images containing trapezoids, half would be orange and half would be some other color. The attribute matching (e.g. triangle = red, trapezoid = orange, etc.) was assigned randomly.

In each dataset version, training data contained 7 classes per input feature type (shape, color, texture), with 3 classes each held out. As in the uncorrelated trifeature datasets (above), validation set stimuli always had at least one of the non-target features held-out; for example, if the target feature was texture, generalization would be evaluated on shapes and colors that were not encountered during training. Feature correlations were matched in the validation set.111Not enforcing feature correlations for the validation set would also be reasonable, but our approach matches the many real-world datasets where train and validation data are sampled from the same distribution.

Binary Features datasets.
We created (non-vision) datasets containing features that differed in difficulty. We define a feature’s difficulty in terms of the minimum complexity of a network required to extract it. We considered two features: an “easy”/“linear” one extractable by a single linear layer, and a “difficult”/“nonlinear” one requiring an XOR computation which must be implemented by a multi-layer perceptron [[26](#bib.bib26)]. The inputs in our dataset were 32-element binary (1 or 0) vectors in which the first 16 elements instantiated the easy/linear feature, and the second 16 instantiated difficult/nonlinear feature. The label was probabilistically determined by these two features — that is, the inputs were sampled so that each feature matched the label a certain percentage of the time. Specifically, we varied the predictivity of the easy feature (p​(label|easy feature)𝑝conditionallabeleasy featurep\left(\text{label}|\text{easy feature}\right)) between 0.5 (chance) and 1 (perfectly predictive). We fixed the predictivity of the harder feature at 0.9. We trained on 256 examples, and used 512 for evaluation and decoding.

### 4.1 What if two features redundantly predict the label?

!(/html/2006.12433/assets/figures/cst_experiments/cst_custom_alexnet_accuracy_perfect_correlation_decoding.png)

Figure 4: When two features redundantly predict the label, models
preferentially learn one feature. Color is more decodable than shape, and shape is more decodable than texture, when AlexNet is trained on perfectly predictive pairs.

Using the Trifeature (Correlated) datasets with conditional probabilities of 1.0, and methods as in Section [3.2](#S3.SS2 "3.2 What’s decodable after training? ‣ 3 Does feature selection happen by enhancement or suppression? ‣ What shapes feature representations? Exploring datasets, architectures, and training"), we trained models on classification tasks in which pairs of features (shape and color, or shape and texture) redundantly predicted the label, while the third feature was task-irrelevant. We then tested decodability of the predictive features (see Appendix [B.1.1](#A2.SS1.SSS1 "B.1.1 Decoding from vision models (Sections 3 and 4) ‣ B.1 Decoding ‣ Appendix B Methods ‣ What shapes feature representations? Exploring datasets, architectures, and training")).

Results.
Across architectures, rather than representing both features equally, one of the two predictive features was always substantially more decodable from the trained model than was the other. Intriguingly, the less decodable, but still perfectly predictive, feature was sometimes even suppressed relative to an untrained model. In AlexNet (Figure [4](#S4.F4 "Figure 4 ‣ 4.1 What if two features redundantly predict the label? ‣ 4 What if multiple features are predictive? ‣ What shapes feature representations? Exploring datasets, architectures, and training")), when shape and color were correlated, color was nearly perfectly decodable, whereas shape was considerably less so, and was in fact less decodable
from pool3 and fc6 than in an untrained model (Figure [A.7](#A1.F7 "Figure A.7 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")). When shape and texture were correlated, decodability of shape was considerably higher than decodability of texture, and texture was less decodable from pool3 than in an untrained model. The same rank-order decodability patterns held in ResNet-50 (Figure [A.6](#A1.F6 "Figure A.6 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training"),  [A.8](#A1.F8 "Figure A.8 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")). Overall, these results suggest that, when two features redundantly predict the label, rather than learning both, models privilege color over shape, and shape over texture. This preference structure aligns with the rank-order decodability of features from untrained models (Figures [2](#S3.F2 "Figure 2 ‣ 3.2 What’s decodable after training? ‣ 3 Does feature selection happen by enhancement or suppression? ‣ What shapes feature representations? Exploring datasets, architectures, and training") and [A.2](#A1.F2 "Figure A.2 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")).

### 4.2 What if one feature perfectly predicts the label, but another only partially predicts it?

In these experiments, in which we used Trifeature (Correlated) datasets with conditional probabilities of 0.1 to 0.9222For reasons of computational resources, for ResNet-50 models, we sampled conditional probabilities as [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]., one feature perfectly predicted the label, and a second feature (correlated non-target) was correlated with the target feature to varying degrees across datasets. The third feature was uncorrelated with the others. We hypothesized that the decodability of the correlated non-target feature would increase as a function of its correlation with the target feature, possibly becoming enhanced relative to the untrained model.

Results.
The target feature was always enhanced (higher decodability from the trained than the untrained model), whereas the correlated non-target feature was generally preserved (no change) or suppressed (Figure [5](#S4.F5 "Figure 5 ‣ 4.2 What if one feature perfectly predicts the label, but another only partially predicts it? ‣ 4 What if multiple features are predictive? ‣ What shapes feature representations? Exploring datasets, architectures, and training")). Surprisingly, the level of suppression was more or less constant across degree of correlation, except at high correlations. When the conditional probability was 0.9, the correlated non-target features were slightly enhanced.
A similar pattern was observed in the post-pool layer of ResNet-50 (Figure [A.3](#A1.F3 "Figure A.3 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")). Figures [A.4](#A1.F4 "Figure A.4 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training") and [A.5](#A1.F5 "Figure A.5 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training") show raw decoding accuracies.

!(/html/2006.12433/assets/figures/cst_experiments/decoding_correlated.png)

Figure 5: Non-target features correlated with the target feature are generally suppressed. For both datasets in which shape and color (left), and shape and texture (right), are correlated, non-target features correlated with the target feature (“correlated”) are suppressed (red region) relative to the untrained model, whereas target features are enhanced (yellow region). Suppression of the correlated non-target feature is largely constant across correlation strengths (x axis), until very high correlations.

### 4.3 Do models prefer reliable but difficult features, or easy but less reliable ones?

Above, we tested what vision models learn when one feature perfectly predicts the label. Here, we consider the non-visual Binary Features datasets and a simpler model architecture (a 5-layer MLP), and consider what happens when no feature is perfectly predictive. We vary both the difficulty and predictivity of features, and test whether a model prefers a feature that is easier to learn (linearly decodable from the input) but only moderately predictive of the label, or a feature that is more difficult (XOR/parity, not linearly correlated with the input) but more predictive of the label. Will a model trade off predictivity for learnability?

!(/html/2006.12433/assets/figures/toy_experiments/toy_easy_vs_hard_compressed.png)

Figure 6: More reliable, but difficult, features can be suppressed by less reliable, but easier, features. The difficult feature (purple) had fixed predictivity of 0.9, while the easy feature (green) had varied predictivity (x-axis). The left panel evaluates the model’s use of each feature (using a test set with the other feature made unpredictive). The middle panel shows decodability of each feature from the penultimate layer. The right panel shows decodability of a single input unit’s value (another linear feature) from the inputs associated with each feature. (5 runs per condition; lines are linear fits.)

Models.
We used 5-layer MLPs (layer sizes 256, 128, 64, and 64). Hidden-layer nonlinearities were leaky rectifiers; output was a sigmoid. Training was via a cross-entropy loss, by full batch gradient descent (lr =0.001absent0.001=0.001). Decoding and representation analyses were performed at the final hidden layer (see Appendix [B.1.2](#A2.SS1.SSS2 "B.1.2 Decoding from models trained on binary features (Section 4.3) ‣ B.1 Decoding ‣ Appendix B Methods ‣ What shapes feature representations? Exploring datasets, architectures, and training")). We assessed feature reliance by testing on a dataset where the other feature was made unpredictive.

Results.
The more reliable feature can be suppressed by a less reliable, but easier, one (Figure [6](#S4.F6 "Figure 6 ‣ 4.3 Do models prefer reliable but difficult features, or easy but less reliable ones? ‣ 4 What if multiple features are predictive? ‣ What shapes feature representations? Exploring datasets, architectures, and training")). When the easy feature had low predictivity, the model relied on the difficult feature (Figure [6](#S4.F6 "Figure 6 ‣ 4.3 Do models prefer reliable but difficult features, or easy but less reliable ones? ‣ 4 What if multiple features are predictive? ‣ What shapes feature representations? Exploring datasets, architectures, and training"), left panel). As the easier feature became more predictive, the model relied on it more, and less on the difficult feature. This switch happened before the predictivity of the features was matched; for example, the model preferred the easy feature when that feature’s predictivity was 0.8, despite the harder feature having greater predictivity (0.9). Again, this preference for the easier feature reflects its greater decodability from an untrained model (see Figure [A.9](#A1.F9 "Figure A.9 ‣ A.1 Binary feature tasks ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")). Thus, in this case the model appears to trade off predictivity for ease of learning.

Which features are represented? The easy feature could be decoded with relatively high accuracy even if it was not predictive, but the difficult feature could be decoded reliably only when the feature-specific tests showed it was being used. (Even non-linear decoders could not recover the difficult feature much more accurately, see Figure [A.10](#A1.F10 "Figure A.10 ‣ A.1 Binary feature tasks ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training").) Intriguingly, a single input unit from either set of inputs could be decoded with around 80%percent8080\% accuracy even if the associated feature was not predictive (Figure [6](#S4.F6 "Figure 6 ‣ 4.3 Do models prefer reliable but difficult features, or easy but less reliable ones? ‣ 4 What if multiple features are predictive? ‣ What shapes feature representations? Exploring datasets, architectures, and training"), right panel), showing that irrelevant inputs were not completely suppressed. However, complex, nonlinear combinations of those input units were difficult to recover.

## 5 What affects the representational similarity between models?

!(/html/2006.12433/assets/figures/cst_experiments/cst_basic_rsa.png)

(a) Similarity by architecture and task pairing.

!(/html/2006.12433/assets/figures/cst_experiments/cst_RSA_by_tasks.png)

(b) Details of cross-task similarity.

Figure 7: Representational similarity of Trifeature (Uncorrelated) models. ([7(a)](#S5.F7.sf1 "In Figure 7 ‣ 5 What affects the representational similarity between models? ‣ What shapes feature representations? Exploring datasets, architectures, and training")) Representational similarity of the same layer across two runs, of different layers within an architecture (e.g. pool3 and fc6), and of different layers across architectures (AlexNet pool3 and ResNet-50 post-pool).
([7(b)](#S5.F7.sf2 "In Figure 7 ‣ 5 What affects the representational similarity between models? ‣ What shapes feature representations? Exploring datasets, architectures, and training")) Representational similarity across layers and architectures on each possible pair of tasks. Similarity is highest between pairs of models trained on the same versus different tasks, though within tasks, it is lowest for texture. (Results from 5 runs per condition; errorbars are bootstrap 95%-CIs.)

!(/html/2006.12433/assets/figures/toy_experiments/toy_task_rsa_heatmap_rsa_correlation.png)

Figure 8: Representational similarity of Binary Feature models. We compare an untrained model, a model trained with an unpredictive, or perfectly predictive, easy feature (p​(label|easy feature)=0.5𝑝conditionallabeleasy feature0.5p(\text{label}|\text{easy feature})=0.5 or 1.01.01.0), and a multitask model. While similarity is generally highest between pairs of models trained on the same task, the numerical magnitude varies considerably.
Multi-task models closely resemble the model trained with the predictive easier feature, even though they are also trained to output the more difficult feature. (5 runs per condition.)

Our findings above suggest that which features a model represents depends on both the predictivity of features, and their availability in an untrained model/easiness to learn. In the following experiments, we tested whether models trained on the same task have consistent representations of inputs, and how similar their representations are to those of untrained models and models trained on a different task. To quantify similarity, we used Representational Similarity Analysis (RSA) [[24](#bib.bib24)], a method widely used in neuroscience, e.g. to compare models to neural (e.g. fMRI and physiology) data.

Representation analyses.
For each model layer, for each training regime, we constructed a representational dissimilarity matrix (RDM), 𝐃S×Ssuperscript𝐃𝑆𝑆\mathbf{D}^{S\times S}, where S𝑆S is the number of stimuli (dataset inputs), and each entry 𝐃i​jsubscript𝐃𝑖𝑗\mathbf{D}\_{ij} contains a measure of the dissimilarity of patterns of activations elicited by stimuli i𝑖i and j𝑗j. We used correlation distance as our dissimilarity measure. We then tested the level of correspondence between pairs of model layers by computing a Pearson’s correlation of the upper right triangles of their RDMs. See Appendix [B.4](#A2.SS4 "B.4 Representational Similarity Analyses ‣ Appendix B Methods ‣ What shapes feature representations? Exploring datasets, architectures, and training") for results with intermediate easy-feature predictivities, as well as with other analyses and metrics.

Visual tasks.
In Figure [7](#S5.F7 "Figure 7 ‣ 5 What affects the representational similarity between models? ‣ What shapes feature representations? Exploring datasets, architectures, and training") we show RSA results for the uncorrelated Trifeature tasks (see Figure [A.11](#A1.F11 "Figure A.11 ‣ A.2 RSA ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training") for the Navon tasks). Models trained on matching tasks had much more similar representations than models trained on different tasks; in fact, models trained on one task were more similar to *untrained* models than to models trained on a different task. The magnitude of the similarity varied substantially, however. Similarity between two models trained on the (more difficult) texture task was lower (mean 0.408, bootstrap 95%-CI [0.394, 0.420]) than similarity between two models trained on one of the other tasks (e.g. shape, 0.624 [0.615, 0.632]), or even two untrained models (0.619 [0.542, 0.702]). The influence of architecture or layer on similarity was small compared to the influence of task. Similarity was most sensitive to the target feature up to relatively high correlations of other features (Figure [A.12](#A1.F12 "Figure A.12 ‣ A.2 RSA ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")).

Binary tasks.
Figure [8](#S5.F8 "Figure 8 ‣ 5 What affects the representational similarity between models? ‣ What shapes feature representations? Exploring datasets, architectures, and training") contains RSA results in the binary features domain (see also Figure [A.13](#A1.F13 "Figure A.13 ‣ A.2 RSA ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")). The similarity between two models trained with a highly predictive easy feature is much greater than the similarity between two models trained with an unpredictive easy feature. That is, models seem to find more representationally variable solutions to extracting a nonlinear feature than extracting a linear feature. In Figure [A.14](#A1.F14 "Figure A.14 ‣ A.2 RSA ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training") we show some intuitions for why this might be the case. Furthermore, the untrained model and multi-task trained model have representations that are more similar to models trained with a predictive easy feature. In fact, the multi-task model, which is trained to classify *both* features, produces representations that closely resemble those of the model trained with a highly-predictive easy feature. This suggests that the easier feature dominates the representations of the multi-task model, even when it also represents the more difficult feature. This finding suggests that there are cases in which the value of a representational similarity metric may be dominated by the representational structures induced by one feature, for example an easier one, even if another feature is still represented.

## 6 Conclusion

Overall, our results sketch a picture of how a model’s representations are shaped by its inductive biases and training. In the presence of multiple redundantly predictive features, the model may choose to principally enhance one, and this choice will favor the feature that is most decodable from an untrained model. Indeed, the model will sometimes favor features that are easier in this sense over ones that are harder, even if the latter are more predictive of the label. Furthermore, features that are not used for classification are actively suppressed rather than preserved from the initial state, even in some cases when they perfectly predict the label. Still, these features are not completely compressed away – in many cases they are still partially decodable. Finally, representational similarity is dominated by easier features, meaning that it may be misleading in complex or multi-task settings. We have shown these results across multiple architectures and datasets, but future work should explore broader settings.

Our findings suggest that, first, practitioners should not assume that label-relevant features will be used, or even represented by, a model. This may help explain shortcut learning, and the variable benefits of pretraining, since features useful for downstream tasks may be suppressed by pretraining. Second, analyzing untrained model representations can be a cheap way to predict which features a model might use. Finally, representational similarity analyses should be interpreted with care when applied to a model (or brain) trained on multiple features or multiple tasks.

## Broader Impact

Understanding the representations that models use to make their decisions is an important part of ensuring the correctness and safety of these decisions, particularly in situations where models operate on inputs different in distribution from those on which they were trained. Furthermore, many ethical concerns about machine learning models hinge on their reliance on features that encourage bias on socially discriminatory features; understanding and being able to predict which features are used is an important step in solving this problem.

## Acknowledgments and Disclosure of Funding

We thank Jay McClelland for useful conversations. KLH was supported by NSF GRFP grant DGE-1656518. AKL was supported by NSF GRFP grant DGE-114747.

## References

* [1]

  ImageNet training script for `torchvision` model implementations (used
  with modification).
  <https://github.com/pytorch/examples/blob/master/imagenet/main.py>.
* [2]

  Adi, Y., Kermany, E., Belinkov, Y., Lavi, O., and Goldberg, Y.
  Fine-grained analysis of sentence embeddings using auxiliary
  prediction tasks.
  International Conference on Learning Representations (2017).
* [3]

  Arora, S., Cohen, N., Hu, W., and Luo, Y.
  Implicit regularization in deep matrix factorization.
  In Advances in Neural Information Processing Systems (2019),
  pp. 7413–7424.
* [4]

  Belkin, M., Hsu, D., Ma, S., and Mandal, S.
  Reconciling modern machine-learning practice and the classical
  bias–variance trade-off.
  Proceedings of the National Academy of Sciences 116, 32 (2019),
  15849–15854.
* [5]

  Cichy, R. M., Khosla, A., Pantazis, D., and Oliva, A.
  Dynamics of scene representations in the human brain revealed by
  magnetoencephalography and deep neural networks.
  NeuroImage 153 (2017), 346–358.
* [6]

  Cichy, R. M., Khosla, A., Pantazis, D., Torralba, A., and Oliva, A.
  Comparison of deep neural networks to spatio-temporal cortical
  dynamics of human visual object recognition reveals hierarchical
  correspondence.
  Scientific reports 6 (2016), 27755.
* [7]

  Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K.
  Bert: Pre-training of deep bidirectional transformers for language
  understanding.
  arXiv preprint arXiv:1810.04805 (2018).
* [8]

  Frankle, J., Schwab, D. J., and Morcos, A. S.
  Training batchnorm and only batchnorm: On the expressive power of
  random features in cnns.
  arXiv preprint arXiv:2003.00152 (2020).
* [9]

  Gaier, A., and Ha, D.
  Weight agnostic neural networks.
  In Advances in Neural Information Processing Systems (2019),
  pp. 5365–5379.
* [10]

  Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W.,
  Bethge, M., and Wichmann, F. A.
  Shortcut learning in deep neural networks.
  arXiv preprint arXiv:2004.07780 (2020).
* [11]

  Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., and
  Brendel, W.
  Imagenet-trained cnns are biased towards texture; increasing shape
  bias improves accuracy and robustness.
  arXiv preprint arXiv:1811.12231 (2018).
* [12]

  He, K., Girshick, R., and Dollár, P.
  Rethinking imagenet pre-training.
  In Proceedings of the IEEE International Conference on Computer
  Vision (2019), pp. 4918–4927.
* [13]

  Hermann, K. L., and Kornblith, S.
  Exploring the origins and prevalence of texture bias in convolutional
  neural networks.
  arXiv preprint arXiv:1911.09071 (2019).
* [14]

  Hong, H., Yamins, D. L., Majaj, N. J., and DiCarlo, J. J.
  Explicit information for category-orthogonal object properties
  increases along the ventral stream.
  Nature neuroscience 19, 4 (2016), 613.
* [15]

  Huh, M., Agrawal, P., and Efros, A. A.
  What makes imagenet good for transfer learning?
  arXiv preprint arXiv:1608.08614 (2016).
* [16]

  Islam, M. A., Jia, S., and Bruce, N. D.
  How much position information do convolutional neural networks
  encode?
  arXiv preprint arXiv:2001.08248 (2020).
* [17]

  Jarrett, K., Kavukcuoglu, K., Ranzato, M., and LeCun, Y.
  What is the best multi-stage architecture for object recognition?
  In 2009 IEEE 12th international conference on computer vision
  (2009), IEEE, pp. 2146–2153.
* [18]

  Kell, A. J., Yamins, D. L., Shook, E. N., Norman-Haignere, S. V., and
  McDermott, J. H.
  A task-optimized neural network replicates human auditory behavior,
  predicts brain responses, and reveals a cortical processing hierarchy.
  Neuron 98, 3 (2018), 630–644.
* [19]

  Khaligh-Razavi, S.-M., and Kriegeskorte, N.
  Deep supervised, but not unsupervised, models may explain it cortical
  representation.
  PLoS computational biology 10, 11 (2014).
* [20]

  Kingma, D. P., and Ba, J.
  Adam: A method for stochastic optimization.
  arXiv preprint arXiv:1412.6980 (2014).
* [21]

  Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G.,
  Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al.
  Overcoming catastrophic forgetting in neural networks.
  Proceedings of the national academy of sciences 114, 13 (2017),
  3521–3526.
* [22]

  Kornblith, S., Norouzi, M., Lee, H., and Hinton, G.
  Similarity of neural network representations revisited.
  arXiv preprint arXiv:1905.00414 (2019).
* [23]

  Kornblith, S., Shlens, J., and Le, Q. V.
  Do better imagenet models transfer better?
  In Proceedings of the IEEE conference on computer vision and
  pattern recognition (2019), pp. 2661–2671.
* [24]

  Kriegeskorte, N., Mur, M., and Bandettini, P. A.
  Representational similarity analysis-connecting the branches of
  systems neuroscience.
  Frontiers in systems neuroscience 2 (2008), 4.
* [25]

  Lampinen, A. K., and Ganguli, S.
  An analytic theory of generalization dynamics and transfer learning
  in deep linear networks.
  Proceedings of the International Conference on Learning
  Representations (2019).
* [26]

  Marvin, M., and Seymour, A. P.
  Perceptrons, 1969.
* [27]

  McClelland, J. L., McNaughton, B. L., and Lampinen, A. K.
  Integration of new information in memory: new insights from a
  complementary learning systems perspective.
  Philosophical Transactions of the Royal Society B 375, 1799
  (2020), 20190637.
* [28]

  McCloskey, M., and Cohen, N. J.
  Catastrophic interference in connectionist networks: The sequential
  learning problem.
  In Psychology of learning and motivation, vol. 24. Elsevier,
  1989, pp. 109–165.
* [29]

  Mehrer, J., Spoerer, C. J., Kriegeskorte, N., and Kietzmann, T. C.
  Individual differences among deep neural network models.
  bioRxiv (2020).
* [30]

  Navon, D.
  Forest before trees: The precedence of global features in visual
  perception.
  Cognitive psychology 9, 3 (1977), 353–383.
* [31]

  Parascandolo, G., Neitz, A., Orvieto, A., Gresele, L., and Schölkopf,
  B.
  Learning explanations that are hard to vary.
  arXiv preprint arXiv:2009.00329 (2020).
* [32]

  Ramanujan, V., Wortsman, M., Kembhavi, A., Farhadi, A., and Rastegari, M.
  What’s hidden in a randomly weighted neural network?
  arXiv preprint arXiv:1911.13299 (2019).
* [33]

  Rostami, M., Kolouri, S., McClelland, J., and Pilly, P.
  Generative continual concept learning.
  arXiv preprint arXiv:1906.03744 (2019).
* [34]

  Saxe, A. M., Bansal, Y., Dapello, J., Advani, M., Kolchinsky, A., Tracey,
  B. D., and Cox, D. D.
  On the information bottleneck theory of deep learning.
  Journal of Statistical Mechanics: Theory and Experiment 2019,
  12 (2019), 124020.
* [35]

  Saxe, A. M., McClelland, J. L., and Ganguli, S.
  Exact solutions to the nonlinear dynamics of learning in deep linear
  neural networks.
  arXiv preprint arXiv:1312.6120 (2013).
* [36]

  Saxe, A. M., McClelland, J. L., and Ganguli, S.
  A mathematical theory of semantic development in deep neural
  networks.
  Proceedings of the National Academy of Sciences 116, 23 (2019),
  11537–11546.
* [37]

  Shah, H., Tamuly, K., Raghunathan, A., Jain, P., and Netrapalli, P.
  The pitfalls of simplicity bias in neural networks.
  arXiv preprint arXiv:2006.07710 (2020).
* [38]

  Shwartz-Ziv, R., and Tishby, N.
  Opening the black box of deep neural networks via information.
  arXiv preprint arXiv:1703.00810 (2017).
* [39]

  Storrs, K. R., Kietzmann, T. C., Walther, A., Mehrer, J., and
  Kriegeskorte, N.
  Diverse deep neural networks all predict human it well, after
  training and fitting.
  bioRxiv (2020).
* [40]

  Ulyanov, D., Vedaldi, A., and Lempitsky, V.
  Deep image prior.
  In Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition (2018), pp. 9446–9454.
* [41]

  Valle-Pérez, G., Camargo, C. Q., and Louis, A. A.
  Deep learning generalizes because the parameter-function map is
  biased towards simple functions.
  Proceedings of the International Conference on Learning
  Representations (2019).
* [42]

  `torchvision` contributors.
  PyTorch implementations of AlexNet, VGG16, ResNet-50.
  <https://pytorch.org/docs/stable/torchvision/models.html>.
* [43]

  Xu, Y., and Vaziri-Pashkam, M.
  Limited correspondence in visual representation between the human
  brain and convolutional neural networks.
  bioRxiv (2020).
* [44]

  Yamins, D. L., Hong, H., Cadieu, C. F., Solomon, E. A., Seibert, D., and
  DiCarlo, J. J.
  Performance-optimized hierarchical models predict neural responses in
  higher visual cortex.
  Proceedings of the National Academy of Sciences 111, 23 (2014),
  8619–8624.
* [45]

  Zemel, R., Wu, Y., Swersky, K., Pitassi, T., and Dwork, C.
  Learning fair representations.
  In International Conference on Machine Learning (2013),
  pp. 325–333.
* [46]

  Zenke, F., Poole, B., and Ganguli, S.
  Continual learning through synaptic intelligence.
  In Proceedings of the 34th International Conference on Machine
  Learning-Volume 70 (2017), JMLR. org, pp. 3987–3995.
* [47]

  Zhang, B. H., Lemoine, B., and Mitchell, M.
  Mitigating unwanted biases with adversarial learning.
  In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics,
  and Society (2018), pp. 335–340.

Supplementary Material for
  
“What shapes feature representations?
  
Exploring datasets, architectures, and training”

## Appendix A Supplemental Figures

!(/html/2006.12433/assets/figures/navon_experiments/navon_custom_resnet50_trained_vs_untrained_decoding.png)

Figure A.1: Feature decodability in models with a ResNet-50 architecture trained on the Navon dataset. Accuracy decoding features (shape, texture) from an untrained model (left) versus from shape- (center) and texture-trained (right) models. Results corresponding to trained models are mean across models trained on 5 cv splits. Chance = 123123\frac{1}{23} = 4.3% (dashed line). Target features are enhanced relative to the untrained model, whereas non-target features are suppressed.

!(/html/2006.12433/assets/figures/cst_experiments/cst_custom_resnet50_trained_vs_untrained_decoding.png)

Figure A.2: Non-target features are suppressed in the post-pool layer of models with a ResNet-50 architecture trained on the Trifeature dataset. Accuracy decoding features (shape, texture, color) for models trained to classify shape (left), texture (center), or color (right). Chance = 1717\frac{1}{7} = 14.3%. As observed with the Navon dataset and in the AlexNet models, non-target features are suppressed in trained models relative to the untrained model.

!(/html/2006.12433/assets/figures/cst_experiments/cst_custom_resnet50_difference_by_prob_decoding.png)

Figure A.3: Non-target features that are correlated with the target feature are suppressed in ResNet-50. For both datasets in which shape and color (upper row) and shape and texture (lower row) are correlated, target features are enhanced (left column), whereas non-target features correlated with the the target feature (“correlated non-target feature”, right column) are suppressed. As observed in experiments using an AlexNet architecture, suppression of the correlated non-target feature is largely constant across correlation strengths (x axis).

!(/html/2006.12433/assets/figures/cst_experiments/cst_custom_alexnet_accuracy_by_prob_decoding.png)

Figure A.4: Decoding accuracy for models with an AlexNet architecture trained on the Trifeature (Correlated) datasets. See Figure [5](#S4.F5 "Figure 5 ‣ 4.2 What if one feature perfectly predicts the label, but another only partially predicts it? ‣ 4 What if multiple features are predictive? ‣ What shapes feature representations? Exploring datasets, architectures, and training") for these results plotted in terms of enhancement/suppression relative to an untrained model.

!(/html/2006.12433/assets/figures/cst_experiments/cst_custom_resnet50_accuracy_by_prob_decoding.png)

Figure A.5: Decoding accuracy for models with a ResNet-50 architecture trained on the Trifeature (Correlated) datasets. as a function of the degree of correlation of the non-target feature with the target feature in See Figure [A.3](#A1.F3 "Figure A.3 ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training") for these results plotted in terms of enhancement/suppression relative to an untrained model.

!(/html/2006.12433/assets/figures/cst_experiments/cst_custom_resnet50_accuracy_perfect_correlation_decoding.png)

Figure A.6: When two features redundantly predict the label, models with a ResNet-50 architecture preferentially learn one feature. Color is more decodable than shape, and shape is more decodable than texture, when ResNet-50 is trained on perfectly predictive pairs, consistent with the pattern we observed in AlexNet (Figure [4](#S4.F4 "Figure 4 ‣ 4.1 What if two features redundantly predict the label? ‣ 4 What if multiple features are predictive? ‣ What shapes feature representations? Exploring datasets, architectures, and training")).

!(/html/2006.12433/assets/figures/cst_experiments/cst_custom_alexnet_difference_perfect_correlation_decoding.png)

Figure A.7: Models with an AlexNet architecture sometimes suppress features that perfectly predict the label. Models trained on a dataset in which shape and color redundantly predict the label suppress shape in pool3 and fc6 relative to an untrained model. Models trained on a dataset in which shape and texture predict the label suppress texture in pool3.

!(/html/2006.12433/assets/figures/cst_experiments/cst_custom_resnet50_difference_perfect_correlation_decoding.png)

Figure A.8: Models with a ResNet-50 architecture sometimes suppress features that perfectly predict the label. Models trained on a dataset in which shape and color redundantly predict the label suppress shape in the post-pool layer relative to an untrained model. For reasons of computational resources, we did not decode from the pre-pool layer.

### A.1 Binary feature tasks

!(/html/2006.12433/assets/figures/toy_experiments/toy_decoding_suppression_timecourse.png)

Figure A.9: The dynamics of feature learning when the easy feature has relatively low predictivity (0.65) and the hard feature has high predictivity (0.9). The easy feature is decodable above chance (∼75%similar-toabsentpercent75\sim 75\%, chance is 50%) before training (epoch 0), while the difficult feature is not. Perhaps because of this, the easy feature is learned first, and test performance on this feature, as well as its decodability, spike early. As the more difficult feature is learned, the decodability and use of the easier feature declines, although not to chance, and the more difficult and predictive feature is still decodable and used less than 80% of the time. (Averages across 5 runs.)

!(/html/2006.12433/assets/figures/toy_experiments/toy_decoding_nonlinear_values.png)

(a) Nonlinear decoder accuracy.

!(/html/2006.12433/assets/figures/toy_experiments/toy_decoding_nonlinear_advantage.png)

(b) Advantage over linear decoder.

Figure A.10: Nonlinear decoders trained on a larger dataset (2048 examples) are still unable to recover the more difficult feature when it is suppressed by the easier feature. ([10(a)](#A1.F10.sf1 "In Figure A.10 ‣ A.1 Binary feature tasks ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")) The nonlinear decoding accuracy of the difficult feature still declines drastically when the easy feature is more predictive. ([10(b)](#A1.F10.sf2 "In Figure A.10 ‣ A.1 Binary feature tasks ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")) While the nonlinear decoders do have some advantage over linear decoders (trained with the same 2048 examples), particularly when the easy feature has moderately high predictivity, the magnitude of the advantage is small. (Results are from same models reported in Fig. [6](#S4.F6 "Figure 6 ‣ 4.3 Do models prefer reliable but difficult features, or easy but less reliable ones? ‣ 4 What if multiple features are predictive? ‣ What shapes feature representations? Exploring datasets, architectures, and training"), with only the decoders trained on the larger dataset. The nonlinear decoders were a 2-layer fully connected network, with 64 hidden units. Panel [10(a)](#A1.F10.sf1 "In Figure A.10 ‣ A.1 Binary feature tasks ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training") contains a linear model fit, while panel [10(b)](#A1.F10.sf2 "In Figure A.10 ‣ A.1 Binary feature tasks ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training") contains a loess curve.)

### A.2 RSA

!(/html/2006.12433/assets/figures/navon_experiments/navon_basic_rsa.png)

(a) Similarity by architecture and task pairing.

!(/html/2006.12433/assets/figures/navon_experiments/navon_RSA_by_tasks.png)

(b) Details of cross-task similarity.

Figure A.11: Two views of the RSA results on the Navon tasks. ([11(a)](#A1.F11.sf1 "In Figure A.11 ‣ A.2 RSA ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")) The results across the two architectures we considered, and different possible training task pairings: two different training tasks, an untrained model vs. a trained one, and two models trained on the same task. In general, whether models are trained on the same task significantly drives RSA results, although untrained models do have some similarity to trained models. ([11(b)](#A1.F11.sf2 "In Figure A.11 ‣ A.2 RSA ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")) Representational similarity between all models and layers on each possible pair of tasks. While similarity is highest between models trained on the same task, the magnitude of that similarity varies across tasks. For instance, two texture-trained models are much less similar to one another than shape trained models. See Fig. [7](#S5.F7 "Figure 7 ‣ 5 What affects the representational similarity between models? ‣ What shapes feature representations? Exploring datasets, architectures, and training") for equivalent analyses on the trifeature tasks. The overall results are similar, but the Navon dataset shows slightly higher similarity within models than the trifeature dataset, and slightly more of an effect of architecture on representational similarity.
(Results are from 5 runs per condition.)

!(/html/2006.12433/assets/figures/cst_experiments/cst_corr_features_task_similarity.png)

Figure A.12: The effect of feature correlations (between shape and texture) on representational similarity in the trifeature dataset. The representational similarity analysis is most sensitive to the target feature, even when another feature is relatively strongly correlated with it. (Results from 5 runs per condition, with matched architecture and layer.)

!(/html/2006.12433/assets/figures/toy_experiments/toy_task_rsa_rsa_correlation.png)

(a) RSA with correlation distance similarity.

!(/html/2006.12433/assets/figures/toy_experiments/toy_task_rsa_rsa_euclidean.png)

(b) RSA with Euclidean similarity.

!(/html/2006.12433/assets/figures/toy_experiments/toy_task_rsa_cka.png)

(c) CKA.

Figure A.13: Further representation analyses for the toy tasks. Here we expand the results from the main text figure Fig. [8](#S5.F8 "Figure 8 ‣ 5 What affects the representational similarity between models? ‣ What shapes feature representations? Exploring datasets, architectures, and training"), including both comparisons to intermediate easy feature predictivity values, and different analysis approaches. The results for intermediate easy feature predictivities interpolate between the cases shown in the main text, but this interpolation reflects the bias towards the easier features. In the main text (Fig. [8](#S5.F8 "Figure 8 ‣ 5 What affects the representational similarity between models? ‣ What shapes feature representations? Exploring datasets, architectures, and training")) we used RSA with correlation distance as the metric. We show an expanded version of this figure in [13(a)](#A1.F13.sf1 "In Figure A.13 ‣ A.2 RSA ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")). We also show that ([13(b)](#A1.F13.sf2 "In Figure A.13 ‣ A.2 RSA ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")) RSA with Euclidean distance as the metric, and ([13(c)](#A1.F13.sf3 "In Figure A.13 ‣ A.2 RSA ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")) CKA [[22](#bib.bib22)] both yield quite similar patterns of results. Our conclusions do not appear to be limited to the particular analysis we considered in the main text.

| A | B |
| --- | --- |
| 0 | 0 |
| 0 | 1 |
| 1 | 0 |
| 1 | 1 |

BABAABA∧\wedgeBA∨\veeBA⊕direct-sum\oplusBA∨\veeBA∧\wedgeBRDM[112011]matrixmissing-subexpression112missing-subexpressionmissing-subexpression01missing-subexpressionmissing-subexpressionmissing-subexpression1\begin{bmatrix}\phantom{11}&1&1&\sqrt{2}\\
&&0&1\\
&&&1\\
&&&\\
\end{bmatrix}ABA∧\wedge¬\negB¬\negA∧\wedgeBA⊕direct-sum\oplusBA∧\wedge¬\negB¬\negA∧\wedgeBRDM[110211]matrixmissing-subexpression110missing-subexpressionmissing-subexpression21missing-subexpressionmissing-subexpressionmissing-subexpression1\begin{bmatrix}\phantom{11}&1&1&0\\
&&\sqrt{2}&1\\
&&&1\\
&&&\\
\end{bmatrix}Correlation: -0.8

Figure A.14: An intuitive example of why representational similarity might be lower on nonlinear tasks — there are multiple solutions resulting in different RDMs. We consider two possible ways that a network with two hidden units could compute the XOR of two binary inputs: either by an AND and an OR (top row), or by units that select each of the valid combinations (bottom row). When the inputs are passed through these networks, and representations are computed at the hidden layer, the patterns are different. In fact, when the representational dissimilarity matrices are computed, and then the correlation is taken between the two networks, it is actually negative! This raises the question of why positive representational similarities are observed at all between different networks computing XOR. The answer likely lies in overparameterization — a very large hidden layer will likely contain units which partially represent both solutions. However, the network will still likely favor one or the other, thus resulting in lower correlations between RDMs than on simpler tasks. (This will likely be exacerbated by other features being partially represented, and so on.)

## Appendix B Methods

### B.1 Decoding

#### B.1.1 Decoding from vision models (Sections [3](#S3 "3 Does feature selection happen by enhancement or suppression? ‣ What shapes feature representations? Exploring datasets, architectures, and training") and [4](#S4 "4 What if multiple features are predictive? ‣ What shapes feature representations? Exploring datasets, architectures, and training"))

Layer definitions. For AlexNet, we decoded from “pool3” (the output of the final convolutional layer, including the max pool), “fc6” (the first linear layer of the classifier, including the ReLU), or “fc7” (the second linear layer of the classifier, incluing the ReLU). For ResNet-50, we decoded from “pre-pool” (the output of the final convolutional layer prior to the global average pool) and “post-pool” (after the global average pool).

Training procedure. The inputs to a decoder were activations from some layer of a trained, frozen model in response to images normalized by the statistics of the dataset on which the model had been trained (images were unnormalized when decoding from untrained models). We trained and validated decoders on either a version of the Trifeature dataset (when decoding from models trained on a Trifeature task) or a version of the Navon dataset; in both, sets of features were uncorrelated.

We trained decoders to minimize cross-entropy loss for 250 epochs using Adam optimization [[20](#bib.bib20)] and a batch size of 64 for each of 6 learning rates (10−6,10−4,10−3,10−2,10−1,1

superscript106superscript104superscript103superscript102superscript101110^{-6},10^{-4},10^{-3},10^{-2},10^{-1},1) and 6 weight decays (0,10−6,10−4,10−3,10−2,10−1

0superscript106superscript104superscript103superscript102superscript1010,10^{-6},10^{-4},10^{-3},10^{-2},10^{-1}). We selected the decoder with the highest validation accuracy (when using trained models, we had trained models on 5 cv-splits, so took the mean validation accuracy across the 5 trained models) over the training period over the set of hyperparameter combinations.

#### B.1.2 Decoding from models trained on binary features (Section [4.3](#S4.SS3 "4.3 Do models prefer reliable but difficult features, or easy but less reliable ones? ‣ 4 What if multiple features are predictive? ‣ What shapes feature representations? Exploring datasets, architectures, and training"))

Decoders were trained for 5000 epochs with a learning rate of 10−3superscript10310^{-3}. We did not use weight decay or search over hyperparameters for the binary feature datasets. For the nonlinear decoding analysis, we used decoders with a single hidden layer with 64 units and a leaky-rectifier nonlinearity, trained for 20000 epochs.

### B.2 Datasets

### B.3 Trifeature

In Fig. [B.1](#A2.F1 "Figure B.1 ‣ B.3 Trifeature ‣ Appendix B Methods ‣ What shapes feature representations? Exploring datasets, architectures, and training") we show sample stimuli illustrating the range of features in the trifeature dataset.

!(/html/2006.12433/assets/figures/cst_demo_stims/triangle_solid_red.png)

!(/html/2006.12433/assets/figures/cst_demo_stims/square_stripes_green.png)

!(/html/2006.12433/assets/figures/cst_demo_stims/plus_grid_blue.png)

!(/html/2006.12433/assets/figures/cst_demo_stims/circle_hexgrid_yellow.png)

!(/html/2006.12433/assets/figures/cst_demo_stims/tee_dots_pink.png)

!(/html/2006.12433/assets/figures/cst_demo_stims/rhombus_noise_cyan.png)

!(/html/2006.12433/assets/figures/cst_demo_stims/pentagon_triangles_purple.png)

!(/html/2006.12433/assets/figures/cst_demo_stims/star_zigzags_ocean.png)

!(/html/2006.12433/assets/figures/cst_demo_stims/fivesquare_rain_orange.png)

!(/html/2006.12433/assets/figures/cst_demo_stims/trapezoid_pluses_white.png)

Figure B.1: Sample images showing all colors, shapes, and textures used in the trifeature dataset.

#### B.3.1 Binary features

As described in the main text, the datasets were composed of inputs that were 32-element binary vectors, and outputs (labels) that were single binary scalars (except in the multi-task case). We divided the inputs into two domains of 16 inputs each, and the labels were probabilistically related to a feature extracted from each domain. Each feature had a predictivity (p​(label|feature)𝑝conditionallabelfeaturep\left(\text{label}|\text{feature}\right)). We sampled the datasets by first assigning a label of 1 to half of the inputs, and then for each domain flipping the label with probability 1−predictivity1predictivity1-\text{predictivity}, and then sampling a domain input uniformly from the set of inputs that matched that feature value. For example, if the label was 1, and we flipped the label for the XOR feature, we would sample an input where the first two inputs of the XOR domain had parity 0.

### B.4 Representational Similarity Analyses

For the Trifeature datasets, representational similarity analyses were performed using 10 examples per combination of features of the 100,000 uncorrelated images. These images would sometimes have some small overlap with the train and val sets, but it was difficult to exclude the sets for all images, and generally due to the small sizes of the train sets this overlap would be no more than a 10% of the RSA dataset. Similarly, for the Navon dataset, we performed the representational similarity analyses on all images. We found that RSA on only train or only validation data did not produce very different results between Navon models for which it was easy to make the comparison.

For the binary feature datasets, representational similarity analyses were performed on a new dataset sampled independently from the training and evaluation data for the models; the predictivity of the features in this dataset did not matter, since the labels are not used in RSA, only the patterns of activation produced by the inputs.

We also compared to CKA [[22](#bib.bib22)], as well as RSA with Euclidean distance as the RDM metric, and found similar results in both cases (Fig. [A.13](#A1.F13 "Figure A.13 ‣ A.2 RSA ‣ Appendix A Supplemental Figures ‣ What shapes feature representations? Exploring datasets, architectures, and training")), and similarly with Spearman rank correlation rather than Pearson between the RDMs.
