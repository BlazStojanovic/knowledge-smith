---
arxiv: '2303.05506'
authors:
- Alan Jeffares
- Tennison Liu
- Jonathan Crabbé
- Fergus Imrie
- Mihaela van der Schaar
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization
  and Specialization'
url: http://arxiv.org/abs/2303.05506v1
year: 2023
---

[2303.05506] TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization














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



# TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization

Alan Jeffares
  
University of Cambridge
  
aj659@cam.ac.uk
  
&Tennison Liu††footnotemark: 
  
University of Cambridge
  
tl522@cam.ac.uk
  
&Jonathan Crabbé
  
University of Cambridge
  
jc2133@cam.ac.uk
  
\ANDFergus Imrie
  
University of California, Los Angeles
  
imrie@ucla.edu
  
&Mihaela van der Schaar
  
University of Cambridge
  
Alan Turing Institute
  
mv472@cam.ac.uk
  
Equal contribution

###### Abstract

Despite their success with unstructured data, deep neural networks are not yet a panacea for structured tabular data. In the tabular domain, their efficiency crucially relies on various forms of regularization to prevent overfitting and provide strong generalization performance. Existing regularization techniques include broad modelling decisions such as choice of architecture, loss functions, and optimization methods. In this work, we introduce Tabular Neural Gradient Orthogonalization and Specialization (TANGOS), a novel framework for regularization in the tabular setting built on latent unit attributions. The gradient attribution of an activation with respect to a given input feature suggests how the neuron *attends* to that feature, and is often employed to interpret the predictions of deep networks. In TANGOS, we take a different approach and incorporate neuron attributions directly into training to encourage orthogonalization and specialization of *latent attributions* in a fully-connected network. Our regularizer encourages neurons to focus on sparse, non-overlapping input features and results in a set of diverse and specialized latent units. In the tabular domain, we demonstrate that our approach can lead to improved out-of-sample generalization performance, outperforming other popular regularization methods. We provide insight into why our regularizer is effective and demonstrate that TANGOS can be applied jointly with existing methods to achieve even greater generalization performance.

## 1 Introduction

Despite its relative under-representation in deep learning research, tabular data is ubiquitous in many salient application areas including medicine, finance, climate science, and economics. Beyond raw performance gains, deep learning provides a number of promising advantages over non-neural methods including multi-modal learning, meta-learning, and certain interpretability methods, which we expand upon in depth in [Appendix C](#A3 "Appendix C Motivation for Deep Learning on Tabular Data ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). Additionally, it is a domain in which general-purpose regularizers are of particular importance. Unlike areas such as computer vision or natural language processing, architectures for tabular data generally do not exploit the inherent structure in the input features (i.e. locality in images and sequential text, respectively) and lack the resulting inductive biases in their design. Consequentially, improvement over non-neural ensemble methods has been less pervasive. Regularization methods that implicitly or explicitly encode inductive biases thus play a more significant role. Furthermore, adapting successful strategies from the ensemble literature to neural networks may provide a path to success in the tabular domain (e.g. Wen et al., [2020](#bib.bib81)). Recent work in Kadra et al. ([2021](#bib.bib43)) has demonstrated that suitable regularization is essential to outperforming such methods and, furthermore, a balanced cocktail of regularizers results in neural network superiority.

Regularization methods employed in practice can be categorized into those that prevent overfitting through data augmentation (Krizhevsky et al., [2012](#bib.bib45); Zhang et al., [2018](#bib.bib85)), network architecture choices (Hinton et al., [2012](#bib.bib34); Ioffe & Szegedy, [2015](#bib.bib40)), and penalty terms that explicitly influence parameter learning (Hoerl & Kennard, [1970](#bib.bib35); Tibshirani, [1996](#bib.bib76); Jin et al., [2020](#bib.bib41)), to name just a few. While all such methods are unified in attempting to improve out-of-sample generalization, this is often achieved in vastly different ways. For example, L​1𝐿1L1 and L​2𝐿2L2 penalties favor sparsity and shrinkage, respectively, on model weights, thus choosing more parsimonious solutions. Data perturbation techniques, on the other hand, encourage smoothness in the system assuming that small perturbations in the input should not result in large changes in the output. Which method works best for a given task is generally not known *a priori* and considering different classes of regularizer is recommended in practice. Furthermore, combining multiple forms of regularization simultaneously is often effective, especially in lower data regimes (see e.g. Brigato & Iocchi, [2021](#bib.bib8) and Hu et al., [2017](#bib.bib38)).

![Refer to caption](/html/2303.05506/assets/figures/l-gos-p1.png)


Figure 1: TANGOS encourages specialization and orthogonalization. TANGOS penalizes neuron attributions during training. Here,   indicates strong positive attribution and   indicates strong negative attribution, while interpolating colors reflect weaker attributions. Neurons are regularized to be *specialized* (attend to sparser features) and *orthogonal* (attend to non-overlapping features).

Neuroscience research has suggested that neurons are both *selective* (Johnston & Dark, [1986](#bib.bib42)) and have *limited capacity* (Cowan et al., [2005](#bib.bib16)) in reacting to specific physiological stimuli. Specifically, neurons selectively choose to focus on a few chunks of information in the input stimulus. In deep learning, a similar concept, commonly described as a *receptive field*, is employed in convolutional layers (Luo et al., [2016](#bib.bib56)). Here, each convolutional unit has multiple filters, and each filter is only sensitive to specialized features in a local region. The output of the filter will activate more strongly if the feature is present. This stands in contrast to fully-connected networks, where the all-to-all relationships between neurons mean each unit depends on the entire input to the network. We leverage this insight to propose a regularization method that can encourage artificial neurons to be more specialized and orthogonal to each other.

Contributions. (1) Novel regularization method for deep tabular models. In this work, we propose TANGOS, a novel method based on regularizing neuron attributions. A visual depiction is given in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization").
Specifically, each neuron is more *specialized*, attending to sparse input features while its attributions are more *orthogonal* to those of other neurons. In effect, different neurons pay attention to non-overlapping subsets of input features resulting in better generalization performance. We demonstrate that this novel regularization method results in excellent generalization performance on tabular data when compared to other popular regularizers.
(2) Distinct regularization objective. We explore how TANGOS results in distinct emergent characteristics in the model weights. We further show that its improved performance is linked to increased diversity among weak learners in an ensemble of latent units, which is generally in contrast to existing regularizers.
(3) Combination with other regularizers. Based upon these insights, we demonstrate that deploying TANGOS *in tandem* with other regularizers can further improve generalization of neural networks in the tabular setting beyond that of any individual regularizer.

## 2 Related Work

Gradient Attribution Regularization.
A number of methods exist which incorporate a regularisation term to penalize the network gradients in some way.
Penalizing gradient attributions is a natural approach for achieving various desirable properties in a neural network. Such methods have been in use at least since Drucker & Le Cun ([1992](#bib.bib19)), where the authors improve robustness by encouraging invariance to small perturbations in the input space. More recently, gradient attribution regularization has been successfully applied across a broad range of application areas. Some notable examples include encouraging the learning of robust features in auto-encoders (Rifai et al., [2011](#bib.bib65)), improving stability in the training of generative adversarial networks (Gulrajani et al., [2017](#bib.bib28)), and providing robustness to adversarial perturbations (Moosavi-Dezfooli et al., [2019](#bib.bib58)). While many works have applied a shrinkage penalty (L2) to input gradients, Ross et al. ([2017a](#bib.bib66)) explore the effects of encouraging sparsity by considering an L1 penalty term. Gradient penalties may also be leveraged to compel a network to attend to particular human-annotated input features (Ross et al., [2017b](#bib.bib67)). A related line of work considers the use of gradient aggregation methods such as Integrated Gradients (Sundararajan et al., [2017](#bib.bib73)) and, typically, penalizes their deviation from a given target value (see e.g. Liu & Avci ([2019](#bib.bib52)) and Chen et al. ([2019](#bib.bib10))). In contrast to these works, we do not require manually annotated regions upon which we constrain the network to attend. Similarly, Erion et al. ([2021](#bib.bib21)) provide methods for encoding domain knowledge such as smoothness between adjacent pixels in an image. We note that while these works have investigated penalizing a predictive model’s output attributions, we are the first to regularize attributions on latent neuron activations. We provide an extended discussion of related works on neural network regularization more generally in [Appendix A](#A1 "Appendix A Extended Related Works ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization").

## 3 TANGOS

### 3.1 Problem Formulation

We operate in the standard supervised learning setting, with dXsubscript𝑑𝑋d\_{X}-dimensional input variables X∈𝒳⊆ℝdX𝑋𝒳superscriptℝsubscript𝑑𝑋X\in\mathcal{X}\subseteq\mathbb{R}^{d\_{X}} and target output variable Y∈𝒴⊆ℝ𝑌𝒴ℝY\in\mathcal{Y}\subseteq\mathbb{R}. Let PX​Ysubscript𝑃𝑋𝑌P\_{XY} denote the joint distribution between input and target variables. The goal of the supervised learning algorithm is to find a predictive model, fθ:𝒳→𝒴:subscript𝑓𝜃→𝒳𝒴f\_{\theta}:\mathcal{X}\rightarrow\mathcal{Y} with learnable parameters θ∈Θ𝜃Θ\theta\in\Theta. The predictive model belongs to a hypothesis space fθ∈ℋsubscript𝑓𝜃ℋf\_{\theta}\in\mathcal{H} that can map from the input space to the output space.

The predictive function is usually learned by optimizing a loss function ℒ:Θ→ℝ:ℒ→Θℝ\mathcal{L}:\Theta\rightarrow\mathbb{R} using *empirical risk minimization* (ERM). The empirical risk cannot be directly minimized since the data distribution PX​Ysubscript𝑃𝑋𝑌P\_{XY} is not known. Instead, we use a finite number of iid samples (x,y)∼PX​Ysimilar-to𝑥𝑦subscript𝑃𝑋𝑌(x,y)\sim P\_{XY}, which we refer to as the training data 𝒟={(xi,yi)}i=1N𝒟superscriptsubscriptsubscript𝑥𝑖subscript𝑦𝑖𝑖1𝑁\mathcal{D}=\{(x\_{i},y\_{i})\}\_{i=1}^{N}.

Once the predictive model is trained on 𝒟𝒟\mathcal{D}, it should ideally predict well on out-of-sample data generated from the same distribution. However, overfitting can occur if the hypothesis space ℋℋ\mathcal{H} is too complex and the sampling of training data does not fully represent the underlying distribution PX​Ysubscript𝑃𝑋𝑌P\_{XY}. Regularization is an approach that reduces the complexity of the hypothesis space so that more generalized functions are learned to explain the data. This leads to the following ERM:

|  |  |  |  |
| --- | --- | --- | --- |
|  | θ∗=arg​minθ∈Θ⁡1|𝒟|​∑(x,y)∈𝒟ℒ​(fθ​(x),y)+ℛ​(θ,x,y),superscript𝜃subscriptargmin𝜃Θ1𝒟subscript𝑥𝑦𝒟ℒsubscript𝑓𝜃𝑥𝑦ℛ𝜃𝑥𝑦\theta^{\*}=\operatorname\*{arg\,min}\_{\theta\in\Theta}\frac{1}{|\mathcal{D}|}\sum\_{(x,y)\in\mathcal{D}}\mathcal{L}(f\_{\theta}(x),y)+\mathcal{R}(\theta,x,y), |  | (1) |

that includes an additional regularization term ℛℛ\mathcal{R} which, generally, is a function of input x𝑥x, the label y𝑦y, the model parameters θ𝜃\theta, and reflects prior assumptions about the model. For example, L​1𝐿1L1 regularization reflects the belief that sparse solutions in parameter space are more desirable.

### 3.2 Neuron Attributions

Formally, attribution methods aim to uncover the importance of each input feature of a given sample to the prediction of the neural network. Recent works have demonstrated that feature attribution methods can be incorporated into the training process (Lundberg & Lee, [2017](#bib.bib55); Erion et al., [2021](#bib.bib21)). These *attribution priors* optimize attributions to have desirable characteristics, including interpretability as well as smoothness and sparsity in predictions. However, these methods have exclusively investigated *output* attributions, i.e., contributions of input features to the output of a model. To the best of our knowledge, we are the first work to investigate regularization of *latent attributions*.

![Refer to caption](/html/2303.05506/assets/x1.png)


Figure 2: Method illustration. TANGOS regularizes the gradients with respect to each of the latent units.

We rewrite our predictive function f𝑓f using function composition f=l∘g𝑓𝑙𝑔f=l\circ g. Here g:𝒳→ℋ:𝑔→𝒳ℋg:\mathcal{X}\rightarrow\mathcal{H} maps the input to a representation h=g​(x)∈ℋℎ𝑔𝑥ℋh=g(x)\in\mathcal{H}, where ℋ⊆ℝdHℋsuperscriptℝsubscript𝑑𝐻\mathcal{H}\subseteq\mathbb{R}^{d\_{H}} is a dHsubscript𝑑𝐻d\_{H}-dimensional latent space. Additionally, l:ℋ→𝒴:𝑙→ℋ𝒴l:\mathcal{H}\rightarrow\mathcal{Y} maps the latent representation to a label space y=l​(h)∈𝒴𝑦𝑙ℎ𝒴y=l(h)\in\mathcal{Y}. We let hi=gi​(x)subscriptℎ𝑖subscript𝑔𝑖𝑥h\_{i}=g\_{i}(x), for, i∈[dH]𝑖delimited-[]subscript𝑑𝐻i\in[d\_{H}] denote the it​hsuperscript𝑖𝑡ℎi^{th} neuron in the hidden layer of interest. Additionally, we use aji​(x)∈ℝsubscriptsuperscript𝑎𝑖𝑗𝑥ℝa^{i}\_{j}(x)\in\mathbb{R} to denote the attribution of the it​hsuperscript𝑖𝑡ℎi^{th} neuron w.r.t. the feature xjsubscript𝑥𝑗x\_{j}. With this notation, upper indices correspond to latent units and lower indices to features. In some cases, it will be convenient to stack all the feature attributions together in the attribution vector ai​(x)=[aji​(x)]j=1dX∈ℝdXsuperscript𝑎𝑖𝑥superscriptsubscriptdelimited-[]subscriptsuperscript𝑎𝑖𝑗𝑥𝑗1subscript𝑑𝑋superscriptℝsubscript𝑑𝑋a^{i}(x)=[a^{i}\_{j}(x)]\_{j=1}^{d\_{X}}\in\mathbb{R}^{d\_{X}}.

Attribution methods work by using gradient signals to evaluate the contributions of the input features. In the most simplistic setting:

|  |  |  |  |
| --- | --- | --- | --- |
|  | aji​(x)≡∂hi​(x)∂xj.subscriptsuperscript𝑎𝑖𝑗𝑥subscriptℎ𝑖𝑥subscript𝑥𝑗a^{i}\_{j}(x)\equiv\frac{\partial h\_{i}(x)}{\partial x\_{j}}. |  | (2) |

This admits a simple interpretation through a first-order Taylor expansion: if the input feature xjsubscript𝑥𝑗x\_{j} were to increase by some small number ϵ∈ℝ+italic-ϵsuperscriptℝ\epsilon\in\mathbb{R}^{+}, the neuron activation would change by ϵ⋅aji​(x)+𝒪​(ϵ2)⋅italic-ϵsubscriptsuperscript𝑎𝑖𝑗𝑥𝒪superscriptitalic-ϵ2\epsilon\cdot a^{i}\_{j}(x)+\mathcal{O}(\epsilon^{2}). The larger the absolute value of the gradient, the stronger the effect of a change in the input feature. We emphasize that our method is *agnostic* to the gradient attribution method, as different methods may be more appropriate for different tasks. For a comprehensive review of different methods, assumptions, and trade-offs, see Ancona et al. ([2017](#bib.bib2)). For completeness, we also note another category of attribution methods is built around *perturbations*: this class of methods evaluates contributions of individual features through repeated perturbations. Generally speaking, they are more computationally inefficient due to the multiple forward passes through the neural network and are difficult to include directly in the training objective.

### 3.3 Rewarding Orthogonalization and Specialization

The main contribution of this work is proposing regularization on neuron attributions. In the most general sense, any function of any neuron attribution method could be used as a regularization term, thus encoding prior knowledge about the properties a model should have.

Specifically, the regularization term is a function of the network parameters θ𝜃\theta and x𝑥x, i.e., ℛ​(θ,x)ℛ𝜃𝑥\mathcal{R}(\theta,x), and encourages prior assumptions on desired behavior of the learned function. Biological sensory neurons are highly specialized. For example, certain visual neurons respond to a specific set of visual features including edges and orientations within a single receptive field. They are thus highly *selective* with *limited capacity* to react to specific physiological stimuli (Johnston & Dark, [1986](#bib.bib42); Cowan et al., [2005](#bib.bib16)). Similarly, we hypothesize that neurons that are more specialized and pay attention to sparser signals should exhibit better generalization performance. We propose the following desiderata and corresponding regularization terms:

* •

  Specialization. The contribution of input features to the activation of a particular neuron should be sparse, i.e., ‖ai​(x)‖normsuperscript𝑎𝑖𝑥||a^{i}(x)|| is small for all i∈[dH]𝑖delimited-[]subscript𝑑𝐻i\in[d\_{H}] and x∈𝒳𝑥𝒳x\in\mathcal{X}. Intuitively, in higher-dimensional settings, a few features should account for a large percentage of total attributions while others are near zero, resulting in more *specialized* neurons. We write this as a regularization term for mini-batch training:

  |  |  |  |
  | --- | --- | --- |
  |  | ℒspec​(x)=1B​∑b=1B1dH​∑i=1dH∥ai​(xb)∥1,subscriptℒspec𝑥1𝐵superscriptsubscript𝑏1𝐵1subscript𝑑𝐻superscriptsubscript𝑖1subscript𝑑𝐻subscriptdelimited-∥∥superscript𝑎𝑖subscript𝑥𝑏1\mathcal{L}\_{\mathrm{spec}}(x)=\frac{1}{B}\sum\_{b=1}^{B}\frac{1}{d\_{H}}\sum\_{i=1}^{d\_{H}}\lVert a^{i}(x\_{b})\rVert\_{1},\\ |  |

  where b∈[B]𝑏delimited-[]𝐵b\in[B] is the batch index of xb∈𝒳subscript𝑥𝑏𝒳x\_{b}\in\mathcal{X} and ∥⋅∥1subscriptdelimited-∥∥⋅1\lVert\cdot\rVert\_{1} denotes the l1subscript𝑙1l\_{1} norm.
* •

  Orthogonalization. Different neurons should attend to non-overlapping subsets of input features given a particular input sample. To encourage this, we penalize the correlation between neuron attributions ρ​[ai​(x),aj​(x)]𝜌superscript𝑎𝑖𝑥superscript𝑎𝑗𝑥\rho[a^{i}(x),a^{j}(x)] for all i≠j𝑖𝑗i\neq j and x∈𝒳𝑥𝒳x\in\mathcal{X}. In other words, for each particular input, we want to discipline the latent units to attend to different aspects of the input. Then, expressing this as a regularization term for mini-batch training, we obtain:

  |  |  |  |
  | --- | --- | --- |
  |  | ℒorth​(x)=1B​∑b=1B1C​∑i=2dH∑j=1i−1ρ​[ai​(xb),aj​(xb)].subscriptℒorth𝑥1𝐵superscriptsubscript𝑏1𝐵1𝐶superscriptsubscript𝑖2subscript𝑑𝐻superscriptsubscript𝑗1𝑖1𝜌superscript𝑎𝑖subscript𝑥𝑏superscript𝑎𝑗subscript𝑥𝑏\mathcal{L}\_{\mathrm{orth}}(x)=\frac{1}{B}\sum\_{b=1}^{B}\frac{1}{C}\sum\_{i=2}^{d\_{H}}\sum\_{j=1}^{i-1}\rho\left[a^{i}(x\_{b}),a^{j}(x\_{b})\right]. |  |

  Here, C𝐶C is the number of pairwise correlations, C=dH⋅(dH−1)2𝐶⋅subscript𝑑𝐻subscript𝑑𝐻12C=\frac{d\_{H}\cdot(d\_{H}-1)}{2}, and ρ​[ai​(xb),aj​(xb)]∈[0,1]𝜌superscript𝑎𝑖subscript𝑥𝑏superscript𝑎𝑗subscript𝑥𝑏01\rho[a^{i}(x\_{b}),a^{j}(x\_{b})]\in[0,1] is calculated using the cosine similarity |ai⊺​(xb)​aj​(xb)|‖ai​(xb)‖2​‖aj​(xb)‖2superscript𝑎limit-from𝑖⊺subscript𝑥𝑏superscript𝑎𝑗subscript𝑥𝑏subscriptnormsuperscript𝑎𝑖subscript𝑥𝑏2subscriptnormsuperscript𝑎𝑗subscript𝑥𝑏2\frac{|a^{i\intercal}(x\_{b})\ a^{j}(x\_{b})|}{||a^{i}(x\_{b})||\_{2}||a^{j}(x\_{b})||\_{2}} where ∥⋅∥2subscriptdelimited-∥∥⋅2\lVert\cdot\rVert\_{2} denotes the l2subscript𝑙2l\_{2} norm.

These terms can be combined into a single regularization term and incorporated into the training objective. The resulting TANGOS regularizer can be expressed as:

|  |  |  |
| --- | --- | --- |
|  | ℛTANGOS​(x)=λ1​ℒspec​(x)+λ2​ℒorth​(x),subscriptℛTANGOS𝑥subscript𝜆1subscriptℒspec𝑥subscript𝜆2subscriptℒorth𝑥\mathcal{R}\_{\texttt{TANGOS}}(x)=\lambda\_{1}\mathcal{L}\_{\mathrm{spec}}(x)+\lambda\_{2}\mathcal{L}\_{\mathrm{orth}}(x), |  |

where λ1subscript𝜆1\lambda\_{1}, λ2subscript𝜆2\lambda\_{2} ∈ℝabsentℝ\in\mathbb{R} act as weighting terms. As this expression is computed using gradient signals, it can be efficiently implemented and minimized in any auto-grad framework.

## 4 *How* and *Why* Does TANGOS Work?

To the best of our knowledge, TANGOS is the only work to explicitly regularize latent neuron attributions. A natural question to ask is (1) How is TANGOS different from other regularization? While intuitively it makes sense to enforce *specialization* of each unit and *orthogonalization* between units, we empirically investigate if other regularizers can achieve similar effects, revealing that our method regularizes a unique objective. Having established that the TANGOS objective is unique, the next question is (2) Why does it work? To investigate this question, we frame the set of neurons as an ensemble, and demonstrate that our regularization improves diversity among *weak learners*, resulting in improved out-of-sample generalization.

### 4.1 TANGOS Regularizes a Unique Objective

TANGOS encourages generalization by explicitly decorrelating and sparsifying the attributions of latent units. A reasonable question to ask is if this is unique, or if other regularizers might achieve the same objective implicitly. Two alternative regularizers that one might consider are L​2𝐿2L2 weight regularization and D​r​o​p​o​u​t𝐷𝑟𝑜𝑝𝑜𝑢𝑡Dropout. Like TANGOS, weight regularization methods implicitly and partially penalize the gradients by shrinking the weights in the neural network. Additionally, D​r​o​p​o​u​t𝐷𝑟𝑜𝑝𝑜𝑢𝑡Dropout trains an ensemble of learners by forcing each neuron to be more independent. In Figure [3](#S4.F3 "Figure 3 ‣ 4.1 TANGOS Regularizes a Unique Objective ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we provide these results on the UCI temperature forecast dataset (Cho et al., [2020](#bib.bib12)), in which data from 25 weather stations in South Korea is used to predict next-day peak temperature. We train a fully connected neural network for each regularization method. Specifically, we plot ℒs​p​e​csubscriptℒ𝑠𝑝𝑒𝑐\mathcal{L}\_{spec} and ℒo​r​t​hsubscriptℒ𝑜𝑟𝑡ℎ\mathcal{L}\_{orth} for neurons in the penultimate layers and the corresponding generalization performance. We supply an extended selection of these results on additional datasets and regularizers in Appendix [J](#A10 "Appendix J Insights - Extended Results ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization").

First, we observe that TANGOS significantly decreases correlation between different neuron attributions while other regularization terms, in fact, increase them. For L​2𝐿2L2 weight regularization, this suggests that as the neural network weights are made smaller, the neurons increasingly attend to the same input features. A similar effect is observed for D​r​o​p​o​u​t𝐷𝑟𝑜𝑝𝑜𝑢𝑡Dropout - which has a logical explanation. Indeed, D​r​o​p​o​u​t𝐷𝑟𝑜𝑝𝑜𝑢𝑡Dropout creates redundancy by forcing each latent unit to be independent of others.
Naturally, this encourages individual neurons to attend to overlapping features. In contrast, TANGOS aims to achieve specialization, such that neurons pay attention to sparse, non-overlapping features.

Additionally, we note that no alternative regularizers achieve greater attribution sparsity. This does not come as a surprise for D​r​o​p​o​u​t𝐷𝑟𝑜𝑝𝑜𝑢𝑡Dropout, where the aim to induce redundancy in each neuron will naturally encourage individual neurons to attend to more features. While L​2𝐿2L2 does achieve a similar level of sparsity, this is paired with a high ℒo​r​t​hsubscriptℒ𝑜𝑟𝑡ℎ\mathcal{L}\_{orth} term indicating that, although the latent units do attend to sparse features, they appear to collapse to a solution in which they all attend to the same weighted subset of the input features. This, as we will discover in §4.2, is unlikely to be optimal for out-of-sample generalization.

Therefore, we conclude that the pairing of the specialization and orthogonality objectives in TANGOS regularizes a unique objective.

![Refer to caption](/html/2303.05506/assets/x2.png)

![Refer to caption](/html/2303.05506/assets/x3.png)

Figure 3: Comparison of regularization objectives. (Top) Generalization performance of key regularization techniques, (Bottom) corresponding neuron attributions evaluated on the test set. L​2𝐿2L2 and D​O𝐷𝑂DO can reduce overfitting, but neuron attributions are in fact becoming more correlated. TANGOS achieves the best generalization performance by penalizing a different objective.

### 4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units

Having established how TANGOS differs from existing regularization objectives, we now turn to answer *why* it works. In this section, we provide an alternative perspective on the effect of TANGOS regularization in the context of ensemble learning. A predictive model f​(x)𝑓𝑥f(x) may be considered as an ensemble model if it can be written in the form f​(x)=∑Tk∈𝒯αk​Tk​(x)𝑓𝑥subscriptsubscript𝑇𝑘𝒯subscript𝛼𝑘subscript𝑇𝑘𝑥f(x)=\sum\_{{T}\_{k}\in\mathcal{T}}\alpha\_{k}T\_{k}(x), where 𝒯𝒯\mathcal{T} represents a set of basis functions sometimes referred to as weak learners and the αksubscript𝛼𝑘\alpha\_{k}’s represent their respective scalar weights. It is therefore clear that each output of a typical neural network may be considered an ensemble predictor with every latent unit in its penultimate layer acting as a weak learner in their contribution to the model’s output. More formally, in this setting Tk​(x)subscript𝑇𝑘𝑥T\_{k}(x) is the activation of latent unit k𝑘k with respect to an input x𝑥x and αksubscript𝛼𝑘\alpha\_{k} is the subsequent connection to the output activation. With this in mind, we present the following definition.

###### Definition 4.1.

Consider an ensemble regressor f​(x)=∑Tk∈𝒯αk​Tk​(x)𝑓𝑥subscriptsubscript𝑇𝑘𝒯subscript𝛼𝑘subscript𝑇𝑘𝑥f(x)=\sum\_{{T}\_{k}\in\mathcal{T}}\alpha\_{k}T\_{k}(x) trained on 𝒟={(xi,yi)}i=1N𝒟superscriptsubscriptsubscript𝑥𝑖subscript𝑦𝑖𝑖1𝑁\mathcal{D}=\{(x\_{i},y\_{i})\}\_{i=1}^{N} where each (x,y)𝑥𝑦(x,y) is drawn randomly from PX​Ysubscript𝑃𝑋𝑌P\_{XY}. Additionally, the weights are constrained such that ∑kαk=1subscript𝑘subscript𝛼𝑘1\sum\_{k}\alpha\_{k}=1. Then, for a given input-label pair (x,y)𝑥𝑦(x,y), we define:

1. (a)

   The overall ensemble error as: Err=(f​(x)−y)2Errsuperscript𝑓𝑥𝑦2\mathrm{Err}=(f(x)-y)^{2}.
2. (b)

   The weighted errors of the weak learners as: Err¯=∑kαk​(Tk​(x)−y)2¯Errsubscript𝑘subscript𝛼𝑘superscriptsubscript𝑇𝑘𝑥𝑦2\overline{\mathrm{Err}}=\sum\_{k}\alpha\_{k}(T\_{k}(x)-y)^{2}.
3. (c)

   The ensemble diversity as: Div=∑kαk​(Tk​(x)−f​(x))2Divsubscript𝑘subscript𝛼𝑘superscriptsubscript𝑇𝑘𝑥𝑓𝑥2\mathrm{Div}=\sum\_{k}\alpha\_{k}(T\_{k}(x)-f(x))^{2}.

Intuitively, Err¯¯Err\overline{\text{Err}} provides a measure of the strength of the ensemble members while Div measures the diversity of their outputs. To understand the relationship between these two terms and the overall ensemble performance, we consider Proposition [3](#S4.E3 "Equation 3 ‣ Proposition 1 (Krogh & Vedelsby 1994). ‣ 4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization").

###### Proposition 1 (Krogh & Vedelsby [1994](#bib.bib46)).

The overall ensemble error for an input-label pair (x,y)𝑥𝑦(x,y) can be decomposed into the weighted errors of the weak learners and the ensemble diversity such that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Err=Err¯−Div.Err¯ErrDiv\mathrm{Err}=\overline{\mathrm{Err}}-\mathrm{Div}. |  | (3) |

![Refer to caption](/html/2303.05506/assets/x4.png)


Figure 4: Neuron Diversity. Overall ensemble error and decomposition in terms of diversity and average error of the weak learners. Note while all methods achieve low overall error, TANGOS is the only method that does so by increasing the diversity among the latent units.

This decomposition provides a fundamental insight into the success of ensemble methods: an ensemble’s overall error is reduced by decreasing the average error of the individual weak learners and increasing the diversity of their outputs. Successful ensemble methods explicitly increase ensemble diversity when training weak learners by, for example, sub-sampling input features (random forest, Breiman, [2001](#bib.bib7)), sub-sampling from the training data (bagging, Breiman, [1996](#bib.bib6)) or error-weighted input importance (boosting, Bühlmann, [2012](#bib.bib9)).

Returning to the specific case of neural networks, it is clear that TANGOS provides a similar mechanism of increasing diversity among the latent units that act as weak learners in the penultimate layer. By forcing the latent units to attend to sparse, uncorrelated selections of features, the learned ensemble is encouraged to produce diverse learners whilst maintaining coverage of the entire input space in aggregate. In Figure [4](#S4.F4 "Figure 4 ‣ 4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we demonstrate this phenomenon in practice by returning to the UCI temperature forecast regression task. We provide extended results in Appendix [J](#A10 "Appendix J Insights - Extended Results ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). We train a fully connected neural network with two hidden layers with the output layer weights constrained such that they sum to 1. We observe that regularizing with TANGOS increases diversity of the latent activations resulting in improved out-of-sample generalization. This is in contrast to other typical regularization approaches which also improve model performance, but exclusively by attempting to reduce the error of the individual ensemble members.
This provides additional motivation for applying TANGOS in the tabular domain, an area where traditional ensemble methods have performed particularly well.

Table 1: Stand-Alone Regularization. Comparison of regularizers on regression and classification in terms of test MSE and NLL. All models are trained on real-world datasets using 5-fold cross-validation and final evaluation reported on a held-out test set. Bold indicates the best performance. The average rank of each method across both regression and classification is included in the final row of the respective tables.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Baseline | L1 | L2 | DO | BN | IN | MU | TANGOS |
| Regression (Mean Squared Error) | | | | | | | | |
| FB | 0.037 | 0.081 | 0.029 | 0.060 | 0.699 | 0.043 | 0.147 | 0.032 |
| BH | 0.192 | 0.197 | 0.183 | 0.209 | 0.190 | 0.215 | 0.286 | 0.166 |
| WE | 0.118 | 0.096 | 0.099 | 0.097 | 0.090 | 0.101 | 0.146 | 0.093 |
| BC | 0.323 | 0.263 | 0.277 | 0.282 | 0.294 | 0.308 | 0.323 | 0.244 |
| WQ | 0.673 | 0.641 | 0.644 | 0.658 | 0.639 | 0.669 | 0.713 | 0.637 |
| SC | 0.422 | 0.408 | 0.411 | 0.423 | 0.410 | 0.434 | 0.547 | 0.387 |
| FF | 1.274 | 1.280 | 1.274 | 1.266 | 1.330 | 1.201 | 1.289 | 1.276 |
| PR | 0.624 | 0.611 | 0.580 | 0.592 | 0.647 | 0.591 | 0.745 | 0.573 |
| ST | 0.419 | 0.416 | 0.418 | 0.387 | 0.461 | 0.539 | 0.380 | 0.382 |
| AB | 0.345 | 0.319 | 0.332 | 0.312 | 0.348 | 0.355 | 0.366 | 0.325 |
| Avg Rank | 5.4 | 3.8 | 3.4 | 4.0 | 5.0 | 5.5 | 7.1 | 1.9 |
| Classification (Mean Negative Log-likelihood) | | | | | | | | |
| HE | 0.490 | 0.472 | 0.431 | 0.428 | 0.459 | 0.435 | 0.416 | 0.426 |
| BR | 0.074 | 0.070 | 0.070 | 0.078 | 0.080 | 0.071 | 0.095 | 0.069 |
| CE | 0.519 | 0.395 | 0.407 | 0.436 | 0.604 | 0.457 | 0.472 | 0.408 |
| CR | 0.464 | 0.405 | 0.402 | 0.456 | 0.460 | 0.481 | 0.448 | 0.369 |
| HC | 0.320 | 0.222 | 0.226 | 0.237 | 0.257 | 0.312 | 0.248 | 0.215 |
| AU | 0.448 | 0.442 | 0.385 | 0.405 | 0.549 | 0.479 | 0.478 | 0.379 |
| TU | 1.649 | 1.633 | 1.613 | 1.621 | 1.484 | 1.646 | 1.657 | 1.495 |
| EN | 1.040 | 1.040 | 1.042 | 1.058 | 1.098 | 1.072 | 1.065 | 0.974 |
| TH | 0.700 | 0.506 | 0.500 | 0.714 | 0.785 | 0.638 | 0.618 | 0.513 |
| SO | 0.606 | 0.238 | 0.382 | 0.567 | 0.484 | 0.540 | 0.412 | 0.371 |
| Avg Rank | 6.4 | 3.0 | 2.7 | 4.8 | 6.3 | 6.0 | 5.2 | 1.7 |

## 5 Experiments

In this section, we empirically evaluate TANGOS as a regularization method for improving generalization performance. We present our benchmark methods and training architecture, followed by extensive results on real-world datasets. There are a few main aspects that deserve empirical investigation, which we investigate in turn: ▶▶\blacktriangleright Stand-alone performance. §5.1 Comparing the performance of TANGOS, where the focus is on applying it as a stand-alone regularizer, to a variety of benchmarks on a suite of real-world datasets. ▶▶\blacktriangleright In tandem performance. §5.2 Motivated by our unique regularization objective and our analysis in §4, we demonstrate that applying TANGOS *in conjunction* with other regularizers can lead to even greater gains in generalization performance. ▶▶\blacktriangleright Modern architectures. §5.3 We evaluate performance on a state-of-the-art tabular architecture and compare to boosting. All experiments were run on NVIDIA RTX A4000 GPUs. Code is provided on Github111<https://github.com/alanjeffares/TANGOS>222<https://github.com/vanderschaarlab/TANGOS>.

TANGOS. We train TANGOS regularized models as described in [Algorithm 1](#alg1 "In Appendix F Approximation and Algorithm ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") in Appendix [F](#A6 "Appendix F Approximation and Algorithm ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). For the specialization parameter we search for λ1∈{1,10,100}subscript𝜆1110100\lambda\_{1}\in\{1,10,100\} and for the orthogonalization parameter we search for λ2∈{0.1,1}subscript𝜆20.11\lambda\_{2}\in\{0.1,1\}. For computational efficiency, we apply a sub-sampling scheme where 50 neuron pairs are randomly sampled for each input (for further details see Appendix [F](#A6 "Appendix F Approximation and Algorithm ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization")).

Benchmarks. We evaluate TANGOS against a selection of popular regularizer benchmarks. First, we consider weight decay methods L1 and L2 regularization, which sparsify and shrink the learnable parameters. For the regularizers coefficients, we search for λ∈{0.1,0.01,0.001}𝜆0.10.010.001\lambda\in\{0.1,0.01,0.001\} where regularization is applied to all layers.
Next, we consider Dropout (DO), with drop rate p∈{10%,25%,50%}𝑝percent10percent25percent50p\in\{10\%,25\%,50\%\}, and apply DO after every dense layer during training. We also consider implicit regularization in batch normalization (BN). Lastly, we evaluate data augmentation techniques Input Noise (IN), where we use additive Gaussian noise with mean 0 and standard deviation σ∈{0.1,0.05,0.01}𝜎0.10.050.01\sigma\in\{0.1,0.05,0.01\} and MixUp (MU). Furthermore, each training run applies early stopping with patience of 30 epochs. In all experiments, we use 5-fold cross-validation to train and validate each benchmark. We select the model which achieves the lowest validation error and provide a final evaluation on a held-out test set.

### 5.1 Generalization: Stand-Alone Regularization

For the first set of experiments, we are interested in investigating the individual regularization effect of TANGOS. To ensure a fair comparison, we evaluate the generalization performance on held-out test sets across a variety of datasets.

Datasets. We employ 202020 real-world tabular datasets from the UCI machine learning repository. Each dataset is split into 80%percent8080\% for cross-validation and the remaining 20%percent2020\% for testing. The splits are standardized on just the training data, such that features have mean 00 and standard deviation 111 and categorical variables are one-hot encoded. See Appendix [L](#A12 "Appendix L Dataset and Regularizer Details ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") for further details on the 202020 datasets used.

Training and Evaluation. To ensure a fair comparison, all regularizers are applied to an MLP with two ReLU-activated hidden layers, where each hidden layer has dH+1subscript𝑑𝐻1d\_{H}+1 neurons. The models are trained using Adam optimizer with a dataset-dependent learning rate from {0.01,0.001,0.0001}0.010.0010.0001\{0.01,0.001,0.0001\} and are trained for up to a maximum of 200200200 epochs. For regression tasks, we report the average Mean Square Error (MSE) and, on classification tasks, we report the average negative log-likelihood (NLL).

Results.
Table [1](#S4.T1 "Table 1 ‣ 4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") provides the benchmarking results for individual regularizers. We observe that TANGOS achieves the best performance on 10/20102010/20 of the datasets. We also observe that on 666 of the remaining datasets, TANGOS ranks second. This is also illustrated by the ranking plot in Appendix [H](#A8 "Appendix H Ranking Plot ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). There we also provide a table displaying standard errors. As several results have overlapping error intervals, we also assess the magnitude of improvement by performing a non-parametric Wilcoxon signed-rank sum test (Wilcoxon, [1992](#bib.bib82)) paired at the dataset level. We compare TANGOS to the best-performing baseline method (L2) as a one-tailed test for both the regression and classification results obtaining p-values of 0.006 and 0.026 respectively. This can be interpreted as strong evidence to suggest the difference is statistically significant in both cases.
Note that a single regularizer is seldom used by itself. In addition to a stand-alone method, it remains to be shown that TANGOS brings value when used with other regularization methods. This is explored in the next section.

### 5.2 Generalisation: In Tandem Regularization

Motivated by the insights described in §4, a natural next question is whether TANGOS can be applied in conjunction with existing regularization to unlock even greater generalization performance. In this set of experiments, we investigate this question.

Setup.
The setting for this experiment is identical to §5.1 except now we consider the six baseline regularizers *in tandem* with TANGOS. We examine if pairing our proposed regularizer with existing methods results in even greater generalization performance. We again run 5-fold cross-validation, searching over the same hyperparameters, with the final models evaluated on a held-out test set.

Results.
We summarize the aggregated results over the datasets for each of the six baseline regularizers in combination with TANGOS in Figure [5](#S5.F5 "Figure 5 ‣ 5.2 Generalisation: In Tandem Regularization ‣ 5 Experiments ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). Consistently across all regularizers in both the regression and the classification settings, we observe that adding TANGOS regularization improves test performance. We provide the full table of results in the supplementary material. We also note an apparent interaction effect between certain regularizers (i.e. input noise for regression and dropout for classification), where methods that seemed to not be particularly effective as stand-alone regularizers become the best-performing method when evaluated in tandem. The relationship between such regularizers provides an interesting direction for future work.

![Refer to caption](/html/2303.05506/assets/x5.png)


(a)

![Refer to caption](/html/2303.05506/assets/x6.png)


(b)

Figure 5: In Tandem Regularization. Aggregated errors across the 10 regression datasets (left) and the 10 classification datasets (right). In all cases, the addition of TANGOS provides superior performance over the standalone regularizer.

### 5.3 Closing the Gap on Boosting

In this experiment, we apply TANGOS regularization to a state-of-the-art deep learning architecture for tabular data (Gorishniy et al., [2021](#bib.bib25)) and evaluate its contribution towards producing competitive performance against leading boosting methods. We provide an extended description of this experiment in [Appendix B](#A2 "Appendix B Tabular Architectures and Boosting ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") and results in [Table 2](#S5.T2 "In 5.3 Closing the Gap on Boosting ‣ 5 Experiments ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). We find that TANGOS provides moderate gains in this setting, improving performance relative to state-of-the-art boosting methods. Although boosting approaches still match or outperform deep learning in this setting, in [Appendix C](#A3 "Appendix C Motivation for Deep Learning on Tabular Data ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") we argue that deep learning may also be worth pursuing in the tabular modality for its other distinct advantages.

Table 2: FT-Transformer Architecture and Boosting. Adding TANGOS regularization can contribute to closing the gap between state-of-the-art tabular architectures and leading boosting methods. We report mean accuracy ±plus-or-minus\pm{} standard deviation.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Setting | Dataset | FT-Transformer | | Boosting | |
| Baseline | + TANGOS | XGBoost | CatBoost |
| Default | Jannis | 0.714±0.002plus-or-minus0.7140.0020.714\pm{0.002} | 0.720±0.000plus-or-minus0.7200.000\mathbf{0.720}\pm{0.000} | 0.711±0.000plus-or-minus0.7110.0000.711\pm{0.000} | 0.724±0.001plus-or-minus0.7240.001\mathbf{0.724}\pm{0.001} |
| Higgs | 0.721±0.002plus-or-minus0.7210.0020.721\pm{0.002} | 0.723±0.000plus-or-minus0.7230.000\mathbf{0.723}\pm{0.000} | 0.717±0.000plus-or-minus0.7170.0000.717\pm{0.000} | 0.728±0.001plus-or-minus0.7280.001\mathbf{0.728}\pm{0.001} |
| Tuned | Jannis | 0.720±0.001plus-or-minus0.7200.0010.720\pm{0.001} | 0.727±0.001plus-or-minus0.7270.001\mathbf{0.727}\pm{0.001} | 0.724±0.000plus-or-minus0.7240.0000.724\pm{0.000} | 0.727±0.001plus-or-minus0.7270.001\mathbf{0.727}\pm{0.001} |
| Higgs | 0.727±0.002plus-or-minus0.7270.0020.727\pm{0.002} | 0.729±0.002plus-or-minus0.7290.002\mathbf{0.729}\pm{0.002} | 0.728±0.001plus-or-minus0.7280.0010.728\pm{0.001} | 0.729±0.002plus-or-minus0.7290.002\mathbf{0.729}\pm{0.002} |

## 6 Discussion

In this work, we have introduced TANGOS, a novel regularization method that promotes specialization and orthogonalization among the gradient attributions of the latent units of a neural network.
We showed *how* this regularization objective is distinct from other popular methods and motivated *why* it provides out-of-sample generalization. We empirically demonstrated TANGOS utility with extensive experiments.
This work raises several exciting avenues for future work including (1) developing TANGOS beyond the tabular setting (e.g. images), (2) investigating alternative efficient methods for achieving specialization and orthogonalization, (3) proposing other latent gradient attribution regularizers, (4) augmenting TANGOS for specific applications such as multi-modal learning or increased interpretability (see [Appendix C](#A3 "Appendix C Motivation for Deep Learning on Tabular Data ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization")).

## Acknowledgments

We thank the anonymous ICLR reviewers as well as members of the van der Schaar lab for many insightful comments and suggestions. Alan Jeffares is funded by the Cystic Fibrosis Trust. Tennison Liu would like to thank AstraZeneca for their sponsorship and support.
Fergus Imrie and Mihaela van der Schaar are supported by the National Science Foundation (NSF, grant number 1722516). Mihaela van der Schaar is additionally supported by the Office of Naval Research (ONR).

## Reproducibility Statement

We have attempted to make our experimental results easily reproducible by both a detailed description of our experimental procedure and providing the code used to produce our results (<https://github.com/alanjeffares/TANGOS>). Experiments are described in Section [5](#S5 "5 Experiments ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") with further details in Appendices [F](#A6 "Appendix F Approximation and Algorithm ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") and [L](#A12 "Appendix L Dataset and Regularizer Details ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). All datasets used in this work can be freely downloaded from the UCI repository (Dua et al., [2017](#bib.bib20)) with specific details provided in Appendix [L](#A12 "Appendix L Dataset and Regularizer Details ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization").

## References

* Acosta et al. (2022)

  Julián N Acosta, Guido J Falcone, Pranav Rajpurkar, and Eric J Topol.
  Multimodal biomedical AI.
  *Nature Medicine*, 28(9):1773–1784, 2022.
* Ancona et al. (2017)

  Marco Ancona, Enea Ceolini, Cengiz Öztireli, and Markus Gross.
  Towards better understanding of gradient-based attribution methods
  for deep neural networks.
  *arXiv preprint arXiv:1711.06104*, 2017.
* Bahri et al. (2021)

  Dara Bahri, Heinrich Jiang, Yi Tay, and Donald Metzler.
  SCARF: Self-supervised contrastive learning using random feature
  corruption.
  *arXiv preprint arXiv:2106.15147*, 2021.
* Baldi et al. (2014)

  Pierre Baldi, Peter Sadowski, and Daniel Whiteson.
  Searching for exotic particles in high-energy physics with deep
  learning.
  *Nature Communications*, 5(1):1–9, 2014.
* Bansal et al. (2018)

  Nitin Bansal, Xiaohan Chen, and Zhangyang Wang.
  Can we gain more from orthogonality regularizations in training deep
  networks?
  *Advances in Neural Information Processing Systems*, 31, 2018.
* Breiman (1996)

  Leo Breiman.
  Bagging predictors.
  *Machine learning*, 24(2):123–140, 1996.
* Breiman (2001)

  Leo Breiman.
  Random forests.
  *Machine learning*, 45(1):5–32, 2001.
* Brigato & Iocchi (2021)

  Lorenzo Brigato and Luca Iocchi.
  A close look at deep learning with small data.
  In *2020 25th International Conference on Pattern Recognition
  (ICPR)*, pp.  2490–2497. IEEE, 2021.
* Bühlmann (2012)

  Peter Bühlmann.
  Bagging, boosting and ensemble methods.
  In *Handbook of computational statistics*, pp.  985–1022.
  Springer, 2012.
* Chen et al. (2019)

  Jiefeng Chen, Xi Wu, Vaibhav Rastogi, Yingyu Liang, and Somesh Jha.
  Robust attribution regularization.
  *Advances in Neural Information Processing Systems*, 32, 2019.
* Chen & Guestrin (2016)

  Tianqi Chen and Carlos Guestrin.
  Xgboost: A scalable tree boosting system.
  In *Proceedings of the 22nd acm sigkdd international conference
  on knowledge discovery and data mining*, pp.  785–794, 2016.
* Cho et al. (2020)

  Dongjin Cho, Cheolhee Yoo, Jungho Im, and Dong-Hyun Cha.
  Comparative assessment of various machine learning-based bias
  correction methods for numerical weather prediction model forecasts of
  extreme air temperatures in urban areas.
  *Earth and Space Science*, 7(4):e2019EA000740, 2020.
* Cortez & Morais (2007)

  Paulo Cortez and Aníbal de Jesus Raimundo Morais.
  A data mining approach to predict forest fires using meteorological
  data.
  2007.
* Cortez & Silva (2008)

  Paulo Cortez and Alice Maria Gonçalves Silva.
  Using data mining to predict secondary school student performance.
  2008.
* Cortez et al. (2009)

  Paulo Cortez, António Cerdeira, Fernando Almeida, Telmo Matos, and José
  Reis.
  Modeling wine preferences by data mining from physicochemical
  properties.
  *Decision support systems*, 47(4):547–553,
  2009.
* Cowan et al. (2005)

  Nelson Cowan, Emily M Elliott, J Scott Saults, Candice C Morey, Sam Mattox,
  Anna Hismjatullina, and Andrew RA Conway.
  On the capacity of attention: Its estimation and its role in working
  memory and cognitive aptitudes.
  *Cognitive psychology*, 51(1):42–100, 2005.
* Crabbé & van der Schaar (2022)

  Jonathan Crabbé and Mihaela van der Schaar.
  Label-free explainability for unsupervised models.
  *arXiv preprint arXiv:2203.01928*, 2022.
* Crabbé et al. (2021)

  Jonathan Crabbé, Zhaozhi Qian, Fergus Imrie, and Mihaela van der Schaar.
  Explaining latent representations with a corpus of examples.
  *Advances in Neural Information Processing Systems*,
  34:12154–12166, 2021.
* Drucker & Le Cun (1992)

  Harris Drucker and Yann Le Cun.
  Improving generalization performance using double backpropagation.
  *IEEE Transactions on Neural Networks*, 3(6):991–997, 1992.
* Dua et al. (2017)

  Dheeru Dua, Casey Graff, et al.
  UCI machine learning repository.
  2017.
* Erion et al. (2021)

  Gabriel Erion, Joseph D Janizek, Pascal Sturmfels, Scott M Lundberg, and Su-In
  Lee.
  Improving performance of deep learning models with axiomatic
  attribution priors and expected gradients.
  *Nature machine intelligence*, 3(7):620–631, 2021.
* Fernandes et al. (2017)

  Kelwin Fernandes, Jaime S Cardoso, and Jessica Fernandes.
  Transfer learning with partial observability applied to cervical
  cancer screening.
  In *Iberian conference on pattern recognition and image
  analysis*, pp.  243–250. Springer, 2017.
* Fisher & Schlimmer (1988)

  Douglas H Fisher and Jeffrey C Schlimmer.
  Concept simplification and prediction accuracy.
  In *Machine Learning Proceedings 1988*, pp.  22–28. Elsevier,
  1988.
* Gorishniy et al. (2022)

  Yura Gorishniy, Ivan Rubachev, and Artem Babenko.
  On embeddings for numerical features in tabular deep learning.
  *arXiv preprint arXiv:2203.05556*, 2022.
* Gorishniy et al. (2021)

  Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
  Revisiting deep learning models for tabular data.
  *Advances in Neural Information Processing Systems*,
  34:18932–18943, 2021.
* Grinsztajn et al. (2022)

  Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux.
  Why do tree-based models still outperform deep learning on tabular
  data?
  *arXiv preprint arXiv:2207.08815*, 2022.
* Grisoni et al. (2015)

  Francesca Grisoni, Viviana Consonni, Sara Villa, Marco Vighi, and Roberto
  Todeschini.
  Qsar models for bioconcentration: is the increase in the complexity
  justified by more accurate predictions?
  *Chemosphere*, 127:171–179, 2015.
* Gulrajani et al. (2017)

  Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C
  Courville.
  Improved training of Wasserstein GANs.
  *Advances in Neural Information Processing Systems*, 30, 2017.
* Guo et al. (2019)

  Wenzhong Guo, Jianwen Wang, and Shiping Wang.
  Deep multimodal representation learning: A survey.
  *IEEE Access*, 7:63373–63394, 2019.
* Guyon et al. (2019)

  Isabelle Guyon, Lisheng Sun-Hosoya, Marc Boullé, Hugo Jair Escalante,
  Sergio Escalera, Zhengying Liu, Damir Jajetic, Bisakha Ray, Mehreen Saeed,
  Michèle Sebag, et al.
  Analysis of the automl challenge series.
  *Automated Machine Learning*, pp.  177, 2019.
* Harrison Jr & Rubinfeld (1978)

  David Harrison Jr and Daniel L Rubinfeld.
  Hedonic housing prices and the demand for clean air.
  *Journal of environmental economics and management*, 5(1):81–102, 1978.
* He et al. (2015)

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  Delving deep into rectifiers: Surpassing human-level performance on
  imagenet classification.
  In *Proceedings of the IEEE international conference on computer
  vision*, pp.  1026–1034, 2015.
* He et al. (2022)

  Yuanqin He, Yan Kang, Jiahuan Luo, Lixin Fan, and Qiang Yang.
  A hybrid self-supervised learning framework for vertical federated
  learning.
  *arXiv preprint arXiv:2208.08934*, 2022.
* Hinton et al. (2012)

  Geoffrey E Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and
  Ruslan R Salakhutdinov.
  Improving neural networks by preventing co-adaptation of feature
  detectors.
  *arXiv preprint arXiv:1207.0580*, 2012.
* Hoerl & Kennard (1970)

  Arthur E Hoerl and Robert W Kennard.
  Ridge regression: Biased estimation for nonorthogonal problems.
  *Technometrics*, 12(1):55–67, 1970.
* Hoffmann et al. (2018)

  Georg Hoffmann, Andreas Bietenbeck, Ralf Lichtinghagen, and Frank Klawonn.
  Using machine learning techniques to generate laboratory diagnostic
  pathways—a case study.
  *J Lab Precis Med*, 3:58, 2018.
* Hospedales et al. (2021)

  Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey.
  Meta-learning in neural networks: A survey.
  *IEEE transactions on pattern analysis and machine
  intelligence*, 44(9):5149–5169, 2021.
* Hu et al. (2017)

  Guosheng Hu, Xiaojiang Peng, Yongxin Yang, Timothy M Hospedales, and Jakob
  Verbeek.
  Frankenstein: Learning deep face representations using small data.
  *IEEE Transactions on Image Processing*, 27(1):293–303, 2017.
* Hussain et al. (2018)

  Sadiq Hussain, Rasha Atallah, Amirrudin Kamsin, and Jiten Hazarika.
  Classification, clustering and association rule mining in educational
  datasets using data mining tools: A case study.
  In *Computer Science On-line Conference*, pp.  196–211.
  Springer, 2018.
* Ioffe & Szegedy (2015)

  Sergey Ioffe and Christian Szegedy.
  Batch normalization: Accelerating deep network training by reducing
  internal covariate shift.
  In *International conference on machine learning*, pp. 448–456. PMLR, 2015.
* Jin et al. (2020)

  Gaojie Jin, Xinping Yi, Liang Zhang, Lijun Zhang, Sven Schewe, and Xiaowei
  Huang.
  How does weight correlation affect generalisation ability of deep
  neural networks?
  *Advances in Neural Information Processing Systems*,
  33:21346–21356, 2020.
* Johnston & Dark (1986)

  William A Johnston and Veronica J Dark.
  Selective attention.
  *Annual review of psychology*, 37(1):43–75,
  1986.
* Kadra et al. (2021)

  Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka.
  Well-tuned simple nets excel on tabular datasets.
  *Advances in Neural Information Processing Systems*, 34, 2021.
* Kim et al. (2018)

  Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda
  Viegas, et al.
  Interpretability beyond feature attribution: Quantitative testing
  with concept activation vectors (tcav).
  In *International conference on machine learning*, pp. 2668–2677. PMLR, 2018.
* Krizhevsky et al. (2012)

  Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.
  Imagenet classification with deep convolutional neural networks.
  *Advances in Neural Information Processing Systems*, 25, 2012.
* Krogh & Vedelsby (1994)

  Anders Krogh and Jesper Vedelsby.
  Neural network ensembles, cross validation, and active learning.
  *Advances in Neural Information Processing Systems*, 7, 1994.
* Kukačka et al. (2017)

  Jan Kukačka, Vladimir Golkov, and Daniel Cremers.
  Regularization for deep learning: A taxonomy.
  *arXiv preprint arXiv:1710.10686*, 2017.
* LeCun et al. (1998)

  Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.
  Gradient-based learning applied to document recognition.
  *Proceedings of the IEEE*, 86(11):2278–2324, 1998.
* Lee et al. (2022)

  Changhee Lee, Fergus Imrie, and Mihaela van der Schaar.
  Self-supervision enhanced feature selection with correlated gates.
  In *International Conference on Learning Representations*, 2022.
* Levin et al. (2022)

  Roman Levin, Valeriia Cherepanova, Avi Schwarzschild, Arpit Bansal, C Bayan
  Bruss, Tom Goldstein, Andrew Gordon Wilson, and Micah Goldblum.
  Transfer learning with deep tabular models.
  *arXiv preprint arXiv:2206.15306*, 2022.
* Liang et al. (2022)

  Dong Liang, Jun Wang, Xiaoyu Gao, Jiahui Wang, Xiaoyong Zhao, and Lei Wang.
  Self-supervised pretraining isolated forest for outlier detection.
  In *2022 International Conference on Big Data, Information and
  Computer Network (BDICN)*, pp.  306–310. IEEE, 2022.
* Liu & Avci (2019)

  Frederick Liu and Besim Avci.
  Incorporating priors with feature attribution on text classification.
  *arXiv preprint arXiv:1906.08286*, 2019.
* Liu et al. (2021)

  Weiyang Liu, Rongmei Lin, Zhen Liu, James M Rehg, Liam Paull, Li Xiong,
  Le Song, and Adrian Weller.
  Orthogonal over-parameterized training.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition*, pp.  7251–7260, 2021.
* Loshchilov & Hutter (2017)

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  *arXiv preprint arXiv:1711.05101*, 2017.
* Lundberg & Lee (2017)

  Scott M Lundberg and Su-In Lee.
  A unified approach to interpreting model predictions.
  *Advances in Neural Information Processing Systems*, 30, 2017.
* Luo et al. (2016)

  Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard Zemel.
  Understanding the effective receptive field in deep convolutional
  neural networks.
  *Advances in Neural Information Processing Systems*, 29, 2016.
* Michalski et al. (1986)

  Ryszard S Michalski, Igor Mozetic, Jiarong Hong, and Nada Lavrac.
  The multi-purpose incremental learning system aq15 and its testing
  application to three medical domains.
  In *Proc. AAAI*, volume 1986, pp.  1–041, 1986.
* Moosavi-Dezfooli et al. (2019)

  Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Jonathan Uesato, and Pascal
  Frossard.
  Robustness via curvature regularization, and vice versa.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition*, pp.  9078–9086, 2019.
* Moro et al. (2016)

  Sérgio Moro, Paulo Rita, and Bernardo Vala.
  Predicting social media performance metrics and evaluation of the
  impact on brand building: A data mining approach.
  *Journal of Business Research*, 69(9):3341–3351, 2016.
* Paszke et al. (2019)

  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
  Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.
  PyTorch: An imperative style, high-performance deep learning
  library.
  *Advances in Neural Information Processing Systems*, 32, 2019.
* Pereyra et al. (2017)

  Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, and Geoffrey
  Hinton.
  Regularizing neural networks by penalizing confident output
  distributions.
  *arXiv preprint arXiv:1701.06548*, 2017.
* Prokhorenkova et al. (2018)

  Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush,
  and Andrey Gulin.
  Catboost: unbiased boosting with categorical features.
  *Advances in Neural Information Processing Systems*, 31, 2018.
* Quinlan (1987)

  J. Ross Quinlan.
  Simplifying decision trees.
  *International journal of man-machine studies*, 27(3):221–234, 1987.
* Ramachandram & Taylor (2017)

  Dhanesh Ramachandram and Graham W Taylor.
  Deep multimodal learning: A survey on recent advances and trends.
  *IEEE Signal Processing Magazine*, 34(6):96–108, 2017.
* Rifai et al. (2011)

  Salah Rifai, Pascal Vincent, Xavier Muller, Xavier Glorot, and Yoshua Bengio.
  Contractive auto-encoders: Explicit invariance during feature
  extraction.
  In *Icml*, 2011.
* Ross et al. (2017a)

  Andrew Ross, Isaac Lage, and Finale Doshi-Velez.
  The neural LASSO: Local linear sparsity for interpretable
  explanations.
  In *Workshop on Transparent and Interpretable Machine Learning
  in Safety Critical Environments, 31st Conference on Neural Information
  Processing Systems*, volume 4, 2017a.
* Ross et al. (2017b)

  Andrew Slavin Ross, Michael C Hughes, and Finale Doshi-Velez.
  Right for the right reasons: Training differentiable models by
  constraining their explanations.
  *arXiv preprint arXiv:1703.03717*, 2017b.
* Rubachev et al. (2022)

  Ivan Rubachev, Artem Alekberov, Yury Gorishniy, and Artem Babenko.
  Revisiting pretraining objectives for tabular deep learning.
  *arXiv preprint arXiv:2207.03208*, 2022.
* Seedat et al. (2023)

  Nabeel Seedat, Alan Jeffares, Fergus Imrie, and Mihaela van der Schaar.
  Improving adaptive conformal prediction using self-supervised
  learning.
  *arXiv preprint arXiv:2302.12238*, 2023.
* Shwartz-Ziv & Armon (2022)

  Ravid Shwartz-Ziv and Amitai Armon.
  Tabular data: Deep learning is not all you need.
  *Information Fusion*, 81:84–90, 2022.
* Street et al. (1993)

  W Nick Street, William H Wolberg, and Olvi L Mangasarian.
  Nuclear feature extraction for breast tumor diagnosis.
  In *Biomedical Image Processing and Biomedical Visualization*,
  volume 1905, pp.  861–870. SPIE, 1993.
* Sun et al. (2019)

  Baohua Sun, Lin Yang, Wenhan Zhang, Michael Lin, Patrick Dong, Charles Young,
  and Jason Dong.
  SuperTML: Two-dimensional word embedding for the precognition on
  structured tabular data.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition Workshops*, pp.  0–0, 2019.
* Sundararajan et al. (2017)

  Mukund Sundararajan, Ankur Taly, and Qiqi Yan.
  Axiomatic attribution for deep networks.
  In *International conference on machine learning*, pp. 3319–3328. PMLR, 2017.
* Tang et al. (2020)

  Michelle Tang, Pulkit Kumar, Hao Chen, and Abhinav Shrivastava.
  Deep multimodal learning for the diagnosis of autism spectrum
  disorder.
  *Journal of Imaging*, 6(6):47, 2020.
* Thompson et al. (2013)

  Joseph J Thompson, Mark R Blair, Lihan Chen, and Andrew J Henrey.
  Video game telemetry as a critical tool in the study of complex skill
  learning.
  *PloS one*, 8(9):e75129, 2013.
* Tibshirani (1996)

  Robert Tibshirani.
  Regression shrinkage and selection via the lasso.
  *Journal of the Royal Statistical Society: Series B
  (Methodological)*, 58(1):267–288, 1996.
* Ucar et al. (2021)

  Talip Ucar, Ehsan Hajiramezanali, and Lindsay Edwards.
  SubTab: Subsetting features of tabular data for self-supervised
  representation learning.
  *Advances in Neural Information Processing Systems*,
  34:18853–18865, 2021.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  *Advances in Neural Information Processing Systems*, 30, 2017.
* Wang & Sun (2022)

  Zifeng Wang and Jimeng Sun.
  TransTab: Learning transferable tabular transformers across tables.
  *Advances in Neural Information Processing Systems*, 2022.
  URL <https://openreview.net/forum?id=A1yGs_SWiIi>.
* Waugh (1995)

  Samuel George Waugh.
  *Extending and benchmarking Cascade-Correlation: extensions to
  the Cascade-Correlation architecture and benchmarking of feed-forward
  supervised artificial neural networks*.
  PhD thesis, University of Tasmania, 1995.
* Wen et al. (2020)

  Yeming Wen, Dustin Tran, and Jimmy Ba.
  BatchEnsemble: an alternative approach to efficient ensemble and
  lifelong learning.
  *arXiv preprint arXiv:2002.06715*, 2020.
* Wilcoxon (1992)

  Frank Wilcoxon.
  Individual comparisons by ranking methods.
  In *Breakthroughs in Statistics*, pp.  196–202. Springer,
  1992.
* Wu et al. (2022)

  Xinglong Wu, Mengying Li, Xin-wu Cui, and Guoping Xu.
  Deep multimodal learning for lymph node metastasis prediction of
  primary thyroid cancer.
  *Physics in Medicine & Biology*, 67(3):035008, 2022.
* Yoon et al. (2020)

  Jinsung Yoon, Yao Zhang, James Jordon, and Mihaela van der Schaar.
  VIME: Extending the success of self-and semi-supervised learning to
  tabular domain.
  *Advances in Neural Information Processing Systems*,
  33:11033–11043, 2020.
* Zhang et al. (2018)

  Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz.
  mixup: Beyond empirical risk minimization.
  In *International Conference on Learning Representations*, 2018.
* Zhang et al. (2021)

  Yu Zhang, Peter Tiňo, Aleš Leonardis, and Ke Tang.
  A survey on neural network interpretability.
  *IEEE Transactions on Emerging Topics in Computational
  Intelligence*, 2021.
* Zhu et al. (2021)

  Yitan Zhu, Thomas Brettin, Fangfang Xia, Alexander Partin, Maulik Shukla,
  Hyunseung Yoo, Yvonne A Evrard, James H Doroshow, and Rick L Stevens.
  Converting tabular data into images for deep learning with
  convolutional neural networks.
  *Scientific reports*, 11(1):11325, 2021.
* Zikeba et al. (2013)

  Maciej Zikeba, Jakub M Tomczak, Marek Lubicz, and Jerzy ’Swikatek.
  Boosted SVM for extracting rules from imbalanced data in
  application to prediction of the post-operative life expectancy in the lung
  cancer patients.
  *Applied Soft Computing*, 2013.
* Zou & Hastie (2005)

  Hui Zou and Trevor Hastie.
  Regularization and variable selection via the elastic net.
  *Journal of the royal statistical society: series B (statistical
  methodology)*, 67(2):301–320, 2005.

## Appendix A Extended Related Works

Neural Network Regularization. Regularization methods seek to penalize complexity and impose a form of smoothness on a model. This may be cast as expressing a prior belief over the hypothesis space of a neural network which attempts to aid generalization. ▶▶\blacktriangleright Categories. A vast array of regularization methods have been proposed throughout the literature (for a comprehensive taxonomy see e.g. Kukačka et al., [2017](#bib.bib47)). Modern nomenclature typically includes broad modeling decisions such as choice of architecture, loss function, and optimization method under the umbrella of regularization. Additionally, many regularization techniques augment the training data using methods such as input noise (Krizhevsky et al., [2012](#bib.bib45)) or MixUp (Zhang et al., [2018](#bib.bib85)). Dropout (Hinton et al., [2012](#bib.bib34)) and related approaches that augment a hidden representation of the input may also be included in this category. Possibly a more conventional category of regularization is that which adds explicit penalty term(s) to the loss function. These terms might penalize the network weights directly to shrink or sparsify their values as in L2 (Hoerl & Kennard, [1970](#bib.bib35)) and L1 (Tibshirani, [1996](#bib.bib76)), respectively. Alternatively, network outputs may be penalized to, for example, reduce overconfidence (Pereyra et al., [2017](#bib.bib61)).
▶▶\blacktriangleright Weight Orthogonalization. A number of works have studied the orthogonalization of network weights via various weight penalization methods (Bansal et al., [2018](#bib.bib5)). More recent work in Liu et al. ([2021](#bib.bib53)) proposed to learn an orthogonal transformation of the randomly initialized incoming weights to a given neuron. In contrast, this work seeks to ensure that the gradients of different latent neurons with respect to a given input vector are orthogonal.
▶▶\blacktriangleright Combination. Compositions of multiple regularization methods are extensively applied in practice. An early example in the regression setting is the elastic net penalty (Zou & Hastie, [2005](#bib.bib89)) which attempts to combine sparsity with shrinkage in the coefficients. More recent work has demonstrated the effectiveness of combining several regularization terms on tabular data (Kadra et al., [2021](#bib.bib43)), a domain in which neural networks superiority had previously been less convincing.

## Appendix B Tabular Architectures and Boosting

While non-neural methods such as XGBoost (Chen & Guestrin, [2016](#bib.bib11)) and CatBoost (Prokhorenkova et al., [2018](#bib.bib62)) are still considered state of the art for tabular data (Grinsztajn et al., [2022](#bib.bib26)), much progress has been made in recent years to close the gap. Furthermore, differing learning paradigms have various strengths and weaknesses outside of maximum generalization performance, which is often a consideration in practical applications. While boosting methods boast excellent computational efficiency and strong out-of-the-box performance, neural networks have unique utility in, for example, multi-modal learning (Ramachandram & Taylor, [2017](#bib.bib64)), meta-learning (Hospedales et al., [2021](#bib.bib37)) and certain interpretability methods (Zhang et al., [2021](#bib.bib86)).
In this section, we provide additional experiments applying TANGOS to a state-of-the-art transformer architecture for tabular data proposed in Gorishniy et al. ([2021](#bib.bib25)). Specifically, this architecture combines a Feature Tokenizer which transforms features into embeddings with a multi-layer Transformer (Vaswani et al., [2017](#bib.bib78)). We compare this FT-Transformer architecture to boosting methods in the default setting where we evaluate out-of-the-box performance and the tuned setting where we jointly optimize the Transformer along with its baseline regularizers. We describe these two settings in more detail next.

Default Setting. In this setting, we use a 3-layer Transformer with a 32-dimensional feature embedding size and 4 attention heads. Following the original paper we use Reglu activations, a hidden layer size of 43 corresponding to a ratio of 4343\frac{4}{3} with the embedding size, Kaiming initialization (He et al., [2015](#bib.bib32)), and AdamW optimizer (Loshchilov & Hutter, [2017](#bib.bib54)). Finally, we apply a learning rate of 0.001. We compare this architecture with and without TANGOS regularization applied which we refer to as “Baseline” and “+ TANGOS” respectively. We set λ1=1subscript𝜆11\lambda\_{1}=1 and λ2=0.01subscript𝜆20.01\lambda\_{2}=0.01 which were found to be reasonable default values for specialization and orthogonalization in our experiments in Section [5](#S5 "5 Experiments ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization").

Tuned Setting. Here we apply ten iterations of random search tuning over the same hyperparameters as in the original work with those achieving the best validation performance selected. We then evaluate this combination by training over three seeds and perform their final evaluations on a held-out test set. We search using the same distributions as in the original work and consider the following ranges. L2 regularization ∈[1​e−06,1​e−03]absent1𝑒061𝑒03\in[1e-06,1e-03], residual dropout ∈[0.0,0.2]absent0.00.2\in[0.0,0.2], hidden layer dropout ∈[0.0,0.5]absent0.00.5\in[0.0,0.5], attention dropout ∈[0.0,0.5]absent0.00.5\in[0.0,0.5], hidden layer to feature embedding dimension ratio ∈[1.0,3.0]absent1.03.0\in[1.0,3.0], embedding dimension ∈[16,48]absent1648\in[16,48], number of layers ∈[1,3]absent13\in[1,3], learning rate ∈[1​e−04,1​e−03]absent1𝑒041𝑒03\in[1e-04,1e-03]. In the “+ TANGOS” setting we also include λ1∈[0.001,10]subscript𝜆10.00110\lambda\_{1}\in[0.001,10] and λ2∈[0.0001,1]subscript𝜆20.00011\lambda\_{2}\in[0.0001,1] with a log uniform distribution. All remaining architecture choices are consistent with the default setting and the original work.

We ran our experiments on the Jannis (Guyon et al., [2019](#bib.bib30)) and Higgs (Baldi et al., [2014](#bib.bib4)) datasets. These are both classification datasets consisting of 837338373383733 and 980509805098050 examples respectively. These datasets were selected as they represent a significant number of input examples along with a middling number of input features relative to the other tabular datasets explored in this work (545454 and 282828 respectively). We follow the experimental protocol of the boosting comparison in Grinsztajn et al. ([2022](#bib.bib26)) using the same training, validation, and test splits and reporting mean test accuracy over three runs. Therefore we obtain the same results for boosting as reported in that work.

The results of this experiment are reported in Table [2](#S5.T2 "Table 2 ‣ 5.3 Closing the Gap on Boosting ‣ 5 Experiments ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") where we find that TANGOS does indeed have a positive effect on the FT-Transformer performance although, consistent with the original work, we found that regularization only provides modest gains at best with this architecture. While we do not claim that TANGOS regularization results in neural networks that outperform Boosting methods, these results indicate that TANGOS regularization can contribute to closing the gap and may play a key role when combined with other methods as highlighted in Kadra et al. ([2021](#bib.bib43)). We believe this to be an important area for future research and, in particular, expect that architecture-specific developments of the ideas presented in this work may provide further improvements on the results obtained in this section.

## Appendix C Motivation for Deep Learning on Tabular Data

Several works have argued that boosting methods generally achieve superior performance to even state-of-the-art deep learning architectures for tabular data (Grinsztajn et al., [2022](#bib.bib26); Shwartz-Ziv & Armon, [2022](#bib.bib70)). However, this is in contrast to recent findings for transformer style architectures in Gorishniy et al. ([2021](#bib.bib25)), especially with appropriate feature embeddings (Gorishniy et al., [2022](#bib.bib24)) and sufficient pretraining (Rubachev et al., [2022](#bib.bib68)). We defer from this discussion to highlight a selection of reasons to consider deep learning methods for tabular data beyond straightforward improvements in predictive performance. In particular, we include a number of deep learning paradigms that are difficult to analogize for non-neural models and have been successfully applied to tabular data.

Multi-modal learning refers to the task of modeling data inputs that consist of multiple data modalities (e.g. image, text, tabular). As one might intuit, jointly modeling these multiple modalities can result in better performance than independently predicting from each of them (Ramachandram & Taylor, [2017](#bib.bib64); Guo et al., [2019](#bib.bib29)). Deep learning provides a uniquely natural method of combining modalities with the advantages of (1) modality-specific encoders, (2) that are fused into a joint downstream representation and trained end-to-end with backpropagation, and (3) superior modeling performance in many modalities such as images and natural language. Healthcare is a domain in which multi-modal learning is particularly salient (Acosta et al., [2022](#bib.bib1)). Recent work in Wu et al. ([2022](#bib.bib83)) showed that jointly modeling tabular clinical records using an MLP together with medical images using a CNN outperforms the non-multi-modal baselines. Elsewhere in Tang et al. ([2020](#bib.bib74)), a multi-modal approach is taken in combining input modalities based on the preprocessing of functional magnetic resonance imaging and region of interest time series data for the diagnosis of autism spectrum disorder. A resnet-18 encodes one modality while an MLP encodes the other, resulting in superior performance when analyzed in an ablation study. In this setting, progress in modeling each of the individual modalities is likely to result in better performance of the system as a whole. Interestingly, Ramachandram & Taylor ([2017](#bib.bib64)) identified regularization techniques for improved cross-modality learning as an important research direction. We believe that further development of the ideas presented in this work could provide a powerful tool for balancing how models attend to multiple input modalities.

Meta-learning aims to distill the experience of multiple learning episodes across a distribution of related tasks to improve learning performance on future tasks (Hospedales et al., [2021](#bib.bib37)). Deep learning-based approaches have seen great success as a solution to this problem in a variety of fields. In the tabular domain, with careful consideration of the shared information between tasks, recent works have also shown promising results in this direction by developing methods for transferring deep tabular models across tables (Wang & Sun, [2022](#bib.bib79); Levin et al., [2022](#bib.bib50)). In particular, in Levin et al. ([2022](#bib.bib50)) it was noted that “representation learning with deep tabular models provides significant gains over strong GBDT baselines”, also finding that “the gains are especially pronounced in low data regimes”.

Interpretability is an important area of deep learning research aiming to provide users with the ability to understand and reason about model outputs. Certain classes of interpretability methods have recently been developed that provide distinct forms of interpretability relying on the hidden representations of neural networks. In such models, probing the representation space of a deep model permits a new type of interpretation. For instance, Kim et al. ([2018](#bib.bib44)) studies how human concepts are represented by deep classifiers. This makes it possible to analyze how the classes predicted by the model relate to human understandable concepts. For example, one can verify if the stripe concept is relevant for a CNN classifier to identify a zebra, as demonstrated in the paper. Another example is Crabbé et al. ([2021](#bib.bib18)), which proposes to explain a given example with reference to a freely selected set of other examples (potentially from the same dataset). A user study was carried out in this work which concluded that, among non-technical users, this method of explanation does affect their confidence in the model’s prediction. These powerful methods crucially rely on the model’s representation space, which effectively assumes that the model is a deep neural network.

Representation learning more generally provides access to several other methods from deep learning to the tabular domain. A number of works have used deep learning approaches to map inputs to embeddings which can be useful for downstream applications. SuperTML (Sun et al., [2019](#bib.bib72)) and Zhu et al. ([2021](#bib.bib87)) map tabular inputs to image-like embeddings that can therefore be passed to image architectures such as CNNs. Other self-supervised methods include VIME (Yoon et al., [2020](#bib.bib84)) which applies input reconstruction, SubTab (Ucar et al., [2021](#bib.bib77)) which suggests a multi-view reconstruction task and SCARF (Bahri et al., [2021](#bib.bib3)) which takes a contrastive approach. Representation learning approaches such as these have proven successful on downstream tabular data tasks such as uncertainty quantification (Seedat et al., [2023](#bib.bib69)), federated learning (He et al., [2022](#bib.bib33)), anomaly detection (Liang et al., [2022](#bib.bib51)), and feature selection (Lee et al., [2022](#bib.bib49)).

## Appendix D TANGOS Behavior Analysis

In this section, we apply TANGOS to a simple image classification task using a convolutional neural network (CNN) and provide a qualitative analysis of the behavior of the learned network. This analysis is conducted on the MNIST dataset (LeCun et al., [1998](#bib.bib48)) using the recommended split resulting in 60,000 training and 10,000 validation examples.

In this experiment, we train a standard CNN architecture (as described in Table [3](#A4.T3 "Table 3 ‣ Appendix D TANGOS Behavior Analysis ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization")) with a penultimate hidden layer of 10 neurons for 10 epochs with Adam optimizer and a learning rate of 0.001. We also apply L2 regularization with weight 0.001. After each epoch, the model is evaluated on the validation set where the epoch achieving the best validation performance is stored for further analysis. Two models are trained under this protocol. One model which applied TANGOS to the penultimate hidden layer with λ1=100subscript𝜆1100\lambda\_{1}=100, λ2=0.1subscript𝜆20.1\lambda\_{2}=0.1 and M=25𝑀25M=25 and a baseline model which does not apply TANGOS.

Table 3: MNIST Convolutional Neural Network Architecture.

Layer Type
Hyperparameters
Activation Function

Conv2d



Input Channels:1 ; Output Channels:16 ; Kernel Size:5 ; Stride:2 ; Padding:1
ReLU

Conv2d



Input Channels:16 ; Output Channels:32 ; Kernel Size:5 ; Stride:2 ; Padding:1
ReLU

Flatten
Start Dimension:1

Linear
Input Dimension: 512 ; Output Dimension: 256
ReLU

Linear
Input Dimension: 256 ; Output Dimension: 10

Linear
Input Dimension: 10 ; Output Dimension: 10

In this section, we examine the gradients of each of the 10 neurons in the penultimate hidden layer with respect to each of the input dimensions of a given image. TANGOS is designed to reward orthogonalization and specialization of these gradient attributions which can be evaluated qualitatively by inspection. In all plots that follow we apply a min-max scaling across all hidden units for a fair comparison. Both strong positive and strong negative values for attributions may be interpreted as a latent unit attending to a given input dimension. In Figure [6](#A4.F6 "Figure 6 ‣ Appendix D TANGOS Behavior Analysis ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") we provide results for the baseline model applied to a test image where, in line with similar analyses in previous works such as Crabbé & van der Schaar ([2022](#bib.bib17)), we note that the way in which hidden units attend to the input is highly entangled. In contrast to this, in Figure [7](#A4.F7 "Figure 7 ‣ Appendix D TANGOS Behavior Analysis ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we include the same plot for the TANGOS trained model on the same image. In this case, each hidden unit does indeed produce relatively sparse and orthogonal attributions as desired. These results were consistent across the test images.

![Refer to caption](/html/2303.05506/assets/x7.png)


Figure 6: Without TANGOS Training. Gradient attributions with respect to each of the 10 hidden neurons. These results suggest significant overlap among the gradient attributions.

![Refer to caption](/html/2303.05506/assets/x8.png)


Figure 7: With TANGOS Training. TANGOS encourages gradient attributions to be sparse with minimal overlap.

We can glean further insight into the TANGOS trained model by examining the role of individual neurons across multiple test images. In Figure [8](#A4.F8 "Figure 8 ‣ Appendix D TANGOS Behavior Analysis ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we provide the gradient attributions for hidden neuron 5 (H5) from our previous discussion across twelve test images. This neuron appears to discriminate between an open or a closed loop at the lower left of the digit. Indeed this is a key aspect of distinction between the set of digits {2,6,8,0}2680\{2,6,8,0\} (first row) and {9,5,3}953\{9,5,3\} (second row). We also include digits where this visual feature is less useful as they contain no lower-left loop either open or closed (third row). This hypothesis can be further examined by analyzing the values of these activations. We note that the first two rows typically have higher magnitude with opposite signs while the third row has lower magnitude activations. In Table [4](#A4.T4 "Table 4 ‣ Appendix D TANGOS Behavior Analysis ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") we summarize the effect of these activation scores on class probabilities by accounting for the weights connecting to each of the ten classes. As one might expect, the weight connections between the hidden neuron and classes on the first row and the second row have opposite signs indicating that neuron 5 does indeed discriminate between these classes.

![Refer to caption](/html/2303.05506/assets/x9.png)


Figure 8: Hidden Neuron 5. This neuron attempts to discriminate whether inputs contain a closed loop on the lower left of their digit. Inputs with a closed lower loop incur highly positive activations (first row). Inputs with open lower loops incur highly negative activations (second row). While ambiguous inputs with no lower loop at all tend to produce low-magnitude activations (third row).




Table 4: Neuron 5 Class Weights. Weights connecting neuron 5 to each of the ten classes and a summary of their combined effect with the neuron activation. The classification influence column provides a categorical indication of the magnitude of the contribution to each class output for a fixed activation magnitude. This is determined by the magnitude of the connecting weight where: Low ∈[0,0.59]absent00.59\in[0,0.59], Medium ∈[0.59,1.17]absent0.591.17\in[0.59,1.17], and High ∈[1.17,1.76]absent1.171.76\in[1.17,1.76].

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Class label | Weight connection | |  | | --- | | Increases class probability | | if activation is | | Classification influence |
| 0 | 1.0883 | Positive | Medium |
| 1 | -0.6826 | Negative | Medium |
| 2 | 1.5298 | Positive | High |
| 3 | -1.5862 | Negative | High |
| 4 | -0.2516 | Negative | Low |
| 5 | -0.6008 | Negative | Medium |
| 6 | 0.9065 | Positive | Medium |
| 7 | 0.3524 | Positive | Low |
| 8 | 0.7362 | Positive | Medium |
| 9 | -1.7608 | Negative | High |

## Appendix E Performance with Increasing Data Size

In this section, we evaluate TANGOS performance with an increasing number of input examples. To do this we use the Dionis dataset, which was the largest benchmark dataset proposed in Kadra et al. ([2021](#bib.bib43)) with 416,188 examples. As in that work, we set aside 20% for testing with the remaining data further split into 80% training and 20% validation. The data was standardized to have zero mean and unit variance with statistics calculated on the training data. We then consider using various proportions (10%, 50%, 100%) of the training data to train an MLP with and without TANGOS regularization. We also evaluate the best-performing regularization method, L2, from our experiments in Section [5](#S5 "5 Experiments ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). For both regularization methods, we train three hyperperameter settings at each proportion and evaluate the best performing of the three on the test set. For TANGOS we consider {(λ1=1,λ2=0.01),(λ1=1,λ2=0.1),(λ1=10,λ2=0.1)}formulae-sequencesubscript𝜆11subscript𝜆20.01formulae-sequencesubscript𝜆11subscript𝜆20.1formulae-sequencesubscript𝜆110subscript𝜆20.1\{(\lambda\_{1}=1,\lambda\_{2}=0.01),(\lambda\_{1}=1,\lambda\_{2}=0.1),(\lambda\_{1}=10,\lambda\_{2}=0.1)\} and for L2 we consider λ∈{0.01,0.001,0.0001}𝜆0.010.0010.0001\lambda\in\{0.01,0.001,0.0001\}. We repeat this procedure for 6 runs and report the mean test accuracy. The MLP contained three ReLU-activated hidden layers of 400, 100, and 10 hidden units, respectively.

We include the results of this experiment in Figure [9](#A5.F9 "Figure 9 ‣ Appendix E Performance with Increasing Data Size ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). Consistent with our experiments in Section [5](#S5 "5 Experiments ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we find that TANGOS outperforms both the baseline model and the strongest baseline regularization method across all proportions of the data. These results are indicative that TANGOS remains similarly effective across both small and large datasets in the tabular domain.

![Refer to caption](/html/2303.05506/assets/x10.png)


Figure 9: Performance Gains With Increasing Data Size. Training with various proportions of training data from the 416,188 examples of the Dionis dataset, we find the relative boost in performance from TANGOS to be consistent.

## Appendix F Approximation and Algorithm

Calculating the attribution of the latent units with respect to the input involves computing the Jacobian matrix, which can be computed in 𝒪​(1)𝒪1\mathcal{O}(1) time and has memory complexity 𝒪​(dH​dX)𝒪subscript𝑑𝐻subscript𝑑𝑋\mathcal{O}(d\_{H}d\_{X}). The computational complexity of calculating ℒo​r​t​hsubscriptℒ𝑜𝑟𝑡ℎ\mathcal{L}\_{orth} is 𝒪​(dH2)𝒪superscriptsubscript𝑑𝐻2\mathcal{O}(d\_{H}^{2}) (i.e. all pairwise computation between latent units). While the calculation can be efficiently parallelized, this still becomes impractically expensive with higher dimensional layers. To address this, we introduce a relaxation by randomly subsampling pairs of neurons to calculate attribution similarity. We denote by I𝐼I denote the set of all possible pairs of neuron indices, I={(i,j)​∀i,j∈[dH]​and​i≠j}𝐼

𝑖𝑗for-all𝑖𝑗
delimited-[]subscript𝑑𝐻and𝑖𝑗I=\{(i,j)\>\forall\>i,j\in[d\_{H}]\>\text{and}\>i\neq j\}. Further, we let M𝑀M denote a randomly sampled subset of I𝐼I, M⊆I𝑀𝐼M\subseteq I. We devise an approximation to the regularization term, denoted by ℒo​r​t​h′subscriptsuperscriptℒ′𝑜𝑟𝑡ℎ\mathcal{L}^{\prime}\_{orth}, by estimating the penalty on the subset M𝑀M, where the size of M𝑀M can be chosen to balance computational burden with more faithful estimation:

|  |  |  |
| --- | --- | --- |
|  | ℒo​r​t​h′​(x)=1B​∑b=1B1|M|​∑(i,j)∈Mρ​[ai​(xb),aj​(xb)]subscriptsuperscriptℒ′𝑜𝑟𝑡ℎ𝑥1𝐵superscriptsubscript𝑏1𝐵1𝑀subscript𝑖𝑗𝑀𝜌superscript𝑎𝑖subscript𝑥𝑏superscript𝑎𝑗subscript𝑥𝑏\mathcal{L}^{\prime}\_{orth}(x)=\frac{1}{B}\sum\_{b=1}^{B}\frac{1}{|M|}\sum\_{(i,j)\in M}\rho[a^{i}(x\_{b}),a^{j}(x\_{b})] |  |

This reduces the complexity of calculating ℒo​r​t​hsubscriptℒ𝑜𝑟𝑡ℎ\mathcal{L}\_{orth} from 𝒪​(dH2)𝒪superscriptsubscript𝑑𝐻2\mathcal{O}(d\_{H}^{2}) to 𝒪​(M)𝒪𝑀\mathcal{O}(M). For our experimental results described in [Tables 1](#S4.T1 "In 4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), [6](#A9.T6 "Table 6 ‣ Appendix I Stand-Alone Uncertainty ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") and [7](#A11.T7 "Table 7 ‣ Appendix K In Tandem Results ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we use |M|=50𝑀50|M|=50. We empirically demonstrate that this approximation still leads to strong results in real-world experiments. The overall training procedure is described in [Algorithm 1](#alg1 "In Appendix F Approximation and Algorithm ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization").

Algorithm 1  TANGOS regularization

Result:  Learned parameters θ𝜃\theta

Input:  λ1,λ2,training data ​𝒟,learning rate ​η

subscript𝜆1subscript𝜆2training data 𝒟learning rate 𝜂\lambda\_{1},\lambda\_{2},\textrm{training data }\mathcal{D},\textrm{learning rate }\eta;

Initialise θ𝜃\theta;

while not converged do

Sample 𝒟m​i​n​isubscript𝒟𝑚𝑖𝑛𝑖\mathcal{D}\_{mini} from 𝒟𝒟\mathcal{D};

ℒ^​(fθ​(x),y)=𝔼(x,y)∼𝒟m​i​n​i​[ℒ​(fθ​(x),y)]^ℒsubscript𝑓𝜃𝑥𝑦subscript𝔼similar-to𝑥𝑦subscript𝒟𝑚𝑖𝑛𝑖delimited-[]ℒsubscript𝑓𝜃𝑥𝑦\hat{\mathcal{L}}(f\_{\theta}(x),y)=\mathbb{E}\_{(x,y)\sim\mathcal{D}\_{mini}}[\mathcal{L}(f\_{\theta}(x),y)];

ℛ^​(x)=λ1​𝔼x∼𝒟m​i​n​i​[ℒs​p​e​c​(x)]+λ2​𝔼x∼𝒟m​i​n​i​[ℒo​r​t​h′​(x)]^ℛ𝑥subscript𝜆1subscript𝔼similar-to𝑥subscript𝒟𝑚𝑖𝑛𝑖delimited-[]subscriptℒ𝑠𝑝𝑒𝑐𝑥subscript𝜆2subscript𝔼similar-to𝑥subscript𝒟𝑚𝑖𝑛𝑖delimited-[]superscriptsubscriptℒ𝑜𝑟𝑡ℎ′𝑥\hat{\mathcal{R}}(x)=\lambda\_{1}\mathbb{E}\_{x\sim\mathcal{D}\_{mini}}[\mathcal{L}\_{spec}(x)]+\lambda\_{2}\mathbb{E}\_{x\sim\mathcal{D}\_{mini}}[\mathcal{L}\_{orth}^{\prime}(x)];

θ←θ+η​∇θ[ℒ^​(fθ​(x),y)+ℛ^​(x)]←𝜃𝜃𝜂subscript∇𝜃^ℒsubscript𝑓𝜃𝑥𝑦^ℛ𝑥\theta\leftarrow\theta+\eta\nabla\_{\theta}\left[\hat{\mathcal{L}}(f\_{\theta}(x),y)+\hat{\mathcal{R}}(x)\right];

end while

Additionally, we provide an empirical analysis of TANGOS designed to evaluate the effectiveness of our proposed subsampling approximation with respect to generalization performance and computational efficiency as the number of sampled neuron pairs M𝑀M grows. Furthermore, we analyze the computational efficiency of our method as the number of latent units grows, evaluating the method’s capacity to scale to large models.

All experiments are run on the BC dataset which we split into 80% training and 20% validation. We fix λ1=100subscript𝜆1100\lambda\_{1}=100 and λ2=0.1subscript𝜆20.1\lambda\_{2}=0.1 throughout our experiments. We run each experiment over 10 random seeds and report the mean and standard deviation. All remaining experimental details are consistent with our experiments in Section [5](#S5 "5 Experiments ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). We note that our implementation of TANGOS is not optimized to the same extent as the Pytorch (Paszke et al., [2019](#bib.bib60)) implementation of L2 to which we compare, and therefore we may consider our relative computational performance to be a loose upper bound on a truly optimized version.

In Figure [10](#A6.F10 "Figure 10 ‣ Appendix F Approximation and Algorithm ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") (left), we report the relative increase in compute time per epoch as we increase the number of sampled pairs. As theory would suggest, this growth is linear. A natural follow-up question is the extent to which model performance is affected by decreasing the number of sampled pairs. In Figure [10](#A6.F10 "Figure 10 ‣ Appendix F Approximation and Algorithm ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") (right), we observe that even very low sampling rates still result in excellent performance. Based on these results, our recommendation for practitioners is that while increasing the sampling rate can lead to marginal improvements in performance, relatively low sampling rates appear to be generally sufficient and do not require prohibitive computational overhead.

![Refer to caption](/html/2303.05506/assets/x11.png)


(a)

![Refer to caption](/html/2303.05506/assets/x12.png)


(b)

Figure 10: Sampling efficiency. Runtime increases linearly with the number of sampled pairs (left) while better generalization performance is maintained even for low sampling rates (right). The benefits of TANGOS can be realized using our proposed sampling approximation with comparable runtime to even the most efficient existing regularization approaches.

Given the results in Figure [10](#A6.F10 "Figure 10 ‣ Appendix F Approximation and Algorithm ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we next wish to evaluate if the proposed sampling scheme enables TANGOS to scale to much bigger models. In order to evaluate this we vary the number of neurons in the relevant hidden layer while maintaining a fixed sampling rate of 50 pairs (consistent with our experiments in Section [5](#S5 "5 Experiments ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization")). Other experimental parameters are consistent with the previous experiment. The results are provided in Figure [11](#A6.F11 "Figure 11 ‣ Appendix F Approximation and Algorithm ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") where we observe a relatively slow increase in runtime as the model grows. These results demonstrate that TANGOS can efficiently be applied to much larger models by using our proposed sampling scheme.

![Refer to caption](/html/2303.05506/assets/x13.png)


Figure 11: Scaling to large models. With a subsampling rate fixed at M=50𝑀50M=50, TANGOS incurs only a small percentage increase in runtime as the number of neurons in the penultimate hidden layer increases dramatically.

## Appendix G Ablation Study

TANGOS is designed with joint application of both regularization on specialization and orthogonalization in mind. Having empirically demonstrated strong overall results, an immediate question is the dynamics of the two regularizers, and how they interact to affect performance. Specifically, we consider the performance gain due to joint regularization effects over applying each regularizer separately.

This includes three separate settings: 1)1) when the specialization regularizer is applied independently (SpecOnly), here we set λ2=0subscript𝜆20\lambda\_{2}=0 and search over λ1∈{1,10,100}subscript𝜆1110100\lambda\_{1}\in\{1,10,100\}; 2)2) when the orthogonalization is applied separately (OrthOnly), we set λ1=0subscript𝜆10\lambda\_{1}=0 and search over λ2∈{0.1,1}subscript𝜆20.11\lambda\_{2}\in\{0.1,1\}; and lastly 3)3) when both are applied jointly (TANGOS), i.e. searching over λ1∈{1,10,100}subscript𝜆1110100\lambda\_{1}\in\{1,10,100\} and λ2∈{0.1,1}subscript𝜆20.11\lambda\_{2}\in\{0.1,1\}. We report the result of the ablation study in [Table 5](#A7.T5 "In Appendix G Ablation Study ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). We empirically observe that the joint effects of both regularizers (i.e. TANGOS) are crucial to achieve consistently good performance.

Combining these results with what we observed in [Figure 3](#S4.F3 "In 4.1 TANGOS Regularizes a Unique Objective ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we hypothesize that applying just specialization regularization, with no regard for diversity, can inadvertently force the neurons to *attend to overlapping regions* in the input space. Correspondingly, simply enforcing orthogonalization, with no regard for sparsity, will likely result in neurons attending to non-overlapping yet *spurious* regions in the input. Thus, we conclude that the two regularizers have distinct, but complementary, effects that work together to achieve the desired regularization effect.

Table 5: Ablation study. Generalization performance on different ablation settings.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Classification (Mean NLL) | | | Regression (MSE) | | |
| Dataset | BR | CR | HC | BC | BH | WQ |
| NoReg | 0.07260.07260.0726 | 0.46330.46330.4633 | 0.33210.33210.3321 | 0.33430.33430.3343 | 0.19770.19770.1977 | 0.67320.67320.6732 |
| SpecOnly | 0.07420.07420.0742 | 0.44660.44660.4466 | 0.38370.38370.3837 | 0.30990.30990.3099 | 0.18420.18420.1842 | 0.67140.67140.6714 |
| OrthOnly | 0.07160.07160.0716 | 0.36960.36960.3696 | 0.20730.2073\mathbf{0.2073} | 0.26920.26920.2692 | 0.19160.19160.1916 | 0.65290.65290.6529 |
| TANGOS | 0.0700.070\mathbf{0.070} | 0.36330.3633\mathbf{0.3633} | 0.21910.21910.2191 | 0.24720.2472\mathbf{0.2472} | 0.18260.1826\mathbf{0.1826} | 0.63790.6379\mathbf{0.6379} |

## Appendix H Ranking Plot

In [Table 1](#S4.T1 "In 4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we reported the generalization performance of TANGOS compared to other regularizers in a stand-alone setting. To gain a better understanding of relative performance, we visually depict the relative ranking of regularizers across all 202020 datasets. [Figure 12](#A8.F12 "In Appendix H Ranking Plot ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") demonstrates that TANGOS consistently ranks as one of the better-performing regularizers, while performance of benchmark methods tend to fluctuate depending on the dataset.

![Refer to caption](/html/2303.05506/assets/x14.png)


Figure 12: Ranking of stand-alone regularizers. Relative ranking of regularizer performance across the 202020 datasets, as reported in Table [1](#S4.T1 "Table 1 ‣ 4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). TANGOS consistently ranks among the best-performing regularizers.

## Appendix I Stand-Alone Uncertainty

In [Table 6](#A9.T6 "In Appendix I Stand-Alone Uncertainty ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we report the standard deviation on generalization performance reported in [Table 1](#S4.T1 "In 4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). The standard errors are computed using 101010 seeded runs.

Table 6: Standard error on generalization performance. Standard errors with respect to the random seed after retraining models from Table [1](#S4.T1 "Table 1 ‣ 4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") experiments 101010 times.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Baseline | L1 | L2 | DO | BN | IN | MU | TANGOS |
| Regression (Mean Squared Error) | | | | | | | | |
| FB | 0.051 | 0.016 | 0.009 | 0.051 | 0.627 | 0.076 | 0.041 | 0.042 |
| BH | 0.023 | 0.021 | 0.029 | 0.022 | 0.025 | 0.023 | 0.011 | 0.028 |
| WE | 0.006 | 0.010 | 0.008 | 0.008 | 0.008 | 0.013 | 0.010 | 0.009 |
| BC | 0.013 | 0.007 | 0.005 | 0.009 | 0.020 | 0.024 | 0.012 | 0.009 |
| WQ | 0.016 | 0.019 | 0.005 | 0.014 | 0.021 | 0.015 | 0.019 | 0.008 |
| SC | 0.026 | 0.014 | 0.019 | 0.025 | 0.023 | 0.168 | 0.067 | 0.017 |
| FF | 0.034 | 0.029 | 0.035 | 0.033 | 0.041 | 0.036 | 0.031 | 0.035 |
| PR | 0.042 | 0.029 | 0.020 | 0.031 | 0.032 | 0.072 | 0.016 | 0.026 |
| ST | 0.090 | 0.084 | 0.085 | 0.064 | 0.076 | 0.080 | 0.082 | 0.029 |
| AB | 0.016 | 0.016 | 0.008 | 0.011 | 0.026 | 0.014 | 0.012 | 0.006 |
| Classification (Mean Negative Log-likelihood) | | | | | | | | |
| HE | 0.057 | 0.049 | 0.009 | 0.033 | 0.163 | 0.038 | 0.067 | 0.032 |
| BR | 0.086 | 0.005 | 0.002 | 0.133 | 0.034 | 0.060 | 0.010 | 0.031 |
| CE | 0.060 | 0.007 | 0.008 | 0.043 | 0.051 | 0.044 | 0.056 | 0.023 |
| CR | 0.029 | 0.094 | 0.004 | 0.034 | 0.041 | 0.027 | 0.027 | 0.019 |
| HC | 0.091 | 0.014 | 0.019 | 0.026 | 0.054 | 0.111 | 0.024 | 0.014 |
| AU | 0.038 | 0.030 | 0.002 | 0.030 | 0.081 | 0.031 | 0.037 | 0.019 |
| TU | 0.075 | 0.075 | 0.087 | 0.077 | 0.087 | 0.096 | 0.078 | 0.064 |
| EN | 0.036 | 0.038 | 0.033 | 0.045 | 0.082 | 0.049 | 0.050 | 0.046 |
| TH | 0.048 | 0.021 | 0.002 | 0.065 | 0.096 | 0.047 | 0.039 | 0.001 |
| SO | 0.038 | 0.028 | 0.016 | 0.046 | 0.029 | 0.025 | 0.023 | 0.008 |

## Appendix J Insights - Extended Results

In this section, we present extended results of the decomposition of overall model error into diversity and weighted error among an ensemble of latent units from Section [4.2](#S4.SS2 "4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). We include all eight regularizers as described in Table [9](#A12.T9 "Table 9 ‣ Appendix L Dataset and Regularizer Details ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") and three datasets (WE, ST, and BC) as described in Table [8](#A12.T8 "Table 8 ‣ Appendix L Dataset and Regularizer Details ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). The results are included in Figure [13](#A10.F13 "Figure 13 ‣ Appendix J Insights - Extended Results ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization").

![Refer to caption](/html/2303.05506/assets/x15.png)


(a) Weather (WE) dataset.

![Refer to caption](/html/2303.05506/assets/x16.png)


(b) Student (ST) dataset.

![Refer to caption](/html/2303.05506/assets/x17.png)


(c) Bioconcentration (BC) dataset.

Figure 13: Neuron diversity. Further examples of ensemble decomposition Err=Err¯−DivErr¯ErrDiv\mathrm{Err}=\overline{\mathrm{Err}}-\mathrm{Div} as discussed in Section [4.2](#S4.SS2 "4.2 TANGOS Generalizes by Increasing Diversity Among Latent Units ‣ 4 How and Why Does TANGOS Work? ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization").

## Appendix K In Tandem Results

In [Table 7](#A11.T7 "In Appendix K In Tandem Results ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), we provide a detailed breakdown of [Figure 5](#S5.F5 "In 5.2 Generalisation: In Tandem Regularization ‣ 5 Experiments ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"), specifically by reporting in tandem performance when benchmarks are paired with TANGOS across all datasets.

Table 7: In tandem performance. Mean ±plus-or-minus\pm standard deviation of generalization performance when each regularizer is employed in tandem with TANGOS.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset | L1 | L2 | DO | BN | IN | MU |
| Regression (Mean Squared Error) | | | | | | |
| FB | 0.033 ±0.018plus-or-minus0.018\pm{0.018} | 0.018 ±0.009plus-or-minus0.009\pm{0.009} | 0.023 ±0.215plus-or-minus0.215\pm{0.215} | 0.042 ±0.268plus-or-minus0.268\pm{0.268} | 0.028 ±0.047plus-or-minus0.047\pm{0.047} | 0.046 ±0.073plus-or-minus0.073\pm{0.073} |
| BH | 0.192 ±0.022plus-or-minus0.022\pm{0.022} | 0.176 ±0.024plus-or-minus0.024\pm{0.024} | 0.196 ±0.024plus-or-minus0.024\pm{0.024} | 0.220 ±0.021plus-or-minus0.021\pm{0.021} | 0.191 ±0.023plus-or-minus0.023\pm{0.023} | 0.178 ±0.019plus-or-minus0.019\pm{0.019} |
| WE | 0.093 ±0.011plus-or-minus0.011\pm{0.011} | 0.091 ±0.009plus-or-minus0.009\pm{0.009} | 0.092 ±0.009plus-or-minus0.009\pm{0.009} | 0.076 ±0.009plus-or-minus0.009\pm{0.009} | 0.081 ±0.012plus-or-minus0.012\pm{0.012} | 0.077 ±0.013plus-or-minus0.013\pm{0.013} |
| WQ | 0.637 ±0.014plus-or-minus0.014\pm{0.014} | 0.639 ±0.006plus-or-minus0.006\pm{0.006} | 0.628 ±0.008plus-or-minus0.008\pm{0.008} | 0.630 ±0.015plus-or-minus0.015\pm{0.015} | 0.644 ±0.011plus-or-minus0.011\pm{0.011} | 0.649 ±0.018plus-or-minus0.018\pm{0.018} |
| BC | 0.227 ±0.011plus-or-minus0.011\pm{0.011} | 0.236 ±0.011plus-or-minus0.011\pm{0.011} | 0.243 ±0.013plus-or-minus0.013\pm{0.013} | 0.275 ±0.027plus-or-minus0.027\pm{0.027} | 0.235 ±0.009plus-or-minus0.009\pm{0.009} | 0.260 ±0.014plus-or-minus0.014\pm{0.014} |
| SC | 0.407 ±0.044plus-or-minus0.044\pm{0.044} | 0.399 ±0.072plus-or-minus0.072\pm{0.072} | 0.370 ±0.031plus-or-minus0.031\pm{0.031} | 0.408 ±0.049plus-or-minus0.049\pm{0.049} | 0.391 ±0.317plus-or-minus0.317\pm{0.317} | 0.425 ±0.061plus-or-minus0.061\pm{0.061} |
| AB | 0.309 ±0.028plus-or-minus0.028\pm{0.028} | 0.312 ±0.028plus-or-minus0.028\pm{0.028} | 0.312 ±0.026plus-or-minus0.026\pm{0.026} | 0.311 ±0.027plus-or-minus0.027\pm{0.027} | 0.319 ±0.037plus-or-minus0.037\pm{0.037} | 0.308 ±0.03plus-or-minus0.03\pm{0.03} |
| FF | 1.281 ±0.029plus-or-minus0.029\pm{0.029} | 1.276 ±0.043plus-or-minus0.043\pm{0.043} | 1.268 ±0.028plus-or-minus0.028\pm{0.028} | 1.297 ±0.027plus-or-minus0.027\pm{0.027} | 1.203 ±0.046plus-or-minus0.046\pm{0.046} | 1.207 ±0.011plus-or-minus0.011\pm{0.011} |
| PR | 0.553 ±0.047plus-or-minus0.047\pm{0.047} | 0.572 ±0.031plus-or-minus0.031\pm{0.031} | 0.642 ±0.093plus-or-minus0.093\pm{0.093} | 0.561 ±0.069plus-or-minus0.069\pm{0.069} | 0.565 ±0.025plus-or-minus0.025\pm{0.025} | 0.568 ±0.081plus-or-minus0.081\pm{0.081} |
| ST | 0.392 ±0.006plus-or-minus0.006\pm{0.006} | 0.382 ±0.006plus-or-minus0.006\pm{0.006} | 0.388 ±0.016plus-or-minus0.016\pm{0.016} | 0.446 ±0.018plus-or-minus0.018\pm{0.018} | 0.417 ±0.012plus-or-minus0.012\pm{0.012} | 0.447 ±0.011plus-or-minus0.011\pm{0.011} |
| Classification (Mean Negative Log-likelihood) | | | | | | |
| HE | 0.441 ±0.046plus-or-minus0.046\pm{0.046} | 0.427 ±0.049plus-or-minus0.049\pm{0.049} | 0.454 ±0.095plus-or-minus0.095\pm{0.095} | 0.407 ±0.047plus-or-minus0.047\pm{0.047} | 0.377 ±0.067plus-or-minus0.067\pm{0.067} | 0.397 ±0.075plus-or-minus0.075\pm{0.075} |
| BR | 0.074 ±0.005plus-or-minus0.005\pm{0.005} | 0.070 ±0.002plus-or-minus0.002\pm{0.002} | 0.068 ±0.006plus-or-minus0.006\pm{0.006} | 0.062 ±0.003plus-or-minus0.003\pm{0.003} | 0.065 ±0.01plus-or-minus0.01\pm{0.01} | 0.078 ±0.011plus-or-minus0.011\pm{0.011} |
| CE | 0.389 ±0.007plus-or-minus0.007\pm{0.007} | 0.396 ±0.007plus-or-minus0.007\pm{0.007} | 0.394 ±0.033plus-or-minus0.033\pm{0.033} | 0.446 ±0.024plus-or-minus0.024\pm{0.024} | 0.422 ±0.075plus-or-minus0.075\pm{0.075} | 0.422 ±0.062plus-or-minus0.062\pm{0.062} |
| CR | 0.362 ±0.021plus-or-minus0.021\pm{0.021} | 0.366 ±0.015plus-or-minus0.015\pm{0.015} | 0.364 ±0.003plus-or-minus0.003\pm{0.003} | 0.406 ±0.023plus-or-minus0.023\pm{0.023} | 0.367 ±0.01plus-or-minus0.01\pm{0.01} | 0.384 ±0.036plus-or-minus0.036\pm{0.036} |
| HC | 0.200 ±0.056plus-or-minus0.056\pm{0.056} | 0.179 ±0.012plus-or-minus0.012\pm{0.012} | 0.185 ±0.009plus-or-minus0.009\pm{0.009} | 0.186 ±0.021plus-or-minus0.021\pm{0.021} | 0.211 ±0.008plus-or-minus0.008\pm{0.008} | 0.181 ±0.037plus-or-minus0.037\pm{0.037} |
| AU | 0.368 ±0.028plus-or-minus0.028\pm{0.028} | 0.360 ±0.138plus-or-minus0.138\pm{0.138} | 0.352 ±0.011plus-or-minus0.011\pm{0.011} | 0.344 ±0.013plus-or-minus0.013\pm{0.013} | 0.379 ±0.016plus-or-minus0.016\pm{0.016} | 0.368 ±0.041plus-or-minus0.041\pm{0.041} |
| TU | 1.519 ±0.082plus-or-minus0.082\pm{0.082} | 1.506 ±0.045plus-or-minus0.045\pm{0.045} | 1.481 ±0.048plus-or-minus0.048\pm{0.048} | 1.506 ±0.067plus-or-minus0.067\pm{0.067} | 1.522 ±0.049plus-or-minus0.049\pm{0.049} | 1.503 ±0.087plus-or-minus0.087\pm{0.087} |
| SO | 0.227 ±0.036plus-or-minus0.036\pm{0.036} | 0.268 ±0.071plus-or-minus0.071\pm{0.071} | 0.233 ±0.021plus-or-minus0.021\pm{0.021} | 0.353 ±0.052plus-or-minus0.052\pm{0.052} | 0.257 ±0.032plus-or-minus0.032\pm{0.032} | 0.304 ±0.054plus-or-minus0.054\pm{0.054} |
| EN | 0.990 ±0.024plus-or-minus0.024\pm{0.024} | 0.971 ±0.002plus-or-minus0.002\pm{0.002} | 0.945 ±0.001plus-or-minus0.001\pm{0.001} | 1.004 ±0.026plus-or-minus0.026\pm{0.026} | 0.995 ±0.001plus-or-minus0.001\pm{0.001} | 1.007 ±0.04plus-or-minus0.04\pm{0.04} |
| TH | 0.506 ±0.031plus-or-minus0.031\pm{0.031} | 0.503 ±0.008plus-or-minus0.008\pm{0.008} | 0.513 ±0.01plus-or-minus0.01\pm{0.01} | 0.524 ±0.008plus-or-minus0.008\pm{0.008} | 0.512 ±0.045plus-or-minus0.045\pm{0.045} | 0.514 ±0.03plus-or-minus0.03\pm{0.03} |

## Appendix L Dataset and Regularizer Details

We perform our experiments on 20 real-world publicly available datasets obtained from (Dua et al., [2017](#bib.bib20)). They are summarized in Table [8](#A12.T8 "Table 8 ‣ Appendix L Dataset and Regularizer Details ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization").
Further information and the source files used for each of the respective datasets can be found at: <https://archive.ics.uci.edu/ml/machine-learning-databases/<UCISource>/> where <UCI Source> denotes the datasets unique identifier as listed in Table [8](#A12.T8 "Table 8 ‣ Appendix L Dataset and Regularizer Details ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization"). Standard preprocessing was applied including standardization of features, one hot encoding of categorical variables, median imputation of missing values and log transformations of highly skewed feature distributions. Furthermore, for computational feasibility, datasets with over 100010001000 samples were reduced in size. In these cases the first 100010001000 samples from the original UCI Source file were used. In Table [9](#A12.T9 "Table 9 ‣ Appendix L Dataset and Regularizer Details ‣ TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization") we summarize the regularizers considered in this work.

Table 8: Dataset descriptions. Summary of the datasets considered in this work.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Dataset | UCI Source | Type | Feature size | Sample Size | Reference |
| Facebook (FB) | “00368” | Regression | 21 | 495 | Moro et al. ([2016](#bib.bib59)) |
| Boston (BH) | [1] | Regression | 13 | 506 | Harrison Jr & Rubinfeld ([1978](#bib.bib31)) |
| Weather (WE) | “00514” | Regression | 45 | 1000 | Cho et al. ([2020](#bib.bib12)) |
| Wine Quality (WQ) | “wine-quality” | Regression | 11 | 1000 | Cortez et al. ([2009](#bib.bib15)) |
| Bioconcentration (BC) | “00510” | Regression | 45 | 779 | Grisoni et al. ([2015](#bib.bib27)) |
| Skillcraft (SC) | “00272” | Regression | 18 | 1000 | Thompson et al. ([2013](#bib.bib75)) |
| Forest Fire (FF) | “forest-fires” | Regression | 39 | 517 | Cortez & Morais ([2007](#bib.bib13)) |
| Protein (PR) | “00265” | Regression | 9 | 1000 | Dua et al. ([2017](#bib.bib20)) |
| Student (ST) | “00320” | Regression | 56 | 649 | Cortez & Silva ([2008](#bib.bib14)) |
| Abalone (AB) | “abalone” | Regression | 9 | 1000 | Waugh ([1995](#bib.bib80)) |
| Heart (HE) | “statlog” | Classification | 20 | 270 | Dua et al. ([2017](#bib.bib20)) |
| Breast (BR) | “breast-cancer-wisconsin” | Classification | 9 | 699 | Street et al. ([1993](#bib.bib71)) |
| Cervical (CE) | “00383” | Classification | 136 | 858 | Fernandes et al. ([2017](#bib.bib22)) |
| Credit (CR) | “credit-screening” | Classification | 40 | 677 | Dua et al. ([2017](#bib.bib20)) |
| HCV (HC) | “00571” | Classification | 12 | 615 | Hoffmann et al. ([2018](#bib.bib36)) |
| Australian (AU) | “statlog” | Classification | 55 | 690 | Quinlan ([1987](#bib.bib63)) |
| Tumor (TU) | “primary-tumor” | Classification | 25 | 339 | Michalski et al. ([1986](#bib.bib57)) |
| Entrance (EN) | “00582” | Classification | 38 | 666 | Hussain et al. ([2018](#bib.bib39)) |
| Thoracic (TH) | “00277” | Classification | 24 | 470 | Zikeba et al. ([2013](#bib.bib88)) |
| Soybean (SO) | “soybean” | Classification | 484 | 683 | Fisher & Schlimmer ([1988](#bib.bib23)) |

* [1] This dataset has now been removed due to ethical issues. For more information see the following url <https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>




Table 9: Overview of regularizers. Description of benchmarks considered in this work and their implementations.

|  |  |  |
| --- | --- | --- |
| Regularizer | Reference | Implementation |
| Baseline | NA | Paszke et al. ([2019](#bib.bib60)) |
| L1 | Tibshirani ([1996](#bib.bib76)) | Paszke et al. ([2019](#bib.bib60)) |
| L2 | Hoerl & Kennard ([1970](#bib.bib35)) | Paszke et al. ([2019](#bib.bib60)) |
| Dropout | Hinton et al. ([2012](#bib.bib34)) | Paszke et al. ([2019](#bib.bib60)) |
| Batch Norm | Ioffe & Szegedy ([2015](#bib.bib40)) | Paszke et al. ([2019](#bib.bib60)) |
| Input Noise | Krizhevsky et al. ([2012](#bib.bib45)) | Paszke et al. ([2019](#bib.bib60)) |
| Mixup | Zhang et al. ([2018](#bib.bib85)) | Zhang et al. ([2018](#bib.bib85)) |
| TANGOS | This work | Paszke et al. ([2019](#bib.bib60)) |

[◄](/html/2303.05505)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2303.05506)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2303.05506)
[View original  
on arXiv](https://arxiv.org/abs/2303.05506)[►](/html/2303.05507)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Feb 29 20:53:06 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
