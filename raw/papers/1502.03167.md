---
arxiv: '1502.03167'
authors:
- Sergey Ioffe
- Christian Szegedy
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal
  Covariate Shift'
url: http://arxiv.org/abs/1502.03167v3
year: 2015
---

[1502.03167] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift














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



# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

Sergey Ioffe
  
Google Inc., sioffe@google.com
  
Christian Szegedy
  
Google Inc., szegedy@google.com

###### Abstract

Training Deep Neural Networks is complicated by the fact that the
distribution of each layer’s inputs changes during training, as the
parameters of the previous layers change. This slows down the
training by requiring lower learning rates and careful parameter
initialization, and makes it notoriously hard to train models with
saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing
layer inputs. Our method draws its strength from making normalization
a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use
much higher learning rates and be less careful about
initialization. It also acts as a regularizer, in some cases
eliminating the need for Dropout. Applied to a state-of-the-art image
classification model, Batch Normalization achieves the same accuracy
with 14 times fewer training steps, and beats the original model by a
significant margin.
Using an ensemble of batch-normalized networks, we
improve upon the best published result on ImageNet classification:
reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy
of human raters.

## 1 Introduction

Deep learning has dramatically advanced the state of the art in vision, speech,
and many other areas. Stochastic gradient descent (SGD) has proved to be an effective way of training
deep networks, and SGD variants such as momentum Sutskever et al. ([2013](#bib.bib20)) and Adagrad Duchi et al. ([2011](#bib.bib4)) have been used to
achieve state of the art performance. SGD optimizes the parameters ΘΘ\Theta of
the network, so as to minimize the loss

|  |  |  |
| --- | --- | --- |
|  | Θ=arg⁡minΘ⁡1N​∑i=1Nℓ​(xi,Θ)ΘsubscriptΘ1𝑁superscriptsubscript𝑖1𝑁ℓsubscriptx𝑖Θ\Theta=\arg\min\_{\Theta}\frac{1}{N}\sum\_{i=1}^{N}\ell(\mathrm{x}\_{i},\Theta) |  |

where
x1​…​Nsubscriptx1…𝑁\mathrm{x}\_{1\ldots N} is the training data set. With SGD, the training
proceeds in steps, and at each step we consider a mini-batch
x1​…​msubscriptx1…𝑚\mathrm{x}\_{1\ldots m} of size m𝑚m. The mini-batch is used
to approximate the gradient of the loss function with respect to the parameters,
by computing

|  |  |  |
| --- | --- | --- |
|  | 1m​∂ℓ​(xi,Θ)∂Θ.1𝑚ℓsubscriptx𝑖ΘΘ\frac{1}{m}\frac{\partial\ell(\mathrm{x}\_{i},\Theta)}{\partial\Theta}. |  |

Using
mini-batches of examples, as opposed to one example at a time, is helpful in
several ways. First, the gradient of the loss over a mini-batch is an estimate
of the gradient over the training set, whose quality improves
as the batch size increases. Second, computation over a batch can be much more
efficient than m𝑚m computations for individual examples, due to the parallelism
afforded by the modern computing platforms.

While stochastic gradient is simple and effective, it requires careful tuning of
the model hyper-parameters, specifically the learning rate used in optimization,
as well as the initial values for the model parameters. The training is
complicated by the fact that the inputs to each layer are affected by the
parameters of all preceding layers – so that
small changes to the network parameters amplify as the network becomes
deeper.

The change in the distributions of layers’ inputs presents a problem
because the layers need to continuously adapt to the new
distribution. When the input distribution to a learning system
changes, it is said to experience covariate shift
Shimodaira ([2000](#bib.bib18)). This is typically handled via domain
adaptation Jiang ([2008](#bib.bib8)). However, the notion of
covariate shift can be extended beyond the learning system as a whole,
to apply to its parts, such as a sub-network or a layer. Consider a
network computing

|  |  |  |
| --- | --- | --- |
|  | ℓ=F2​(F1​(u,Θ1),Θ2)ℓsubscript𝐹2subscript𝐹1usubscriptΘ1subscriptΘ2\ell=F\_{2}(F\_{1}(\mathrm{u},\Theta\_{1}),\Theta\_{2}) |  |

where
F1subscript𝐹1F\_{1} and F2subscript𝐹2F\_{2} are arbitrary transformations, and the parameters
Θ1,Θ2

subscriptΘ1subscriptΘ2\Theta\_{1},\Theta\_{2} are to be learned so as to minimize the loss
ℓℓ\ell. Learning Θ2subscriptΘ2\Theta\_{2} can be viewed as if the inputs
x=F1​(u,Θ1)xsubscript𝐹1usubscriptΘ1\mathrm{x}=F\_{1}(\mathrm{u},\Theta\_{1}) are fed into the sub-network

|  |  |  |
| --- | --- | --- |
|  | ℓ=F2​(x,Θ2).ℓsubscript𝐹2xsubscriptΘ2\ell=F\_{2}(\mathrm{x},\Theta\_{2}). |  |

For example, a gradient descent step

|  |  |  |
| --- | --- | --- |
|  | Θ2←Θ2−αm​∑i=1m∂F2​(xi,Θ2)∂Θ2←subscriptΘ2subscriptΘ2𝛼𝑚superscriptsubscript𝑖1𝑚subscript𝐹2subscriptx𝑖subscriptΘ2subscriptΘ2\Theta\_{2}\leftarrow\Theta\_{2}-\frac{\alpha}{m}\sum\_{i=1}^{m}\frac{\partial F\_{2}(\mathrm{x}\_{i},\Theta\_{2})}{\partial\Theta\_{2}} |  |

(for batch size m𝑚m and learning
rate α𝛼\alpha) is exactly equivalent to that for a stand-alone network
F2subscript𝐹2F\_{2} with input xx\mathrm{x}. Therefore, the input distribution properties
that make training more efficient – such as having the same
distribution between the training and test data – apply to training
the sub-network as well. As such it is advantageous for the
distribution of xx\mathrm{x} to remain fixed over time. Then, Θ2subscriptΘ2\Theta\_{2} does
not have to readjust to compensate for the change in the distribution
of xx\mathrm{x}.

Fixed distribution of inputs to a sub-network would have positive
consequences for the layers outside the sub-network, as
well. Consider a layer with a sigmoid activation function z=g​(W​u+b)z𝑔𝑊ub\mathrm{z}=g(W\mathrm{u}+\mathrm{b}) where uu\mathrm{u} is the layer input, the weight matrix W𝑊W and
bias vector bb\mathrm{b} are the layer parameters to be learned, and g​(x)=11+exp⁡(−x)𝑔𝑥11𝑥g(x)=\frac{1}{1+\exp(-x)}. As |x|𝑥|x| increases, g′​(x)superscript𝑔′𝑥g^{\prime}(x) tends to zero. This
means that for all dimensions of x=W​u+bx𝑊ub\mathrm{x}=W\mathrm{u}+\mathrm{b} except those with
small absolute values, the gradient flowing down to uu\mathrm{u} will vanish
and the model will train slowly. However, since xx\mathrm{x} is affected by
W,b

𝑊bW,\mathrm{b} and the parameters of all the layers below, changes to those
parameters during training will likely move many dimensions of xx\mathrm{x}
into the saturated regime of the nonlinearity and slow down the
convergence. This effect is amplified as the network depth
increases. In practice, the saturation problem and the resulting
vanishing gradients are usually addressed by using Rectified Linear
Units Nair & Hinton ([2010](#bib.bib12)) R​e​L​U​(x)=max⁡(x,0)𝑅𝑒𝐿𝑈𝑥𝑥0ReLU(x)=\max(x,0), careful initialization
Bengio & Glorot ([2010](#bib.bib1)); Saxe et al. ([2013](#bib.bib17)), and small learning rates. If,
however, we could ensure that the distribution of nonlinearity inputs
remains more stable as the network trains, then the optimizer would be
less likely to get stuck in the saturated regime, and the training
would accelerate.

We refer to the change in the distributions of internal nodes of a
deep network, in the course of training, as Internal Covariate Shift. Eliminating it offers
a promise of faster training. We propose a new mechanism, which we
call Batch Normalization, that takes a step towards reducing
internal covariate shift, and in doing so dramatically accelerates the
training of deep neural nets. It accomplishes this via a normalization
step that fixes the means and variances of layer inputs. Batch
Normalization also has a beneficial effect on the gradient flow
through the network, by reducing the dependence of gradients on the
scale of the parameters or of their initial values. This allows us to
use much higher learning rates without the risk of
divergence. Furthermore, batch normalization regularizes the model and
reduces the need for Dropout Srivastava et al. ([2014](#bib.bib19)). Finally, Batch
Normalization makes it possible to use saturating nonlinearities by
preventing the network from getting stuck in the saturated modes.

In Sec. [4.2](#S4.SS2 "4.2 ImageNet classification ‣ 4 Experiments ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"), we apply Batch Normalization to the
best-performing ImageNet classification network, and show that we can
match its performance using only 7% of the training steps, and can
further exceed its accuracy by a substantial margin. Using an
ensemble of such networks trained with Batch Normalization, we achieve
the top-5 error rate that improves upon the best known results on
ImageNet classification.

## 2 Towards Reducing Internal Covariate Shift

We define Internal Covariate Shift as
the change in the
distribution of network activations due to the change in network parameters during training. To improve the
training, we seek to reduce the internal covariate shift. By fixing
the distribution of the layer inputs xx\mathrm{x} as the training progresses,
we expect to improve the training speed.
It has been long known LeCun et al. ([1998b](#bib.bib10)); Wiesler & Ney ([2011](#bib.bib22))
that the network training converges faster if its inputs are whitened – i.e.,
linearly transformed to have zero means and unit variances, and decorrelated. As each layer
observes the inputs produced by the layers below, it would be advantageous to
achieve the same whitening of the inputs of each layer. By whitening the
inputs to each layer, we would take a step towards achieving the fixed
distributions of inputs that would remove the ill effects of the internal covariate shift.

We could consider whitening activations at every training step or at
some interval, either by modifying the network directly or by changing
the parameters of the optimization algorithm to depend on the network
activation values Wiesler et al. ([2014](#bib.bib23)); Raiko et al. ([2012](#bib.bib15)); Povey et al. ([2014](#bib.bib14)); [Desjardins & Kavukcuoglu](#bib.bib3) . However, if these modifications are interspersed with
the optimization steps, then the gradient descent step may attempt to
update the parameters in a way that requires the normalization to be
updated, which reduces the effect of the gradient step. For example,
consider a layer with the input u𝑢u that adds the learned bias b𝑏b,
and normalizes the result by subtracting the mean of the activation
computed over the training data: x^=x−E​[x]^𝑥𝑥𝐸delimited-[]𝑥\widehat{x}=x-E[x] where x=u+b𝑥𝑢𝑏x=u+b,
 𝒳={x1​…​N}𝒳subscript𝑥1…𝑁\mathcal{X}=\{x\_{1\ldots N}\} is the set of values of x𝑥x over the
training set, and E​[x]=1N​∑i=1NxiEdelimited-[]𝑥1𝑁superscriptsubscript𝑖1𝑁subscript𝑥𝑖\text{E}[x]=\frac{1}{N}\sum\_{i=1}^{N}x\_{i}. If a gradient descent step ignores the dependence of E​[x]Edelimited-[]𝑥\text{E}[x] on b𝑏b, then it will
update b←b+Δ​b←𝑏𝑏Δ𝑏b\leftarrow b+\Delta b, where Δ​b∝−∂ℓ/∂x^proportional-toΔ𝑏ℓ^𝑥\Delta b\propto-\partial{\ell}/\partial{\widehat{x}}. Then u+(b+Δ​b)−E​[u+(b+Δ​b)]=u+b−E​[u+b]𝑢𝑏Δ𝑏Edelimited-[]𝑢𝑏Δ𝑏𝑢𝑏Edelimited-[]𝑢𝑏u+(b+\Delta b)-\text{E}[u+(b+\Delta b)]=u+b-\text{E}[u+b]. Thus, the combination of the update to b𝑏b and subsequent change in
normalization led to no change in the output of the layer nor,
consequently, the loss. As the training continues, b𝑏b will grow
indefinitely while the loss remains fixed. This problem can get worse
if the normalization not only centers but also scales the activations.
We have observed this empirically in initial experiments, where the
model blows up when the normalization parameters are computed outside
the gradient descent step.

The issue with the above approach is that the gradient descent
optimization does not take into account the fact that the
normalization takes place. To address this issue, we would like to
ensure that, for any parameter values, the network always
produces activations with the desired distribution. Doing so would
allow the gradient of the loss with respect to the model parameters to
account for the normalization, and for its dependence on the model
parameters ΘΘ\Theta. Let again xx\mathrm{x} be a layer input, treated as a
vector, and 𝒳𝒳\mathcal{X} be the set of these inputs over the training data
set. The normalization can then be written as a transformation

|  |  |  |
| --- | --- | --- |
|  | x^=Norm​(x,𝒳)^xNormx𝒳\widehat{\mathrm{x}}=\text{Norm}(\mathrm{x},\mathcal{X}) |  |

which depends not only on the given
training example xx\mathrm{x} but on all examples 𝒳𝒳\mathcal{X} – each of which
depends on ΘΘ\Theta if xx\mathrm{x} is generated by another layer. For
backpropagation, we would need to compute the Jacobians

|  |  |  |
| --- | --- | --- |
|  | ∂Norm​(x,𝒳)∂x​ and ​∂Norm​(x,𝒳)∂𝒳;Normx𝒳x and Normx𝒳𝒳\frac{\partial\text{Norm}(\mathrm{x},\mathcal{X})}{\partial\mathrm{x}}\text{\, and\, }\frac{\partial\text{Norm}(\mathrm{x},\mathcal{X})}{\partial\mathcal{X}}; |  |

ignoring the latter term would lead to the explosion described above.
Within this framework, whitening the layer inputs is expensive, as it requires
computing the covariance matrix Cov​[x]=Ex∈𝒳​[xxT]−E​[x]​E​[x]TCovdelimited-[]xsubscriptEx𝒳delimited-[]superscriptxx𝑇Edelimited-[]xEsuperscriptdelimited-[]x𝑇\text{Cov}[\mathrm{x}]=\text{E}\_{\mathrm{x}\in\mathcal{X}}[\mathrm{x}\mathrm{x}^{T}]-\text{E}[\mathrm{x}]\text{E}[\mathrm{x}]^{T} and its
inverse square root, to produce the whitened activations Cov​[x]−1/2​(x−E​[x])Covsuperscriptdelimited-[]x12xEdelimited-[]x\text{Cov}[\mathrm{x}]^{-1/2}(\mathrm{x}-\text{E}[\mathrm{x}]),
as well as the derivatives of these transforms for backpropagation.
This motivates us to seek an
alternative that performs input normalization in a way that is
differentiable and does not require the analysis of the entire
training set after every parameter update.

Some of the previous approaches
(e.g. Lyu & Simoncelli ([2008](#bib.bib11))) use statistics computed over a single
training example, or, in the case of image networks, over different
feature maps at a given location. However, this changes the
representation ability of a network by discarding the absolute scale
of activations. We want to a preserve the information in the network,
by normalizing the activations in a training example relative to the
statistics of the entire training data.

## 3 Normalization via Mini-Batch Statistics

Since the full whitening of each layer’s inputs is costly and not
everywhere differentiable, we make two necessary simplifications. The first
is that instead of whitening the features in layer
inputs and outputs jointly, we will normalize each scalar feature
independently, by making it have the mean of zero and the variance of
1. For a layer with d𝑑d-dimensional input x=(x(1)​…​x(d))xsuperscript𝑥1…superscript𝑥𝑑\mathrm{x}=(x^{(1)}\ldots x^{(d)}), we
will normalize each dimension

|  |  |  |
| --- | --- | --- |
|  | x^(k)=x(k)−E​[x(k)]Var​[x(k)]superscript^𝑥𝑘superscript𝑥𝑘Edelimited-[]superscript𝑥𝑘Vardelimited-[]superscript𝑥𝑘\widehat{x}^{(k)}=\frac{x^{(k)}-\text{E}[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}]}} |  |

where the expectation and variance are
computed over the training data set. As shown in
LeCun et al. ([1998b](#bib.bib10)), such normalization speeds up convergence,
even when the features are not decorrelated.

Note that simply normalizing each input of a layer may change what the
layer can represent. For instance, normalizing the inputs of a
sigmoid would constrain them to the linear
regime of the nonlinearity. To address this, we make sure that the transformation inserted in
the network can represent the identity transform. To
accomplish this, we introduce, for each activation x(k)superscript𝑥𝑘x^{(k)}, a pair of
parameters γ(k),β(k)

superscript𝛾𝑘superscript𝛽𝑘\gamma^{(k)},\beta^{(k)}, which scale and shift the
normalized value:

|  |  |  |
| --- | --- | --- |
|  | y(k)=γ(k)​x^(k)+β(k).superscript𝑦𝑘superscript𝛾𝑘superscript^𝑥𝑘superscript𝛽𝑘y^{(k)}=\gamma^{(k)}\widehat{x}^{(k)}+\beta^{(k)}. |  |

These parameters are learned along with the original model
parameters, and restore the representation power of the
network. Indeed, by setting γ(k)=Var​[x(k)]superscript𝛾𝑘Vardelimited-[]superscript𝑥𝑘\gamma^{(k)}=\sqrt{\text{Var}[x^{(k)}]} and
β(k)=E​[x(k)]superscript𝛽𝑘Edelimited-[]superscript𝑥𝑘\beta^{(k)}=\text{E}[x^{(k)}], we could recover the original activations, if that were the optimal thing to do.

In the batch setting where each training step is based on the entire training
set, we would use the whole set to normalize activations. However, this is
impractical when using stochastic optimization. Therefore, we make the second
simplification: since we use mini-batches in stochastic gradient training, each mini-batch produces estimates of the mean and variance of each
activation. This way, the statistics used for normalization can fully
participate in the gradient backpropagation.
Note that the use of mini-batches is enabled by computation of
per-dimension variances rather than joint covariances; in the joint case,
regularization would be required since the mini-batch size is likely to be
smaller than the number of activations being whitened, resulting in singular
covariance matrices.

Consider a mini-batch ℬℬ\mathcal{B} of size m𝑚m. Since the normalization is applied to
each activation independently, let us focus on a particular activation x(k)superscript𝑥𝑘x^{(k)} and omit k𝑘k for clarity. We have m𝑚m values of this activation
in the mini-batch,

|  |  |  |
| --- | --- | --- |
|  | ℬ={x1​…​m}.ℬsubscript𝑥1…𝑚\mathcal{B}=\{x\_{1\ldots m}\}. |  |

Let the normalized values be
x^1​…​msubscript^𝑥1…𝑚\widehat{x}\_{1\ldots m}, and their linear transformations be y1​…​msubscript𝑦1…𝑚y\_{1\ldots m}. We refer to the transform

|  |  |  |
| --- | --- | --- |
|  | BNγ,β:x1​…​m→y1​…​m:subscriptBN  𝛾𝛽→subscript𝑥1…𝑚subscript𝑦1…𝑚\text{BN}\_{\gamma,\beta}:x\_{1\ldots m}\rightarrow y\_{1\ldots m} |  |

as the Batch Normalizing Transform.
We present the BN Transform in Algorithm [1](#alg1 "Algorithm 1 ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"). In the algorithm, ϵitalic-ϵ\epsilon is a constant added to the mini-batch variance for numerical stability.

Algorithm 1  Batch Normalizing Transform, applied to activation x𝑥x over a mini-batch.

0:

|  |
| --- |
| Values of x𝑥x over a mini-batch: ℬ={x1​…​m}ℬsubscript𝑥1…𝑚\mathcal{B}=\{x\_{1\ldots m}\}; |
| Parameters to be learned: γ𝛾\gamma, β𝛽\beta |

0:  {yi=BNγ,β​(xi)}subscript𝑦𝑖subscriptBN

𝛾𝛽subscript𝑥𝑖\{y\_{i}=\text{BN}\_{\gamma,\beta}(x\_{i})\}

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | μℬsubscript𝜇ℬ\displaystyle\mu\_{\mathcal{B}} | ←1m​∑i=1mxi←absent1𝑚superscriptsubscript𝑖1𝑚subscript𝑥𝑖\displaystyle\leftarrow\frac{1}{m}\sum\_{i=1}^{m}x\_{i} | // mini-batch mean |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | σℬ2superscriptsubscript𝜎ℬ2\displaystyle\sigma\_{\mathcal{B}}^{2} | ←1m​∑i=1m(xi−μℬ)2←absent1𝑚superscriptsubscript𝑖1𝑚superscriptsubscript𝑥𝑖subscript𝜇ℬ2\displaystyle\leftarrow\frac{1}{m}\sum\_{i=1}^{m}(x\_{i}-\mu\_{\mathcal{B}})^{2} | // mini-batch variance |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | x^isubscript^𝑥𝑖\displaystyle\widehat{x}\_{i} | ←xi−μℬσℬ2+ϵ←absentsubscript𝑥𝑖subscript𝜇ℬsuperscriptsubscript𝜎ℬ2italic-ϵ\displaystyle\leftarrow\frac{x\_{i}-\mu\_{\mathcal{B}}}{\sqrt{\sigma\_{\mathcal{B}}^{2}+\epsilon}} | // normalize |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | yisubscript𝑦𝑖\displaystyle y\_{i} | ←γ​x^i+β≡BNγ,β​(xi)←absent𝛾subscript^𝑥𝑖𝛽subscriptBN  𝛾𝛽subscript𝑥𝑖\displaystyle\leftarrow\gamma\widehat{x}\_{i}+\beta\equiv\text{BN}\_{\gamma,\beta}(x\_{i}) | // scale and shift |  |

The BN transform can be added to a network to manipulate any activation. In the notation y=BNγ,β​(x)𝑦subscriptBN

𝛾𝛽𝑥y=\text{BN}\_{\gamma,\beta}(x), we indicate that the parameters γ𝛾\gamma and β𝛽\beta are to be learned, but it should be noted that the BN transform does not independently process the activation in each training example. Rather, BNγ,β​(x)subscriptBN

𝛾𝛽𝑥\text{BN}\_{\gamma,\beta}(x) depends both on the training example and the other examples in the mini-batch.
The scaled and shifted values y𝑦y are passed to other network layers. The normalized activations x^^𝑥\widehat{x} are internal to our transformation, but their presence is crucial. The distributions of values of any x^^𝑥\widehat{x} has the
expected value of 00 and the variance of 111, as long as the elements of each mini-batch are
sampled from the same distribution, and if we neglect ϵitalic-ϵ\epsilon. This can be seen by observing that ∑i=1mx^i=0superscriptsubscript𝑖1𝑚subscript^𝑥𝑖0\sum\_{i=1}^{m}\widehat{x}\_{i}=0 and
1m​∑i=1mx^i2=11𝑚superscriptsubscript𝑖1𝑚superscriptsubscript^𝑥𝑖21\frac{1}{m}\sum\_{i=1}^{m}\widehat{x}\_{i}^{2}=1, and taking expectations. Each normalized activation x^(k)superscript^𝑥𝑘\widehat{x}^{(k)} can be viewed as an input to a sub-network composed of the linear transform y(k)=γ(k)​x^(k)+β(k)superscript𝑦𝑘superscript𝛾𝑘superscript^𝑥𝑘superscript𝛽𝑘y^{(k)}=\gamma^{(k)}\widehat{x}^{(k)}+\beta^{(k)}, followed by the other processing done by the original network. These sub-network inputs all have fixed means and variances, and although the joint distribution of these normalized x^(k)superscript^𝑥𝑘\widehat{x}^{(k)} can change over the course of training, we expect that the introduction of normalized inputs accelerates the training of the sub-network and, consequently, the network as a whole.

During training we need to backpropagate the gradient of loss ℓℓ\ell through
this transformation, as well as compute the gradients with respect to the parameters of the BN transform. We use chain rule, as follows (before
simplification):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂ℓ∂x^iℓsubscript^𝑥𝑖\displaystyle\textstyle\frac{\partial\ell}{\partial\widehat{x}\_{i}} | =∂ℓ∂yi⋅γabsent⋅ℓsubscript𝑦𝑖𝛾\displaystyle\textstyle=\frac{\partial\ell}{\partial y\_{i}}\cdot\gamma |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂ℓ∂σℬ2ℓsuperscriptsubscript𝜎ℬ2\displaystyle\textstyle\frac{\partial\ell}{\partial\sigma\_{\mathcal{B}}^{2}} | =∑i=1m∂ℓ∂x^i⋅(xi−μℬ)⋅−12​(σℬ2+ϵ)−3/2absentsuperscriptsubscript𝑖1𝑚⋅ℓsubscript^𝑥𝑖subscript𝑥𝑖subscript𝜇ℬ12superscriptsuperscriptsubscript𝜎ℬ2italic-ϵ32\displaystyle\textstyle=\sum\_{i=1}^{m}\frac{\partial\ell}{\partial\widehat{x}\_{i}}\cdot(x\_{i}-\mu\_{\mathcal{B}})\cdot\frac{-1}{2}(\sigma\_{\mathcal{B}}^{2}+\epsilon)^{-3/2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂ℓ∂μℬℓsubscript𝜇ℬ\displaystyle\textstyle\frac{\partial\ell}{\partial\mu\_{\mathcal{B}}} | =(∑i=1m∂ℓ∂x^i⋅−1σℬ2+ϵ)+∂ℓ∂σℬ2⋅∑i=1m−2​(xi−μℬ)mabsentsuperscriptsubscript𝑖1𝑚⋅ℓsubscript^𝑥𝑖1superscriptsubscript𝜎ℬ2italic-ϵ⋅ℓsuperscriptsubscript𝜎ℬ2superscriptsubscript𝑖1𝑚2subscript𝑥𝑖subscript𝜇ℬ𝑚\displaystyle\textstyle=\bigg{(}\sum\_{i=1}^{m}\frac{\partial\ell}{\partial\widehat{x}\_{i}}\cdot\frac{-1}{\sqrt{\sigma\_{\mathcal{B}}^{2}+\epsilon}}\bigg{)}+\frac{\partial\ell}{\partial\sigma\_{\mathcal{B}}^{2}}\cdot\frac{\sum\_{i=1}^{m}-2(x\_{i}-\mu\_{\mathcal{B}})}{m} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂ℓ∂xiℓsubscript𝑥𝑖\displaystyle\textstyle\frac{\partial\ell}{\partial x\_{i}} | =∂ℓ∂x^i⋅1σℬ2+ϵ+∂ℓ∂σℬ2⋅2​(xi−μℬ)m+∂ℓ∂μℬ⋅1mabsent⋅ℓsubscript^𝑥𝑖1superscriptsubscript𝜎ℬ2italic-ϵ⋅ℓsuperscriptsubscript𝜎ℬ22subscript𝑥𝑖subscript𝜇ℬ𝑚⋅ℓsubscript𝜇ℬ1𝑚\displaystyle\textstyle=\frac{\partial\ell}{\partial\widehat{x}\_{i}}\cdot\frac{1}{\sqrt{\sigma\_{\mathcal{B}}^{2}+\epsilon}}+\frac{\partial\ell}{\partial\sigma\_{\mathcal{B}}^{2}}\cdot\frac{2(x\_{i}-\mu\_{\mathcal{B}})}{m}+\frac{\partial\ell}{\partial\mu\_{\mathcal{B}}}\cdot\frac{1}{m} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂ℓ∂γℓ𝛾\displaystyle\textstyle\frac{\partial\ell}{\partial\gamma} | =∑i=1m∂ℓ∂yi⋅x^iabsentsuperscriptsubscript𝑖1𝑚⋅ℓsubscript𝑦𝑖subscript^𝑥𝑖\displaystyle\textstyle=\sum\_{i=1}^{m}\frac{\partial\ell}{\partial y\_{i}}\cdot\widehat{x}\_{i} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂ℓ∂βℓ𝛽\displaystyle\textstyle\frac{\partial\ell}{\partial\beta} | =∑i=1m∂ℓ∂yiabsentsuperscriptsubscript𝑖1𝑚ℓsubscript𝑦𝑖\displaystyle\textstyle=\sum\_{i=1}^{m}\frac{\partial\ell}{\partial y\_{i}} |  |

Thus, BN transform is a differentiable transformation that introduces normalized activations
into the network. This ensures that as the model is training, layers can continue learning on input distributions that exhibit less internal covariate shift, thus accelerating the training.
Furthermore, the learned affine transform applied to these normalized activations allows the BN transform to represent the identity transformation and preserves the network capacity.

### 3.1 Training and Inference with Batch-Normalized Networks

To Batch-Normalize a network, we specify a subset of activations
and insert the BN transform for each of them, according to
Alg. [1](#alg1 "Algorithm 1 ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"). Any layer that previously received x𝑥x as the
input, now receives BN​(x)BN𝑥\text{BN}(x). A model employing Batch
Normalization can be trained using batch gradient descent, or
Stochastic Gradient Descent with a mini-batch size m>1𝑚1m>1, or with any
of its variants such as Adagrad Duchi et al. ([2011](#bib.bib4)). The normalization of activations
that depends on the mini-batch allows efficient training, but is
neither necessary nor desirable during inference; we want the output
to depend only on the input, deterministically. For this, once the
network has been trained, we use the
normalization

|  |  |  |
| --- | --- | --- |
|  | x^=x−E​[x]Var​[x]+ϵ^𝑥𝑥Edelimited-[]𝑥Vardelimited-[]𝑥italic-ϵ\widehat{x}=\frac{x-\text{E}[x]}{\sqrt{\text{Var}[x]+\epsilon}} |  |

using
the population, rather than mini-batch, statistics. Neglecting
ϵitalic-ϵ\epsilon, these normalized activations have the same mean 0 and
variance 1 as during training. We use the unbiased variance estimate
Var​[x]=mm−1⋅Eℬ​[σℬ2]Vardelimited-[]𝑥⋅𝑚𝑚1subscriptEℬdelimited-[]superscriptsubscript𝜎ℬ2\text{Var}[x]=\frac{m}{m-1}\cdot\text{E}\_{\mathcal{B}}[\sigma\_{\mathcal{B}}^{2}], where the
expectation is over training mini-batches of size m𝑚m and
σℬ2superscriptsubscript𝜎ℬ2\sigma\_{\mathcal{B}}^{2} are their sample variances. Using moving averages
instead, we can track the accuracy of a model as it trains. Since the
means and variances are fixed during inference, the normalization is
simply a linear transform applied to each activation. It may further
be composed with the scaling by γ𝛾\gamma and shift by β𝛽\beta, to
yield a single linear transform that replaces BN​(x)BN𝑥\text{BN}(x).
Algorithm [2](#alg2 "Algorithm 2 ‣ 3.1 Training and Inference with Batch-Normalized Networks ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift") summarizes the procedure for training
batch-normalized networks.

Algorithm 2  Training a Batch-Normalized Network

0:

|  |
| --- |
| Network N with trainable parameters ΘΘ\Theta; |
| subset of activations {x(k)}k=1Ksuperscriptsubscriptsuperscript𝑥𝑘𝑘1𝐾\{x^{(k)}\}\_{k=1}^{K} |

0:  Batch-normalized network for inference, NBNinfsuperscriptsubscriptNBNinf\text{\sl N}\_{\mathrm{BN}}^{\mathrm{inf}}

1:  NBNtr←N←superscriptsubscriptNBNtrN\text{\sl N}\_{\mathrm{BN}}^{\mathrm{tr}}\leftarrow\text{\sl N}  // Training BN network

2:  for k=1​…​K𝑘1…𝐾k=1\ldots K do

3:     Add transformation y(k)=BNγ(k),β(k)​(x(k))superscript𝑦𝑘subscriptBN

superscript𝛾𝑘superscript𝛽𝑘superscript𝑥𝑘y^{(k)}=\text{BN}\_{\gamma^{(k)},\beta^{(k)}}(x^{(k)}) to NBNtrsuperscriptsubscriptNBNtr\text{\sl N}\_{\mathrm{BN}}^{\mathrm{tr}} (Alg. [1](#alg1 "Algorithm 1 ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"))

4:     Modify each layer in NBNtrsuperscriptsubscriptNBNtr\text{\sl N}\_{\mathrm{BN}}^{\mathrm{tr}} with input x(k)superscript𝑥𝑘x^{(k)} to take y(k)superscript𝑦𝑘y^{(k)} instead

5:  end for

6:  Train NBNtrsuperscriptsubscriptNBNtr\text{\sl N}\_{\mathrm{BN}}^{\mathrm{tr}} to optimize the parameters Θ∪{γ(k),β(k)}k=1KΘsuperscriptsubscriptsuperscript𝛾𝑘superscript𝛽𝑘𝑘1𝐾\Theta\cup\{\gamma^{(k)},\beta^{(k)}\}\_{k=1}^{K}

7:  
NBNinf←NBNtr←superscriptsubscriptNBNinfsuperscriptsubscriptNBNtr\text{\sl N}\_{\mathrm{BN}}^{\mathrm{inf}}\leftarrow\text{\sl N}\_{\mathrm{BN}}^{\mathrm{tr}}

|  |
| --- |
| // Inference BN network with frozen |
| // parameters |

8:  for k=1​…​K𝑘1…𝐾k=1\ldots K do

9:     // For clarity, x≡x(k),γ≡γ(k),μℬ≡μℬ(k)formulae-sequence𝑥superscript𝑥𝑘formulae-sequence𝛾superscript𝛾𝑘subscript𝜇ℬsuperscriptsubscript𝜇ℬ𝑘x\equiv x^{(k)},\gamma\equiv\gamma^{(k)},\mu\_{\mathcal{B}}\equiv\mu\_{\mathcal{B}}^{(k)}, etc.

10:     Process multiple training mini-batches ℬℬ\mathcal{B}, each of size m𝑚m, and average over them:

|  |  |  |  |
| --- | --- | --- | --- |
|  | E​[x]Edelimited-[]𝑥\displaystyle\text{E}[x] | ←Eℬ​[μℬ]←absentsubscriptEℬdelimited-[]subscript𝜇ℬ\displaystyle\leftarrow\text{E}\_{\mathcal{B}}[\mu\_{\mathcal{B}}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Var​[x]Vardelimited-[]𝑥\displaystyle\text{Var}[x] | ←mm−1​Eℬ​[σℬ2]←absent𝑚𝑚1subscriptEℬdelimited-[]superscriptsubscript𝜎ℬ2\displaystyle\leftarrow\textstyle\frac{m}{m-1}\text{E}\_{\mathcal{B}}[\sigma\_{\mathcal{B}}^{2}] |  |

11:     
In NBNinfsuperscriptsubscriptNBNinf\text{\sl N}\_{\mathrm{BN}}^{\mathrm{inf}}, replace the transform y=BNγ,β​(x)𝑦subscriptBN

𝛾𝛽𝑥y=\text{BN}\_{\gamma,\beta}(x) with  y=γVar​[x]+ϵ⋅x+(β−γ​E​[x]Var​[x]+ϵ)𝑦⋅𝛾Vardelimited-[]𝑥italic-ϵ𝑥𝛽𝛾Edelimited-[]𝑥Vardelimited-[]𝑥italic-ϵy=\frac{\gamma}{\sqrt{\text{Var}[x]+\epsilon}}\cdot x+\big{(}\beta-\frac{\gamma\,\text{E}[x]}{\sqrt{\text{Var}[x]+\epsilon}}\big{)}

12:  end for

### 3.2 Batch-Normalized Convolutional Networks

Batch Normalization can be applied to any set of activations in the network. Here, we focus on transforms that
consist of an affine transformation followed by an element-wise
nonlinearity:

|  |  |  |
| --- | --- | --- |
|  | z=g​(W​u+b)z𝑔𝑊ub\mathrm{z}=g(W\mathrm{u}+\mathrm{b}) |  |

where W𝑊W and bb\mathrm{b} are learned parameters of the
model, and g​(⋅)𝑔⋅g(\cdot) is the nonlinearity such as sigmoid or
ReLU. This formulation covers both fully-connected and convolutional layers. We add the BN transform immediately before the nonlinearity, by normalizing x=W​u+bx𝑊ub\mathrm{x}=W\mathrm{u}+\mathrm{b}. We could have also normalized the layer inputs uu\mathrm{u}, but
since uu\mathrm{u} is likely the output of another nonlinearity, the
shape of its distribution is likely to change during training, and constraining its first and second moments would not eliminate the covariate shift.
In contrast, W​u+b𝑊ubW\mathrm{u}+\mathrm{b} is more likely to have a symmetric, non-sparse distribution,
that is “more Gaussian” Hyvärinen & Oja ([2000](#bib.bib7)); normalizing it is likely to produce activations with a stable distribution.

Note that, since we normalize W​u+b𝑊ubW\mathrm{u}+\mathrm{b}, the bias bb\mathrm{b} can be ignored since its
effect will be canceled by the subsequent mean subtraction (the role of the bias is subsumed by β𝛽\beta in Alg. [1](#alg1 "Algorithm 1 ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift")). Thus, z=g​(W​u+b)z𝑔𝑊ub\mathrm{z}=g(W\mathrm{u}+\mathrm{b}) is replaced with

|  |  |  |
| --- | --- | --- |
|  | z=g​(BN​(W​u))z𝑔BN𝑊u\mathrm{z}=g(\text{BN}(W\mathrm{u})) |  |

where the BN transform is applied independently to each dimension of x=W​ux𝑊u\mathrm{x}=W\mathrm{u}, with a separate pair of learned parameters γ(k)superscript𝛾𝑘\gamma^{(k)}, β(k)superscript𝛽𝑘\beta^{(k)} per dimension.

For convolutional layers, we additionally want the normalization
to obey the convolutional property – so that different elements
of the same feature map, at different locations, are normalized in the
same way. To achieve this, we jointly normalize all the activations in
a mini-batch, over all locations. In Alg. [1](#alg1 "Algorithm 1 ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"), we let
ℬℬ\mathcal{B} be the set of all values in a feature map across both the
elements of a mini-batch and spatial locations – so for a mini-batch
of size m𝑚m and feature maps of size p×q𝑝𝑞p\times q, we use the effective mini-batch of size m′=|ℬ|=m⋅p​qsuperscript𝑚′ℬ⋅𝑚𝑝𝑞m^{\prime}=|\mathcal{B}|=m\cdot p\,q. We learn a pair of parameters γ(k)superscript𝛾𝑘\gamma^{(k)} and β(k)superscript𝛽𝑘\beta^{(k)} per feature map, rather than per activation.
Alg. [2](#alg2 "Algorithm 2 ‣ 3.1 Training and Inference with Batch-Normalized Networks ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift") is modified similarly, so that during inference the BN transform applies the same linear transformation to each activation in a given feature map.

### 3.3 Batch Normalization enables higher learning rates

In
traditional deep networks, too-high learning rate may result in the gradients that explode or vanish, as well as getting stuck in poor local minima. Batch Normalization helps address these issues. By normalizing activations throughout the network, it prevents small changes to the parameters from amplifying into larger and suboptimal changes in activations in gradients; for instance, it prevents the training from getting stuck in the saturated regimes of nonlinearities.

Batch Normalization also makes training more resilient to the parameter scale. Normally, large learning rates
may increase the scale of layer parameters, which then amplify the gradient during backpropagation and lead to the model explosion.
However, with Batch Normalization, backpropagation through a layer is unaffected by the scale of its parameters. Indeed, for a scalar a𝑎a,

|  |  |  |
| --- | --- | --- |
|  | BN​(W​u)=BN​((a​W)​u)BN𝑊uBN𝑎𝑊u\text{BN}(W\mathrm{u})=\text{BN}((aW)\mathrm{u}) |  |

and we can show that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂BN​((a​W)​u)∂uBN𝑎𝑊uu\displaystyle\textstyle\frac{\partial\text{BN}((aW)\mathrm{u})}{\partial\mathrm{u}} | =∂BN​(W​u)∂uabsentBN𝑊uu\displaystyle=\textstyle\frac{\partial\text{BN}(W\mathrm{u})}{\partial\mathrm{u}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂BN​((a​W)​u)∂(a​W)BN𝑎𝑊u𝑎𝑊\displaystyle\textstyle\frac{\partial\text{BN}((aW)\mathrm{u})}{\partial(aW)} | =1a⋅∂BN​(W​u)∂Wabsent⋅1𝑎BN𝑊u𝑊\displaystyle\textstyle=\frac{1}{a}\cdot\frac{\partial\text{BN}(W\mathrm{u})}{\partial W} |  |

The scale does not affect the layer Jacobian nor, consequently, the gradient propagation. Moreover, larger weights lead to smaller gradients, and Batch Normalization will stabilize the parameter growth.

We further conjecture that Batch Normalization may lead the layer Jacobians to
have singular values close to 1, which is known to be beneficial for training Saxe et al. ([2013](#bib.bib17)). Consider two consecutive layers with normalized inputs, and the transformation between these normalized vectors:
z^=F​(x^)^z𝐹^x\widehat{\mathrm{z}}=F(\widehat{\mathrm{x}}). If we assume that x^^x\widehat{\mathrm{x}} and z^^z\widehat{\mathrm{z}} are Gaussian and uncorrelated, and
that F​(x^)≈J​x^𝐹^x𝐽^xF(\widehat{\mathrm{x}})\approx J\widehat{\mathrm{x}} is a linear transformation for the given model
parameters, then both x^^x\widehat{\mathrm{x}} and z^^z\widehat{\mathrm{z}} have unit covariances, and I=Cov​[z^]=J​Cov​[x^]​JT=J​JT𝐼Covdelimited-[]^z𝐽Covdelimited-[]^xsuperscript𝐽𝑇𝐽superscript𝐽𝑇I=\text{Cov}[\widehat{\mathrm{z}}]=J\text{Cov}[\widehat{\mathrm{x}}]J^{T}=JJ^{T}. Thus, J​JT=I𝐽superscript𝐽𝑇𝐼JJ^{T}=I, and so all
singular values of J𝐽J are equal to 1, which preserves the
gradient magnitudes during backpropagation. In reality, the transformation is
not linear, and the normalized values are not guaranteed to be Gaussian nor independent, but we nevertheless expect Batch Normalization to help make gradient propagation better behaved. The precise effect of Batch
Normalization on gradient propagation remains an area of further study.

### 3.4 Batch Normalization regularizes the model

When training with Batch Normalization, a training example is seen in
conjunction with other examples in the mini-batch, and the training network no longer
producing deterministic values for a given training example. In our
experiments, we found this effect to be advantageous to the
generalization of the network. Whereas Dropout Srivastava et al. ([2014](#bib.bib19)) is
typically used to reduce overfitting, in a batch-normalized network
we found that it can be either removed or reduced in strength.

## 4 Experiments

### 4.1 Activations over time

To verify the effects of internal covariate shift on training, and the ability of Batch
Normalization to combat it, we considered the problem of predicting the digit
class on the MNIST dataset LeCun et al. ([1998a](#bib.bib9)). We used a very simple network, with a 28x28
binary image as input, and 3 fully-connected hidden layers with 100 activations each.
Each hidden layer computes y=g​(W​u+b)y𝑔𝑊ub\mathrm{y}=g(W\mathrm{u}+\mathrm{b}) with sigmoid nonlinearity, and the weights W𝑊W initialized to small random Gaussian values. The last hidden layer is followed by a fully-connected layer with 10 activations (one per class) and cross-entropy loss. We trained the network for 50000
steps, with 60 examples per mini-batch. We added Batch Normalization to each hidden layer of the network, as in Sec. [3.1](#S3.SS1 "3.1 Training and Inference with Batch-Normalized Networks ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift").
We were interested in the
comparison between the baseline and batch-normalized networks, rather than
achieving the state of the art performance on MNIST (which the described
architecture does not).

|  |  |  |
| --- | --- | --- |
| Refer to caption | Refer to caption | Refer to caption |
| (a) | (b) Without BN | (c) With BN |

Figure 1: (a) The test accuracy of the MNIST network trained with and without Batch Normalization, vs. the number of training steps. Batch Normalization helps the network train faster and achieve higher accuracy. (b, c) The evolution of input distributions to a typical sigmoid, over the course of training, shown as {15,50,85}155085\{15,50,85\}th percentiles. Batch Normalization makes the distribution more stable and reduces the internal covariate shift.

Figure [1](#S4.F1 "Figure 1 ‣ 4.1 Activations over time ‣ 4 Experiments ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift")(a) shows the fraction of correct predictions by the two networks on
held-out test data, as training progresses. The batch-normalized network enjoys the higher test accuracy. To investigate why, we studied inputs to the sigmoid, in the original network N and batch-normalized network NBNtrsuperscriptsubscriptNBNtr\text{\sl N}\_{\mathrm{BN}}^{\mathrm{tr}} (Alg. [2](#alg2 "Algorithm 2 ‣ 3.1 Training and Inference with Batch-Normalized Networks ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift")) over the course of training. In Fig. [1](#S4.F1 "Figure 1 ‣ 4.1 Activations over time ‣ 4 Experiments ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift")(b,c) we show, for one typical activation from the last hidden layer of each network, how its distribution evolves. The distributions in the original network
change significantly over time, both in their mean and the variance, which complicates the training of the subsequent layers. In
contrast, the distributions in the batch-normalized network are much more stable
as training progresses, which aids the training.

### 4.2 ImageNet classification

We applied Batch Normalization to a new variant of the Inception network Szegedy et al. ([2014](#bib.bib21)),
trained on the ImageNet classification task Russakovsky et al. ([2014](#bib.bib16)). The network has a large
number of convolutional and pooling layers, with a softmax layer to predict the
image class, out of 1000 possibilities. Convolutional layers use ReLU as the
nonlinearity. The main difference to the network described in Szegedy et al. ([2014](#bib.bib21)) is that
the 5×5555\times 5 convolutional layers are replaced by two consecutive layers of 3×3333\times 3 convolutions
with up to 128128128 filters. The network contains 13.6⋅106⋅13.6superscript10613.6\cdot 10^{6} parameters, and, other than the top softmax layer, has no fully-connected layers. More details are given in the Appendix. We refer to this model as Inception in the rest of the text. The model was trained using a version of Stochastic Gradient Descent with momentum
Sutskever et al. ([2013](#bib.bib20)), using the mini-batch size of 32. The training was performed using a large-scale, distributed architecture (similar to Dean et al. ([2012](#bib.bib2))).
All networks are evaluated as training progresses by computing the validation accuracy @​1@1@1, i.e. the
probability of predicting the correct label out of 1000 possibilities, on a held-out set, using a single crop per image.

|  |
| --- |
| Refer to caption |

Figure 2: Single crop validation accuracy of Inception and its
batch-normalized variants, vs. the number of training steps.

| Model | Steps to 72.2% | Max accuracy |
| --- | --- | --- |
| Inception | 31.0⋅106⋅31.0superscript10631.0\cdot 10^{6} | 72.2% |
| BN-Baseline | 13.3⋅106⋅13.3superscript10613.3\cdot 10^{6} | 72.7% |
| BN-x5 | 2.1⋅106⋅2.1superscript1062.1\cdot 10^{6} | 73.0% |
| BN-x30 | 2.7⋅106⋅2.7superscript1062.7\cdot 10^{6} | 74.8% |
| BN-x5-Sigmoid |  | 69.8% |

Figure 3: For Inception and the batch-normalized variants, the number of training steps required to reach the maximum accuracy of Inception (72.2%), and the maximum accuracy achieved by the network.

In our experiments, we evaluated several modifications of Inception with Batch Normalization. In all cases, Batch Normalization was applied to the
input of each nonlinearity, in a convolutional way, as described in section
[3.2](#S3.SS2 "3.2 Batch-Normalized Convolutional Networks ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"), while keeping the rest of the architecture constant.

#### 4.2.1 Accelerating BN Networks

Simply adding Batch Normalization to a network does not take full advantage of our method. To do so, we further changed the network and its training parameters, as follows:

Increase learning rate. In a batch-normalized model, we have been able to achieve a training speedup from higher learning rates, with no ill side effects (Sec. [3.3](#S3.SS3 "3.3 Batch Normalization enables higher learning rates ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift")).

Remove Dropout. As described in Sec. [3.4](#S3.SS4 "3.4 Batch Normalization regularizes the model ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"), Batch Normalization fulfills some of the same goals as Dropout. Removing Dropout from Modified BN-Inception speeds up training, without increasing overfitting.

Reduce the L2subscript𝐿2L\_{2} weight regularization. While in Inception an L2subscript𝐿2L\_{2} loss on the model parameters controls overfitting, in Modified BN-Inception the weight of this loss is reduced by a factor of 5. We find that this improves the accuracy on the held-out validation data.

Accelerate the learning rate decay. In training Inception, learning rate was decayed exponentially. Because our network trains faster than Inception, we lower the learning rate 6 times faster.

Remove Local Response Normalization While Inception and other networks Srivastava et al. ([2014](#bib.bib19)) benefit from it, we found that with Batch Normalization it is not necessary.

Shuffle training examples more thoroughly. We enabled within-shard shuffling of the training data, which prevents the same examples from always appearing in a mini-batch together. This led to about 1% improvements in the validation accuracy, which is consistent with the view of Batch Normalization as a regularizer (Sec. [3.4](#S3.SS4 "3.4 Batch Normalization regularizes the model ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift")): the randomization inherent in our method should be most beneficial when it affects an example differently each time it is seen.

Reduce the photometric distortions. Because batch-normalized networks train faster and observe each training example fewer times, we let the trainer focus on more “real” images by distorting them less.

#### 4.2.2 Single-Network Classification

We evaluated the following networks, all trained on the LSVRC2012 training data, and tested on the validation data:

Inception: the network described at the beginning of Section [4.2](#S4.SS2 "4.2 ImageNet classification ‣ 4 Experiments ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"), trained with the initial learning rate of 0.0015.

BN-Baseline: Same as Inception with Batch Normalization before each nonlinearity.

BN-x5: Inception with Batch Normalization and the modifications in Sec. [4.2.1](#S4.SS2.SSS1 "4.2.1 Accelerating BN Networks ‣ 4.2 ImageNet classification ‣ 4 Experiments ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"). The initial learning rate was increased by a factor of 5, to 0.0075. The same learning rate increase with original Inception caused the model parameters to reach machine infinity.

BN-x30: Like BN-x5, but with the initial learning rate 0.045 (30 times that of Inception).

BN-x5-Sigmoid: Like BN-x5, but with sigmoid nonlinearity g​(t)=11+exp⁡(−x)𝑔𝑡11𝑥g(t)=\frac{1}{1+\exp(-x)} instead of ReLU.
We also attempted to train the original Inception with sigmoid, but the model remained at the accuracy equivalent to chance.

In Figure [3](#S4.F3 "Figure 3 ‣ 4.2 ImageNet classification ‣ 4 Experiments ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"), we show the validation accuracy of the
networks, as a function of the number of training steps. Inception
reached the accuracy of 72.2% after 31⋅106⋅31superscript10631\cdot 10^{6} training steps. The
Figure [3](#S4.F3 "Figure 3 ‣ 4.2 ImageNet classification ‣ 4 Experiments ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift") shows, for each network, the number of training
steps required to reach the same 72.2% accuracy, as well as the
maximum validation accuracy reached by the network and the number of steps
to reach it.

By only using Batch Normalization (BN-Baseline), we match the accuracy of Inception in less than half the number of training steps. By applying the modifications in Sec. [4.2.1](#S4.SS2.SSS1 "4.2.1 Accelerating BN Networks ‣ 4.2 ImageNet classification ‣ 4 Experiments ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"), we significantly increase the training speed of the network. BN-x5 needs 14 times fewer steps than Inception to reach the 72.2% accuracy.
Interestingly, increasing the learning rate further (BN-x30) causes the model to train somewhat slower initially, but allows it to reach a higher final accuracy. It reaches 74.8% after 6⋅106⋅6superscript1066\cdot 10^{6} steps, i.e. 5 times fewer steps than required by Inception to reach 72.2%.

We also verified that the reduction in internal covariate shift allows deep networks with Batch Normalization to be trained when sigmoid is used as the nonlinearity, despite the well-known difficulty of training such networks. Indeed, BN-x5-Sigmoid achieves the accuracy of 69.8%. Without Batch Normalization, Inception with sigmoid never achieves better than 1/1000110001/1000 accuracy.

#### 4.2.3 Ensemble Classification

| Model | Resolution | Crops | Models | Top-1 error | Top-5 error |
| --- | --- | --- | --- | --- | --- |
| GoogLeNet ensemble | 224 | 144 | 7 | - | 6.67% |
| Deep Image low-res | 256 | - | 1 | - | 7.96% |
| Deep Image high-res | 512 | - | 1 | 24.88 | 7.42% |
| Deep Image ensemble | variable | - | - | - | 5.98% |
| BN-Inception single crop | 224 | 1 | 1 | 25.2% | 7.82% |
| BN-Inception multicrop | 224 | 144 | 1 | 21.99% | 5.82% |
| BN-Inception ensemble | 224 | 144 | 6 | 20.1% | 4.9%\* |

Figure 4: Batch-Normalized Inception comparison with previous state of the art on the provided validation set comprising 50000 images.
\*BN-Inception ensemble has reached 4.82% top-5 error on the 100000 images of the test set of the ImageNet as reported by the test server.

The current reported best results on the ImageNet Large Scale Visual Recognition
Competition are reached by the Deep Image ensemble of traditional models
Wu et al. ([2015](#bib.bib24)) and the ensemble model of He et al. ([2015](#bib.bib6)). The latter reports the
top-5 error of 4.94%, as evaluated by the ILSVRC server. Here we report a top-5
validation error of 4.9%, and test error of 4.82% (according to the ILSVRC
server). This improves upon the previous best result,
and
exceeds the estimated accuracy of human raters according to Russakovsky et al. ([2014](#bib.bib16)).

For our ensemble, we used 6 networks. Each was based on BN-x30,
modified via some of the following: increased initial weights in the
convolutional layers; using Dropout (with the Dropout probability of
5% or 10%, vs. 40% for the original Inception); and using
non-convolutional, per-activation Batch Normalization with last hidden
layers of the model. Each network achieved its maximum accuracy after
about 6⋅106⋅6superscript1066\cdot 10^{6} training steps. The ensemble prediction was based on
the arithmetic average of class probabilities predicted by the
constituent networks. The details of ensemble and multicrop inference
are similar to Szegedy et al. ([2014](#bib.bib21)).

We demonstrate in Fig. [4](#S4.F4 "Figure 4 ‣ 4.2.3 Ensemble Classification ‣ 4.2 ImageNet classification ‣ 4 Experiments ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift") that batch
normalization allows us to set new state-of-the-art by a healthy
margin on the ImageNet classification challenge benchmarks.

## 5 Conclusion

We have presented a novel mechanism for dramatically accelerating the
training of deep networks. It is based on the premise that covariate
shift, which is known to complicate the training of machine learning
systems, also applies to sub-networks and layers, and removing it from
internal activations of the network may aid in training. Our proposed
method draws its power from normalizing activations, and from
incorporating this normalization in the network architecture
itself. This ensures that the normalization is appropriately handled
by any optimization method that is being used to train the network. To
enable stochastic optimization methods commonly used in deep network
training, we perform the normalization for each mini-batch, and
backpropagate the gradients through the normalization
parameters. Batch Normalization adds only two extra parameters per
activation, and in doing so preserves the representation ability of
the network. We presented an algorithm for constructing, training, and
performing inference with batch-normalized networks. The resulting
networks can be trained with saturating nonlinearities, are more
tolerant to increased training rates, and often do not require Dropout
for regularization.

Merely adding Batch Normalization to a state-of-the-art image
classification model yields a substantial speedup in training. By
further increasing the learning rates, removing Dropout, and applying
other modifications afforded by Batch Normalization, we reach the
previous state of the art with only a small fraction of training steps
– and then beat the state of the art in single-network image
classification. Furthermore, by combining multiple models trained with
Batch Normalization, we perform better than the best known system on
ImageNet, by a significant margin.

Interestingly, our method bears similarity to the standardization layer of
Gülçehre & Bengio ([2013](#bib.bib5)), though the two methods stem from very different goals, and
perform different tasks. The goal of Batch Normalization is to achieve a stable
distribution of activation values throughout training, and in our experiments we
apply it before the nonlinearity since that is where matching the first and
second moments is more likely to result in a stable distribution. On the
contrary, Gülçehre & Bengio ([2013](#bib.bib5)) apply the standardization layer to the output of
the nonlinearity, which results in sparser activations. In our large-scale image
classification experiments, we have not observed the nonlinearity inputs
to be sparse, neither with nor without Batch Normalization. Other notable
differentiating characteristics of Batch Normalization include the learned scale
and shift that allow the BN transform to represent identity (the standardization
layer did not require this since it was followed by the learned linear transform
that, conceptually, absorbs the necessary scale and shift), handling of
convolutional layers, deterministic inference that does not depend on the
mini-batch, and batch-normalizing each convolutional layer in the network.

In this work, we have not explored the full range of possibilities
that Batch Normalization potentially enables. Our future work includes
applications of our method to Recurrent Neural Networks
Pascanu et al. ([2013](#bib.bib13)), where the internal covariate shift and the
vanishing or exploding gradients may be especially severe, and which
would allow us to more thoroughly test the hypothesis that
normalization improves gradient propagation (Sec. [3.3](#S3.SS3 "3.3 Batch Normalization enables higher learning rates ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift")). We
plan to investigate whether Batch Normalization can help with domain
adaptation, in its traditional sense – i.e. whether the normalization
performed by the network would allow it to more easily generalize to
new data distributions, perhaps with just a recomputation of the
population means and variances (Alg. [2](#alg2 "Algorithm 2 ‣ 3.1 Training and Inference with Batch-Normalized Networks ‣ 3 Normalization via Mini-Batch Statistics ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift")). Finally, we
believe that further theoretical analysis of the algorithm would allow
still more improvements and applications.

## References

* Bengio & Glorot (2010)

  Bengio, Yoshua and Glorot, Xavier.
  Understanding the difficulty of training deep feedforward neural
  networks.
  In *Proceedings of AISTATS 2010*, volume 9, pp.  249–256, May
  2010.
* Dean et al. (2012)

  Dean, Jeffrey, Corrado, Greg S., Monga, Rajat, Chen, Kai, Devin, Matthieu, Le,
  Quoc V., Mao, Mark Z., Ranzato, Marc’Aurelio, Senior, Andrew, Tucker, Paul,
  Yang, Ke, and Ng, Andrew Y.
  Large scale distributed deep networks.
  In *NIPS*, 2012.
* (3)

  Desjardins, Guillaume and Kavukcuoglu, Koray.
  Natural neural networks.
  (unpublished).
* Duchi et al. (2011)

  Duchi, John, Hazan, Elad, and Singer, Yoram.
  Adaptive subgradient methods for online learning and stochastic
  optimization.
  *J. Mach. Learn. Res.*, 12:2121–2159, July 2011.
  ISSN 1532-4435.
* Gülçehre & Bengio (2013)

  Gülçehre, Çaglar and Bengio, Yoshua.
  Knowledge matters: Importance of prior information for optimization.
  *CoRR*, abs/1301.4083, 2013.
* He et al. (2015)

  He, K., Zhang, X., Ren, S., and Sun, J.
  Delving Deep into Rectifiers: Surpassing Human-Level Performance on
  ImageNet Classification.
  *ArXiv e-prints*, February 2015.
* Hyvärinen & Oja (2000)

  Hyvärinen, A. and Oja, E.
  Independent component analysis: Algorithms and applications.
  *Neural Netw.*, 13(4-5):411–430, May 2000.
* Jiang (2008)

  Jiang, Jing.
  A literature survey on domain adaptation of statistical classifiers,
  2008.
* LeCun et al. (1998a)

  LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P.
  Gradient-based learning applied to document recognition.
  *Proceedings of the IEEE*, 86(11):2278–2324, November 1998a.
* LeCun et al. (1998b)

  LeCun, Y., Bottou, L., Orr, G., and Muller, K.
  Efficient backprop.
  In Orr, G. and K., Muller (eds.), *Neural Networks: Tricks of
  the trade*. Springer, 1998b.
* Lyu & Simoncelli (2008)

  Lyu, S and Simoncelli, E P.
  Nonlinear image representation using divisive normalization.
  In *Proc. Computer Vision and Pattern Recognition*, pp.  1–8.
  IEEE Computer Society, Jun 23-28 2008.
  doi: 10.1109/CVPR.2008.4587821.
* Nair & Hinton (2010)

  Nair, Vinod and Hinton, Geoffrey E.
  Rectified linear units improve restricted boltzmann machines.
  In *ICML*, pp.  807–814. Omnipress, 2010.
* Pascanu et al. (2013)

  Pascanu, Razvan, Mikolov, Tomas, and Bengio, Yoshua.
  On the difficulty of training recurrent neural networks.
  In *Proceedings of the 30th International Conference on Machine
  Learning, ICML 2013, Atlanta, GA, USA, 16-21 June 2013*, pp.  1310–1318,
  2013.
* Povey et al. (2014)

  Povey, Daniel, Zhang, Xiaohui, and Khudanpur, Sanjeev.
  Parallel training of deep neural networks with natural gradient and
  parameter averaging.
  *CoRR*, abs/1410.7455, 2014.
* Raiko et al. (2012)

  Raiko, Tapani, Valpola, Harri, and LeCun, Yann.
  Deep learning made easier by linear transformations in perceptrons.
  In *International Conference on Artificial Intelligence and
  Statistics (AISTATS)*, pp.  924–932, 2012.
* Russakovsky et al. (2014)

  Russakovsky, Olga, Deng, Jia, Su, Hao, Krause, Jonathan, Satheesh, Sanjeev, Ma,
  Sean, Huang, Zhiheng, Karpathy, Andrej, Khosla, Aditya, Bernstein, Michael,
  Berg, Alexander C., and Fei-Fei, Li.
  ImageNet Large Scale Visual Recognition Challenge, 2014.
* Saxe et al. (2013)

  Saxe, Andrew M., McClelland, James L., and Ganguli, Surya.
  Exact solutions to the nonlinear dynamics of learning in deep linear
  neural networks.
  *CoRR*, abs/1312.6120, 2013.
* Shimodaira (2000)

  Shimodaira, Hidetoshi.
  Improving predictive inference under covariate shift by weighting the
  log-likelihood function.
  *Journal of Statistical Planning and Inference*, 90(2):227–244, October 2000.
* Srivastava et al. (2014)

  Srivastava, Nitish, Hinton, Geoffrey, Krizhevsky, Alex, Sutskever, Ilya, and
  Salakhutdinov, Ruslan.
  Dropout: A simple way to prevent neural networks from overfitting.
  *J. Mach. Learn. Res.*, 15(1):1929–1958,
  January 2014.
* Sutskever et al. (2013)

  Sutskever, Ilya, Martens, James, Dahl, George E., and Hinton, Geoffrey E.
  On the importance of initialization and momentum in deep learning.
  In *ICML (3)*, volume 28 of *JMLR Proceedings*, pp. 1139–1147. JMLR.org, 2013.
* Szegedy et al. (2014)

  Szegedy, Christian, Liu, Wei, Jia, Yangqing, Sermanet, Pierre, Reed, Scott,
  Anguelov, Dragomir, Erhan, Dumitru, Vanhoucke, Vincent, and Rabinovich,
  Andrew.
  Going deeper with convolutions.
  *CoRR*, abs/1409.4842, 2014.
* Wiesler & Ney (2011)

  Wiesler, Simon and Ney, Hermann.
  A convergence analysis of log-linear training.
  In Shawe-Taylor, J., Zemel, R.S., Bartlett, P., Pereira, F.C.N., and
  Weinberger, K.Q. (eds.), *Advances in Neural Information Processing
  Systems 24*, pp.  657–665, Granada, Spain, December 2011.
* Wiesler et al. (2014)

  Wiesler, Simon, Richard, Alexander, Schlüter, Ralf, and Ney, Hermann.
  Mean-normalized stochastic gradient for large-scale deep learning.
  In *IEEE International Conference on Acoustics, Speech, and
  Signal Processing*, pp.  180–184, Florence, Italy, May 2014.
* Wu et al. (2015)

  Wu, Ren, Yan, Shengen, Shan, Yi, Dang, Qingqing, and Sun, Gang.
  Deep image: Scaling up image recognition, 2015.

## Appendix

### Variant of the Inception Model Used

| type | patch size/        stride | output     size | depth | #​1×1#11\#1{\times}1 | #​3×3#33\#3{\times}3    reduce | #​3×3#33\#3{\times}3 | double #​3×3#33\#3{\times}3         reduce | double      #​3×3#33\#3{\times}3 | Pool +proj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| convolution\* | 7×7/27727{\times}7/2 | 112×112×6411211264112{\times}112{\times}64 | 1 |  |  |  |  |  |  |
| max pool | 3×3/23323{\times}3/2 | 56×56×6456566456{\times}56{\times}64 | 0 |  |  |  |  |  |  |
| convolution | 3×3/13313{\times}3/1 | 56×56×192565619256{\times}56{\times}192 | 1 |  | 64 | 192 |  |  |  |
| max pool | 3×3/23323{\times}3/2 | 28×28×192282819228{\times}28{\times}192 | 0 |  |  |  |  |  |  |
| inception (3a) |  | 28×28×256282825628{\times}28{\times}256 | 3 | 64 | 64 | 64 | 64 | 96 | avg + 32 |
| inception (3b) |  | 28×28×320282832028{\times}28{\times}320 | 3 | 64 | 64 | 96 | 64 | 96 | avg + 64 |
| inception (3c) | stride 2 | 28×28×576282857628{\times}28{\times}576 | 3 | 0 | 128 | 160 | 64 | 96 | max + pass through |
| inception (4a) |  | 14×14×576141457614{\times}14{\times}576 | 3 | 224 | 64 | 96 | 96 | 128 | avg + 128 |
| inception (4b) |  | 14×14×576141457614{\times}14{\times}576 | 3 | 192 | 96 | 128 | 96 | 128 | avg + 128 |
| inception (4c) |  | 14×14×576141457614{\times}14{\times}576 | 3 | 160 | 128 | 160 | 128 | 160 | avg + 128 |
| inception (4d) |  | 14×14×576141457614{\times}14{\times}576 | 3 | 96 | 128 | 192 | 160 | 192 | avg + 128 |
| inception (4e) | stride 2 | 14×14×10241414102414{\times}14{\times}1024 | 3 | 0 | 128 | 192 | 192 | 256 | max + pass through |
| inception (5a) |  | 7×7×10247710247{\times}7{\times}1024 | 3 | 352 | 192 | 320 | 160 | 224 | avg + 128 |
| inception (5b) |  | 7×7×10247710247{\times}7{\times}1024 | 3 | 352 | 192 | 320 | 192 | 224 | max + 128 |
| avg pool | 7×7/17717{\times}7/1 | 1×1×10241110241{\times}1{\times}1024 | 0 |  |  |  |  |  |  |

Figure 5: Inception architecture

Figure [5](#Sx1.F5 "Figure 5 ‣ Variant of the Inception Model Used ‣ Appendix ‣ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift") documents the changes that were performed compared to the architecture with respect to the GoogleNet archictecture. For the interpretation of this table, please consult Szegedy et al. ([2014](#bib.bib21)). The notable architecture changes compared to the GoogLeNet model include:

* •

  The 5×{\times}5 convolutional layers are replaced by two consecutive 3×{\times}3 convolutional layers. This increases the maximum depth of the network by 9 weight layers. Also it increases the number of parameters by 25% and the computational cost is increased by about 30%.
* •

  The number 28×{\times}28 inception modules is increased from 2 to 3.
* •

  Inside the modules, sometimes average, sometimes maximum-pooling is employed. This is indicated in the entries corresponding to the pooling layers of the table.
* •

  There are no across the board pooling layers between any two Inception modules, but stride-2 convolution/pooling layers are employed before the filter concatenation in the modules 3c, 4e.

Our model employed separable convolution with depth multiplier 888 on the first convolutional layer. This reduces the computational cost while increasing the memory consumption at training time.

[◄](/html/1502.03166)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1502.03167)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1502.03167)
[View original  
on arXiv](https://arxiv.org/abs/1502.03167)[►](/html/1502.03170)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Mar 16 02:39:22 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
