---
arxiv: '1910.03225'
authors:
- Tony Duan
- Anand Avati
- Daisy Yi Ding
- Khanh K. Thai
- Sanjay Basu
- Andrew Y. Ng
- Alejandro Schuler
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'NGBoost: Natural Gradient Boosting for Probabilistic Prediction'
url: http://arxiv.org/abs/1910.03225v4
year: 2019
---

[1910.03225] NGBoost: Natural Gradient Boosting for Probabilistic Prediction















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



# NGBoost: Natural Gradient Boosting for Probabilistic Prediction

Tony Duan
  
Anand Avati
  
Daisy Yi Ding
  
Khanh K. Thai
  
Sanjay Basu
  
Andrew Ng
  
Alejandro Schuler

###### Abstract

We present Natural Gradient Boosting (NGBoost), an algorithm for generic probabilistic prediction via gradient boosting. Typical regression models return a point estimate, conditional on covariates, but probabilistic regression models output a full probability distribution over the outcome space, conditional on the covariates. This allows for predictive uncertainty estimation — crucial in applications like healthcare and weather forecasting.
NGBoost generalizes gradient boosting to probabilistic regression by treating the parameters of the conditional distribution as targets for a multiparameter boosting algorithm. Furthermore, we show how the *Natural Gradient* is required to correct the training dynamics of our multiparameter boosting approach. NGBoost can be used with any base learner, any family of distributions with continuous parameters, and any scoring rule. NGBoost matches or exceeds the performance of existing methods for probabilistic prediction while offering additional benefits in flexibility, scalability, and usability. An open-source implementation is available at [github.com/stanfordmlgroup/ngboost](https://github.com/stanfordmlgroup/ngboost).

gradient boosting, probabilistic regression

## 1 Introduction

![Refer to caption](/html/1910.03225/assets/x1.png)


Figure 1: Prediction intervals for a toy 1-dimensional probabilistic regression problem, fit via NGBoost. The dots represent data points. The thick black line is the predicted mean after fitting the model. The thin gray lines are the upper and lower quantiles covering 95% of the prediction distribution.

x𝑥xBase Learners {f(m)​(x)}m=1Msuperscriptsubscriptsuperscript𝑓𝑚𝑥𝑚1𝑀\left\{f^{(m)}(x)\right\}\_{m=1}^{M}Distribution Pθ​(y|x)subscript𝑃𝜃conditional𝑦𝑥P\_{\theta}(y|x)Scoring Rule 𝒮​(Pθ,y)𝒮subscript𝑃𝜃𝑦\mathcal{S}(P\_{\theta},y)y𝑦yθ𝜃\thetaFit Natural Gradient ∇~θsubscript~∇𝜃\tilde{\nabla}\_{\theta}


Figure 2: NGBoost is modular with respect to choice of base learner, distribution, and scoring rule.

Many important supervised machine learning problems are regression problems. Weather forecasting (predicting temperature of the next day based on today’s atmospheric variables (Gneiting and Katzfuss,, [2014](#bib.bib14))) and clinical prediction (predicting time to mortality with survival prediction on structured medical records of the patient (Avati et al.,, [2018](#bib.bib3))) are important examples.

Most machine learning methods tackle this problem with point prediction, returning a single “best guess” prediction (e.g. the temperature tomorrow will be 16°C). However, in these fields it is often important to be able to quantify uncertainty in the prediction or be able to answer multiple questions on the fly (e.g. what’s the probability it will be between 18°C and 20°C? What about <<15°C?) (Kruchten,, [2016](#bib.bib19)).

In order to answer arbitrary questions about the probability of events conditional on covariates, we must estimate the conditional probability distribution P​(y|x)𝑃conditional𝑦𝑥P(y|x) for each value of x𝑥x instead of producing a point estimate like 𝔼​[y|x]𝔼delimited-[]conditional𝑦𝑥\mathbb{E}[y|x]. This is called *probabilistic regression*. Probabilistic regression is increasingly being used in fields like meteorology and healthcare (Gneiting and Raftery,, [2007](#bib.bib15); Avati et al.,, [2019](#bib.bib2)).

Probabilistic estimation is already the norm in *classification* problems. Although some classifiers (e.g. standard support vector machines) only return a predicted class label, most are capable of returning estimated probabilities for each class; effectively, a conditional probability mass function.

However, existing methods for probabilistic *regression* are either inflexible, slow, or inaccessible to non-experts. Any mean-estimating regression method can be made probabilistic by assuming homoscedasticity and estimating an unconditional noise model, but homoscedasticity is a strong assumption and the process requires some statistical know-how. Generalized Additive Models for Shape, Scale, and Location (GAMLSS) allow heteroscedasticity but are restricted to a pre-specified model form (Stasinopoulos et al.,, [2007](#bib.bib28)). Bayesian methods naturally generate predictive uncertainty estimates by integrating predictions over the posterior, but exact solutions to Bayesian models are limited to simple models, and calculating the posterior distribution of more powerful models such as Neural Networks (NN) (Neal,, [1996](#bib.bib23)) and Bayesian Additive Regression Trees (BART) (Chipman et al.,, [2010](#bib.bib6)) is difficult. Inference in these models requires computationally expensive approximation via, for example, MCMC sampling. Moreover, sampling-based inference requires some statistical expertise and thus limits the ease-of-use of Bayesian methods. Bayesian approaches often also scale poorly to large datasets (Rasmussen and Williams,, [2005](#bib.bib25)). Bayesian Deep Learning is gaining popularity (Graves,, [2011](#bib.bib16); Blundell et al.,, [2015](#bib.bib4); Hernández-Lobato and Adams,, [2015](#bib.bib17)) but, while neural networks have empirically excelled at perception tasks (such as with visual and audio input), they usually perform only on par with traditional methods when data are limited in size or tabular. Extensive hyper-parameter tuning and informative prior specification are also challenges for Bayesian Deep Learning which make it difficult to use out-of-the-box.

Meanwhile, Gradient Boosting Machines (GBMs) (Friedman,, [2001](#bib.bib10); Chen and Guestrin,, [2016](#bib.bib5)) are a set of highly-modular methods that work out-of-the-box and perform well on structured input data, even with relatively small datasets. This can be seen in their empirical success on Kaggle and other data science competitions (Chen and Guestrin,, [2016](#bib.bib5)). In classification tasks, their predictions are probabilistic by default (by use of the sigmoid or softmax link function). But in regression tasks, they output only a scalar value. Under a squared-error loss these scalars can be interpreted as the mean of a conditional Gaussian distribution with some (unknown) constant variance. However, such probabilistic interpretations have little use if the variance is assumed constant. The predicted distributions need to have at least two degrees of freedom (two parameters) to effectively convey both the magnitude and the uncertainty of the predictions, as illustrated in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction"). It is precisely this problem of simultaneous boosting of multiple parameters from the base learners which makes probabilistic forecasting with GBMs a challenge, and NGBoost addresses this with a multiparameter boosting approach and the use of natural gradients (Amari,, [1998](#bib.bib1)).

## 2 Summary of Contributions

1. i.

   We present Natural Gradient Boosting, a modular algorithm for probabilistic regression (section [3.4](#S3.SS4 "3.4 NGBoost: Natural Gradient Boosting ‣ 3 Natural Gradient Boosting ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction")) which uses multiparameter boosting and natural gradients to integrate any choice of:

   * •

     Base learner (e.g. Regression Tree),
   * •

     Parametric probability distribution (Normal, Laplace, etc.), and
   * •

     Scoring rule (MLE, CRPS, etc.).
2. ii.

   We present a generalization of the natural gradient to other scoring rules such as CRPS (section [3.2](#S3.SS2 "3.2 The Generalized Natural Gradient ‣ 3 Natural Gradient Boosting ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction")).
3. iii.

   We demonstrate empirically that NGBoost performs competitively relative to other models in its predictive uncertainty estimates, as well as in its point estimates (section [4](#S4 "4 Experiments ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction")).

## 3 Natural Gradient Boosting

In standard prediction settings, the object of interest is an estimate of a scalar function like 𝔼​[y|x]𝔼delimited-[]conditional𝑦𝑥\mathbb{E}[y|x], where x𝑥x is a vector of observed features and y𝑦y is the prediction target. In our setting we are interested in producing a probability distribution Pθ​(y|x)subscript𝑃𝜃conditional𝑦𝑥P\_{\theta}(y|x) (with CDF Fθsubscript𝐹𝜃F\_{\theta}). Our approach is to assume Pθ​(y|x)subscript𝑃𝜃conditional𝑦𝑥P\_{\theta}(y|x) is of a specified parametric form, then estimate the p𝑝p parameters θ∈ℝp𝜃superscriptℝ𝑝\theta\in\mathbb{R}^{p} of the distribution as functions of x𝑥x.

### 3.1 Proper Scoring Rules

To begin, we need a learning objective. In point prediction, the predictions are compared to the observed data with a loss function. The analogue in probabilistic regression is a *scoring rule*, which compares the estimated probability distribution to the observed data.

A *proper* scoring rule 𝒮𝒮\mathcal{S} takes as input a forecasted probability distribution P𝑃P and one observation y𝑦y (outcome), and assigns a score 𝒮​(P,y)𝒮𝑃𝑦\mathcal{S}(P,y) to the forecast such that the true distribution of the outcomes gets the best score in expectation (Gneiting and Raftery,, [2007](#bib.bib15)). In mathematical notation, a scoring rule 𝒮𝒮\mathcal{S} is a proper scoring rule if and only if it satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼y∼Q​[𝒮​(Q,y)]≤𝔼y∼Q​[𝒮​(P,y)]∀P,Q,  subscript𝔼similar-to𝑦𝑄delimited-[]𝒮𝑄𝑦subscript𝔼similar-to𝑦𝑄delimited-[]𝒮𝑃𝑦for-all𝑃𝑄\mathbb{E}\_{y\sim Q}[\mathcal{S}(Q,y)]\quad\leq\quad\mathbb{E}\_{y\sim Q}[\mathcal{S}(P,y)]\quad\forall P,Q, |  | (1) |

where Q𝑄Q represents the true distribution of outcomes y𝑦y, and P𝑃P is any other distribution (such as the probabilistic forecast from a model). Since we are working with parametric distributions, we can identify each distribution with its parameters and write the score as 𝒮​(θ,y)𝒮𝜃𝑦\mathcal{S}(\theta,y).

The most commonly used proper scoring rule is the logarithmic score ℒℒ\mathcal{L}, which, when minimized, gives the MLE:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(θ,y)=−log⁡Pθ​(y).ℒ𝜃𝑦subscript𝑃𝜃𝑦\mathcal{L}(\theta,y)=-\log P\_{\theta}(y). |  | (2) |

Another example is CRPS, which is generally considered a robust alternative to MLE (Gebetsberger et al.,, [2018](#bib.bib13)). The CRPS (denoted 𝒞𝒞\mathcal{C}) is defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒞​(θ,y)=∫−∞yFθ​(z)2​𝑑z+∫y∞(1−Fθ​(z))2​𝑑z.𝒞𝜃𝑦superscriptsubscript𝑦subscript𝐹𝜃superscript𝑧2differential-d𝑧superscriptsubscript𝑦superscript1subscript𝐹𝜃𝑧2differential-d𝑧\mathcal{C}(\theta,y)=\int\_{-\infty}^{y}F\_{\theta}(z)^{2}dz+\int\_{y}^{\infty}(1-F\_{\theta}(z))^{2}dz. |  | (3) |

### 3.2 The Generalized Natural Gradient

![Refer to caption](/html/1910.03225/assets/x2.png)

![Refer to caption](/html/1910.03225/assets/x3.png)

Figure 3: Proper scoring rules and corresponding gradients for fitting a Normal distribution on samples ∼N​(0,1)similar-toabsent𝑁01\sim N(0,1). For each scoring rule, the landscape of the score (colors and contours) is identical, but the gradient fields (arrows) are markedly different depending on which kind of gradient is used.

We take a standard gradient descent approach to find the parameters that minimize the scoring rule by descending along the negative gradient of the score relative to the parameters at each point x𝑥x. The (ordinary) gradient of a scoring rule 𝒮𝒮\mathcal{S} over a parameterized probability distribution Pθsubscript𝑃𝜃P\_{\theta} with parameter θ𝜃\theta and outcome y𝑦y with respect to the parameters is denoted ∇𝒮​(θ,y)∇𝒮𝜃𝑦\nabla\mathcal{S}(\theta,y). It is the direction of steepest ascent, such that moving the parameters an infinitesimally small amount in that direction of the gradient (as opposed to any other direction) will increase the scoring rule the most. That is,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇𝒮​(θ,y)∇𝒮𝜃𝑦\displaystyle\nabla\mathcal{S}(\theta,y) | ∝limϵ→0arg⁡maxd:‖d‖=ϵ​𝒮​(θ+d,y).proportional-toabsentsubscript→italic-ϵ0:𝑑norm𝑑italic-ϵ𝒮𝜃𝑑𝑦\displaystyle\propto\lim\_{\epsilon\to 0}\underset{d:\|d\|=\epsilon}{\arg\max}\mathcal{S}(\theta+d,y). |  | (4) |

This gradient is *not* invariant to reparameterization. Consider reparameterizing Pθsubscript𝑃𝜃P\_{\theta} to Pz​(θ)​(y)subscript𝑃𝑧𝜃𝑦P\_{z(\theta)}(y) so Pθ​(y∈A)=Pψ​(y∈A)subscript𝑃𝜃𝑦𝐴subscript𝑃𝜓𝑦𝐴P\_{\theta}(y\in A)=P\_{\psi}(y\in A) for all events A𝐴A when ψ=z​(θ)𝜓𝑧𝜃\psi=z(\theta). If the gradient is calculated relative to θ𝜃\theta and an infinitesimal step is taken in that direction, say from θ𝜃\theta to θ+d​θ𝜃𝑑𝜃\theta+d\theta the resulting distribution will be different than if the gradient had been calculated relative to ψ𝜓\psi and a step was taken from ψ𝜓\psi to ψ+d​ψ𝜓𝑑𝜓\psi+d\psi. In other words, Pθ+d​θ​(y∈A)≠Pψ+d​ψ​(y∈A)subscript𝑃𝜃𝑑𝜃𝑦𝐴subscript𝑃𝜓𝑑𝜓𝑦𝐴P\_{\theta+d\theta}(y\in A)\neq P\_{\psi+d\psi}(y\in A). Thus the choice of parameterization can drastically impact the training dynamics, even though the minima are unchanged.

The problem is that “distance” between two parameter values does not correspond to an appropriate “distance” between the distributions that those parameters identify. This motivates the natural gradient (denoted ∇~~∇\tilde{\nabla}), which originated in information geometry (Amari,, [1998](#bib.bib1)).

#### Divergences.

Every proper scoring rule induces a *divergence* that can serve as local distance metric in the space of distributions. A proper scoring rule by definition satisfies the inequality of Eqn [1](#S3.E1 "In 3.1 Proper Scoring Rules ‣ 3 Natural Gradient Boosting ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction"). The excess score of the right hand side over the left is the divergence induced by that scoring rule (Dawid and Musio,, [2014](#bib.bib9)):

|  |  |  |  |
| --- | --- | --- | --- |
|  | D𝒮​(Q∥P)=𝔼y∼Q​[𝒮​(P,y)]−𝔼y∼Q​[𝒮​(Q,y)],subscript𝐷𝒮conditional𝑄𝑃subscript𝔼similar-to𝑦𝑄delimited-[]𝒮𝑃𝑦subscript𝔼similar-to𝑦𝑄delimited-[]𝒮𝑄𝑦D\_{\mathcal{S}}(Q\|P)=\mathbb{E}\_{y\sim Q}[\mathcal{S}(P,y)]-\mathbb{E}\_{y\sim Q}[\mathcal{S}(Q,y)], |  | (5) |

which is necessarily non-negative, and can be interpreted as a measure of difference from one distribution Q𝑄Q to another P𝑃P. The MLE scoring rule induces the Kullback-Leibler divergence (KL divergence, or DK​Lsubscript𝐷𝐾𝐿D\_{KL}), while CRPS induces the L2superscript𝐿2L^{2} divergence (Dawid,, [2007](#bib.bib8); Machete,, [2013](#bib.bib21)).

The divergences DK​Lsubscript𝐷𝐾𝐿D\_{KL} and DL2subscript𝐷superscript𝐿2D\_{L^{2}} are invariant to how Q𝑄Q and P𝑃P are parameterized. Though divergences in general are not symmetric, for small changes of the parameters they are almost symmetric and can serve as a local distance metric. When used as such, a divergence induces a statistical manifold where each point in the manifold corresponds to a probability distribution (Dawid and Musio,, [2014](#bib.bib9)).

#### Natural Gradient.

The generalized natural gradient is the direction of steepest ascent in Riemannian space, which is invariant to parametrization, and is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇~​𝒮​(θ,y)∝limϵ→0arg⁡maxd:D𝒮(Pθ||Pθ+d)=ϵ​𝒮​(θ+d,y).\tilde{\nabla}\mathcal{S}(\theta,y)\propto\lim\_{\epsilon\to 0}\underset{d:D\_{\mathcal{S}}(P\_{\theta}||P\_{\theta+d})=\epsilon}{\arg\max}\mathcal{S}(\theta+d,y). |  | (6) |

If we solve the corresponding optimization problem, we obtain the natural gradient of the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇~​𝒮​(θ,y)∝ℐ𝒮​(θ)−1​∇𝒮​(θ,y)  ~∇𝒮𝜃𝑦proportional-tosubscriptℐ𝒮superscript𝜃1∇𝒮𝜃𝑦\tilde{\nabla}\mathcal{S}(\theta,y)\quad\propto\quad\mathcal{I}\_{\mathcal{S}}(\theta)^{-1}\nabla\mathcal{S}(\theta,y) |  | (7) |

where ℐ𝒮​(θ)subscriptℐ𝒮𝜃\mathcal{I}\_{\mathcal{S}}(\theta) is the Riemannian metric of the statistical manifold at θ𝜃\theta, which is induced by the scoring rule 𝒮𝒮\mathcal{S}. While the natural gradient was originally defined for the statistical manifold with the distance measure induced by DK​Lsubscript𝐷𝐾𝐿D\_{KL} (Martens,, [2014](#bib.bib22)), we provide a more general treatment here that applies to any divergence that corresponds to some proper scoring rule.

By choosing 𝒮=ℒ𝒮ℒ\mathcal{S}=\mathcal{L} (i.e. MLE) and solving the above optimization problem, we get:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇~​ℒ​(θ,y)∝ℐℒ​(θ)−1​∇ℒ​(θ,y)  ~∇ℒ𝜃𝑦proportional-tosubscriptℐℒsuperscript𝜃1∇ℒ𝜃𝑦\tilde{\nabla}\mathcal{L}(\theta,y)\quad\propto\quad\mathcal{I}\_{\mathcal{L}}(\theta)^{-1}\nabla\mathcal{L}(\theta,y) |  | (8) |

where ℐℒ​(θ)subscriptℐℒ𝜃\mathcal{I}\_{\mathcal{L}}(\theta) is the Fisher Information carried by an observation about Pθsubscript𝑃𝜃P\_{\theta}, which is defined as:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℐℒ​(θ)subscriptℐℒ𝜃\displaystyle\mathcal{I}\_{\mathcal{L}}(\theta) | =𝔼y∼Pθ​[∇θℒ​(θ,y)​∇θℒ​(θ,y)T]absentsubscript𝔼similar-to𝑦subscript𝑃𝜃delimited-[]subscript∇𝜃ℒ𝜃𝑦subscript∇𝜃ℒsuperscript𝜃𝑦𝑇\displaystyle=\mathbb{E}\_{y\sim P\_{\theta}}\left[\nabla\_{\theta}\mathcal{L}(\theta,y)\nabla\_{\theta}\mathcal{L}(\theta,y)^{T}\right] |  | (9) |

Similarly, by choosing 𝒮=𝒞𝒮𝒞\mathcal{S}=\mathcal{C} (i.e. CRPS) and solving the above optimization problem, we get:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇~​𝒞​(θ,y)∝ℐ𝒞​(θ)−1​∇𝒞​(θ,y)  ~∇𝒞𝜃𝑦proportional-tosubscriptℐ𝒞superscript𝜃1∇𝒞𝜃𝑦\tilde{\nabla}\mathcal{C}(\theta,y)\quad\propto\quad\mathcal{I}\_{\mathcal{C}}(\theta)^{-1}\nabla\mathcal{C}(\theta,y) |  | (10) |

where ℐ𝒞​(θ)subscriptℐ𝒞𝜃\mathcal{I}\_{\mathcal{C}}(\theta) is the Riemannian metric of the statistical manifold that uses DL2subscript𝐷superscript𝐿2D\_{L^{2}} as the local distance measure, given by (Dawid,, [2007](#bib.bib8)):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℐ𝒞​(θ)subscriptℐ𝒞𝜃\displaystyle\mathcal{I}\_{\mathcal{C}}(\theta) | =2​∫−∞∞∇θFθ​(z)​∇θFθ​(z)T​𝑑z.absent2superscriptsubscriptsubscript∇𝜃subscript𝐹𝜃𝑧subscript∇𝜃subscript𝐹𝜃superscript𝑧𝑇differential-d𝑧\displaystyle=2\int\_{-\infty}^{\infty}\nabla\_{\theta}F\_{\theta}(z)\nabla\_{\theta}F\_{\theta}(z)^{T}dz. |  | (11) |

Using the natural gradient for learning the parameters makes the optimization problem invariant to parametrization and leads to more efficient and stable learning dynamics (Amari,, [1998](#bib.bib1)). Figure [3](#S3.F3 "Figure 3 ‣ 3.2 The Generalized Natural Gradient ‣ 3 Natural Gradient Boosting ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction") shows the vector field of gradients and natural gradients for ℒℒ\mathcal{L} and 𝒞𝒞\mathcal{C} on the parameter space of a Normal distribution parameterized by μ𝜇\mu (mean) and log⁡σ𝜎\log\sigma (logarithm of the standard deviation).

### 3.3 Gradient Boosting

Gradient boosting (Friedman,, [2001](#bib.bib10)) is a supervised learning technique where several weak learners (or base learners) are combined in an additive ensemble. The model is learnt sequentially, where the next base learner is fit against the training objective residual of the current ensemble. The output of the fitted base learner is then scaled by a learning rate and added into the ensemble.

The boosting framework can be generalized to any choice of base learner but most popular implementations use shallow decision trees because they work well in practice (Chen and Guestrin,, [2016](#bib.bib5); Ke et al.,, [2017](#bib.bib18)).

When fitting a decision tree to the gradient, the algorithm partitions the data into axis-aligned slices. Each slice of the partition is associated with a leaf node of the tree, and is made as homogeneous in its response variable (the gradients at that set of data points) as possible. The criterion of homogeneity is typically the sample variance. The prediction value of the leaf node (which is common to all the examples ending up in the leaf node) is then set to be the additive component to the predictions that minimizes the loss the most. This is equivalent to doing a “line search” in the functional optimization problem for each leaf node, and, for some losses, closed form solutions are available. For example, for squared error, the response variables are residuals, and the result of the line search will yield the sample mean of the response variables in the leaf.

We now consider adapting gradient boosting for prediction of parameters θ𝜃\theta in the probabilistic regression context.

### 3.4 NGBoost: Natural Gradient Boosting

The NGBoost algorithm is a supervised learning method for probabilistic prediction that uses boosting to estimate the parameters of a conditional probability distribution P​(y|x)𝑃conditional𝑦𝑥P(y|x) as functions of x𝑥x. Here y𝑦y could be one of several types ({±1}plus-or-minus1\{\pm 1\}, ℝℝ\mathbb{R}, {1,…,K}1…𝐾\{1,\ldots,K\}, ℝ+subscriptℝ\mathbb{R}\_{+}, ℕℕ\mathbb{N}, etc.) and x𝑥x is a vector in ℝdsuperscriptℝ𝑑\mathbb{R}^{d}. In our experiments we focus on real valued outputs, though all of our methods are applicable to other modalities such as classification and survival prediction.

The algorithm has three modular components, which are chosen upfront as a configuration:

* •

  Base learner (f𝑓f),
* •

  Parametric probability distribution (Pθsubscript𝑃𝜃P\_{\theta}), and
* •

  Proper scoring rule (𝒮𝒮\mathcal{S}).

Data: Dataset 𝒟={xi,yi}i=1n𝒟superscriptsubscriptsubscript𝑥𝑖subscript𝑦𝑖𝑖1𝑛\mathcal{D}=\{x\_{i},y\_{i}\}\_{i=1}^{n}.

Input: Boosting iterations M𝑀M, Learning rate η𝜂\eta, Probability distribution with parameter θ𝜃\theta, Proper scoring rule 𝒮𝒮\mathcal{S}, Base learner f𝑓f.

Output: Scalings and base learners {ρ(m),f(m)}m=1M.superscriptsubscriptsuperscript𝜌𝑚superscript𝑓𝑚𝑚1𝑀\{\rho^{(m)},f^{(m)}\}\_{m=1}^{M}.

θ(0)←arg⁡minθ​∑i=1n𝒮​(θ,yi)←superscript𝜃0subscript𝜃superscriptsubscript𝑖1𝑛𝒮𝜃subscript𝑦𝑖\theta^{(0)}\leftarrow\arg\min\_{\theta}\sum\_{i=1}^{n}\mathcal{S}(\theta,y\_{i}) {initialize to marginal}

for *m←1,…,M←𝑚

1…𝑀m\leftarrow 1,\ldots,M* do

for *i←1,…,n←𝑖

1…𝑛i\leftarrow 1,\ldots,n* do

gi(m)←ℐ𝒮​(θi(m−1))−1​∇θ𝒮​(θi(m−1),yi)←superscriptsubscript𝑔𝑖𝑚subscriptℐ𝒮superscriptsuperscriptsubscript𝜃𝑖𝑚11subscript∇𝜃𝒮superscriptsubscript𝜃𝑖𝑚1subscript𝑦𝑖g\_{i}^{(m)}\leftarrow\mathcal{I}\_{\mathcal{S}}\left(\theta\_{i}^{(m-1)}\right)^{-1}{\nabla\_{\theta}}\mathcal{S}\left(\theta\_{i}^{(m-1)},{y}\_{i}\right)

end for

f(m)←𝖿𝗂𝗍​({xi,gi(m)}i=1n)←superscript𝑓𝑚𝖿𝗂𝗍superscriptsubscriptsubscript𝑥𝑖superscriptsubscript𝑔𝑖𝑚𝑖1𝑛f^{(m)}\leftarrow\mathsf{fit}\left(\left\{{x}\_{i},g\_{i}^{(m)}\right\}\_{i=1}^{{n}}\right)  
ρ(m)←arg⁡minρ​∑i=1n𝒮​(θi(m−1)−ρ⋅f(m)​(xi),yi)←superscript𝜌𝑚subscript𝜌superscriptsubscript𝑖1𝑛𝒮superscriptsubscript𝜃𝑖𝑚1⋅𝜌superscript𝑓𝑚subscript𝑥𝑖subscript𝑦𝑖\rho^{(m)}\leftarrow\arg\min\_{\rho}\sum\_{i=1}^{{n}}\mathcal{S}\left(\theta\_{i}^{(m-1)}-\rho\cdot f^{(m)}({x}\_{i}),{y}\_{i}\right)
for *i←1,…,n←𝑖

1…𝑛i\leftarrow 1,\ldots,n* do

θi(m)←θi(m−1)−η​(ρ(m)⋅f(m)​(xi))←superscriptsubscript𝜃𝑖𝑚superscriptsubscript𝜃𝑖𝑚1𝜂⋅superscript𝜌𝑚superscript𝑓𝑚subscript𝑥𝑖\theta\_{i}^{(m)}\leftarrow\theta\_{i}^{(m-1)}-\eta\left(\rho^{(m)}\cdot f^{(m)}({x}\_{i})\right)

end for

end for

Algorithm 1  NGBoost for probabilistic prediction

A prediction y|xconditional𝑦𝑥y|x on a new input x𝑥x is made in the form of a conditional distribution Pθsubscript𝑃𝜃P\_{\theta}, whose parameters θ𝜃\theta are obtained by an additive combination of M𝑀M base learner outputs (corresponding to the M𝑀M gradient boosting stages) and an initial θ(0)superscript𝜃0\theta^{(0)}. Note that θ𝜃\theta can be a vector of parameters (not limited to be scalar valued), and they completely determine the probabilistic prediction y|xconditional𝑦𝑥y|x. For example, when using the Normal distribution, θ=(μ,log⁡σ)𝜃𝜇𝜎\theta=(\mu,\log\sigma) in our experiments. To obtain the predicted parameter θ𝜃\theta for some x𝑥x, each of the base learners f(m)superscript𝑓𝑚f^{(m)} take x𝑥x as their input. Here f(m)superscript𝑓𝑚f^{(m)} collectively refers to the set of base learners, one per parameter, of stage m𝑚m. For example, for a Normal distribution with parameters μ𝜇\mu and log⁡σ𝜎\log\sigma, there will be two base learners, fμ(m)superscriptsubscript𝑓𝜇𝑚f\_{\mu}^{(m)} and flog⁡σ(m)superscriptsubscript𝑓𝜎𝑚f\_{\log\sigma}^{(m)} per stage, collectively denoted as f(m)=(fμ(m),flog⁡σ(m))superscript𝑓𝑚superscriptsubscript𝑓𝜇𝑚superscriptsubscript𝑓𝜎𝑚f^{(m)}=\left(f\_{\mu}^{(m)},f\_{\log\sigma}^{(m)}\right). The predicted outputs are scaled with stage-specific scaling factors ρ(m)superscript𝜌𝑚\rho^{(m)}, and a common learning rate η𝜂\eta:

|  |  |  |
| --- | --- | --- |
|  | y|x∼Pθ​(x),θ=θ(0)−η​∑m=1Mρ(m)⋅f(m)​(x).formulae-sequencesimilar-toconditional𝑦𝑥subscript𝑃𝜃𝑥𝜃superscript𝜃0𝜂superscriptsubscript𝑚1𝑀⋅superscript𝜌𝑚superscript𝑓𝑚𝑥y|x\sim P\_{\theta}(x),\quad\quad\theta=\theta^{(0)}-\eta\sum\_{m=1}^{M}\rho^{(m)}\cdot f^{(m)}(x). |  |

Each scaling factor ρ(m)superscript𝜌𝑚\rho^{(m)} is a single scalar, even if the distribution has multiple parameters. The model is learnt sequentially, a set of base learners f(m)superscript𝑓𝑚f^{(m)} and a scaling factor ρ(m)superscript𝜌𝑚\rho^{(m)} per stage. The learning algorithm starts by first estimating a common θ(0)superscript𝜃0\theta^{(0)} such that it minimizes the sum of the scoring rule 𝒮𝒮\mathcal{S} over the response variables from all training examples, essentially fitting the marginal distribution of y𝑦y. This becomes the initial predicted parameter θ(0)superscript𝜃0\theta^{(0)} for all examples.

In each iteration m𝑚m, the algorithm calculates, for each example i𝑖i, the natural gradients gi(m)superscriptsubscript𝑔𝑖𝑚g\_{i}^{(m)} of the scoring rule 𝒮𝒮\mathcal{S} with respect to the predicted parameters of that example up to that stage, θi(m−1)subscriptsuperscript𝜃𝑚1𝑖\theta^{(m-1)}\_{i}. Note that gi(m)superscriptsubscript𝑔𝑖𝑚g\_{i}^{(m)} has the same dimension as θ𝜃\theta. A set of base learners for that iteration f(m)superscript𝑓𝑚f^{(m)} are fit to predict the corresponding components of the natural gradients gi(m)superscriptsubscript𝑔𝑖𝑚g\_{i}^{(m)} of each example xisubscript𝑥𝑖x\_{i}.

The output of the fitted base learner is the projection of the natural gradient on to the range of the base learner class. This projected gradient is then scaled by a scaling factor ρ(m)superscript𝜌𝑚\rho^{(m)} since local approximations might not hold true very far away from the current parameter position. The scaling factor is chosen to minimize the overall true scoring rule loss along the direction of the projected gradient in the form of a line search. In practice, we found that implementing this line search by successive halving of ρ𝜌\rho (starting with ρ=1𝜌1\rho=1) until the scaled gradient update results in a lower overall loss relative to the previous iteration works reasonably well and is easy to implement.

Once the scaling factor ρ(m)superscript𝜌𝑚\rho^{(m)} is determined, the predicted per-example parameters are updated to θi(m)subscriptsuperscript𝜃𝑚𝑖\theta^{(m)}\_{i} by adding to each θi(m−1)subscriptsuperscript𝜃𝑚1𝑖\theta^{(m-1)}\_{i} the negative scaled projected gradient for i𝑖i, ρ(m)⋅f(m)​(xi)⋅superscript𝜌𝑚superscript𝑓𝑚subscript𝑥𝑖\rho^{(m)}\cdot f^{(m)}(x\_{i}) which is further scaled by a small learning rate η𝜂\eta (typically 0.1 or 0.01).

The pseudo-code is presented in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.4 NGBoost: Natural Gradient Boosting ‣ 3 Natural Gradient Boosting ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction"). For very large datasets computational performance can be easily improved by simply randomly sub-sampling mini-batches within the 𝖿𝗂𝗍​()𝖿𝗂𝗍

\mathsf{fit}() operation.

### 3.5 Analysis and Discussion

(a) 0% fit

![Refer to caption](/html/1910.03225/assets/x4.png)

![Refer to caption](/html/1910.03225/assets/x5.png)

(b) 33% fit

![Refer to caption](/html/1910.03225/assets/x6.png)

![Refer to caption](/html/1910.03225/assets/x7.png)

(c) 67% fit

![Refer to caption](/html/1910.03225/assets/x8.png)

![Refer to caption](/html/1910.03225/assets/x9.png)

(d) 100% fit

![Refer to caption](/html/1910.03225/assets/x10.png)

![Refer to caption](/html/1910.03225/assets/x11.png)

Figure 4: Contrasting the learning dynamics between using the ordinary gradient (top row) vs. the natural gradient (bottom row) for the purpose of gradient boosting the parameters of a Normal distribution on a toy data set. With ordinary gradients, we observe that “lucky” examples that are accidentally close to the initial predicted mean dominate the learning. This is because, under the ordinary gradient, the variances of those examples that have the correct mean get adjusted much more aggressively than the wrong means of the “unlucky” examples. This results in simultaneous overfitting of the “lucky” examples in the middle and underfitting of the “unlucky” examples at the ends. Under the natural gradient, all the updates are better balanced.

#### Boosting for probabilistic prediction.

Our boosting approach generalizes gradient boosting to predict conditional distributions. For instance, if the user specifies the conditional distribution to be a Normal distribution with a *fixed* variance and uses the logarithmic scoring rule, our approach recovers the standard boosting algorithm with MSE loss (modulo the per-leaf line search). The advantage of NGBoost is that users are also free to specify any other family of distributions identified by a set of real-valued parameters and allow all of those parameters to vary over the covariates, not just the mean. NGBoost thus trivially extends to a variety of use cases, such as negative binomial boosting (for counts), Gamma or Weibull boosting (for survival prediction, with or without right-censored data), etc.

#### Multiparameter boosting.

These wide-ranging extensions of gradient boosting are made possible by turning the distributional prediction problem into a problem of jointly estimating p𝑝p functions of x𝑥x, one per parameter, according to the scoring rule objective. In this setting, an overall line search for the stage multiplier (as opposed to per-leaf line search) is an inevitable consequence. However, our use of natural gradient makes this less of a problem as the gradients of all the examples come “optimally pre-scaled” (in both the relative magnitude between parameters, and across examples) due to the inverse Fisher Information factor. The use of ordinary gradients instead would be sub-optimal, as shown in Figure [4](#S3.F4 "Figure 4 ‣ 3.5 Analysis and Discussion ‣ 3 Natural Gradient Boosting ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction"). With the natural gradient the parameters converge at approximately the same rate despite different conditional means, variances, and “*distances*” from the initial marginal distribution, even while being subjected to a common scaling factor ρ(m)superscript𝜌𝑚\rho^{(m)} in each iteration. We attribute this stability to the “optimal pre-scaling” property of the natural gradient.

#### Parameterization.

When the probability distribution is in the exponential family and the choice of parameterization is the natural parameters of that family, then a Newton-Raphson step is equivalent to a natural gradient descent step. However, in other parameterizations and distributions, the equivalence need not hold. This is especially important in the boosting context because, depending on the inductive biases of the base learners, certain parameterization choices may result in more suitable model spaces than others. For example, one setting we are particularly interested in is the two-parameter Normal distribution. Though it is in the exponential family, we use a mean (μ𝜇\mu) and log-scale (log⁡σ𝜎\log\sigma) parameterization for both ease of implementation and modeling convenience (to disentangle magnitude of predictions from uncertainty estimates). Since natural gradients are invariant to parameterization this does not pose a problem, whereas the Newton-Raphson method would fail as the problem is no longer convex in this parameterization.

#### Computational complexity.

There are two computational differences between our algorithm and a standard boosting algorithm which contribute to complexity. The first is that a series of learners must be fit for each parameter in NGBoost, whereas standard boosting fits only one series of learners. The relative increase in computational cost is thus linear in the number of distributional parameters (p𝑝p). The other difference is that we must compute the natural gradient per observation, which requires as many inversions of a p×p𝑝𝑝p\times p matrix ℐ𝒮subscriptℐ𝒮\mathcal{I}\_{\mathcal{S}} as there are observations (N𝑁N). The cost of doing so scales with p3superscript𝑝3p^{3}, and linearly with N𝑁N. In practice, both costs are minimal because most commonly used distributions have only one or two parameters and distributions with more than five are exceedingly rare. Scaling in terms of p𝑝p is therefore not a significant concern. However, even though these costs are otherwise linear in N𝑁N, it may be prudent to avoid inverting a large number of small matrices by sub-sampling mini-batches of data in each boosting iteration. This is done in most implementations of boosting algorithms. All in all, NGBoost scales exactly like other boosting algorithms in terms of N𝑁N but with larger “constants” that depend on p≈100𝑝superscript100p\approx 10^{0}.

## 4 Experiments

Table 1: Comparison of probabilistic regression performance on regression benchmark UCI datasets as measured by NLL. Results for MC dropout, Deep Ensembles, and Concrete Dropout are reported from Gal and Ghahramani, ([2016](#bib.bib11)); Lakshminarayanan et al., ([2017](#bib.bib20)); Gal et al., ([2017](#bib.bib12)) respectively. NGBoost offers competitive performance in terms of NLL, especially on smaller datasets. The best method for each dataset is bolded, as are those with standard errors that overlap with the best method.

| Dataset | N𝑁N | NGBoost | MC dropout | Deep Ensembles | Concrete Dropout | Gaussian Process | GAMLSS | DistForest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Boston | 506 | 2.43 ±plus-or-minus\pm 0.15 | 2.46 ±plus-or-minus\pm 0.25 | 2.41 ±plus-or-minus\pm 0.25 | 2.72 ±plus-or-minus\pm 0.01 | 2.37 ±plus-or-minus\pm 0.24 | 2.73 ±plus-or-minus\pm 0.56 | 2.67 ±plus-or-minus\pm 0.08 |
| Concrete | 1030 | 3.04 ±plus-or-minus\pm 0.17 | 3.04 ±plus-or-minus\pm 0.09 | 3.06 ±plus-or-minus\pm 0.18 | 3.51 ±plus-or-minus\pm 0.00 | 3.03 ±plus-or-minus\pm 0.11 | 3.24 ±plus-or-minus\pm 0.08 | 3.38 ±plus-or-minus\pm 0.05 |
| Energy | 768 | 0.60 ±plus-or-minus\pm 0.45 | 1.99 ±plus-or-minus\pm 0.09 | 1.38 ±plus-or-minus\pm 0.22 | 2.30 ±plus-or-minus\pm 0.00 | 0.66 ±plus-or-minus\pm 0.17 | 1.24 ±plus-or-minus\pm 0.86 | 1.53 ±plus-or-minus\pm 0.14 |
| Kin8nm | 8192 | -0.49 ±plus-or-minus\pm 0.02 | -0.95 ±plus-or-minus\pm 0.03 | -1.20 ±plus-or-minus\pm 0.02 | -0.65 ±plus-or-minus\pm 0.00 | -1.11 ±plus-or-minus\pm 0.03 | -0.26 ±plus-or-minus\pm 0.02 | -0.40 ±plus-or-minus\pm 0.01 |
| Naval | 11934 | -5.34±plus-or-minus\pm 0.04 | -3.80 ±plus-or-minus\pm 0.05 | -5.63 ±plus-or-minus\pm 0.05 | -5.87 ±plus-or-minus\pm 0.05 | -4.98 ±plus-or-minus\pm 0.02 | -5.56 ±plus-or-minus\pm 0.07 | -4.84 ±plus-or-minus\pm 0.01 |
| Power | 9568 | 2.79 ±plus-or-minus\pm 0.11 | 2.80 ±plus-or-minus\pm 0.05 | 2.79 ±plus-or-minus\pm 0.04 | 2.75 ±plus-or-minus\pm 0.01 | 2.81 ±plus-or-minus\pm 0.05 | 2.86 ±plus-or-minus\pm 0.04 | 2.68 ±plus-or-minus\pm 0.05 |
| Protein | 45730 | 2.81 ±plus-or-minus\pm 0.03 | 2.89 ±plus-or-minus\pm 0.01 | 2.83 ±plus-or-minus\pm 0.02 | 2.81 ±plus-or-minus\pm 0.00 | 2.89 ±plus-or-minus\pm 0.02 | 3.00 ±plus-or-minus\pm 0.01 | 2.59 ±plus-or-minus\pm 0.04 |
| Wine | 1588 | 0.91 ±plus-or-minus\pm 0.06 | 0.93 ±plus-or-minus\pm 0.06 | 0.94 ±plus-or-minus\pm 0.12 | 1.70 ±plus-or-minus\pm 0.00 | 0.95 ±plus-or-minus\pm 0.06 | 0.97 ±plus-or-minus\pm 0.09 | 1.05 ±plus-or-minus\pm 0.15 |
| Yacht | 308 | 0.20 ±plus-or-minus\pm 0.26 | 1.55 ±plus-or-minus\pm 0.12 | 1.18 ±plus-or-minus\pm 0.21 | 1.75 ±plus-or-minus\pm 0.00 | 0.10 ±plus-or-minus\pm 0.26 | 0.80 ±plus-or-minus\pm 0.56 | 2.94 ±plus-or-minus\pm 0.09 |
| Year MSD | 515345 | 3.43 ±plus-or-minus\pm NA | 3.59 ±plus-or-minus\pm NA | 3.35 ±plus-or-minus\pm NA | NA±plus-or-minus\pm NA | NA ±plus-or-minus\pm NA | NA ±plus-or-minus\pm NA | NA ±plus-or-minus\pm NA |




Table 2: Comparison of probabilistic regression performance on regression benchmark UCI datasets as measured by NLL while ablating key components of NGBoost. Multiparameter boosting must be used in tandem with the natural gradient to increase performance. Bolding is as in Table 1.

| Dataset | N | NGBoost | 2nd-Order | Multiparameter | Homoscedastic |
| --- | --- | --- | --- | --- | --- |
| Boston | 506 | 2.43 ±plus-or-minus\pm 0.15 | 3.57 ±plus-or-minus\pm 0.20 | 3.17 ±plus-or-minus\pm 0.13 | 2.79 ±plus-or-minus\pm 0.42 |
| Concrete | 1030 | 3.04 ±plus-or-minus\pm 0.17 | 4.21 ±plus-or-minus\pm 0.05 | 3.94 ±plus-or-minus\pm 0.09 | 3.22 ±plus-or-minus\pm 0.29 |
| Energy | 768 | 0.60 ±plus-or-minus\pm 0.45 | 3.64 ±plus-or-minus\pm 0.06 | 3.24 ±plus-or-minus\pm 0.09 | 0.68 ±plus-or-minus\pm 0.25 |
| Kin8nm | 8192 | -0.49 ±plus-or-minus\pm 0.02 | 0.10 ±plus-or-minus\pm 0.07 | -0.52 ±plus-or-minus\pm 0.03 | -0.37 ±plus-or-minus\pm 0.05 |
| Naval | 11934 | -5.34±plus-or-minus\pm 0.04 | -2.80 ±plus-or-minus\pm 0.01 | -3.46 ±plus-or-minus\pm 0.00 | -4.35 ±plus-or-minus\pm 0.07 |
| Power | 9568 | 2.79 ±plus-or-minus\pm 0.11 | 4.11 ±plus-or-minus\pm 0.03 | 3.79 ±plus-or-minus\pm 0.13 | 2.66 ±plus-or-minus\pm 0.11 |
| Protein | 45730 | 2.81 ±plus-or-minus\pm 0.03 | 3.23 ±plus-or-minus\pm 0.00 | 3.04 ±plus-or-minus\pm 0.02 | 2.86 ±plus-or-minus\pm 0.01 |
| Wine | 1588 | 0.91 ±plus-or-minus\pm 0.06 | 1.21 ±plus-or-minus\pm 0.09 | 0.93 ±plus-or-minus\pm 0.07 | 1.34 ±plus-or-minus\pm 0.67 |
| Yacht | 308 | 0.20 ±plus-or-minus\pm 0.26 | 4.11 ±plus-or-minus\pm 0.17 | 3.29 ±plus-or-minus\pm 0.20 | 2.02 ±plus-or-minus\pm 0.21 |
| Year MSD | 515345 | 3.43 ±plus-or-minus\pm NA | 3.80 ±plus-or-minus\pm 0.00 | 3.60 ±plus-or-minus\pm NA | 3.63 ±plus-or-minus\pm NA |




Table 3: Comparison of point-estimation performance on regression benchmark UCI datasets as measured by RMSE. Although not optimized for point estimation, NGBoost still offers competitive performance. Bolding is as in Table [1](#S4.T1 "Table 1 ‣ 4 Experiments ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction").

| Dataset | N𝑁N | NGBoost | Elastic Net | Random Forest | Gradient Boosting | GAMLSS | Distributional Forest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Boston | 506 | 2.94 ±plus-or-minus\pm 0.53 | 4.08 ±plus-or-minus\pm 0.16 | 2.97 ±plus-or-minus\pm 0.30 | 2.46 ±plus-or-minus\pm 0.32 | 4.32 ±plus-or-minus\pm 1.40 | 3.99 ±plus-or-minus\pm 1.13 |
| Concrete | 1030 | 5.06 ±plus-or-minus\pm 0.61 | 12.1 ±plus-or-minus\pm 0.05 | 5.29 ±plus-or-minus\pm 0.16 | 4.46 ±plus-or-minus\pm 0.29 | 6.72 ±plus-or-minus\pm 0.59 | 6.61 ±plus-or-minus\pm 0.83 |
| Energy | 768 | 0.46 ±plus-or-minus\pm 0.06 | 2.75 ±plus-or-minus\pm 0.03 | 0.52 ±plus-or-minus\pm 0.09 | 0.39 ±plus-or-minus\pm 0.02 | 1.43 ±plus-or-minus\pm 0.32 | 1.11 ±plus-or-minus\pm 0.27 |
| Kin8nm | 8192 | 0.16 ±plus-or-minus\pm 0.00 | 0.20 ±plus-or-minus\pm 0.00 | 0.15 ±plus-or-minus\pm 0.00 | 0.14 ±plus-or-minus\pm 0.00 | 0.20 ±plus-or-minus\pm 0.01 | 0.16 ±plus-or-minus\pm 0.00 |
| Naval | 11934 | 0.00 ±plus-or-minus\pm 0.00 | 0.00 ±plus-or-minus\pm 0.00 | 0.00 ±plus-or-minus\pm 0.00 | 0.00 ±plus-or-minus\pm 0.00 | 0.00 ±plus-or-minus\pm 0.00 | 0.00 ±plus-or-minus\pm 0.00 |
| Power | 9568 | 3.79 ±plus-or-minus\pm 0.18 | 4.42 ±plus-or-minus\pm 0.00 | 3.26 ±plus-or-minus\pm 0.03 | 3.01 ±plus-or-minus\pm 0.10 | 4.25 ±plus-or-minus\pm 0.19 | 3.64 ±plus-or-minus\pm 0.24 |
| Protein | 45730 | 4.33 ±plus-or-minus\pm 0.03 | 5.20 ±plus-or-minus\pm 0.00 | 3.60 ±plus-or-minus\pm 0.00 | 3.95 ±plus-or-minus\pm 0.00 | 5.04 ±plus-or-minus\pm 0.04 | 3.89 ±plus-or-minus\pm 0.04 |
| Wine | 1588 | 0.63 ±plus-or-minus\pm 0.04 | 0.58 ±plus-or-minus\pm 0.00 | 0.50 ±plus-or-minus\pm 0.01 | 0.53 ±plus-or-minus\pm 0.02 | 0.64 ±plus-or-minus\pm 0.04 | 0.67 ±plus-or-minus\pm 0.05 |
| Yacht | 308 | 0.50 ±plus-or-minus\pm 0.20 | 7.65 ±plus-or-minus\pm 0.21 | 0.61 ±plus-or-minus\pm 0.08 | 0.42 ±plus-or-minus\pm 0.09 | 8.29 ±plus-or-minus\pm 2.56 | 4.19±plus-or-minus\pm 0.92 |
| Year MSD | 515345 | 8.94 ±plus-or-minus\pm NA | 9.49 ±plus-or-minus\pm NA | 9.05 ±plus-or-minus\pm NA | 8.73 ±plus-or-minus\pm NA | NA ±plus-or-minus\pm NA | NA ±plus-or-minus\pm NA |

Our experiments use datasets from the UCI Machine Learning Repository, and follow the protocol first proposed in Hernández-Lobato and Adams, ([2015](#bib.bib17)). For all datasets, we hold out a random 10% of the examples as a test set. From the other 90% we initially hold out 20% as a validation set to select M𝑀M (the number of boosting stages) that gives the best log-likelihood, and then retrain on the entire 90% using the chosen M𝑀M. The retrained model is then made to predict on the held-out 10% test set. This entire process is repeated 20 times for all datasets except Protein and Year MSD, for which it is repeated 5 times and 1 time respectively.

For all experiments, NGBoost was configured with the Normal distribution, decision tree base learner with a maximum depth of three levels, and log scoring rule. The Year MSD dataset, being extremely large relative to the rest, was fit using a learning rate η𝜂\eta of 0.1 while the rest of the datasets were fit with a learning rate of 0.01. In general we recommend small learning rates, subject to computational feasibility. For the Year MSD dataset we use a mini-batch size of 10%, for all other datasets we use 100%.

### 4.1 Probabilistic regression.

The quality of predictive uncertainty is captured in the average negative log-likelihood (NLL) (i.e. log⁡Pθ^​(y|x)^subscript𝑃𝜃conditional𝑦𝑥\log\hat{P\_{\theta}}(y|x)) as measured on the test set.

Our comparison in this task is against other probabilistic prediction methods. Namely:

MC dropout fits a neural network to the dataset and interprets Bernoulli dropout as a variational approximation for Bayesian inference, obtaining predictive uncertainty by integrating over Monte Carlo samples (Gal and Ghahramani,, [2016](#bib.bib11)). We use the results from Gal and Ghahramani, ([2016](#bib.bib11)) as our benchmark.

Deep Ensembles fit an ensemble of neural networks to the dataset and obtain predictive uncertainty by making an approximation to the Gaussian mixture arising out of the ensemble (Lakshminarayanan et al.,, [2017](#bib.bib20)). We use the results from Lakshminarayanan et al., ([2017](#bib.bib20)) as our benchmark.

Concrete Dropout improves upon MC dropout by employing a continuous relaxation of the Bernoulli distribution to automatically tune the dropout probability (Gal et al.,, [2017](#bib.bib12)). We use the results from Gal et al., ([2017](#bib.bib12)) as our benchmark.

Gaussian Processes are a nonparametric Bayesian method where the response is interpreted as a multivariate Gaussian distribution with covariance given by some kernel between covariates (Rasmussen and Williams,, [2005](#bib.bib25)). Our experiments used an automatic relevance detection kernel fit via gradient-based optimization of the marginal log-likelihood. Datasets with N>2000𝑁2000N>2000 employed 1000 inducing inputs randomly chosen from the training set, with inducing points fit with variational inference as in Titsias, ([2009](#bib.bib29)). All features and labels were standardized to zero-mean and unit variance for pre-processing. The standardized noise level was tuned via grid search for each dataset, with values ranging between 0.01 and 0.1.

GAMLSS uses generalized (parametric) linear models to fit each distributional parameter instead of boosting (Stasinopoulos et al.,, [2007](#bib.bib28)). Responses were parameterized as Normal distributions N​(μ,σ2)𝑁𝜇superscript𝜎2N(\mu,\sigma^{2}). The mean μ𝜇\mu and log-std log⁡σ𝜎\log\sigma were independently modeled as linear combinations of natural cubic splines of the covariates. No interaction terms were included. All features and labels were standardized to zero-mean and unit variance for pre-processing.

Distributional Forests use trees to estimate distributional parameters in each leaf, which are then averaged across the model (Schlosser et al.,, [2019](#bib.bib26)). Responses were parameterized as Normal distributions N​(μ,σ2)𝑁𝜇superscript𝜎2N(\mu,\sigma^{2}). The mean μ𝜇\mu and log-std log⁡σ𝜎\log\sigma were independently modeled using forests consisting of 200 trees and default hyper-parameters (d𝑑\sqrt{d} covariates sampled per split, minimum 20 examples for a split node, minimum 7 examples in a terminal node). All features and labels were standardized to zero-mean and unit variance for pre-processing.

Our probabilistic regression results are summarized in Table [1](#S4.T1 "Table 1 ‣ 4 Experiments ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction"). Results for the Year MSD dataset are unavailable either because they were not reported or because the necessary computations for gradient-based optimization of hyper-parameters did not fit in memory.

### 4.2 Ablation

We compare the NLL of our full NGBoost algorithm on these data versus that of the following comparators, each tuned in the same fashion:

2nd-Order boosting is NGBoost using 2nd-order gradient descent instead of the natural gradient. This tests the added benefit of using the natural gradient vis-a-vis 2nd-order methods. Recent work has argued that the natural gradient improves training dynamics by approximating the Hessian used in 2nd-order methods (Martens,, [2014](#bib.bib22)). We use the “saddle-free” Newton-Raphson method of Dauphin et al., ([2014](#bib.bib7)) in our implementation of 2nd-order multiparameter boosting to provide a strong baseline.

Multiparameter boosting is NGBoost using the ordinary gradient. This tests the added benefit of using the natural gradient vis-a-vis the standard gradient, but still allows for all of the parameters of the distribution to vary across x𝑥x.

Homoscedastic boosting is NGBoost assuming a homoscedastic variance σ2​(x)=σ2=Var​[r]^superscript𝜎2𝑥superscript𝜎2^Vardelimited-[]𝑟\sigma^{2}(x)=\sigma^{2}=\widehat{\text{Var}[r]} where r𝑟r are the training set residuals from a single-parameter (mean) boosting model. This tests the added benefit of allowing parameters other than the conditional mean to vary across x𝑥x. Note that the natural gradient plays no meaningful role when there is only a single parameter estimated with NGBoost.

Our ablation results are summarized in Table [2](#S4.T2 "Table 2 ‣ 4 Experiments ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction").

### 4.3 Point estimation

Although NGBoost is not specifically designed for point estimation, it is easy to extract point estimates of expectations 𝔼^​[y|x]^𝔼delimited-[]conditional𝑦𝑥\hat{\mathbb{E}}[y|x] from the estimated distributions P^θ​(y|x)subscript^𝑃𝜃conditional𝑦𝑥\hat{P}\_{\theta}(y|x). We use this approach in a third evaluation to compare the same NGBoost models as above to the Scikit-Learn implementations of random forests, standard gradient boosting, and elastic net regression (Pedregosa et al.,, [2011](#bib.bib24)). Predictive performance in this evaluation is captured by the root mean squared-error (RMSE) of the predictions on the test set. We performed hyperparameter tuning for each of the comparator methods using the same validation procedure as described above, although optimizing for RMSE instead of NLL for all methods except NGBoost. We compare to:

Elastic Net: We used the Scikit-Learn implementation of elastic net regularized linear models. We tuned over lasso-ridge mixture parameters of 0.01, 0.7, and 0.99 and over a range of regularization parameters between 0.00005 and 0.01. All other parameters were left to their default values.

Random Forest: We used the Scikit-Learn implementation of random forests. We set the number of trees to 500 and left all other parameters at their default values.

Gradient Boosting: We used the Scikit-Learn implementation of gradient boosted trees. We tuned over learning rates of 0.01, 0.05, and 0.1, tree depths of 3 and 4, and number of boosting iterations between 0 and 1000. All other parameters were left to their default values.

Our point prediction results are summarized in Table [3](#S4.T3 "Table 3 ‣ 4 Experiments ‣ NGBoost: Natural Gradient Boosting for Probabilistic Prediction").

## 5 Conclusions

NGBoost is a method for probabilistic prediction with competitive state-of-the-art performance on a variety of datasets. NGBoost combines a multiparameter boosting algorithm with the natural gradient to efficiently estimate how parameters of the presumed outcome distribution vary with the observed features.

NGBoost performs as well as existing methods for probabilistic regression but retains major advantages: NGBoost is flexible, scalable, and easy-to-use. We have not rigorously quantified these advantages in this paper (since they would be largely irrelevant without first establishing performance), but many of the benefits are self-evident. Unlike problem-specific approaches, NGBoost handles classification, regression, survival problems, etc. using the same software package and interface. NGBoost scales to large numbers of features or observations with the same favorable complexity of traditional boosting algorithms. No expert knowledge of deep learning, Bayesian statistics, or Monte Carlo methods is required to use NGBoost. It works out of the box.

Our ablation experiments demonstrate that multiparameter boosting and the natural gradient work together to improve performance. Assuming a uniform variance across all covariates works reasonably well for some datasets, as would be expected, but this is not always the case. However, using multiparameter boosting to relax the homoscedasticity assumption most often results in worse performance, likely due to poor training dynamics. 2nd-order methods result in even worse performance. NGBoost employs the natural gradient to correct the training dynamics of multiparameter boosting. The superiority to 2nd-order methods demonstrates that this is due to exploiting the curvature of the score in distributional space, not the curvature of the score in parameter space. The result is performance that is almost always better than assuming homoscedasticity, sometimes by a large margin.

Furthermore, the advantages of probabilistic regression come almost “for free”. On point estimation tasks NGBoost performs better than elastic net, about on par with random forests, and within striking distance of gradient boosting. This is despite the fact that the NGBoost models were (a) optimized for NLL, not to minimize RMSE and (b) less aggressively tuned. Thus, although point prediction will always be best with a dedicated model for that purpose, the loss in RMSE is not substantial if NGBoost is used in order to support probabilistic regression instead.

There are many avenues for future work. This paper is focused on regression problems for clarity of exposition, but NGBoost is also applicable to classification and to survival problems with right-censored data (using the censored likelihood as a scoring rule). NGBoost could also be used for *joint* prediction: by modeling two outcomes z𝑧z and y𝑦y with a jointly parameterized conditional distribution Pθ​(z,y|x)subscript𝑃𝜃𝑧conditional𝑦𝑥P\_{\theta}(z,y|x), a single NGBoost model could answer any question like “what is the probability that it rains more than 4 inches *and* the temperature is greater than 17°C tomorrow?”.

Some further technical innovations are also worth exploring. The natural gradient loses its invariance property with finite step sizes, which we can address with differential equation solvers for higher-order invariance (Song et al.,, [2018](#bib.bib27)). Better tree-based base learners and regularization (e.g. Chen and Guestrin, ([2016](#bib.bib5)); Ke et al., ([2017](#bib.bib18))) are also likely to improve performance, especially in terms of scaling to large datasets.

Although we have shown empirically that NGBoost is useful for probabilistic prediction, it remains to be seen whether it is useful for inference problems and under what assumptions. For instance, if we assume that y|x∼𝒟​(θ​(x))similar-toconditional𝑦𝑥𝒟𝜃𝑥y|x\sim\mathcal{D}(\theta(x)) for some distribution 𝒟𝒟\mathcal{D} with parameters θ𝜃\theta and we estimate θ^ngb​(x)subscript^𝜃ngb𝑥\hat{\theta}\_{\text{ngb}}(x) using NGBoost, under what conditions do we have that θ^ngb​(x)→θ​(x)→subscript^𝜃ngb𝑥𝜃𝑥\hat{\theta}\_{\text{ngb}}(x)\rightarrow\theta(x) as sample size increases? Are there conditions where the convergence is uniform in x𝑥x? If the model is misspecified (i.e. 𝒟𝒟\mathcal{D} is not correct), are there conditions under which moment estimates from the model are still consistent? Addressing these questions and others like them would be of significant value.

### Acknowledgements

This work was funded in part by the National Institutes of Health. We thank anonymous reviewers for feedback.

## References

* Amari, (1998)

  Amari, S.-i. (1998).
  Natural Gradient Works Efficiently in Learning.
  Neural Computation, page 29.
* Avati et al., (2019)

  Avati, A., Duan, T., Jung, K., Shah, N. H., and Ng, A. (2019).
  Countdown Regression: Sharp and Calibrated Survival
  Predictions.
  In Uncertainty in Artificial Intelligence.
  arXiv: 1806.08324.
* Avati et al., (2018)

  Avati, A., Jung, K., Harman, S., Downing, L., Ng, A., and Shah, N. H. (2018).
  Improving palliative care with deep learning.
  BMC Medical Informatics and Decision Making, 18(4):122.
* Blundell et al., (2015)

  Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra, D. (2015).
  Weight Uncertainty in Neural Network.
  In International Conference on Machine Learning, pages
  1613–1622.
* Chen and Guestrin, (2016)

  Chen, T. and Guestrin, C. (2016).
  XGBoost: A Scalable Tree Boosting System.
  In Proceedings of the 22nd ACM SIGKDD International
  Conference on Knowledge Discovery and Data Mining, KDD ’16,
  pages 785–794, New York, NY, USA. ACM.
* Chipman et al., (2010)

  Chipman, H. A., George, E. I., and McCulloch, R. E. (2010).
  BART: Bayesian additive regression trees.
  The Annals of Applied Statistics, 4(1):266–298.
* Dauphin et al., (2014)

  Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., and Bengio, Y.
  (2014).
  Identifying and attacking the saddle point problem in
  high-dimensional non-convex optimization.
  In Ghahramani, Z., Welling, M., Cortes, C., Lawrence, N. D., and
  Weinberger, K. Q., editors, Advances in Neural Information
  Processing Systems 27, pages 2933–2941. Curran Associates, Inc.
* Dawid, (2007)

  Dawid, A. P. (2007).
  The geometry of proper scoring rules.
  Annals of the Institute of Statistical Mathematics,
  59(1):77–93.
* Dawid and Musio, (2014)

  Dawid, A. P. and Musio, M. (2014).
  Theory and Applications of Proper Scoring Rules.
  METRON, 72(2):169–183.
  arXiv: 1401.0398.
* Friedman, (2001)

  Friedman, J. H. (2001).
  Greedy Function Approximation: A Gradient Boosting
  Machine.
  The Annals of Statistics, 29(5):1189–1232.
* Gal and Ghahramani, (2016)

  Gal, Y. and Ghahramani, Z. (2016).
  Dropout As a Bayesian Approximation: Representing Model
  Uncertainty in Deep Learning.
  In International Conference on Machine Learning,
  ICML’16, pages 1050–1059. JMLR.org.
* Gal et al., (2017)

  Gal, Y., Hron, J., and Kendall, A. (2017).
  Concrete Dropout.
  In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R.,
  Vishwanathan, S., and Garnett, R., editors, Advances in Neural
  Information Processing Systems 30, pages 3581–3590. Curran
  Associates, Inc.
* Gebetsberger et al., (2018)

  Gebetsberger, M., Messner, J. W., Mayr, G. J., and Zeileis, A. (2018).
  Estimation Methods for Nonhomogeneous Regression Models:
  Minimum Continuous Ranked Probability Score versus Maximum
  Likelihood.
  Monthly Weather Review, 146(12):4323–4338.
* Gneiting and Katzfuss, (2014)

  Gneiting, T. and Katzfuss, M. (2014).
  Probabilistic Forecasting.
  Annual Review of Statistics and Its Application, 1(1):125–151.
* Gneiting and Raftery, (2007)

  Gneiting, T. and Raftery, A. E. (2007).
  Strictly Proper Scoring Rules, Prediction, and Estimation.
  Journal of the American Statistical Association,
  102(477):359–378.
* Graves, (2011)

  Graves, A. (2011).
  Practical Variational Inference for Neural Networks.
  In Shawe-Taylor, J., Zemel, R. S., Bartlett, P. L., Pereira, F., and
  Weinberger, K. Q., editors, Advances in Neural Information
  Processing Systems 24, pages 2348–2356. Curran Associates, Inc.
* Hernández-Lobato and Adams,
  (2015)

  Hernández-Lobato, J. M. and Adams, R. P. (2015).
  Probabilistic Backpropagation for Scalable Learning of
  Bayesian Neural Networks.
  In International Conference on Machine Learning,
  ICML’15, pages 1861–1869. JMLR.org.
* Ke et al., (2017)

  Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu,
  T.-Y. (2017).
  LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree.
  In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R.,
  Vishwanathan, S., and Garnett, R., editors, Advances in Neural
  Information Processing Systems 30, pages 3146–3154. Curran
  Associates, Inc.
* Kruchten, (2016)

  Kruchten, N. (2016).
  Machine learning meets economics.
* Lakshminarayanan et al., (2017)

  Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017).
  Simple and Scalable Predictive Uncertainty Estimation using
  Deep Ensembles.
  In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R.,
  Vishwanathan, S., and Garnett, R., editors, Advances in Neural
  Information Processing Systems 30, pages 6402–6413. Curran
  Associates, Inc.
* Machete, (2013)

  Machete, R. L. (2013).
  Contrasting probabilistic scoring rules.
  Journal of Statistical Planning and Inference,
  143(10):1781–1790.
* Martens, (2014)

  Martens, J. (2014).
  New insights and perspectives on the natural gradient method.
  Technical report.
  arXiv: 1412.1193.
* Neal, (1996)

  Neal, R. M. (1996).
  Bayesian Learning for Neural Networks.
  Springer-Verlag, Berlin, Heidelberg.
* Pedregosa et al., (2011)

  Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
  O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J.,
  Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E.
  (2011).
  Scikit-learn: Machine Learning in Python.
  Journal of Machine Learning Research, 12:2825–2830.
* Rasmussen and Williams, (2005)

  Rasmussen, C. E. and Williams, C. K. I. (2005).
  Gaussian Processes for Machine Learning (Adaptive
  Computation and Machine Learning).
  The MIT Press.
* Schlosser et al., (2019)

  Schlosser, L., Hothorn, T., Stauffer, R., and Zeileis, A. (2019).
  Distributional regression forests for probabilistic precipitation
  forecasting in complex terrain.
  The Annals of Applied Statistics, 13(3):1564–1589.
  Publisher: Institute of Mathematical Statistics.
* Song et al., (2018)

  Song, Y., Song, J., and Ermon, S. (2018).
  Accelerating Natural Gradient with Higher-Order Invariance.
  In International Conference on Machine Learning, pages
  4713–4722.
* Stasinopoulos et al., (2007)

  Stasinopoulos, D. M., Rigby, R. A., et al. (2007).
  Generalized additive models for location scale and shape (gamlss) in
  r.
  Journal of Statistical Software, 23(7):1–46.
* Titsias, (2009)

  Titsias, M. (2009).
  Variational Learning of Inducing Variables in Sparse
  Gaussian Processes.
  In International Conference on Artificial Intelligence and
  Statistics, pages 567–574.

[◄](/html/1910.03224)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1910.03225)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1910.03225)
[View original  
on arXiv](https://arxiv.org/abs/1910.03225)[►](/html/1910.03226)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Mar 2 15:06:14 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
