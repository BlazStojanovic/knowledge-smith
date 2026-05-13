---
arxiv: '2604.01411'
authors:
- Nicholas Roberts
- Sungjun Cho
- Zhiqi Gao
- Tzu-Heng Huang
- Albert Wu
- Gabriel Orlanski
- Avi Trost
- Kelly Buchanan
- Aws Albarghouthi
- Frederic Sala
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Test-Time Scaling Makes Overtraining Compute-Optimal
url: https://arxiv.org/abs/2604.01411
year: 2026
---

[2604.01411] Test-Time Scaling Makes Overtraining Compute-Optimal














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



# Test-Time Scaling Makes Overtraining Compute-Optimal

Nicholas Roberts       Sungjun Cho      Zhiqi Gao      Tzu-Heng Huang      Albert Wu
Corresponding author: [nick11roberts@cs.wisc.edu](mailto:nick11roberts@cs.wisc.edu).
  
Gabriel Orlanski   Avi Trost   Kelly Buchanan   Aws Albarghouthi   Frederic Sala
Affiliation: University of Wisconsin-Madison     Stanford University

###### Abstract

Modern LLMs scale at test-time, e.g. via repeated sampling, where inference cost grows with model size and the number of samples.
This creates a trade-off that pretraining scaling laws, such as Chinchilla, do not address.
We present Train-to-Test (T2T^{2}) scaling laws that jointly optimize model size, training tokens, and number of inference samples under fixed end-to-end budgets.
T2T^{2} modernizes pretraining scaling laws with pass@kk modeling used for test-time scaling, then jointly optimizes pretraining and test-time decisions.
Forecasts from T2T^{2} are robust over distinct modeling approaches: measuring joint scaling effect on the task loss and modeling impact on task accuracy.
Across eight downstream tasks, we find that when accounting for inference cost, optimal pretraining decisions shift radically into the overtraining regime, well-outside of the range of standard pretraining scaling suites.
We validate our results by pretraining heavily overtrained models in the optimal region that T2T^{2} scaling forecasts, confirming their substantially stronger performance compared to pretraining scaling alone.
Finally, as frontier LLMs are post-trained, we show that our findings survive the post-training stage, making T2T^{2} scaling meaningful in modern deployments.

## 1 Introduction

Pretraining scaling laws tell us how to optimally train language models, but not how to deploy them (kaplan2020scaling; hoffmann2022training).
Test-time scaling laws tell us how to optimally allocate compute at deployment, but not how to train models (snell2024scaling; brown2025large).
The two have developed largely in isolation, yet are fundamentally coupled. Model size and training duration determine both the quality and cost of inference samples.
Models designed to reason through frontier research problems will be sampled from hundreds or thousands of times (jaech2024openai; guo2025deepseek); these should be trained differently from chat models that instantly answer everyday questions.

Should parameter and token counts change if you know how your model will be used at test time?
In practice, Chinchilla (hoffmann2022training) scaling laws guide the allocation of pretraining compute for flagship models.
However, modern model releases are families spanning a range of sizes (touvron2023llama; groeneveld2024olmo; qwen2024qwen2), with the lower end intentionally overtrained well beyond Chinchilla-optimal ratios to reduce per-query inference cost.
This makes them natural candidates for test-time scaling, yet nothing connects pretraining decisions to this inference strategy.
No existing scaling law captures the core tradeoff: smaller models are cheaper per sample but weaker per sample, and the benefit of repeated sampling is a highly nonlinear function of per-sample quality.

Unifying pretraining and inference scaling is challenging because the two regimes operate under fundamentally different evaluation criteria.
Pretraining is evaluated using the loss, a smooth, continuous quantity.
Test-time scaling, by contrast, is evaluated through downstream task metrics such as pass@kk—the probability of producing at least one correct answer in kk independent attempts.
Should a unified scaling law across pretraining and test-time scaling model the loss or model the pass@kk accuracy?

Prior work has addressed pieces of this problem but not the whole.
sardana2024beyond extends Chinchilla to account for inference cost, but considers only the aggregate volume of single-pass serving instead of the multiplicative cost and performance gains from repeated sampling.
Recent studies empirically show that allocating more inference compute to smaller models via repeated sampling can match or exceed the performance of larger ones (brown2025large; snell2024scaling), but they treat pretrained models as given and do not address how they should have been trained.
schaeffer2025pretraining develop scaling laws that predict pass@kk from pretraining compute, but treat this as forecasting rather than an optimization problem—they predict what performance *will be* for a given model, not what model *should be* trained for a given budget.
No existing work jointly optimizes model size, training duration, and the number of inference samples under a single compute budget.

In this work, we close the loop between pre-training and test-time scaling.
We propose Train-to-Test (T2T^{2}) scaling laws that predict performance as a function of model size NN, training tokens DD, and number of samples kk, and optimize over all three under a total compute budget that includes both training (6​N​D6ND) and inference (2​N​k2Nk) cost.
Following Chinchilla, we evaluate multiple modeling approaches: whether to model the loss or pass@kk as functions of NN, DD, and kk.
Although the two approaches are quite different, we find that they agree closely: both suggest substantial overtraining and test-time scaling across our evaluations.
We build on an existing set of Chinchilla scaling checkpoints from porian2024resolving, extending it into the overtrained regime and assembling a testbed of over 100 models across 12 compute levels spanning three orders of magnitude.

![Refer to caption](/html/2604.01411/assets/x1.png)


Figure 1: Our T2T^{2} scaling laws combine Chinchilla scaling for pretraining with pass@kk modeling for test-time scaling via repeated sampling to obtain optimal pretraining allocations subject to a test-time scaling budget. T2T^{2} recommends overtraining compared to Chinchilla.

Using T2T^{2} scaling laws, we find that *optimal pretraining decisions shift radically into the overtraining regime* when considering test-time compute.
When we correct for the cost of repeated sampling, the optimal model is substantially smaller and more overtrained than what Chinchilla prescribes.
Our evaluation spans eight tasks covering knowledge, reasoning, and language understanding, on which we investigate three research questions:

1. RQ1

   Should pretraining change if you know your test-time scaling budget? Yes—T2T^{2} scaling consistently recommends small overtrained models. (§[4.1](#S4.SS1 "4.1 RQ1: Should Pretraining Change if You Know Your Test-Time Scaling Budget? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"))
2. RQ2

   Does T2T^{2} extrapolate to overtrained checkpoints? Yes—we overtrain models from scratch and show that they consistently outperform Chinchilla checkpoints. (§[4.2](#S4.SS2 "4.2 RQ2: Does 𝑇² Scaling Extrapolate to Overtrained Checkpoints? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"))
3. RQ3

   Does T2T^{2} scaling survive post-training? Yes—we find that compute-optimal trade-offs derived from base models persist after supervised fine-tuning. (§[4.3](#S4.SS3 "4.3 RQ3: Does 𝑇² Scaling Survive Post-Training? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"))

To answer these questions, we make the following contributions:

Contributions

•

End-to-end scaling: We formalize train-to-test scaling as a joint optimization over model size NN, dataset size DD, and inference compute kk under train and test budgets.
•

Loss and accuracy scaling: We introduce two complementary approaches: (i) loss- and (ii) accuracy-based formulations that explicitly incorporate inference cost.
•

Validation on overtrained checkpoints: We train models in the predicted overtrained regime and show improved performance under a range of fixed inference budgets.
•

Interactions with post-training: The predictions from our scaling approach persist after post-training, even though overtrained models are harder to fine-tune.

## 2 Background

Our work connects two important areas: (i) pretraining scaling laws and (ii) test-time sampling strategies after deployment.
We begin with their setups then dive into our new modeling techniques.
A summary of additional related work can be found in Appendix [A](#A1 "Appendix A Related Work ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").

Chinchilla scaling laws for pretraining.
The Chinchilla scaling law (hoffmann2022training) models the pretraining loss as a function of finite model capacity NN and dataset size DD (number of training tokens): L​(N,D)=E+ANα+BDβL(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}},
where EE represents an irreducible loss floor fit for the given data distribution and evaluation setup while the remaining terms capture reducible contributions from NN and DD.
The parameters AA, BB, α\alpha, β\beta, and EE are all non-negative and are fit empirically from a grid of training runs.
Here, the loss is assumed to be the negative log-likelihood (NLL) over the data distribution: 𝔼(x,y)∼𝒟​[−log⁡(p​(y|x))]\mathbb{E}\_{(x,y)\sim\mathcal{D}}[-\log(p(y|x))] with p​(y|x)p(y|x) being the probability assigned by the model.
Given a pretraining budget Ctrain≈6​N​DC\_{\text{train}}\approx 6ND, the *compute-optima* minimize LL subject to this constraint,
yielding N∗​(Ctrain)∝CtrainaN^{\*}(C\_{\text{train}})\propto C\_{\text{train}}^{a} and D∗​(Ctrain)∝CtrainbD^{\*}(C\_{\text{train}})\propto C\_{\text{train}}^{b} with a≈b≈0.5a\approx b\approx 0.5.
That is, the optimal model size and training tokens should scale at similar rates as a function of the pretraining compute budget.

Pass@k estimation for test-time scaling.
The standard metric for evaluating repeated sampling is pass@kk: draw kk independent samples from a model and succeed if *any* sample is correct.
For a single problem ii with per-sample success probability pip\_{i}, the probability of at least one answer in kk attempts being correct is pass@​ki=1−(1−pi)k\text{pass@}k\_{i}=1-(1-p\_{i})^{k}.
Aggregating over a benchmark 𝒟\mathcal{D} of MM problems gives the expected pass@kk:

|  |  |  |
| --- | --- | --- |
|  | pass@​k𝒟=𝔼i∼𝒟​[pass@​ki]=1M​∑i=1M[1−(1−pi)k].\text{pass@}k\_{\mathcal{D}}=\mathbb{E}\_{i\sim\mathcal{D}}\left[\text{pass@}k\_{i}\right]=\dfrac{1}{M}\sum\_{i=1}^{M}\left[1-(1-p\_{i})^{k}\right]. |  |

## 3 Estimating Optimal Pretraining Allocations for Test-Time Scaling

We present two modeling approaches for T2T^{2} scaling that answer our central research question: should choices made during pretraining change if you know your test-time scaling budget?
In our first approach, we model the impact of repeated sampling on the loss by fitting a parametric function of the negative log pass@kk.
In our second approach, we model the pass@kk accuracy directly by composing Chinchilla scaling with a pass@kk estimator.
In §[4](#S4 "4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"), we show that our findings are robust across both approaches.
Finally, once we establish these two approaches, we answer our main research question by standardizing the test-time scaling budget: using more repeated samples for smaller models and fewer for larger models.
Standardizing the inference budget of test-time scaling across checkpoints allows us to see how optimal pretraining decisions shift in light of test-time scaling considerations.
If the optimal pretraining decisions (model size and the number of training tokens) shift compared to those recommended by standard Chinchilla scaling, then the answer to RQ1 is yes: pretraining decisions should change if you know your test-time scaling budget.

We first describe the optimization objectives of our T2T^{2} approaches.
Given a compute budget for training (CtrainC\_{\text{train}}) and inference (CinfC\_{\text{inf}}), the optimization problem in terms of the NLL is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minN,D,k⁡L​(N,D,k)s.t.6​N​D≤Ctrain​ and ​  2​N​k≤Cinf,\min\_{N,D,k}\;\;L(N,D,k)\qquad\text{s.t.}\quad 6ND\leq C\_{\text{train}}\,\,\text{ and }\,\,2Nk\leq C\_{\text{inf}}, |  | (1) |

or similarly, in terms of the pass@kk accuracy:

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxN,D,k⁡Acc​(N,D,k)s.t.6​N​D≤Ctrain​ and ​  2​N​k≤Cinf.\max\_{N,D,k}\;\;\text{Acc}(N,D,k)\qquad\text{s.t.}\quad 6ND\leq C\_{\text{train}}\,\,\text{ and }\,\,2Nk\leq C\_{\text{inf}}. |  | (2) |

L​(N,D,k)L(N,D,k) and Acc​(N,D,k)\text{Acc}(N,D,k) represent the aggregated NLL and accuracy respectively, as functions of model capacity NN, dataset size DD, and number of sampling attempts kk.

### 3.1 Approach 1: T2T^{2} as a Parametric Model of the Task Loss

Our first approach models the loss as a function of the parameter count NN, training tokens DD, and the number of repeated samples kk used at test-time in order to optimize Equation [1](#S3.E1 "In 3 Estimating Optimal Pretraining Allocations for Test-Time Scaling ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").
First, in order to make repeated sampling compatible with the negative log likelihood (NLL), we rewrite the single-sample probability in terms of the probability that the target outcome is obtained at least once under kk repeated samples, following prior work on pass@kk (chen2021evaluating; brown2025large; ehrlich2025codemonkeys; schaeffer2025how).
That is, working with the definition of pass@​ki\text{pass@}k\_{i} allows us to define the corresponding NLL-style objective under repeated sampling as

|  |  |  |
| --- | --- | --- |
|  | 𝔼i∼𝒟task​[−log⁡pass@​ki]=𝔼i∼𝒟task​[−log⁡(1−(1−pi)k)],\mathbb{E}\_{i\sim\mathcal{D}\_{\text{task}}}[-\log\text{pass@}k\_{i}]=\mathbb{E}\_{i\sim\mathcal{D}\_{\text{task}}}\left[-\log\left(1-(1-p\_{i})^{k}\right)\right], |  |

where 𝒟task\mathcal{D}\_{\text{task}} is a distribution over samples ii representing a downstream task.

With this in place, we can model the negative log pass@kk as an extension of the Chinchilla scaling law, L^​(N,D)\widehat{L}(N,D) by adding a power-law term in kk:

|  |  |  |
| --- | --- | --- |
|  | L^​(N,D,k)=L^​(N,D)+Gkγ=E+ANα+BDβ+Gkγ.\widehat{L}(N,D,k)=\widehat{L}(N,D)+\frac{G}{k^{\gamma}}=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}+\frac{G}{k^{\gamma}}. |  |

We choose this model because prior work has found that the negative log pass@kk contribution from kk yields power law scaling111By Jensen’s inequality, our NLL-style objective acts as an upper-bounding surrogate on the negative log expected pass@kk, which scales as a power law (we minimize the expected negative log pass@kk). Therefore, minimizing our surrogate minimizes the quantity of interest. under an assumption that the task difficulty distribution can be modeled by a Beta distribution, which has been found to hold in practice (brown2025large; schaeffer2025how).
This has convenient properties when combined with the other power law terms in NN and DD in the Chinchilla scaling law:

First, when k=1k=1, we recover standard Chinchilla scaling:

|  |  |  |
| --- | --- | --- |
|  | L^​(N,D,1)=E′+ANα+BDβ=L^​(N,D),\widehat{L}(N,D,1)=E^{\prime}+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}=\widehat{L}(N,D), |  |

where E′=E+GE^{\prime}=E+G absorbs the additional constant.
Second, a property of Chinchilla scaling is that as N,D→∞N,D\to\infty, the model approaches the ‘irreducible loss’ term EE.
Given its power law form, this is still true when kk approaches infinity alongside NN and DD.

### 3.2 Approach 2: T2T^{2} as a Parametric Model of the Task Accuracy

While the previous model is simple, it trades off interpretability—practitioners often value pass@kk forecasts due to their interpretation as the likelihood of solving a problem given a certain compute investment.
Our second approach addresses this by modeling the pass@kk directly as an accuracy-like metric as a function of NN, DD, and kk, which optimizes Equation [2](#S3.E2 "In 3 Estimating Optimal Pretraining Allocations for Test-Time Scaling ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").

A naive approach to modeling pass@kk might be to begin with L^​(N,D)\widehat{L}(N,D), and simply map the NLL to accuracy pp for the same task, then compute pass@​k=1−(1−p)k\text{pass@}k=1-(1-p)^{k}.
Prior work has shown that the relationship between the mean NLL and the mean accuracy can be well approximated using a fitted sigmoid (llama3).
In other words, we can model the mean single-pass task accuracy, 𝔼𝒟task​[Acc​(N,D)]\mathbb{E}\_{\mathcal{D}\_{\text{task}}}[\text{Acc}(N,D)], as σθ​(L^​(N,D))\sigma\_{\theta}(\widehat{L}(N,D)) with a parameterized sigmoid σθ\sigma\_{\theta} fit to pairs of NLL and accuracy values on the task distribution across the model population.
So this naive model of the pass@kk might take the following form:

|  |  |  |
| --- | --- | --- |
|  | Acc^naive​(N,D,k)=1−(1−σθ​(L​(N,D)))k.\widehat{\text{Acc}}\_{\text{naive}}(N,D,k)=1-(1-\sigma\_{\theta}(L(N,D)))^{k}. |  |

However, our goal is instead to obtain an estimator of the mean pass@kk accuracy, 𝔼𝒟task​[Acc​(N,D,k)]\mathbb{E}\_{\mathcal{D}\_{\text{task}}}[\text{Acc}(N,D,k)] that depends on the scaling parameters, rather than the single-pass accuracy, so this naive model overestimates due to the concavity of the pass@kk:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1−(1−𝔼𝒟task​[Acc​(N,D)])k\displaystyle 1-(1-\mathbb{E}\_{\mathcal{D}\_{\text{task}}}[\text{Acc}(N,D)])^{k} | ≥𝔼𝒟task​[1−(1−Acc​(N,D))k]\displaystyle\geq\mathbb{E}\_{\mathcal{D}\_{\text{task}}}[1-(1-\text{Acc}(N,D))^{k}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼𝒟task​[Acc​(N,D,k)].\displaystyle=\mathbb{E}\_{\mathcal{D}\_{\text{task}}}[\text{Acc}(N,D,k)]. |  |

A simple way to avoid overestimating the pass@kk would be to directly use the per-question probabilities from model likelihoods, which would allow us to compute the mean pass@kk exactly.
However, our goal is a scaling law, a parametric model that can forecast pass@kk at unevaluated (N,D,k)(N,D,k) configurations.
This requires us to model the distribution of per-question probabilities and how this distribution varies with model size and training tokens.

Intuitively, we want to account for the natural spread of difficulty between tasks in our data distribution.
We do this by modeling the per-question single-pass accuracies as a Beta distribution, following prior work (kazdan2025efficient).
We model Acc​(N,D)∼Beta​(aN,D,bN,D)\text{Acc}(N,D)\sim\mathrm{Beta}(a\_{N,D},b\_{N,D}),
and parameters aN,Da\_{N,D} with bN,Db\_{N,D} related to NN and DD via the NLL, which we model as a Beta regression problem.
Using the mean (μ\mu) and sample size (ν\nu) parameterization of the Beta distribution, we model μ∈(0,1)\mu\in(0,1) and ν∈(0,∞)\nu\in(0,\infty) using standard link functions from Beta regression: a logit link for the mean (which we rescale with an additional parameter), and a log link for the sample size.
We relate this to the loss by using the Chinchilla loss estimate as our linear predictor.
This yields the following parameterization of aN,Da\_{N,D} and bN,Db\_{N,D}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | μN,D\displaystyle\mu\_{N,D} | =σθ​(L^​(N,D))=θ21+exp⁡(θ1⋅(L^​(N,D)−θ0)),\displaystyle=\sigma\_{\theta}(\widehat{L}(N,D))=\frac{\theta\_{2}}{1+\exp\bigl(\theta\_{1}\cdot(\widehat{L}(N,D)-\theta\_{0})\bigr)}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | νN,D\displaystyle\nu\_{N,D} | =exp⁡(θ3+θ4⋅L^​(N,D)),\displaystyle=\exp(\theta\_{3}+\theta\_{4}\cdot\widehat{L}(N,D)), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | aN,D\displaystyle a\_{N,D} | =μN,D​νN,D,\displaystyle=\mu\_{N,D}\nu\_{N,D}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | bN,D\displaystyle b\_{N,D} | =(1−μN,D)​νN,D.\displaystyle=(1-\mu\_{N,D})\nu\_{N,D}. |  |

Finally, using this model of the single-pass accuracy, we obtain the following pass@kk model via properties of the Beta distribution:222B​(a,b)=Γ​(a)​Γ​(b)Γ​(a+b)\mathrm{B}(a,b)=\frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)} is the Beta function, where Γ\Gamma is the Gamma function.

|  |  |  |  |
| --- | --- | --- | --- |
|  | Acc^​(N,D,k)\displaystyle\widehat{\text{Acc}}(N,D,k) | =𝔼Acc​(N,D)∼Beta​(aN,D,bN,D)​[1−(1−Acc​(N,D))k]\displaystyle=\mathbb{E}\_{\text{Acc}(N,D)\sim\mathrm{Beta}(a\_{N,D},b\_{N,D})}\bigl[1-(1-\text{Acc}(N,D))^{k}\bigr] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−𝔼Acc​(N,D)∼Beta​(aN,D,bN,D)​[(1−Acc​(N,D))k]\displaystyle=1-\mathbb{E}\_{\text{Acc}(N,D)\sim\mathrm{Beta}(a\_{N,D},b\_{N,D})}\bigl[(1-\text{Acc}(N,D))^{k}\bigr] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−B​(aN,D,bN,D+k)B​(aN,D,bN,D)\displaystyle=1-\frac{\mathrm{B}(a\_{N,D},\,b\_{N,D}+k)}{\mathrm{B}(a\_{N,D},\,b\_{N,D})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−B​(μN,D​νN,D,(1−μN,D)​νN,D+k)B​(μN,D​νN,D,(1−μN,D)​νN,D).\displaystyle=1-\frac{\mathrm{B}(\mu\_{N,D}\nu\_{N,D},\,(1-\mu\_{N,D})\nu\_{N,D}+k)}{\mathrm{B}(\mu\_{N,D}\nu\_{N,D},\,(1-\mu\_{N,D})\nu\_{N,D})}. |  |

### 3.3 Inference Cost Correction

We equalize our T2T^{2} scaling laws over an inference budget, CinfC\_{\text{inf}}, measured as the inference FLOPs per-token served.
Just as the pretraining cost, Ctrain=6​N​DC\_{\text{train}}=6ND, scales multiplicatively as a function of NN and the number of training tokens DD, the inference budget CinfC\_{\text{inf}} scales multiplicatively in kk and approximately 2​N2N FLOPs for a forward pass:

|  |  |  |
| --- | --- | --- |
|  | Cinf=2​N​k.C\_{\text{inf}}=2Nk. |  |

Then for a fixed budget CinfC\_{\text{inf}}, this gives us

|  |  |  |
| --- | --- | --- |
|  | k=Cinf2​N,k=\frac{C\_{\text{inf}}}{2N}, |  |

where smaller models are allocated more repeated samples compared to larger models, subject to the same inference budget.
We plug this into both of our T2T^{2} scaling approaches, which gives us our inference-corrected loss model:333Optimization details for fitting Approach 1 and Approach 2 can be found in Appendix [F](#A6 "Appendix F Fitting 𝑇² Scaling ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").

Approach 1

L^​(N,D,Cinf2​N)=L^​(N,D)+Gkγ=E+ANα+BDβ+G(Cinf2​N)γ,\widehat{L}\left(N,D,\frac{C\_{\text{inf}}}{2N}\right)=\widehat{L}(N,D)+\frac{G}{k^{\gamma}}=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}+\frac{G}{\left(\frac{C\_{\text{inf}}}{2N}\right)^{\gamma}},

and our inference-corrected pass@kk accuracy model:

Approach 2

Acc^​(N,D,Cinf2​N)=1−B​(μN,D​νN,D,(1−μN,D)​νN,D+Cinf2​N)B​(μN,D​νN,D,(1−μN,D)​νN,D).\widehat{\text{Acc}}\left(N,D,\frac{C\_{\text{inf}}}{2N}\right)=1-\frac{\mathrm{B}(\mu\_{N,D}\nu\_{N,D},\,(1-\mu\_{N,D})\nu\_{N,D}+\frac{C\_{\text{inf}}}{2N})}{\mathrm{B}(\mu\_{N,D}\nu\_{N,D},\,(1-\mu\_{N,D})\nu\_{N,D})}.

Now for both models, we can choose an inference budget CinfC\_{\text{inf}}, and observe the pretraining decisions that optimize both the pretraining and inference budgets CtrainC\_{\text{train}} and CinfC\_{\text{inf}}.
We represent Approach 1 in blue and Approach 2 in red for consistency with our Figures.

## 4 Experiments

In this section, we provide experimental results addressing the three research questions about our T2T^{2} scaling approaches.First, in §[4.1](#S4.SS1 "4.1 RQ1: Should Pretraining Change if You Know Your Test-Time Scaling Budget? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"), we show that if you know your test-time scaling budget prior to pretraining, you should overtrain significantly beyond the standard Chinchilla recommendation of 20 tokens per parameter.
In §[4.2](#S4.SS2 "4.2 RQ2: Does 𝑇² Scaling Extrapolate to Overtrained Checkpoints? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"), we validate our predictions against overtrained checkpoints that extend standard Chinchilla scaling suites, showing that our scaling approaches extrapolate to the optimal regions that they predict.
Finally, in §[4.3](#S4.SS3 "4.3 RQ3: Does 𝑇² Scaling Survive Post-Training? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"), we show that overtraining predictions from our T2T^{2} approaches persist after post-training.
We fit T2T^{2} scaling to checkpoints from porian2024resolving, which we extend with additional overtrained checkpoints, all trained on RefinedWeb (refinedweb).

Tasks.
We evaluate T2T^{2} across eight real and synthetic tasks that we select to be simple enough for small base models, as all of our checkpoints have fewer than 1B parameters.
The real tasks that we evaluate include the OpenAI variant of LAMBADA (lambada; radford2019language), ARC-Easy (ai2arc), SciQ (SciQ), and OpenBookQA (OpenBookQA2018).
We also evaluate on four synthetic tasks: simple knowledge recall, multi-step arithmetic reasoning, commonsense causal reasoning, and spatial reasoning, each consisting of 1,000 fill-in-the-blank or short completion questions that were generated using GPT-5 and Claude Opus 4.6.
We provide additional task details in Appendix [E](#A5 "Appendix E Evaluation Tasks ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").
Unless otherwise noted, we present macro averaged results over all tasks.

|  |
| --- |
| Refer to caption |

Figure 2: 
Optimal pretraining forecasts predicted by both T2T^{2} approaches, compared to hoffmann2022training. (Left) Optimal tokens per parameter (including the 20 tokens per parameter rule of thumb used by practitioners), (Middle) Optimal model sizes. (Right) Optimal training set sizes. Both T2T^{2} approaches forecast extreme overtraining.

### 4.1 RQ1: Should Pretraining Change if You Know Your Test-Time Scaling Budget?

We evaluate RQ1 by comparing the predictions from T2T^{2} to Chinchilla scaling and find that if you know your test-time scaling budget, you should significantly overtrain.

Setup.
We fit both T2T^{2} approaches to a suite of 106 checkpoints ranging in size from 5M to 901M parameters trained on roughly 50M to 120B tokens.
Next, we set the per-token inference budget Cinf=140​BC\_{\text{inf}}=140\text{B} FLOPs, or approximately the cost of a single forward pass using the 70B Chinchilla model (hoffmann2022training).
Finally, to compare T2T^{2} forecasts to Chinchilla, we extrapolate the predictions from our T2T^{2} approaches and standard Chinchilla scaling beyond our scaling suite to 102510^{25} FLOPs.
Using the same fits, we visualize pretraining isoFLOP profiles for both approaches.
We compare the standard single-pass setting (k=1k{=}1) to the inference-corrected setting with Cinf=2×109C\_{\text{inf}}=2\times 10^{9} FLOPs and k=Cinf2​Nk=\frac{C\_{\text{inf}}}{2N}.
Each of the 12 isoFLOP curves traces out a fixed pretraining budget CtrainC\_{\text{train}} by varying NN and DD subject to Ctrain=6​N​DC\_{\text{train}}=6ND.
We plot the Chinchilla optimal frontier in black and that of T2T^{2} in red.
Results are macro averaged across all eight tasks.
Individual scaling fits for each task across different budgets can be found in Appendix [B](#A2 "Appendix B Per-Task Analysis ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").

Results.
Our results are shown in Figure [2](#S4.F2 "Figure 2 ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") and Figure [3](#S4.F3 "Figure 3 ‣ 4.1 RQ1: Should Pretraining Change if You Know Your Test-Time Scaling Budget? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").
Figure [2](#S4.F2 "Figure 2 ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") shows that we can answer RQ1 in the affirmative: both T2T^{2} approaches forecast models that are dramatically smaller and more overtrained than what Chinchilla prescribes.
We additionally confirm that the Chinchilla scaling fit is consistent with hoffmann2022training by overlaying the 70B Chinchilla hero run model described in their paper, alongside the 20 tokens per parameter rule of thumb.
Despite modeling fundamentally different quantities (NLL vs accuracy), both T2T^{2} recommend extreme overtraining, with Approach 2 recommending more aggressive overtraining than Approach 1.
Figure [3](#S4.F3 "Figure 3 ‣ 4.1 RQ1: Should Pretraining Change if You Know Your Test-Time Scaling Budget? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") shows isoFLOP curves under our T2T^{2} approaches, how the overtraining trend develops within our scaling population.
At every compute scale, the optimal frontier of both T2T^{2} approaches shifts considerably toward smaller overtrained models with more repeated samples compared to the Chinchilla optimum.
When inference-corrected, we see that the Chinchilla optimal frontier exhibits non-monotonic improvement in CtrainC\_{\text{train}}.
This is consistent with the findings of snell2024scaling, showing that smaller models with more test-time compute can outperform larger models.
On the other hand, T2T^{2} shows both stronger and consistently monotonic improvement, as we jointly model pretraining and test-time scaling.
These results confirm that if you know your test-time scaling budget, you should substantially overtrain compared to Chinchilla optimal pretraining.

|  |
| --- |
| Refer to caption |

Figure 3: T2T^{2} scaling across all of our evaluation tasks. Both approaches improve monotonically over Chinchilla scaling, while Chinchilla exhibits non-monotonic scaling in CtrainC\_{\text{train}}.

### 4.2 RQ2: Does T2T^{2} Scaling Extrapolate to Overtrained Checkpoints?

Next, we evaluate RQ2 by fitting both T2T^{2} approaches to standard Chinchilla scaling checkpoints and measuring the performance of extrapolation to overtrained checkpoints.

Setup.
We fit both of our T2T^{2} approaches to a suite of 85 Chinchilla scaling checkpoints from porian2024resolving (which stop short of the optimal overtraining regime that T2T^{2} predicts) and measure the relative absolute error of extrapolating the predictions to 21 overtrained checkpoints that we train using an identical pretraining setup.
We include training details and the exact checkpoint grid in Appendix [C](#A3 "Appendix C Pretraining Details ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").
We also compare the empirical best overtrained checkpoint (among our 21) in the inference-corrected regime and compare it to the empirical Chinchilla optimal checkpoint at a pretraining budget of Ctrain=2.56×1019C\_{\text{train}}=2.56\times 10^{19} across all eight tasks.
We set Cinf=2×109C\_{\text{inf}}=2\times 10^{9} for all of the above.

Results.
Our extrapolation results are shown in Figure [4](#S4.F4 "Figure 4 ‣ 4.2 RQ2: Does 𝑇² Scaling Extrapolate to Overtrained Checkpoints? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") and empirical checkpoint pass@kk results are shown in Table [1](#S4.T1 "Table 1 ‣ Table 2 ‣ 4.2 RQ2: Does 𝑇² Scaling Extrapolate to Overtrained Checkpoints? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").
Figure [4](#S4.F4 "Figure 4 ‣ 4.2 RQ2: Does 𝑇² Scaling Extrapolate to Overtrained Checkpoints? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") shows that our T2T^{2} approaches both extrapolate to the 16 new overtrained checkpoints.
While both approaches somewhat overestimate performance, Approach 1 extrapolates better than Approach 2, with a relative error of 2.8% compared to 8.4%.
Table [1](#S4.T1 "Table 1 ‣ Table 2 ‣ 4.2 RQ2: Does 𝑇² Scaling Extrapolate to Overtrained Checkpoints? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") shows that our best small overtrained checkpoints always outperform the Chinchilla optimal checkpoints when inference corrected, across all eight tasks.
This confirms that T2T^{2} extrapolates to real overtrained checkpoints, and that this phenomenon is not just an artifact of our T2T^{2} approaches.

![Refer to caption](/html/2604.01411/assets/x4.png)
  


Figure 4: Extrapolating porian2024resolving checkpoints to the overtraining regime.



|  |  | Best overtrained | Chinchilla opt. |
| --- | --- | --- | --- |
| Real | LAMBADA OpenAI | 49.90% (37M) | 27.30% (455M) |
| OpenBookQA | 01.40% (37M) | 0.30% (901M) |
| SciQ | 01.20% (37M) | 0.22% (611M) |
| ARC-Easy | 00.14% (149M) | 0.07% (611M) |
| Synthetic | Simple Knowledge | 14.60% (84M) | 5.80% (901M) |
| Simple Reasoning | 57.90% (37M) | 18.40% (901M) |
| Commonsense Causal | 08.10% (37M) | 1.40% (901M) |
| Spatial Reasoning | 06.00% (37M) | 1.10% (901M) |

Table 1: Comparison of overtrained base models vs Chinchilla optimal pass@kk, subject to Ctrain=2.56×1019C\_{\text{train}}=2.56\times 10^{19} and Cinf=2×109C\_{\text{inf}}=2\times 10^{9} FLOPs. Optimal model sizes are shown in parentheses.

|  |  | OpenBookQA | SciQ | ARC-Easy |
| --- | --- | --- | --- | --- |
| FT | Best overtrained | 2.80% (37M) | 56.10% (149M) | 5.60% (149M) |
| Chinchilla opt. | 0.45% (901M) | 29.00% (901M) | 1.50% (901M) |
| SFT | Best overtrained | 2.60% (37M) | 66.80% (84M) | 8.20% (37M) |
| Chinchilla opt. | 0.38% (901M) | 57.60% (347M) | 3.40% (455M) |

Table 2: Post-training comparison of overtraining vs Chinchilla optimal pass@kk, subject to Ctrain=2.56×1019C\_{\text{train}}=2.56\times 10^{19} and Cinf=2×109C\_{\text{inf}}=2\times 10^{9} FLOPs. Optimal model sizes are shown in parentheses.

### 4.3 RQ3: Does T2T^{2} Scaling Survive Post-Training?

Finally, we evaluate RQ3 by showing that our findings persist after post-training.

Setup.
We explore two canonical post-training techniques: standard fine-tuning (FT) and supervised fine-tuning (SFT), where we only fine-tune on the targets.
We post-train on the three real tasks that have a standard training set: ARC-Easy, SciQ, and OpenBookQA, and report improved performance on the test sets for each of these.
Additional post-training details can be found in Appendix [D](#A4 "Appendix D Post-Training Details ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").
We allocate the same number of training steps to each checkpoint, rather than scaling training based on FLOPs, since we ultimately train to convergence.
After post-training, we fit both T2T^{2} approaches to the FT and SFT checkpoints and evaluate their optimal tokens per parameter frontier compared to base models under T2T^{2} scaling and the Chinchilla frontier.
Finally, like in RQ2, we compare the best overtrained FT and SFT checkpoints to the Chinchilla optimal checkpoints for each task.

Results.
Our results are shown in Figure [5](#S4.F5 "Figure 5 ‣ 4.3 RQ3: Does 𝑇² Scaling Survive Post-Training? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") and Table [2](#S4.T2 "Table 2 ‣ 4.2 RQ2: Does 𝑇² Scaling Extrapolate to Overtrained Checkpoints? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").
We see in Figure [5](#S4.F5 "Figure 5 ‣ 4.3 RQ3: Does 𝑇² Scaling Survive Post-Training? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") that the optimal frontier continues to shift toward smaller overtrained models with more test-time samples across all three tasks and methods.
Again, we find that these results are consistent between Approach 1 and Approach 2.
On the other hand, we find that the optimal overtraining recommendation is somewhat subdued compared to T2T^{2} on the base models alone, but not enough to shift it back to the original Chinchilla recommendation.
The finding that it is subdued is consistent with prior work showing that overtrained models are harder to fine-tune (springer2025overtrained).
Finally, we see in Table [2](#S4.T2 "Table 2 ‣ 4.2 RQ2: Does 𝑇² Scaling Extrapolate to Overtrained Checkpoints? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") that our best overtrained checkpoints still outperform the Chinchilla optimal checkpoints after post-training, and that performance improves across the board compared to the same analysis on base models in Table [1](#S4.T1 "Table 1 ‣ Table 2 ‣ 4.2 RQ2: Does 𝑇² Scaling Extrapolate to Overtrained Checkpoints? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").
This confirms that our findings with T2T^{2} scaling persist after post-training.

|  |
| --- |
| Refer to caption |

Figure 5: T2T^{2} overtraining findings survive post-training. The optimal frontier is slightly subdued compared to base models, which is consistent with springer2025overtrained.

## 5 Conclusion

In this work, we have presented T2T^{2} scaling laws that jointly optimize model size, training tokens, and the number of repeated samples at test-time under fixed pretraining and inference budgets.
We find that when test-time compute via repeated sampling is accounted for during pretraining decisions, the optimal model is substantially smaller and more overtrained than what standard Chinchilla scaling prescribes.
This finding is consistent across two complementary modeling approaches: Approach 1 which models the NLL, and Approach 2 which models the pass@kk accuracy directly.
We validated this across eight real and synthetic downstream tasks, validated that T2T^{2} scaling extrapolates to the overtraining regime where its optima are predicted, and that our findings persist after post-training.
Based on our findings, we offer a recommendation to practitioners: if you know your test-time scaling budget with repeated sampling, you should train a smaller model for longer, and T2T^{2} scaling offers a blueprint for doing so.
In future work, we plan to validate our prescribed overtraining recipes at larger scales, account for transformer-specific inference cost models, and explicitly model the role of post-training in T2T^{2} scaling.

## Appendix Roadmap

Our appendix is structured as follows.
We begin with related work in Appendix [A](#A1 "Appendix A Related Work ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"), followed by Appendix [B](#A2 "Appendix B Per-Task Analysis ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"), which presents per-task scaling law analyses.
We next turn to experimental details: Appendix [C](#A3 "Appendix C Pretraining Details ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") and Appendix [D](#A4 "Appendix D Post-Training Details ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") describe our pretraining and post-training setups, respectively, while Appendix [E](#A5 "Appendix E Evaluation Tasks ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") provides descriptions of all evaluation tasks employed in our study.
Finally, Appendix [F](#A6 "Appendix F Fitting 𝑇² Scaling ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") presents the details of our T2T^{2} scaling fitting methodology.

## Appendix A Related Work

Our work sits at the intersection of three research threads: (i) pretraining scaling laws, (ii) test-time scaling, and (iii) overtrained models.

### A.1 Pretraining Scaling Laws

kaplan2020scaling established that model loss follows predictable power laws as a function of model size and training data.
hoffmann2022training (Chinchilla) refined this into compute-optimal training recipes, prescribing how model size and token count should scale together under a fixed compute budget.
Recent extensions has broadened the scope of scaling law modeling: studying data quality and quantity (goyal2024scaling), incorporating downstream task accuracy (isik2024scaling; bhagia2024establishing), decomposing scaling behaviors across knowledge and reasoning skills (roberts2025compute), and extending to multimodal settings (shukor2025scaling).
These frameworks, however, treat inference as an afterthought—optimizing for a model that is trained once and queried once.
sardana2024beyond take a step toward deployment-aware scaling by folding inference serving volume into the compute-optimal recipe, yet their analysis is limited to single-pass queries.
We modernize this line of work, where the optimal training decisions must account for both the cost and the compounding performance gains of drawing multiple inference samples.

### A.2 Test-Time Scaling

Beyond scaling pretraining compute, recent work has increasingly focused on investing computation at inference time (snell2024scaling; zhang2025survey; jaech2024openai; orlanski2025reward).
This test-time paradigm often focuses on the search for a correct reasoning path rather than the model’s inherent knowledge and can broadly be categorized into three regimes:
(i) parallel scaling, which uses consensus through self-consistency (brown2025large), or verification over multiple independent responses (saad2025shrinking);
(ii) sequential scaling, which refines reasoning through iterative improvements or hierarchical pruning (wei2022chain; madaan2023self);
and (iii) internal scaling, which allows the model to dynamically adjust generation depth based on task difficulty (jaech2024openai).
In this work, we focus on parallel repeated sampling—the most common form of test-time scaling—and incorporate pretraining compute budget to jointly optimize allocation decisions.

### A.3 Overtraining

hoffmann2022training (Chinchilla) prescribes a compute-optimal ratio of roughly 20 training tokens per model parameter, yet modern models release routinely deviate from this blueprint by training smaller models on far more tokens than recommended.
This deliberate overtraining is motivated by inference efficiency: a smaller model costs less per query at deployment.
Recent model families illustrate this trend—Llama-2-7B (touvron2023llama) was trained on 2T tokens (∼\sim290×\times the recommended ratio);
Google’s Gemma-7B (team2024gemma) was trained on 6T tokens (∼\sim857×\times), and its successor Gemma 2-9B (team2024gemma) on 8T tokens (∼\sim889×\times)—with OLMo (groeneveld2024olmo) following a similar philosophy.
Our work complements these findings by examining overtraining through a different lens: rather than studying its effect on post-training (springer2025overtrained), we show that overtraining is actively *beneficial* when models are deployed with a repeated-sampling inference budget, and we provide a principled framework for determining how much to overtrain given a joint train-and-test compute allocation.

## Appendix B Per-Task Analysis

We present isoFLOP profiles for each of the individual tasks in our evaluation suite in Figure [6](#A2.F6 "Figure 6 ‣ Appendix B Per-Task Analysis ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") for Approach 1 and Figure [7](#A2.F7 "Figure 7 ‣ Appendix B Per-Task Analysis ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") for Approach 2 .
We find that overtraining predictions are relatively stable across inference budgets for both approaches.

|  |
| --- |
| Refer to caption |
| Refer to caption |
| Refer to caption |
| Refer to caption |
| Refer to caption |
| Refer to caption |
| Refer to caption |
| Refer to caption |

Figure 6: 
Approach 1 IsoFLOP profiles across different scaling budgets for all eight tasks.



|  |
| --- |
| Refer to caption |
| Refer to caption |
| Refer to caption |
| Refer to caption |
| Refer to caption |
| Refer to caption |
| Refer to caption |
| Refer to caption |

Figure 7: 
Approach 2 IsoFLOP profiles across different scaling budgets for all eight tasks.

## Appendix C Pretraining Details

In this section, we provide details of our pretraining setup and scaling grid.

### C.1 Checkpoint Scaling Grid

Figure [8](#A3.F8 "Figure 8 ‣ C.1 Checkpoint Scaling Grid ‣ Appendix C Pretraining Details ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") shows our checkpoint grid, comprising pretrained checkpoints from porian2024resolving alongside additional overtrained checkpoints we pretrained in this work.
Model sizes range from 5M to 901M parameters, and training FLOPs span 1.25×10161.25\times 10^{16} to 2.56×10192.56\times 10^{19}.
Each cell reports the number of tokens per parameter, which characterizes the degree of overtraining.
Typically, a suite of Chinchilla scaling checkpoints contains checkpoints at either side of the typical 20 tokens per parameter recommendation derived from hoffmann2022training.
However, since T2T^{2} suggests overtraining beyond the available set of checkpoints, we train additional checkpoints at higher tokens per parameter ratios.
The overtrained checkpoints (shown in orange) are used to validate our forecasts in §[4.2](#S4.SS2 "4.2 RQ2: Does 𝑇² Scaling Extrapolate to Overtrained Checkpoints? ‣ 4 Experiments ‣ Test-Time Scaling Makes Overtraining Compute-Optimal").

|  |
| --- |
| Refer to caption |

Figure 8: 
Overall checkpoint scaling grid.
Each cell reports the number of tokens per parameter.
Orange cells are overtrained checkpoints we created.

### C.2 Hyperparameters

We train our overtrained checkpoints, shown in Figure [8](#A3.F8 "Figure 8 ‣ C.1 Checkpoint Scaling Grid ‣ Appendix C Pretraining Details ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"), from scratch using the OpenLM framework with same fixed hyperparameters used for the Chinchilla-optimal checkpoints from porian2024resolving.
Specifically, we use their hparams=base, warmup=short, decay=chinchilla configuration.
We use the AdamW optimizer with a learning rate of 3×10−33\times 10^{-3}, β1=0.9\beta\_{1}=0.9, β2=0.95\beta\_{2}=0.95, and a decoupled weight decay of 1×10−41\times 10^{-4}.
Training uses a global batch size of 256 sequences of length 2048 tokens, cosine learning rate decay to zero matched to the token budget of each run, and a warmup period equal in tokens to the model’s parameter count.
We apply gradient clipping with a max norm of 1.0, QK-normalization, z-loss with coefficient 10−410^{-4}, and train in bfloat16 mixed precision.
All hyperparameters are held fixed across model sizes, consistent with the base (untuned) configuration of porian2024resolving.
We train on the RefinedWeb dataset with a vocabulary size of 50,432.

## Appendix D Post-Training Details

We describe our post-training setup and configurations below.
We employ two variants of post-training: (i) standard fine-tuning and (ii) supervised fine-tuning (SFT).
Standard fine-tuning follows the conventional next-token prediction objective, computing loss over both the instruction (question) and completion (answer).
SFT, in contrast, computes loss over the completion only, excluding instruction tokens from parameter updates.

We fine-tune on three tasks—ARC Easy (ai2arc), SciQ (SciQ), and OpenBookQA (OpenBookQA2018)—covering the full population of pretrained checkpoints, including the overtrained ones.
Each model is trained for 6 epochs until convergence using a batch size of 8 and a constant learning rate of 2×10−52\times 10^{-5}, after that we evaluate on the respective test set.
All fine-tuning experiments are conducted on 4 NVIDIA A10 GPUs.
Box [D](#A4 "Appendix D Post-Training Details ‣ Test-Time Scaling Makes Overtraining Compute-Optimal") presents the training data format for each task, where the highlighted tokens indicate the completion portion used in the SFT loss computation.
Their evaluation follows the same format: we measure negative log-likelihood over the correct answer placed in the highlighted placeholder.

Box 1: Training Data Formats

Each format separates the prompt (plain) from the completion (highlighted), which is the only portion used in the SFT loss.
ARC Easy:
[⬇](data:text/plain;base64,UXVlc3Rpb246IHtxdWVzdGlvbn1cbkFuc3dlcjooKkBcY29sb3Jib3h7eWVsbG93ITQwfXtcdGV4dHR0eyBce2Fuc3dlclx9fX1AKik=)
Question: {question}\nAnswer: {answer}
OpenBookQA:
[⬇](data:text/plain;base64,e3F1ZXN0aW9ufSgqQFxjb2xvcmJveHt5ZWxsb3chNDB9e1x0ZXh0dHR7IFx7YW5zd2VyXH19fUAqKQ==)
{question} {answer}
SciQ:
[⬇](data:text/plain;base64,e3N1cHBvcnR9XG5RdWVzdGlvbjoge3F1ZXN0aW9ufVxuQW5zd2VyOigqQFxjb2xvcmJveHt5ZWxsb3chNDB9e1x0ZXh0dHR7IFx7YW5zd2VyXH19fUAqKQ==)
{support}\nQuestion: {question}\nAnswer: {answer}

## Appendix E Evaluation Tasks

Next, we describe the eight downstream tasks used to evaluate T2T^{2} scaling, covering both real-world benchmarks and synthetic tasks.
For all tasks, we measure the NLL of each model over the correct answer.

We evaluate on four real-world benchmarks.

1. 1.

   LAMBADA (lambada) (OpenAI variant): tests long-range language understanding, where the model must predict the final word of a passage given a broad context.
2. 2.

   ARC Easy (ai2arc): consists of elementary-level science questions in a four-way multiple choice format, drawn from standardized tests.
3. 3.

   SciQ (SciQ): contains science exam questions paired with supporting passages, presented in a multiple-choice format.
4. 4.

   OpenBookQA (OpenBookQA2018): requires multi-step reasoning by combining an open book of core science facts with broader common knowledge, presented as four-way multiple choice questions.

In addition to these four benchmarks, we incorporate four synthetic tasks spanning different domains.
These tasks are designed to evaluate models on (i) simple knowledge recall, (ii) multi-step arithmetic reasoning, (iii) commonsense causal reasoning, and (iv) spatial reasoning.
Each task consists of 1,000 fill-in-the-blank or short-completion questions, generated using GPT-5 and Claude Opus 4.6.
Below, we present representative examples from each task along with their evaluation format.
As in Box [D](#A4 "Appendix D Post-Training Details ‣ Test-Time Scaling Makes Overtraining Compute-Optimal"), the token spans used to compute the NLL are highlighted in each example below.

Box 2: Commonsense Causal Reasoning

Example 1:
[⬇](data:text/plain;base64,R3JhbmRwYXJlbnRzIHRlbGwgc3RvcmllcyB0byBncmFuZGNoaWxkcmVuLiBUZWFjaGVycyBleHBsYWluCmNvbmNlcHRzIHRvIHN0dWRlbnRzLiBDb2FjaGVzIGRlbW9uc3RyYXRlIHRlY2huaXF1ZXMgdG8oKkBcY29sb3Jib3h7eWVsbG93ITQwfXtcdGV4dHR0eyBwbGF5ZXJzfX1AKik=)
Grandparents tell stories to grandchildren. Teachers explain
concepts to students. Coaches demonstrate techniques to players
Example 2:
[⬇](data:text/plain;base64,QSBtb3RoZXIgY29tZm9ydHMgYSBjcnlpbmcgYmFieS4gQSB0ZWFjaGVyIGVuY291cmFnZXMgYQpzdHJ1Z2dsaW5nIHN0dWRlbnQuIEEgY29hY2ggbW90aXZhdGVzIGEgZGlzY291cmFnZWQoKkBcY29sb3Jib3h7eWVsbG93ITQwfXtcdGV4dHR0eyBwbGF5ZXJ9fUAqKQ==)
A mother comforts a crying baby. A teacher encourages a
struggling student. A coach motivates a discouraged player


Box 3: Simple Knowledge Recall

Example 1:
[⬇](data:text/plain;base64,VGhlIGNhcGl0YWwgb2YgRWd5cHQgaXMoKkBcY29sb3Jib3h7eWVsbG93ITQwfXtcdGV4dHR0eyBDYWlyb319QCop)
The capital of Egypt is Cairo
Example 2:
[⬇](data:text/plain;base64,VGhlIGZpZnRoIHRhc3RlIGlzKCpAXGNvbG9yYm94e3llbGxvdyE0MH17XHRleHR0dHsgdW1hbWl9fUAqKQ==)
The fifth taste is umami


Box 4: Multi-Step Arithmetic Reasoning

Example 1:
[⬇](data:text/plain;base64,SSBoYXZlIDUgdG95cy4gSSBnaXZlIGF3YXkgMiB0b3lzLiBTdGVwIDE6IEkgc3RhcnRlZCB3aXRoIDUKdG95cy4gU3RlcCAyOiBJIGdhdmUgYXdheSAyIHRveXMuIFN0ZXAgMzogNSBtaW51cyAyIGVxdWFscygqQFxjb2xvcmJveHt5ZWxsb3chNDB9e1x0ZXh0dHR7IDN9fUAqKQ==)
I have 5 toys. I give away 2 toys. Step 1: I started with 5
toys. Step 2: I gave away 2 toys. Step 3: 5 minus 2 equals 3
Example 2:
[⬇](data:text/plain;base64,UGF0dGVybjogMTAsIDIwLCAzMCwgLi4uIFRoaXMgYWRkcyAxMCBlYWNoIHRpbWUuIEFmdGVyIDMwCmNvbWVzKCpAXGNvbG9yYm94e3llbGxvdyE0MH17XHRleHR0dHsgNDB9fUAqKQ==)
Pattern: 10, 20, 30, ... This adds 10 each time. After 30
comes 40


Box 5: Spatial Reasoning

Example 1:
[⬇](data:text/plain;base64,VGhlIGJhYnkgaXMgaW4gdGhlIGNyaWIuIFRoZSBjcmliIGlzIGluIHRoZSBudXJzZXJ5LiBUaGUKbnVyc2VyeSBpcyBpbiB0aGUgaG91c2UuIFNvIHRoZSBiYWJ5IGlzIGluIHRoZSgqQFxjb2xvcmJveHt5ZWxsb3chNDB9e1x0ZXh0dHR7IGhvdXNlfX1AKik=)
The baby is in the crib. The crib is in the nursery. The
nursery is in the house. So the baby is in the house
Example 2:
[⬇](data:text/plain;base64,VGhlIGdsYXNzZXMgYXJlIGluIHRoZSBjYXNlLiBUaGUgY2FzZSBpcyBpbiB0aGUgaGFuZGJhZy4KU28gdGhlIGdsYXNzZXMgYXJlIGluIHRoZSgqQFxjb2xvcmJveHt5ZWxsb3chNDB9e1x0ZXh0dHR7IGhhbmRiYWd9fUAqKQ==)
The glasses are in the case. The case is in the handbag.
So the glasses are in the handbag

## Appendix F Fitting T2T^{2} Scaling

In this section, we describe how each of our T2T^{2} approaches are fit to empirical checkpoints.

#### Fitting Approach 1.

We fit the seven parameters (log⁡A,log⁡B,log⁡E,α,β,log⁡G,γ)(\log A,\log B,\log E,\alpha,\beta,\log G,\gamma) of the additive model by minimizing the sum of squared errors (SSE) between predicted and empirical NLL values across all checkpoints and sampled values of kk.
We use the L-BFGS-B algorithm with 500 random restarts (each with up to 5,000 iterations and a tolerance of 10−1510^{-15}) and we select the run with the lowest objective value.

#### Fitting Approach 2.

We fit the model in two stages.
First, we fit the standard Chinchilla scaling model L^​(N,D)=E+ANα+BDβ\widehat{L}(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}} to the empirical NLL values of all checkpoints.
We profile over a grid of 40 candidate EE values spaced between 0.01⋅min⁡(NLL)0.01\cdot\min(\text{NLL}) and 0.95⋅min⁡(NLL)0.95\cdot\min(\text{NLL}); for each, we optimize the remaining four parameters (log⁡A,log⁡B,α,β)(\log A,\log B,\alpha,\beta) via L-BFGS-B with 50+ random restarts, using inverse-variance weighting across isoFLOP groups.
Second, we fit the Beta regression parameters.
The per-question success probability is modeled as p∼Beta​(aN,D,bN,D)p\sim\text{Beta}(a\_{N,D},b\_{N,D}) where μ=aN,D/(aN,D+bN,D)\mu=a\_{N,D}/(a\_{N,D}+b\_{N,D}) is a scaled logit link and the concentration ν=aN,D+bN,D\nu=a\_{N,D}+b\_{N,D} is parameterized as a log link function.
Together, the five parameters (θ0,θ1,θ2,θ3,θ4)(\theta\_{0},\theta\_{1},\theta\_{2},\theta\_{3},\theta\_{4}) are fit by minimizing SSE between predicted and empirical pass@kk accuracy values over a grid of initializations seeded from a sigmoid baseline, again using L-BFGS-B.

[◄](/html/2604.01409)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2604.01411)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2604.01411)
[View original  
on arXiv](https://arxiv.org/abs/2604.01411)[►](/html/2604.01412)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue May 5 20:05:22 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
