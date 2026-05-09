---
arxiv: '2604.20614'
authors:
- Alessandro Morosini
- Matea Gjika
- Tomaso Poggio
- Pierfrancesco Beneventano
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'Too Sharp, Too Sure: When Calibration Follows Curvature'
url: https://arxiv.org/abs/2604.20614
year: 2026
---

[2604.20614] Too Sharp, Too Sure: When Calibration Follows Curvature















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



# Too Sharp, Too Sure: When Calibration Follows Curvature

Alessandro Morosini
Affiliation: Massachusetts Institute of Technology, Cambridge, MA, USA
Correspondence to:[morosini@mit.edu](mailto:morosini@mit.edu)
  
Matea Gjika
Affiliation: Massachusetts Institute of Technology, Cambridge, MA, USA
  
Tomaso Poggio
Affiliation: Massachusetts Institute of Technology, Cambridge, MA, USA
  
Pierfrancesco Beneventano
Affiliation: Massachusetts Institute of Technology, Cambridge, MA, USA
Correspondence to:[pierb@mit.edu](mailto:pierb@mit.edu)

###### Abstract

Modern neural networks can achieve high accuracy while remaining poorly calibrated, producing confidence estimates that do not match empirical correctness. Yet calibration is often treated as a post-hoc attribute. We take a different perspective: we study calibration as a *training-time* phenomenon on small vision tasks, and ask whether calibrated solutions can be obtained reliably by intervening on the training procedure. We identify a tight coupling between calibration, curvature, and margins during training of deep networks under multiple gradient-based methods. Empirically, Expected Calibration Error (ECE) closely tracks curvature-based sharpness throughout optimization. Mathematically, we show that both ECE and Gauss–Newton curvature are controlled, up to problem-specific constants, by the same margin-dependent exponential tail functional along the trajectory. Guided by this mechanism, we introduce a margin-aware training objective that explicitly targets robust-margin tails and local smoothness, yielding improved out-of-sample calibration across optimizers without sacrificing accuracy.

###### Keywords:

Calibration, copy them

![Refer to caption](/html/2604.20614/assets/x1.png)


Figure 1: Training dynamics for Gradient Descent and Stochastic Gradient Descent across learning rates on CIFAR-10. Expected Calibration Error closely tracks sharpness throughout training: both rise as the model enters the edge of stability regime, peak around the same time, and decay together as training progresses.

## 1 Introduction

Neural networks are now routinely used in settings where a model’s stated uncertainty matters as much as its accuracy, for example, in risk-sensitive domains such as healthcare or autonomous driving. In these contexts, we would like predicted probabilities to reflect empirical correctness: among predictions made with confidence pp, approximately a fraction pp should be correct. However, modern deep networks are often *miscalibrated*, frequently exhibiting overconfidence even when wrong (Guo et al., [2017](#bib.bib12)).

A widely adopted response to overconfidence in neural networks is *post-hoc* calibration: models are trained for accuracy and their predicted probabilities are adjusted afterward. While effective in many regimes, this framing treats calibration as a post-training concern, rather than a property that *emerges during training*.
Recent work, however, suggests that calibration may be influenced by training dynamics. For example, Sharpness-Aware Minimization (SAM), which biases optimization toward flatter regions of the loss landscape, has been observed to generally reduce overconfidence (Tan et al., [2026](#bib.bib48)). Since sharpness is a geometric property shaped along the optimization trajectory, these results hint at a link between calibration and loss geometry during training. Yet this connection remains poorly understood, motivating the following question:

*How does calibration evolve throughout optimization,
  
and what aspects of the training govern it?*

Importantly, answering this question is non-trivial. Calibration is defined in terms of the model’s *predictive confidence distribution*, whereas most training-time analyses characterize optimization through *loss-landscape geometry*, using quantities such as curvature or sharpness to reason about stability and generalization. Bridging these viewpoints is challenging, particularly early in training when predictions are still rapidly evolving. While several recent studies have examined relationships between sharpness, flat minima, and calibration at convergence, the resulting picture is mixed: curvature proxies do not reliably predict calibration across architectures, regularization schemes, or optimizers (Mason-Williams et al., [2024](#bib.bib31)). Crucially, these analyses focus on
converged solutions. By contrast, we study how calibration and loss geometry co-evolve during training—a perspective that not only clarifies their relationship but also reveals dynamics that can be exploited to improve calibration.
After formalizing a training-time connection between calibration and loss geometry, we turn to a prescriptive goal:

*Can we intervene on the training procedure to reliably obtain calibrated solutions?*

#### Contributions.

We study calibration *during* optimization by jointly tracking calibration metrics, such as Expected Calibration Error (ECE), and curvature-based sharpness proxies, such as Gauss–Newton (GN) sharpness, along the training trajectory (Section [3](#S3 "3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")). We conduct this analysis throughout training rather than only at convergence. We observe across multiple gradient-based optimization methods that

Contribution 1.
Calibration error and curvature-based measures exhibit a strong and consistent temporal correlation throughout training.

Next, we probe whether the coupling between calibration and curvature is causal.
We compare optimizers designed to minimize sharpness (i.e., favoring flatter minima) with methods that instead suppress steep descent directions along the trajectory. Despite both affecting curvature, we find that

Contribution 2:
Directional interventions yield consistently better in-sample calibration than flat-minima methods.

This clarifies the distinction between “being flat” and “training in stable directions” and its relation to confidence.

We then provide a unifying explanation through the lens of the *margin* in Section [4](#S4 "4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"). Intuitively, both confidence and curvature are shaped by how strongly the model separates the correct class from its nearest competitor. We formalize this connection by showing mathematically that

Contribution 3:
A single margin-based functional controls both calibration error and Gauss–Newton sharpness, up to problem-dependent constants.

This perspective also clarifies an often-observed phenomenon: training and test calibration can diverge even when accuracy improves (Carrell et al., [2022](#bib.bib5); Wu et al., [2025](#bib.bib51)). Once most examples achieve large positive margins, a relatively small set of near-boundary or negative-margin points can dominate the margin functional, making calibration highly sensitive to how optimization shapes this tail.

Based on the mathematical connection we establish between margins, curvature, and calibration, we design a new robust-margin-aware loss. This yields a principled training-time handle on calibration, which we confirm in Section [5](#S5 "5 From Margin Theory to Calibrated Training ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"):

Contribution 4:
We propose CalMO (CALibration with Margin Objective), a training objective that yields better-calibrated models without sacrificing accuracy.

Our margin-based view also yields concrete diagnostics for optimizer behavior. In particular, we observe that Muon induces unusually large training margins, leading to near-zero training ECE, but severe test-time overconfidence. Consistent with this finding, Muon is also the optimizer that benefits most from CalMO, as robust-margin control directly targets this failure mode.

## 2 Preliminaries and Related Work

#### Calibration.

A model is calibrated if its predicted confidence values reflect empirical correctness frequencies: among all predictions made with confidence pp, approximately a fraction pp should be correct (Niculescu-Mizil & Caruana, [2005](#bib.bib38); DeGroot & Fienberg, [1983](#bib.bib9)). Since calibration is a distributional property, it is typically assessed empirically from finite samples. The most widely used measure is ECE (Naeini et al., [2015](#bib.bib35)), which compares accuracy and confidence after binning predictions by confidence.
Predictions are grouped into MM bins according to their confidence
(i.e., the maximum predicted class probability); we denote by BmB\_{m}
the set of predictions whose confidence falls in bin mm.
For each bin, we compute the empirical accuracy acc​(Bm)\mathrm{acc}(B\_{m}) and the average predicted confidence conf​(Bm)\mathrm{conf}(B\_{m}). ECE is then defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ECE=∑m=1M|Bm|n​|acc​(Bm)−conf​(Bm)|.\mathrm{ECE}=\sum\_{m=1}^{M}\frac{|B\_{m}|}{n}\bigl|\mathrm{acc}(B\_{m})-\mathrm{conf}(B\_{m})\bigr|. |  | (1) |

Calibration is also evaluated through other summary statistics such as the Maximum Calibration Error (Naeini et al., [2015](#bib.bib35)), reliability diagrams (Guo et al., [2017](#bib.bib12)), kernel-based metrics
such as the Kernel Calibration Error (KCE) (Kumar et al., [2018](#bib.bib23)), or similar related metrics. Further details, including the multiclass extension, are provided in Appendix [B.1](#A2.SS1 "B.1 Definitions and basic reductions ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").

#### Mitigating miscalibration.

Existing approaches to mitigating calibration error fall into two broad categories: post-hoc methods and intrinsic training-time methods.
Post-hoc methods modify model outputs without adjusting the parameters, and include techniques such as Platt scaling (Platt, [2000](#bib.bib41)) and temperature scaling (Guo et al., [2017](#bib.bib12)). Intrinsic methods incorporate calibration objectives directly into training by using regularization to penalize overconfident predictions (Pereyra et al., [2017](#bib.bib40)), by incorporating differential proxies for calibration into the loss (Kumar et al., [2018](#bib.bib23); Bohdal et al., [2023](#bib.bib4)), or by applying label smoothing (Müller et al., [2019](#bib.bib34)). A more detailed overview of these methods is provided in Appendix [A](#A1 "Appendix A Further Related Work ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").

An important perspective links calibration to adversarial robustness. Points with small robust margin have been shown to be more likely miscalibrated; motivated by this, R-AdaLS (Qin et al., [2021](#bib.bib42)) bins data points by robust margin and applies stronger label smoothing to low-margin samples. Moreover, state-of-the-art calibration losses can be unified as penalties on logit distances (Liu et al., [2022](#bib.bib28)). These results suggest that calibration errors are tied to local margin geometry, rather than solely to global confidence statistics.

#### Curvature along the trajectory.

A complementary line of work treats curvature not as a static attribute of the final solution, but as a dynamical quantity that governs optimization stability throughout training. Early work established a link between “wide valleys” and generalization (Hochreiter & Schmidhuber, [1997](#bib.bib14)), a picture later reinforced by the sharp-minima account of the large-batch generalization gap (Keskar et al., [2017](#bib.bib20)), motivating the study of Hessian-based sharpness along the training trajectory.
More recent analyses emphasize that curvature generally increases as gradient methods approach an *edge-of-stability* (EoS) regime, where the top curvature direction becomes commensurate with the inverse step size and the dynamics become oscillatory or unstable (Xing et al., [2018](#bib.bib52); Jastrzębski et al., [2018](#bib.bib17), [2019](#bib.bib16); Cohen et al., [2021](#bib.bib7), [2022](#bib.bib8); Andreyev & Beneventano, [2024](#bib.bib1)), before often decreasing toward the end of training.

In parallel, neighborhood-based objectives reshape the training trajectory by explicitly penalizing worst-case loss increases under small weight perturbations (e.g., SAM), thereby suppressing sensitivity to sharp directions (Foret et al., [2021](#bib.bib11); Zhou et al., [2025](#bib.bib57)); such procedures have also been observed to improve confidence estimates under cross-entropy (Tan et al., [2026](#bib.bib48)). At the same time, evidence suggests that the sharpness–calibration relationship can be fragile across architectures and regularization schemes (Mason-Williams et al., [2024](#bib.bib31)), pointing to the importance of *how* curvature directions are traversed, not only where optimization converges. Our trajectory-level study aligns with this viewpoint by jointly tracking calibration and curvature throughout training and by contrasting convergence-to-flatness with explicit suppression of unstable high-curvature directions.

## 3 The Coupling Between Calibration and Sharpness

We track sharpness and ECE throughout training to study how loss landscape geometry relates to calibration. Following Cohen et al. ([2021](#bib.bib7)), we train an MLP (2 hidden layers, 200 units, tanh activation) on CIFAR-10 under cross-entropy (CE) loss. This small-scale setup enables frequent computation of GN sharpness, which serves as a proxy for the top Hessian eigenvalue λmax\lambda\_{\max}.
Models are trained using gradient descent (GD), stochastic gradient descent (SGD),
AdamW (Kingma & Ba, [2015](#bib.bib21); Loshchilov & Hutter, [2019](#bib.bib29)), Muon (Jordan et al., [2024](#bib.bib19)), and SAM (Foret et al., [2021](#bib.bib11)). We monitor GN sharpness and batch sharpness (Andreyev & Beneventano, [2024](#bib.bib1)) as proxies for loss-landscape geometry, alongside ECE, KCE, loss, and accuracy.

### 3.1 Calibration Temporally Correlates with Sharpness

|  |  | GD | SGD | AdamW | Muon | SAM |
| --- | --- | --- | --- | --- | --- | --- |
| Train | ECE | .83±.08.83{\scriptstyle\pm.08} | .84±.07.84{\scriptstyle\pm.07} | .72±.07.72{\scriptstyle\pm.07} | .63±.26.63{\scriptstyle\pm.26} | .92±.04.92{\scriptstyle\pm.04} |
| KCE | .83±.08.83{\scriptstyle\pm.08} | .84±.07.84{\scriptstyle\pm.07} | .70±.08.70{\scriptstyle\pm.08} | .61±.28.61{\scriptstyle\pm.28} | .91±.04.91{\scriptstyle\pm.04} |
| Test | ECE | .96±.02.96{\scriptstyle\pm.02} | .97±.01.97{\scriptstyle\pm.01} | .15±.15.15{\scriptstyle\pm.15} | −.10±.28-.10{\scriptstyle\pm.28} | .98±.01.98{\scriptstyle\pm.01} |
| KCE | .97±.01.97{\scriptstyle\pm.01} | .97±.01.97{\scriptstyle\pm.01} | .21±.22.21{\scriptstyle\pm.22} | −.16±.31-.16{\scriptstyle\pm.31} | .98±.01.98{\scriptstyle\pm.01} |

Table 1: Pearson correlation between calibration metrics (ECE and KCE) and GN sharpness, mean ±\pm std over 4
learning rates.

Figure [1](#S0.F1 "Figure 1 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") shows training dynamics for GD and SGD. Across all CE experiments111We observe a similar temporal correlation on models trained with mean-squared error (MSE) loss.
There, however, both GN sharpness and ECE increase and then plateau at high values. This behavior reflects the fact that MSE is not a proper scoring rule and induces systematic underconfidence; we therefore focus on CE in the main text and defer a detailed discussion of MSE to Appendix [E](#A5 "Appendix E Extension to Mean Squared Error ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").
,
training ECE and GN sharpness follow the same trajectory: both quantities are small at initialization, increase as training enters an EoS regime, and decrease again later in training. This holds across optimizers and learning rates
(see Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Converging to Flat Minima or Following Flat Directions? ‣ 3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") and Appendix [C.1](#A3.SS1 "C.1 Sharpness–Calibration Correlation Analysis ‣ Appendix C Additional Sharpness–Calibration Experiments ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") for additional results). Similar observations extend to CIFAR-100 (Appendix [C.1](#A3.SS1 "C.1 Sharpness–Calibration Correlation Analysis ‣ Appendix C Additional Sharpness–Calibration Experiments ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")). Table [1](#S3.T1 "Table 1 ‣ 3.1 Calibration Temporally Correlates with Sharpness ‣ 3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") quantifies this effect, showing strong Pearson correlations between ECE and (batch) sharpness throughout training; KCE closely matches ECE across all settings, confirming the coupling is not a binning artifact.

As we show in Section [4](#S4 "4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"), both calibration and sharpness converge to zero once all training points are correctly classified, so their coupling at the end of training is expected. What is surprising is the strong correlation during training, well before convergence, when the model is far from interpolation and calibration is nontrivial. To our knowledge, this has not been observed or explained before. These results suggest that calibration does not depend on training-metrics at convergence, but on the trajectory itself: models that stay in lower-sharpness regions remain better calibrated throughout training, not just asymptotically. We formalize this connection in Section [4](#S4 "4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").

### 3.2 Converging to Flat Minima or Following Flat Directions?

The strong temporal correlation between sharpness and calibration established in the previous section raises a causal question: *does calibration improve because optimization converges to flatter minima, or because training dynamics suppress movement along high-curvature directions?* To disentangle these mechanisms, we formulate two competing hypotheses, and empirically find that suppressing directions of steep descent leads to improved in-sample calibration.

###### Hypothesis 1 (Flat Minima for Calibration).

Training procedures that bias optimization toward flat minima lead to lower in-sample calibration error.

###### Hypothesis 2 (Directional Flatness for Calibration).

Training procedures that suppress updates along directions of steep curvature lead to lower in-sample calibration error, even if the final solution is not globally flat.

We test Hypothesis [1](#Thmhyp1 "Hypothesis 1 (Flat Minima for Calibration). ‣ 3.2 Converging to Flat Minima or Following Flat Directions? ‣ 3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") using SAM (Zhou et al., [2025](#bib.bib57)), which explicitly penalizes worst-case loss perturbations within a local neighborhood, and is known to bias optimization toward flatter minima. To test Hypothesis [2](#Thmhyp2 "Hypothesis 2 (Directional Flatness for Calibration). ‣ 3.2 Converging to Flat Minima or Following Flat Directions? ‣ 3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"), we use optimizers that directly suppress high-curvature directions during training. Muon (Jordan et al., [2024](#bib.bib19)) rescales gradient components to equalize their magnitudes, effectively clamping updates along sharp directions while amplifying flatter ones. BulkSGD (Song et al., [2025](#bib.bib44)) achieves a more extreme intervention by projecting gradients onto the subspace orthogonal to the top Hessian eigenvectors, thereby removing the steepest descent directions entirely. A more detailed analysis of the optimizers and their benefits in this experimental setting can be found in Appendix [C.2](#A3.SS2 "C.2 Optimizer Details: SAM, Muon, and BulkSGD ‣ Appendix C Additional Sharpness–Calibration Experiments ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").

Figures [2](#S3.F2 "Figure 2 ‣ 3.2 Converging to Flat Minima or Following Flat Directions? ‣ 3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") and [3](#S3.F3 "Figure 3 ‣ 3.2 Converging to Flat Minima or Following Flat Directions? ‣ 3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") show the training dynamics. Although SAM consistently maintains lower sharpness than GD and SGD, its calibration trajectory closely mirrors that of standard training, with a comparable peak ECE and slower convergence. In contrast, both Muon and BulkSGD achieve substantially lower peak calibration error and faster ECE decay, despite exhibiting markedly different sharpness profiles.

![Refer to caption](/html/2604.20614/assets/x2.png)


Figure 2: Training dynamics for SAM and Muon across learning rates on CIFAR-10.



![Refer to caption](/html/2604.20614/assets/img/main/Bulk_SGD_LR_2_70.png)


(a) BulkSGD with learning rate 270\frac{2}{70}

![Refer to caption](/html/2604.20614/assets/img/main/BulkSGD_LR_2_30.png)


(b) BulkSGD with learning rate 230\frac{2}{30}

Figure 3: Training dynamics for BulkSGD across different learning rates and number of projected-out gradients on CIFAR10.

Notably, Muon maintains low calibration error while operating in regimes that are not globally flat, and BulkSGD improves calibration even in the presence of pronounced instability.
This suggests that calibration is sensitive to how optimization traverses sharp directions, rather than to the absolute flatness of the loss landscape.
However, BulkSGD induces oscillatory dynamics and a sharpness divergence when too many dominant directions are projected out, making it impractical as a standalone optimizer.
Together, these results support Hypothesis [2](#Thmhyp2 "Hypothesis 2 (Directional Flatness for Calibration). ‣ 3.2 Converging to Flat Minima or Following Flat Directions? ‣ 3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") over Hypothesis [1](#Thmhyp1 "Hypothesis 1 (Flat Minima for Calibration). ‣ 3.2 Converging to Flat Minima or Following Flat Directions? ‣ 3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"): suppressing updates along high-curvature directions during training leads to improved in-sample calibration, whereas convergence to flat minima alone does not.

### 3.3 Out-of-Sample Behavior

The sharpness–calibration coupling is less consistent out of sample (Table [1](#S3.T1 "Table 1 ‣ 3.1 Calibration Temporally Correlates with Sharpness ‣ 3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")). Across optimizers, test ECE does not consistently decrease alongside training ECE—in some cases it worsens as training progresses, even after sharpness and training calibration improve (Appendix [C.1](#A3.SS1 "C.1 Sharpness–Calibration Correlation Analysis ‣ Appendix C Additional Sharpness–Calibration Experiments ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")).
Muon is an extreme example: training ECE drops to near zero while test ECE remains high, yielding a negative test correlation despite strong in-sample alignment. This reflects the calibration generalization gap in overparameterized models (Carrell et al., [2022](#bib.bib5); Berta et al., [2025](#bib.bib3); Wu et al., [2025](#bib.bib51)): a model that fits training data well can become overconfident on misclassified test examples, causing test ECE to increase and decouple from sharpness. In Muon’s case, the near-zero training ECE is consistent with the large training margins it induces (Section [4](#S4 "4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")): once these are extreme, the model becomes overconfident on test examples near the decision boundary, precisely where the margin functional is most sensitive.

Together with the findings from Section [3.2](#S3.SS2 "3.2 Converging to Flat Minima or Following Flat Directions? ‣ 3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"), these results point to an important distinction:
on the one hand, directional interventions yield better in-sample calibration than flat-minima methods, suggesting that *how* optimization traverses curvature matters more than where it converges; on the other hand, in-sample improvements do not automatically transfer to test data, pointing to a fundamental train–test gap. In the following section, we formalize this train–test gap and use it to design a training-time intervention that improves out-of-sample calibration.

## 4 Curvature and Calibration in the Separable and Non-separable Regimes

In this section we explain the temporal alignment between calibration error and curvature observed in Section [3](#S3 "3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")
through a common underlying quantity: the (robust) *true logit margin*. Our central claim is that, across training,
both ECE and Gauss–Newton sharpness respond to the evolution of the same margin-dependent
tail functional. This provides a concrete mechanism linking predictive confidence to loss-landscape geometry.

The analysis naturally separates into two regimes. Early in training, the data behave as
*overlap-dominated*: a nontrivial fraction of examples exhibit small or negative true margins, and no uniform separability holds.
The same regime persists at test time whenever accuracy is below 11, since any misclassified example has mθ​(x,y)<0m\_{\theta}(x,y)<0 by definition—and, unlike on training, no cross-entropy mechanism pushes those margins to grow.
In either case, neither calibration error nor curvature is forced to be small, and both can be dominated by a few hard or ambiguous points.
This perspective aligns with observations that loss and curvature are controlled by a small set of strongly opposing examples
(Rosenfeld & Risteski, [2024](#bib.bib43)). Later in training on the *training set*, models trained with cross-entropy typically enter an
*interpolating* regime in which all true margins become strictly positive. In this regime, calibration error and curvature
become tightly coupled: once the margin tail contracts, both quantities are forced to decrease together.

Together, these results provide a mechanism-level explanation for the observed co-evolution of calibration and curvature during
training, and clarify why training and test calibration can diverge even as accuracy improves. We conclude the section by
connecting these theoretical regimes to the empirical training dynamics observed in Section [3](#S3 "3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").

#### Setup and Notation.

Let (X,Y)∼π(X,Y)\sim\pi with Y∈{1,…,K}Y\in\{1,\dots,K\}. A model θ∈ℝd\theta\in\mathbb{R}^{d} produces logits zθ​(x)∈ℝKz\_{\theta}(x)\in\mathbb{R}^{K} and probabilities
pθ​(x)=softmax⁡(zθ​(x))p\_{\theta}(x)=\operatorname{softmax}(z\_{\theta}(x)). Let y^​(x)=arg⁡maxk⁡zθ​(x)k\hat{y}(x)=\arg\max\_{k}z\_{\theta}(x)\_{k} (deterministic tie-break) and confidence P^​(x)=maxk⁡pθ​(x)k\hat{P}(x)=\max\_{k}p\_{\theta}(x)\_{k}.
Define the *true (logit) margin*

|  |  |  |
| --- | --- | --- |
|  | mθ​(x,y):=zθ​(x)y−maxj≠y⁡zθ​(x)j,m\_{\theta}(x,y):=z\_{\theta}(x)\_{y}-\max\_{j\neq y}z\_{\theta}(x)\_{j}, |  |

and the *robust true margin* at radius ε>0\varepsilon>0,

|  |  |  |
| --- | --- | --- |
|  | mε,θ​(x,y):=inf‖δ‖≤εmθ​(x+δ,y).m\_{\varepsilon,\theta}(x,y):=\inf\_{\|\delta\|\leq\varepsilon}m\_{\theta}(x+\delta,y). |  |

Let ECEM\mathrm{ECE}\_{M} denote the population π\pi/sample 𝒟\mathcal{D} binned ECE computed by binning P^​(X)\hat{P}(X) into MM bins.
Let Jθ​(x):=∂zθ​(x)/∂θ∈ℝK×dJ\_{\theta}(x):=\partial z\_{\theta}(x)/\partial\theta\in\mathbb{R}^{K\times d} and, for cross-entropy, Hz​(p):=diag⁡(p)−p​p⊤H\_{z}(p):=\operatorname{diag}(p)-pp^{\top}.
Define the population Gauss–Newton matrix and its curvature proxy

|  |  |  |
| --- | --- | --- |
|  | HGN​(θ;π):=𝔼π​[Jθ​(X)⊤​Hz​(pθ​(X))​Jθ​(X)],H\_{\mathrm{GN}}(\theta;\pi):=\mathbb{E}\_{\pi}\!\big[J\_{\theta}(X)^{\top}H\_{z}(p\_{\theta}(X))J\_{\theta}(X)\big], |  |

|  |  |  |
| --- | --- | --- |
|  | λmax:=λmax​(HGN​(θ;π)).\lambda\_{\max}:=\lambda\_{\max}\!\big(H\_{\mathrm{GN}}(\theta;\pi)\big). |  |

### 4.1 Regime I: overlap-dominated (non-separable) behavior

In this subsection all the quantities (ECE, GN matrix, robust margin, robust margin moment) are considered at a population level. See details in Appendix [B](#A2 "Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"). Define the robust exponential margin moment Q​(θ):=𝔼(X,Y)∼π​[e−mε,θ​(X,Y)]Q(\theta):=\mathbb{E}\_{(X,Y)\sim\pi}\!\big[e^{-m\_{\varepsilon,\theta}(X,Y)}\big].

###### Theorem 4.1 (Overlap regime: robust-margin upper bounds).

For any θ\theta and any distribution π\pi,

|  |  |  |
| --- | --- | --- |
|  | ECEM≤(K−1)​Q​(θ).\mathrm{ECE}\_{M}\ \leq\ (K-1)\,Q(\theta). |  |

If additionally ‖Jθ​(X)‖op≤CJ\|J\_{\theta}(X)\|\_{\mathrm{op}}\leq C\_{J} holds π\pi-a.s., then

|  |  |  |
| --- | --- | --- |
|  | λmax≤ 2​CJ2​(K−1)​Q​(θ).\lambda\_{\max}\ \leq\ 2C\_{J}^{2}\,(K-1)\,Q(\theta). |  |

Proof is in Appendix [B.4](#A2.SS4 "B.4 Proof of Theorem 4.1 ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").

#### Interpretation (two bottlenecks).

Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1 (Overlap regime: robust-margin upper bounds). ‣ 4.1 Regime I: overlap-dominated (non-separable) behavior ‣ 4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") exposes two multiplicative controls:

* •

  a *probability bottleneck* Q​(θ)Q(\theta), dominated by the tail of small/negative robust margins,
* •

  a *geometry bottleneck* CJ2C\_{J}^{2} (how parameter perturbations move logits).

In overlap-dominated regimes (early training or test-time), a persistent set of small robust margins can keep Q​(θ)Q(\theta) bounded away from 0,
so these bounds need not certify vanishing calibration error or curvature even if loss continues to decrease.

### 4.2 Regime II: Interpolating (separable) behavior on the training set

In this subsection all the quantities (ECE, GN matrix, robust margin, robust margin moment) are considered at a finite-sample level. See details in Appendix [B](#A2 "Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"). Let γ​(θ;𝒟):=min1≤i≤n⁡mθ​(xi,yi)\gamma(\theta;\mathcal{D})\;:=\;\min\_{1\leq i\leq n}m\_{\theta}(x\_{i},y\_{i}) and the empirical exponential margin moment Q𝒟​(θ):=1n​∑i=1ne−mθt​(xi,yi)Q\_{\mathcal{D}}(\theta):=\frac{1}{n}\sum\_{i=1}^{n}e^{-m\_{\theta\_{t}}(x\_{i},y\_{i})}.

###### Theorem 4.2 (Interpolating regime: two-sided ECE–margin control and coupling to λmax\lambda\_{\max}).

Assume γ​(θ;𝒟)>0\gamma(\theta;\mathcal{D})>0 (all training points correctly classified with strictly positive true margin).
Then

|  |  |  |
| --- | --- | --- |
|  | 1K​Q𝒟​(θ)≤ECEM≤(K−1)​Q𝒟​(θ)≤(K−1)​e−γ​(θ;𝒟).\begin{split}\frac{1}{K}\,Q\_{\mathcal{D}}(\theta)\ &\leq\ \mathrm{ECE}\_{M}\ \leq\ (K-1)\,Q\_{\mathcal{D}}(\theta)\\ &\leq\ (K-1)\,e^{-\gamma(\theta;\mathcal{D})}.\end{split} |  |

If additionally maxi∈[n]⁡‖Jθ​(xi)‖op≤CJ\max\_{i\in[n]}\|J\_{\theta}(x\_{i})\|\_{\mathrm{op}}\leq C\_{J}, then

|  |  |  |
| --- | --- | --- |
|  | λmax≤ 2​CJ2​(K−1)​Q𝒟​(θ)≤ 2​CJ2​K​(K−1)​ECEM,\lambda\_{\max}\ \leq\ 2C\_{J}^{2}\,(K-1)\,Q\_{\mathcal{D}}(\theta)\ \leq\ 2C\_{J}^{2}\,K(K-1)\,\mathrm{ECE}\_{M}, |  |

equivalently ECEM≥λmax/(2​CJ2​K​(K−1))\ \mathrm{ECE}\_{M}\geq\lambda\_{\max}/(2C\_{J}^{2}K(K-1)).

Proof is in Appendix [B.5](#A2.SS5 "B.5 Proof of Theorem 4.2 ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").

#### Interpretation.

In the interpolating regime, ECEM\mathrm{ECE}\_{M} is equivalent up to constants to the exponential margin moment Q𝒟​(θ)Q\_{\mathcal{D}}(\theta),
and λmax\lambda\_{\max} is controlled by the *same* moment (under bounded Jacobians).
This implies that once the training set is correctly classified, GN sharpness cannot be large without in-sample ECE also being large.
Moreover, in this regime, empirical binning becomes immaterial:
ECEM\mathrm{ECE}\_{M} reduces to the mean misconfidence (formalized in Appendix [B](#A2 "Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")).

### 4.3 Discussion: how the two-regime proxy matches the observed train/test split

Section [3](#S3 "3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") shows strong co-evolution of training ECE and sharpness, well before convergence.
Asymptotic interpolation alone is insufficient to explain this observation.
Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1 (Overlap regime: robust-margin upper bounds). ‣ 4.1 Regime I: overlap-dominated (non-separable) behavior ‣ 4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")–[4.2](#S4.Thmtheorem2 "Theorem 4.2 (Interpolating regime: two-sided ECE–margin control and coupling to 𝜆ₘₐₓ). ‣ 4.2 Regime II: Interpolating (separable) behavior on the training set ‣ 4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") provide a mechanism-level lens: throughout training,
both quantities respond to the evolution of the same margin-dependent tail functional, and in the interpolating regime
this coupling becomes two-sided.

A remaining question is: why test ECE can increase while the logged (predicted) margin increases.
Two caveats explain this:

* •

  The bounds depend on the *true* margin mθ​(x,y)m\_{\theta}(x,y), not the predicted margin zy^−z(2)z\_{\hat{y}}-z\_{(2)}.
  A model can become *more confidently wrong* on a subset of test points: predicted margins increase, accuracy plateaus, and test ECE increases.
* •

  The sharpness control involves a geometry term (Jacobians). Even if Hz​(p)H\_{z}(p) contracts as predictions become more one-hot,
  large Jacobian norms (or failure of uniform Jacobian control) can keep curvature proxies large.

## 5 From Margin Theory to Calibrated Training

Section [4](#S4 "4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") establishes that calibration error is controlled by the margin
functional Q​(θ)=𝔼​[e−mε,θ​(X,Y)]Q(\theta)=\mathbb{E}[e^{-m\_{\varepsilon,\theta}(X,Y)}], with the bound
ECE≤(K−1)​Q​(θ)\mathrm{ECE}\leq(K-1)\,Q(\theta) holding on any distribution. This motivates directly
targeting Q​(θ)Q(\theta) during training. We propose an objective that enforces robust margins.

### 5.1 Calibration with Margin Objective

To minimize the ECE bound, we want large robust margins. For this, we combine two strategies: (i) directly raising the margin at adversarial points, and (ii) ensuring clean margins do not collapse under perturbation:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | LCalMO(θ)=𝔼(x,y)[\displaystyle L\_{\mathrm{CalMO}}(\theta)=\mathbb{E}\_{(x,y)}\Big[ | ℓCE​(θ;x,y)+λr​Rrob​(θ;x,y)\displaystyle\ell\_{\mathrm{CE}}(\theta;x,y)+\lambda\_{r}\,R\_{\mathrm{rob}}(\theta;x,y) |  | (2) |
|  |  | +λsRsmooth(θ;x,y)].\displaystyle+\lambda\_{s}\,R\_{\mathrm{smooth}}(\theta;x,y)\Big]. |  |

where λr,λs≥0\lambda\_{r},\lambda\_{s}\geq 0 are hyperparameters and RrobR\_{\mathrm{rob}} and RsmoothR\_{\mathrm{smooth}} are defined below.

#### Robustness regularizer.

Following TRADES (Zhang et al., [2019](#bib.bib54)), we encourage consistent predictions
between clean and adversarial inputs:

|  |  |  |
| --- | --- | --- |
|  | Rrob​(θ;x,y)=DKL​(pθ​(x)∥pθ​(xadv)),R\_{\mathrm{rob}}(\theta;x,y)=D\_{\mathrm{KL}}\big(p\_{\theta}(x)\,\|\,p\_{\theta}(x\_{\mathrm{adv}})\big), |  |

where xadv∈arg⁡max‖x′−x‖≤ε⁡ℓCE​(θ;x′,y)x\_{\mathrm{adv}}\in\arg\max\_{\|x^{\prime}-x\|\leq\varepsilon}\ell\_{\mathrm{CE}}(\theta;x^{\prime},y).
Combined with cross-entropy at xx, this raises the margin at the worst point in the
ε\varepsilon-neighborhood, directly targeting mε,θ​(x,y)m\_{\varepsilon,\theta}(x,y).

#### Smoothness regularizer.

By Lemma [B.5](#A2.Thmtheorem5 "Lemma B.5 (Robust margin comparisons (trivial upper bound; Lipschitz lower bound)). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"), if the margin has local Lipschitz constant Lm​(x,y)L\_{m}(x,y), then

|  |  |  |  |
| --- | --- | --- | --- |
|  | mε,θ​(x,y)≥mθ​(x,y)−ε​Lm​(x,y),m\_{\varepsilon,\theta}(x,y)\geq m\_{\theta}(x,y)-\varepsilon L\_{m}(x,y), |  | (3) |

implying e−mε,θ≤e−mθ⋅eε​Lme^{-m\_{\varepsilon,\theta}}\leq e^{-m\_{\theta}}\cdot e^{\varepsilon L\_{m}}.
When LmL\_{m} is large, this bound becomes vacuous even for large clean margins.
To prevent this, we penalize:

|  |  |  |
| --- | --- | --- |
|  | Rsmooth​(θ;x,y)=‖∇xmθ​(x,y)‖22.R\_{\mathrm{smooth}}(\theta;x,y)=\|\nabla\_{x}m\_{\theta}(x,y)\|\_{2}^{2}. |  |

For neural networks with Lipschitz activations, mθm\_{\theta} is locally Lipschitz
with Lm​(x,y)=‖∇xmθ​(x,y)‖L\_{m}(x,y)=\|\nabla\_{x}m\_{\theta}(x,y)\| almost everywhere.
This keeps mθ−ε​Lmm\_{\theta}-\varepsilon L\_{m} close to mθm\_{\theta}, ensuring large clean
margins translate to large robust margins.

#### Why naive margin maximization fails.

As training progresses, clean margins mθ​(xi,yi)m\_{\theta}(x\_{i},y\_{i}) on training points grow, Qtrain→0Q\_{\mathrm{train}}\to 0, and training ECE vanishes. Yet test ECE remains high.
The issue is that large clean margins need not imply large *robust* margins: a model can achieve mθ​(x,y)≫0m\_{\theta}(x,y)\gg 0 while the margin collapses at x+δx+\delta for small perturbations. The robust margin mε,θ​(x,y)=inf‖δ‖≤εmθ​(x+δ,y)m\_{\varepsilon,\theta}(x,y)=\inf\_{\|\delta\|\leq\varepsilon}m\_{\theta}(x+\delta,y) captures this distinction. If robust margins at training points are large, nearby test points—which lie within a ε\varepsilon-neighborhood under typical data distributions—inherit reasonable margins. Fragile margins, large only at training locations, provide no such transfer.

### 5.2 Empirical Results with CalMO

#### Setup.

We run a *controlled fixed-budget experiment*: within each optimizer, only the loss function varies; all other training choices are held fixed. Concretely, we train ResNet-20 (He et al., [2016](#bib.bib13)) on CIFAR-10.
We compare four gradient-based optimizers spanning different training dynamics: SGD, AdamW, Muon, and SAM. Learning rates are tuned per optimizer: η=0.01\eta=0.01 for SGD, Muon and SAM, η=0.001\eta=0.001 for AdamW. All runs use batch size 128 for 10,000 steps. We compare standard cross-entropy against CalMO with hyperparameters λr=0.5\lambda\_{r}=0.5 (robustness) and λs=0.01\lambda\_{s}=0.01 (flatness). The adversarial perturbation radius is ε=8/255\varepsilon=8/255, computed via 3-step PGD. Adversarial examples are initialized at the clean input xx (no random start) and updated with step size α=2/255\alpha=2/255. We report test accuracy and ECE computed with 15 bins.

These values are fixed without tuning; our aim is to validate that the robust-margin regularization motivated by Section [4](#S4 "4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") yields calibration gains over CE, and to demonstrate that training-time intervention on the loss landscape can improve calibration without relying on post-hoc corrections. We report test values in Table [2](#S5.T2 "Table 2 ‣ Performance. ‣ 5.2 Empirical Results with CalMO ‣ 5 From Margin Theory to Calibrated Training ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"); training values are reported in Table [A1](#A4.T1 "Table A1 ‣ D.2 Train–Test Calibration Gap ‣ Appendix D CalMO: Extended Results ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") in the Appendix.

#### Performance.

CalMO lowers test ECE across all four optimizers (Table [2](#S5.T2 "Table 2 ‣ Performance. ‣ 5.2 Empirical Results with CalMO ‣ 5 From Margin Theory to Calibrated Training ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")), with reductions ranging from 0.0030.003 (SAM: 0.020→0.0170.020\to 0.017) to 0.0460.046 (Muon: 0.065→0.0190.065\to 0.019); test accuracy is preserved or improves in every case, by between +0.2+0.2 and +4.9+4.9 points, consistent with regularization helping under the fixed 10k-step budget. The Muon gap is the sharpest illustration of the mechanism of Section [4](#S4 "4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"): directional optimization drives training margins to extreme values that collapse under input perturbation (fragile margins), keeping test ECE high, and CalMO’s robust-margin term directly targets this tail.

| Optimizer | Method | Acc (%) | ECE ↓\downarrow |
| --- | --- | --- | --- |
| SGD | CE | 75.2±1.275.2\pm 1.2 | 0.081±0.0210.081\pm 0.021 |
| Flat. (λr=0\lambda\_{r}{=}0) | 80.8±0.1\mathbf{80.8\pm 0.1} | 0.053±0.003\mathbf{0.053\pm 0.003} |
| Rob. (λs=0\lambda\_{s}{=}0) | 78.2±0.978.2\pm 0.9 | 0.062±0.0110.062\pm 0.011 |
| CalMO | 80.1±1.480.1\pm 1.4 | 0.056±0.0010.056\pm 0.001 |
| AdamW | CE | 80.7±0.480.7\pm 0.4 | 0.061±0.0070.061\pm 0.007 |
| Flat. (λr=0\lambda\_{r}{=}0) | 83.3±0.9\mathbf{83.3\pm 0.9} | 0.047±0.0020.047\pm 0.002 |
| Rob. (λs=0\lambda\_{s}{=}0) | 82.4±0.282.4\pm 0.2 | 0.039±0.009\mathbf{0.039\pm 0.009} |
| CalMO | 83.2±1.183.2\pm 1.1 | 0.045±0.0070.045\pm 0.007 |
| Muon | CE | 80.3±0.380.3\pm 0.3 | 0.065±0.0160.065\pm 0.016 |
| Flat. (λr=0\lambda\_{r}{=}0) | 81.0±0.281.0\pm 0.2 | 0.046±0.0120.046\pm 0.012 |
| Rob. (λs=0\lambda\_{s}{=}0) | 81.9±0.4\mathbf{81.9\pm 0.4} | 0.052±0.0100.052\pm 0.010 |
| CalMO | 81.7±0.981.7\pm 0.9 | 0.019±0.002\mathbf{0.019\pm 0.002} |
| SAM | CE | 85.0±0.385.0\pm 0.3 | 0.020±0.0050.020\pm 0.005 |
| Flat. (λr=0\lambda\_{r}{=}0) | 84.1±0.384.1\pm 0.3 | 0.016±0.004\mathbf{0.016\pm 0.004} |
| Rob. (λs=0\lambda\_{s}{=}0) | 85.5±0.4\mathbf{85.5\pm 0.4} | 0.021±0.0060.021\pm 0.006 |
| CalMO | 85.2±0.485.2\pm 0.4 | 0.017±0.0050.017\pm 0.005 |

Table 2: CalMO vs CE. ResNet-20 on CIFAR-10, test set.
Flat.: flatness only (λr=0,λs=0.01\lambda\_{r}{=}0,\lambda\_{s}{=}0.01); Rob.: robustness only (λr=0.5,λs=0\lambda\_{r}{=}0.5,\lambda\_{s}{=}0); CalMO: λr=0.5,λs=0.01\lambda\_{r}{=}0.5,\lambda\_{s}{=}0.01. The relative importance of each term varies with optimizer, but CalMO strikes a balance across all settings.

#### Computational cost.

CalMO incurs additional cost from the robustness and flatness terms. The robustness term requires a 3-step PGD attack per iteration; however, these compute gradients with respect to the input rather than the model parameters, making each PGD step relatively cheap. The flatness term adds one forward pass and one input-gradient computation. This cost can be reduced by using only the flatness term (λr=0\lambda\_{r}=0). All methods are trained for 10k steps, and we report performance at the best validation step. Matching compute by training CE longer does not close the gap: CE reaches its validation optimum well before 10k steps, after which performance degrades.

#### Ablation.

To isolate the roles of robustness and flatness in shaping calibration, we ablate the CalMO objective into its components and evaluate their effects across optimizers. The results reveal a consistent but optimizer-dependent pattern.
For SGD, enforcing flatness alone yields most of the calibration gains, while robustness provides limited benefit.
For AdamW the two terms split roles: robustness yields the largest ECE reduction, while flatness yields the largest accuracy gain, suggesting that its adaptive preconditioning stabilizes parameter-space geometry along the directions relevant for accuracy, while leaving calibration sensitive
to robust-margin control.
Muon exhibits a different behavior: neither robustness nor flatness alone is sufficient to substantially reduce ECE, but their combination leads to the greatest calibration. This supports the theoretical mechanism developed in Section [4](#S4 "4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"), where calibration is governed jointly by the robust-margin tail and geometric sensitivity. From this perspective, CalMO should not be viewed as a universally superior training objective, but rather as a targeted intervention that controls these two terms.

## 6 Conclusion

We studied calibration as a *training-time phenomenon* rather than a static property of a converged model, and showed that calibration and sharpness are tightly coupled along the optimization trajectory across multiple optimizers. This coupling arises from a shared dependence on margin growth, explaining both the temporal co-evolution of Expected Calibration Error and curvature during training, as well as the frequent divergence between train and test calibration in overlap-dominated regimes.

Building on this perspective, we distinguished between two competing mechanisms for improving calibration: convergence to flat minima and suppression of updates along high-curvature directions. Empirically, optimizers that implement directional control, such as Muon and BulkSGD, yielded consistently better in-sample calibration than methods targeting flat minima alone. Guided by the margin-based bounds, we proposed CalMO, a training objective that jointly regulates robust margins and local smoothness, and showed that their combined regulation can substantially improve out-of-sample calibration—most notably for directionally amplified optimizers such as Muon—while preserving predictive accuracy.

#### Limitations.

Our empirical study relies on explicit curvature diagnostics (Gauss–Newton / Hessian-based measurements), which are computationally expensive and constrain the scale of architectures and datasets we can probe. Our theoretical results identify a margin-tail mediator that upper-bounds both calibration error and curvature under a Jacobian-control assumption; we do not claim that these certificates are tight or that the Jacobian bounds hold uniformly in all deep networks. A natural next step is to develop scalable, distributionally robust proxies for the mediator (for example, low-rank spectral estimators, mini-batch surrogates, or input-space stability measurements) and to characterize when they preserve the qualitative regime predictions we derive.

Our experiments also cover small-scale image classification only; extending the trajectory-level analysis to language-model training, where calibration is a central open question, is a natural follow-up. More broadly, we see the trajectory perspective itself—tracking how calibration, curvature, and margins co-evolve rather than inspecting them only at convergence—as a lens applicable to other training-time phenomena in deep networks.

## References

* Andreyev & Beneventano (2024)

  Andreyev, A. and Beneventano, P.
  Edge of stochastic stability: Revisiting the edge of stability for SGD.
  *arXiv preprint arXiv:2412.20553*, 2024.
  doi: 10.48550/arXiv.2412.20553.
  URL <https://arxiv.org/abs/2412.20553>.
* Bartlett et al. (2017)

  Bartlett, P. L., Foster, D. J., and Telgarsky, M.
  Spectrally-normalized margin bounds for neural networks.
  In *Advances in Neural Information Processing Systems*, volume 30, pp. 6240–6249, 2017.
  URL <https://proceedings.neurips.cc/paper_files/paper/2017/hash/b22b257ad0519d4500539da3c8bcf4dd-Abstract.html>.
* Berta et al. (2025)

  Berta, E., Holzmüller, D., Jordan, M. I., and Bach, F.
  Rethinking early stopping: Refine, then calibrate.
  *arXiv preprint arXiv:2501.19195*, 2025.
  doi: 10.48550/arXiv.2501.19195.
  URL <https://arxiv.org/abs/2501.19195>.
* Bohdal et al. (2023)

  Bohdal, O., Yang, Y., and Hospedales, T.
  Meta-calibration: Learning of model calibration using differentiable expected calibration error.
  *Transactions on Machine Learning Research*, 2023.
  URL <https://openreview.net/forum?id=R2hUure38l>.
  Accepted by TMLR.
* Carrell et al. (2022)

  Carrell, A. M., Mallinar, N., Lucas, J., and Nakkiran, P.
  The calibration generalization gap.
  *arXiv preprint arXiv:2210.01964*, 2022.
  URL <https://arxiv.org/abs/2210.01964>.
* Chaudhari et al. (2017)

  Chaudhari, P., Choromanska, A., Soatto, S., LeCun, Y., Baldassi, C., Borgs, C., Chayes, J., Sagun, L., and Zecchina, R.
  Entropy-SGD: Biasing gradient descent into wide valleys.
  In *International Conference on Learning Representations*, 2017.
  URL <https://openreview.net/forum?id=B1YfAfcgl>.
* Cohen et al. (2021)

  Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., and Talwalkar, A.
  Gradient descent on neural networks typically occurs at the edge of stability.
  In *International Conference on Learning Representations*, 2021.
  URL <https://openreview.net/forum?id=jh-rTtvkGeM>.
  ICLR 2021 Poster.
* Cohen et al. (2022)

  Cohen, J. M., Ghorbani, B., Krishnan, S., Agarwal, N., Medapati, S., Badura, M., Suo, D., Cardoze, D., Nado, Z., Dahl, G. E., and Gilmer, J.
  Adaptive gradient methods at the edge of stability, 2022.
  URL <https://arxiv.org/abs/2207.14484>.
* DeGroot & Fienberg (1983)

  DeGroot, M. H. and Fienberg, S. E.
  The comparison and evaluation of forecasters.
  *Journal of the Royal Statistical Society: Series D (The Statistician)*, 32(1-2):12–22, 1983.
  doi: 10.2307/2987588.
* Dinh et al. (2017)

  Dinh, L., Pascanu, R., Bengio, S., and Bengio, Y.
  Sharp minima can generalize for deep nets.
  In Precup, D. and Teh, Y. W. (eds.), *Proceedings of the 34th International Conference on Machine Learning*, volume 70 of *Proceedings of Machine Learning Research*, pp. 1019–1028. PMLR, 2017.
  URL <https://proceedings.mlr.press/v70/dinh17b.html>.
* Foret et al. (2021)

  Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B.
  Sharpness-aware minimization for efficiently improving generalization.
  In *International Conference on Learning Representations*, 2021.
  URL <https://openreview.net/forum?id=6Tm1mposlrM>.
  ICLR 2021 Spotlight.
* Guo et al. (2017)

  Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q.
  On calibration of modern neural networks.
  In *Proceedings of the 34th International Conference on Machine Learning*, volume 70 of *Proceedings of Machine Learning Research*, pp. 1321–1330. PMLR, 2017.
  URL <https://proceedings.mlr.press/v70/guo17a.html>.
* He et al. (2016)

  He, K., Zhang, X., Ren, S., and Sun, J.
  Deep residual learning for image recognition.
  In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.
* Hochreiter & Schmidhuber (1997)

  Hochreiter, S. and Schmidhuber, J.
  Flat minima.
  *Neural Computation*, 9(1):1–42, 1997.
  doi: 10.1162/neco.1997.9.1.1.
* Hoffer et al. (2017)

  Hoffer, E., Hubara, I., and Soudry, D.
  Train longer, generalize better: Closing the generalization gap in large batch training of neural networks.
  In *Advances in Neural Information Processing Systems*, volume 30, pp. 1731–1741, 2017.
  URL <https://proceedings.neurips.cc/paper_files/paper/2017/hash/a5e0ff62be0b08456fc7f1e88812af3d-Abstract.html>.
* Jastrzębski et al. (2019)

  Jastrzębski, S., Kenton, Z., Ballas, N., Fischer, A., Bengio, Y., and Storkey, A.
  On the relation between the sharpest directions of DNN loss and the SGD step length.
  In *International Conference on Learning Representations*, 2019.
  URL <https://openreview.net/forum?id=SkgEaj05t7>.
* Jastrzębski et al. (2018)

  Jastrzębski, S., Kenton, Z., Arpit, D., Ballas, N., Fischer, A., Bengio, Y., and Storkey, A.
  Three Factors Influencing Minima in SGD.
  *arXiv:1711.04623 [cs, stat]*, September 2018.
  URL <http://arxiv.org/abs/1711.04623>.
  arXiv:1711.04623.
* Jiang et al. (2020)

  Jiang, Y., Neyshabur, B., Mobahi, H., Krishnan, D., and Bengio, S.
  Fantastic generalization measures and where to find them.
  In *International Conference on Learning Representations*, 2020.
  URL <https://openreview.net/forum?id=SJgIPJBFvH>.
* Jordan et al. (2024)

  Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., and Bernstein, J.
  Muon: An optimizer for hidden layers in neural networks, 2024.
  URL <https://kellerjordan.github.io/posts/muon/>.
* Keskar et al. (2017)

  Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., and Tang, P. T. P.
  On large-batch training for deep learning: Generalization gap and sharp minima.
  In *International Conference on Learning Representations*, 2017.
  URL <https://openreview.net/forum?id=H1oyRlYgg>.
* Kingma & Ba (2015)

  Kingma, D. P. and Ba, J.
  Adam: A method for stochastic optimization.
  In *International Conference on Learning Representations*, 2015.
  URL <https://arxiv.org/abs/1412.6980>.
* Kull et al. (2019)

  Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, H., and Flach, P.
  Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration.
  In *Advances in Neural Information Processing Systems*, volume 32, pp. 12316–12326, 2019.
  URL <https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html>.
* Kumar et al. (2018)

  Kumar, A., Sarawagi, S., and Jain, U.
  Trainable calibration measures for neural networks from kernel mean embeddings.
  In *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pp. 2805–2814. PMLR, 2018.
  URL <https://proceedings.mlr.press/v80/kumar18a.html>.
* Lengyel et al. (2021)

  Lengyel, D., Jennings, N., Parpas, P., and Kantas, N.
  On flat minima, large margins and generalizability.
  OpenReview (ICLR 2021 submission), 2021.
  URL <https://openreview.net/forum?id=Ki5Mv0iY8C>.
* Li & Sur (2025)

  Li, Y. and Sur, P.
  Optimal and provable calibration in high-dimensional binary classification: Angular calibration and Platt scaling.
  In *Advances in Neural Information Processing Systems*, 2025.
  URL <https://openreview.net/forum?id=SgQAleMecy>.
  NeurIPS 2025 Spotlight.
* Liang et al. (2019)

  Liang, T., Poggio, T., Rakhlin, A., and Stokes, J.
  Fisher-Rao metric, geometry, and complexity of neural networks.
  In *Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics*, volume 89 of *Proceedings of Machine Learning Research*, pp. 888–896. PMLR, 2019.
* Lin et al. (2017)

  Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P.
  Focal loss for dense object detection.
  In *Proceedings of the IEEE international conference on computer vision*, pp. 2980–2988, 2017.
* Liu et al. (2022)

  Liu, B., Ben Ayed, I., Galdran, A., and Dolz, J.
  The devil is in the margin: Margin-based label smoothing for network calibration.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 80–88, 2022.
* Loshchilov & Hutter (2019)

  Loshchilov, I. and Hutter, F.
  Decoupled weight decay regularization.
  In *International Conference on Learning Representations*, 2019.
  URL <https://openreview.net/forum?id=Bkg6RiCqY7>.
* Maddox et al. (2020)

  Maddox, W. J., Benton, G., and Wilson, A. G.
  Rethinking parameter counting in deep models: Effective dimensionality revisited.
  *arXiv preprint arXiv:2003.02139*, 2020.
  URL <https://arxiv.org/abs/2003.02139>.
* Mason-Williams et al. (2024)

  Mason-Williams, I., Ekholm, F., and Huszár, F.
  Explicit regularisation, sharpness and calibration.
  In *NeurIPS 2024 Workshop on Scientific Methods for Understanding Deep Learning (SciForDL)*. OpenReview.net, October 2024.
  URL <https://openreview.net/forum?id=ZQTiGcykl6>.
* Möllenhoff & Khan (2023)

  Möllenhoff, T. and Khan, M. E.
  SAM as an optimal relaxation of Bayes.
  In *International Conference on Learning Representations*, 2023.
  URL <https://openreview.net/forum?id=k4fevFqSQcX>.
* Mukhoti et al. (2020)

  Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P. H. S., and Dokania, P. K.
  Calibrating deep neural networks using focal loss.
  In *Advances in Neural Information Processing Systems*, volume 33, pp. 15288–15299, 2020.
  URL <https://proceedings.neurips.cc/paper/2020/hash/aeb7b30ef1d024a76f21a1d40e30c302-Abstract.html>.
* Müller et al. (2019)

  Müller, R., Kornblith, S., and Hinton, G. E.
  When does label smoothing help?
  In *Advances in Neural Information Processing Systems*, volume 32, pp. 4696–4705, 2019.
  URL <https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html>.
* Naeini et al. (2015)

  Naeini, M. P., Cooper, G. F., and Hauskrecht, M.
  Obtaining well calibrated probabilities using Bayesian binning.
  In *Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence*, pp. 2901–2907, 2015.
  doi: 10.1609/aaai.v29i1.9602.
  URL <https://ojs.aaai.org/index.php/AAAI/article/view/9602>.
* Nagarajan & Kolter (2019)

  Nagarajan, V. and Kolter, J. Z.
  Deterministic PAC-bayesian generalization bounds for deep networks via generalizing noise-resilience.
  In *International Conference on Learning Representations*, 2019.
  URL <https://openreview.net/forum?id=Hygn2o0qKX>.
* Neyshabur et al. (2017)

  Neyshabur, B., Bhojanapalli, S., McAllester, D., and Srebro, N.
  Exploring generalization in deep learning.
  In *Advances in Neural Information Processing Systems*, volume 30, pp. 5947–5956, 2017.
  URL <https://proceedings.neurips.cc/paper_files/paper/2017/hash/10ce03a1ed01077e3e289f3e53c72813-Abstract.html>.
* Niculescu-Mizil & Caruana (2005)

  Niculescu-Mizil, A. and Caruana, R.
  Predicting good probabilities with supervised learning.
  In *Proceedings of the 22nd International Conference on Machine Learning*, pp. 625–632, 2005.
  doi: 10.1145/1102351.1102430.
* Ovadia et al. (2019)

  Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B., and Snoek, J.
  Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift.
  In *Advances in Neural Information Processing Systems*, volume 32, pp. 13991–14002, 2019.
  URL <https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html>.
* Pereyra et al. (2017)

  Pereyra, G., Tucker, G., Chorowski, J., Kaiser, Ł., and Hinton, G.
  Regularizing neural networks by penalizing confident output distributions.
  In *ICLR 2017 Workshop Track Proceedings*. OpenReview.net, 2017.
  URL <https://openreview.net/forum?id=HyhbYrGYe>.
* Platt (2000)

  Platt, J. C.
  Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.
  In Smola, A. J., Bartlett, P. L., Schölkopf, B., and Schuurmans, D. (eds.), *Advances in Large Margin Classifiers*, pp. 61–74. MIT Press, 2000.
* Qin et al. (2021)

  Qin, Y., Wang, X., Beutel, A., and Chi, E.
  Improving calibration through the relationship with adversarial robustness.
  In *Advances in Neural Information Processing Systems*, volume 34, pp. 14358–14369, 2021.
  URL <https://proceedings.neurips.cc/paper/2021/hash/78421a2e0e1168e5cd1b7a8d23773ce6-Abstract.html>.
* Rosenfeld & Risteski (2024)

  Rosenfeld, E. and Risteski, A.
  Outliers with opposing signals have an outsized effect on neural network optimization.
  In *International Conference on Learning Representations*, 2024.
  URL <https://openreview.net/forum?id=kIZ3S3tel6>.
  ICLR 2024 Poster.
* Song et al. (2025)

  Song, M., Ahn, K., and Yun, C.
  Does SGD really happen in tiny subspaces?
  In *International Conference on Learning Representations*, 2025.
  URL <https://openreview.net/forum?id=v6iLQBoIJw>.
  ICLR 2025 Poster.
* Soudry et al. (2018)

  Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., and Srebro, N.
  The implicit bias of gradient descent on separable data.
  *Journal of Machine Learning Research*, 19(70):1–57, 2018.
  URL <https://jmlr.org/papers/v19/18-188.html>.
* Stutz et al. (2020)

  Stutz, D., Hein, M., and Schiele, B.
  Confidence-calibrated adversarial training: Generalizing to unseen attacks.
  In Daumé III, H. and Singh, A. (eds.), *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp. 9155–9166. PMLR, 2020.
  URL <https://proceedings.mlr.press/v119/stutz20a.html>.
* Stutz et al. (2021)

  Stutz, D., Hein, M., and Schiele, B.
  Relating adversarially robust generalization to flat minima.
  In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 7807–7817, 2021.
  URL <https://openaccess.thecvf.com/content/ICCV2021/papers/Stutz_Relating_Adversarially_Robust_Generalization_to_Flat_Minima_ICCV_2021_paper.pdf>.
* Tan et al. (2026)

  Tan, C., Zhou, Y., Ye, H., Dai, G., Liu, J., Song, Z., Zhang, J., Zhao, Z., Hao, Y., and Xu, Y.
  Towards understanding the calibration benefits of sharpness-aware minimization.
  In *International Conference on Learning Representations*, 2026.
  URL <https://openreview.net/forum?id=c0ERcCz6lD>.
  ICLR 2026 Poster.
* Thulasidasan et al. (2019)

  Thulasidasan, S., Chennupati, G., Bilmes, J. A., Bhattacharya, T., and Michalak, S.
  On mixup training: Improved calibration and predictive uncertainty for deep neural networks.
  In *Advances in Neural Information Processing Systems*, volume 32, 2019.
  URL <https://proceedings.neurips.cc/paper_files/paper/2019/hash/36ad8b5f42db492827016448975cc22d-Abstract.html>.
* Tsuzuku et al. (2020)

  Tsuzuku, Y., Sato, I., and Sugiyama, M.
  Normalized flat minima: Exploring scale invariant definition of flat minima for neural networks using PAC-bayesian analysis.
  In Daumé III, H. and Singh, A. (eds.), *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp. 9636–9647. PMLR, 2020.
  URL <https://proceedings.mlr.press/v119/tsuzuku20a.html>.
* Wu et al. (2025)

  Wu, J., Bartlett, P., Telgarsky, M., and Yu, B.
  Benefits of early stopping in gradient descent for overparameterized logistic regression.
  In *Proceedings of the 42nd International Conference on Machine Learning*, volume 267 of *Proceedings of Machine Learning Research*, pp. 67081–67110. PMLR, 2025.
  URL <https://proceedings.mlr.press/v267/wu25b.html>.
* Xing et al. (2018)

  Xing, C., Arpit, D., Tsirigotis, C., and Bengio, Y.
  A walk with SGD.
  *arXiv preprint arXiv:1802.08770*, 2018.
  URL <https://arxiv.org/abs/1802.08770>.
* Zadrozny & Elkan (2002)

  Zadrozny, B. and Elkan, C.
  Transforming classifier scores into accurate multiclass probability estimates.
  In *Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 694–699, 2002.
  doi: 10.1145/775047.775151.
* Zhang et al. (2019)

  Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., and Jordan, M.
  Theoretically principled trade-off between robustness and accuracy.
  In *International conference on machine learning*, pp. 7472–7482. PMLR, 2019.
* Zhang et al. (2020)

  Zhang, J., Kailkhura, B., and Han, T. Y.-J.
  Mix-n-Match: Ensemble and compositional methods for uncertainty calibration in deep learning.
  In Daumé III, H. and Singh, A. (eds.), *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp. 11117–11128. PMLR, 2020.
  URL <https://proceedings.mlr.press/v119/zhang20k.html>.
* Zheng et al. (2021)

  Zheng, Y., Zhang, R., and Mao, Y.
  Regularizing neural networks via adversarial model perturbation.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 8152–8161, 2021.
  doi: 10.1109/CVPR46437.2021.00806.
  URL <https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Regularizing_Neural_Networks_via_Adversarial_Model_Perturbation_CVPR_2021_paper.html>.
* Zhou et al. (2025)

  Zhou, Z., Wang, M., Mao, Y., Li, B., and Yan, J.
  Sharpness-aware minimization efficiently selects flatter minima late in training.
  In *International Conference on Learning Representations*, 2025.
  URL <https://openreview.net/forum?id=aD2uwhLbnA>.
  ICLR 2025 Spotlight.

## Appendix A Further Related Work

### A.1 Mitigating Miscalibration: Extended Discussion

#### Post-hoc calibration.

Post-hoc methods learn a mapping from model scores to probabilities on a held-out set. Classical approaches include Platt scaling and isotonic/binning methods (Platt, [2000](#bib.bib41); Zadrozny & Elkan, [2002](#bib.bib53)), with temperature scaling the de facto recipe for deep networks (Guo et al., [2017](#bib.bib12)). More expressive calibrators—such as Dirichlet calibration and compositional strategies like Mix-n-Match—correct class- or confidence-dependent distortions while preserving accuracy (Kull et al., [2019](#bib.bib22); Zhang et al., [2020](#bib.bib55)). Because post-hoc methods do not influence training dynamics, they provide limited mechanistic insight and can degrade under distribution shift (Ovadia et al., [2019](#bib.bib39)).

#### Intrinsic methods.

Intrinsic methods incorporate calibration objectives directly into training. These include entropy-based regularization (Pereyra et al., [2017](#bib.bib40)), label smoothing (Müller et al., [2019](#bib.bib34)), augmentation schemes such as mixup that soften targets (Thulasidasan et al., [2019](#bib.bib49)), and focal loss—originally introduced for class imbalance—which yields better calibrated classifiers even before post-hoc scaling (Lin et al., [2017](#bib.bib27); Mukhoti et al., [2020](#bib.bib33)). Differentiable surrogates of calibration metrics further enable joint training for accuracy and calibration (Kumar et al., [2018](#bib.bib23); Bohdal et al., [2023](#bib.bib4)).

#### High-dimensional perspectives on miscalibration.

Recent theoretical work highlights that miscalibration can arise intrinsically from high-dimensional statistical effects, even in well-specified problems. Li & Sur ([2025](#bib.bib25)) analyze confidence estimates in high-dimensional classification and show that predictive probabilities can systematically deviate from true correctness likelihoods due to margin concentration and estimation noise, suggesting that miscalibration need not stem from optimization failures alone. Our perspective is complementary: rather than asymptotic statistical limits, we study the finite-sample training-time evolution of margins and curvature.

### A.2 Robust Margins, Sharpness, and Calibration

#### Robust margins and calibration.

Cross-entropy training pushes predictions toward extreme softmax outputs. On linearly separable data, gradient descent drives margins to infinity, converging to a max-margin classifier (Soudry et al., [2018](#bib.bib45)). While large margins aid classification, they also amplify overconfidence: Qin et al. ([2021](#bib.bib42)) showed that inputs with small robust margin are more likely to be miscalibrated, and proposed adaptive label smoothing on such points. Focal loss (Mukhoti et al., [2020](#bib.bib33)) and label smoothing (Müller et al., [2019](#bib.bib34)) can likewise curb overconfidence on hard examples. Foret et al. ([2021](#bib.bib11))’s SAM, which biases optimization toward flatter minima, has been observed to lower calibration error (Zheng et al., [2021](#bib.bib56); Möllenhoff & Khan, [2023](#bib.bib32)). These results share a common theme: controlling the growth or fragility of margins tends to improve calibration. Achieving both robustness and calibration is nevertheless non-trivial—standard adversarial training can degrade calibration without targeted interventions (Stutz et al., [2020](#bib.bib46)).

#### Flat minima and margins.

Loss-landscape geometry has long been linked to generalization, with flat minima hypothesized to be preferable to sharp ones (Hochreiter & Schmidhuber, [1997](#bib.bib14); Keskar et al., [2017](#bib.bib20)). Dinh et al. ([2017](#bib.bib10)) complicated this picture: scaling symmetries in deep networks allow arbitrarily sharp solutions with identical outputs, motivating scale-invariant sharpness measures (Tsuzuku et al., [2020](#bib.bib50); Liang et al., [2019](#bib.bib26)). Under such measures, flatter minima correlate with better generalization in CE-trained models (Jiang et al., [2020](#bib.bib18); Maddox et al., [2020](#bib.bib30)). A structural correlate is the classification margin: flat basins of the CE loss align with large training margins (Lengyel et al., [2021](#bib.bib24); Jiang et al., [2020](#bib.bib18)). Small-batch SGD, which implicitly enlarges margins (Hoffer et al., [2017](#bib.bib15)), also finds flatter solutions than large-batch training (Keskar et al., [2017](#bib.bib20)). Adversarially robust models—which have larger input margins—exhibit lower curvature in weight space (Stutz et al., [2021](#bib.bib47)); conversely, weight-space flatness regularizers such as entropy-SGD (Chaudhari et al., [2017](#bib.bib6)) improve adversarial robustness as a side effect (Stutz et al., [2021](#bib.bib47)).

#### Linear vs. non-linear caveats.

In linear models trained with cross-entropy, the notion of “flat vs. sharp” is less meaningful: on separable data the weight norm grows without bound as margins maximize, driving the Hessian to zero while the classifier becomes arbitrarily confident. Meaningful cross-setting flatness comparisons therefore require correcting for reparameterization invariances (Dinh et al., [2017](#bib.bib10); Neyshabur et al., [2017](#bib.bib37); Tsuzuku et al., [2020](#bib.bib50)). In deep non-linear networks with such corrections, large margins correspond to flatter minima (Lengyel et al., [2021](#bib.bib24)). This distinction explains why margin-based analyses (Bartlett et al., [2017](#bib.bib2); Nagarajan & Kolter, [2019](#bib.bib36)) are often preferred for theoretical guarantees in linear settings.

### A.3 Positioning Relative to Prior Work

The four literatures above—miscalibration under cross-entropy, sharpness/flatness and optimization stability, margin maximization via implicit bias, and robustness–calibration connections—together with recent work on calibration benefits of sharpness-aware optimizers (Tan et al., [2026](#bib.bib48)), provide the backdrop for our contribution.

Our work extends these literatures in three directions:

* •

  Trajectory-level analysis. Most prior work compares final solutions; we track calibration and curvature *pathwise* across training and show that they co-evolve, peaking together near the edge of stability and decaying together.
* •

  A shared margin-tail mediator. We prove that a single exponential margin moment and its robust variant simultaneously upper-bound ECE and Gauss–Newton sharpness, with a two-sided sandwich for ECE in the interpolating regime. The triangulation ECE ↔\leftrightarrow GN sharpness ↔\leftrightarrow robust margins is, to our knowledge, new.
* •

  Directional vs. flat-minima interventions. We distinguish optimizers that bias toward flat minima (SAM) from those that suppress steep descent directions along the trajectory (Muon, BulkSGD), and show empirically that the latter yield more reliable in-sample calibration gains. This motivates CalMO as a principled intervention on the mediator.

## Appendix B Proofs for Section [4](#S4 "4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")

### B.1 Definitions and basic reductions

#### Fixed binning.

Fix M∈ℕM\in\mathbb{N} and deterministic bin edges 0=a0<a1<⋯<aM=10=a\_{0}<a\_{1}<\cdots<a\_{M}=1.
Define bins Im:=(am−1,am]I\_{m}:=(a\_{m-1},a\_{m}] for m=1,…,Mm=1,\dots,M.

#### Population ECE with fixed bins.

Let (X,Y)∼π(X,Y)\sim\pi with Y∈{1,…,K}Y\in\{1,\dots,K\}.
For fixed θ\theta, define the predicted label

|  |  |  |
| --- | --- | --- |
|  | Y^:=arg⁡maxk⁡zθ​(X)k\widehat{Y}:=\arg\max\_{k}z\_{\theta}(X)\_{k} |  |

(using the deterministic tie-break rule from the main text),
and the confidence

|  |  |  |
| --- | --- | --- |
|  | P^:=maxk⁡pθ​(X)k=pθ​(X)Y^.\widehat{P}:=\max\_{k}p\_{\theta}(X)\_{k}=p\_{\theta}(X)\_{\widehat{Y}}. |  |

Let Bm:={P^∈Im}B\_{m}:=\{\widehat{P}\in I\_{m}\}. Define binwise accuracy and confidence by

|  |  |  |
| --- | --- | --- |
|  | acc​(Bm):={ℙ​(Y^=Y∣Bm),ℙ​(Bm)>0,0,ℙ​(Bm)=0,conf​(Bm):={𝔼​[P^∣Bm],ℙ​(Bm)>0,0,ℙ​(Bm)=0.\mathrm{acc}(B\_{m}):=\begin{cases}\mathbb{P}(\widehat{Y}=Y\mid B\_{m}),&\mathbb{P}(B\_{m})>0,\\ 0,&\mathbb{P}(B\_{m})=0,\end{cases}\qquad\mathrm{conf}(B\_{m}):=\begin{cases}\mathbb{E}[\widehat{P}\mid B\_{m}],&\mathbb{P}(B\_{m})>0,\\ 0,&\mathbb{P}(B\_{m})=0.\end{cases} |  |

The population binned calibration error is

|  |  |  |
| --- | --- | --- |
|  | ECEM​(θ;π):=∑m=1Mℙ​(Bm)​|acc​(Bm)−conf​(Bm)|.\mathrm{ECE}\_{M}(\theta;\pi):=\sum\_{m=1}^{M}\mathbb{P}(B\_{m})\,|\mathrm{acc}(B\_{m})-\mathrm{conf}(B\_{m})|. |  |

Equivalently, with Z:=𝟏​{Y^=Y}−P^Z:=\mathbf{1}\{\widehat{Y}=Y\}-\widehat{P},

|  |  |  |
| --- | --- | --- |
|  | ECEM(θ;π)=∑m=1Mℙ(Bm)|𝔼[Z∣Bm]|.\mathrm{ECE}\_{M}(\theta;\pi)=\sum\_{m=1}^{M}\mathbb{P}(B\_{m})\,\big|\mathbb{E}[Z\mid B\_{m}]\big|. |  |

#### Empirical ECE.

Given a dataset 𝒟={(xi,yi)}i=1n\mathcal{D}=\{(x\_{i},y\_{i})\}\_{i=1}^{n}, define

|  |  |  |
| --- | --- | --- |
|  | Y^i:=arg⁡maxk⁡zθ​(xi)k(same deterministic tie-break),P^i:=maxk⁡pθ​(xi)k,\widehat{Y}\_{i}:=\arg\max\_{k}z\_{\theta}(x\_{i})\_{k}\quad\text{(same deterministic tie-break)},\qquad\widehat{P}\_{i}:=\max\_{k}p\_{\theta}(x\_{i})\_{k}, |  |

and bins Bm:={i:P^i∈Im}B\_{m}:=\{i:\widehat{P}\_{i}\in I\_{m}\}.
Let Zi:=𝟏​{Y^i=yi}−P^iZ\_{i}:=\mathbf{1}\{\widehat{Y}\_{i}=y\_{i}\}-\widehat{P}\_{i}.
Then

|  |  |  |
| --- | --- | --- |
|  | ECEM​(θ;𝒟):=∑m=1M|Bm|n​|1|Bm|​∑i∈BmZi|,\mathrm{ECE}\_{M}(\theta;\mathcal{D}):=\sum\_{m=1}^{M}\frac{|B\_{m}|}{n}\,\left|\frac{1}{|B\_{m}|}\sum\_{i\in B\_{m}}Z\_{i}\right|, |  |

with the convention that the inner average is 0 when |Bm|=0|B\_{m}|=0.

### B.2 Core lemmas

###### Lemma B.1 (ECE is bounded by the mean absolute correctness–confidence gap).

(Population).
Let Z:=𝟏​{Y^=Y}−P^Z:=\mathbf{1}\{\widehat{Y}=Y\}-\widehat{P} and let
𝒢:=σ​(B1,…,BM)\mathcal{G}:=\sigma(B\_{1},\dots,B\_{M}) be the σ\sigma-algebra generated by the bin events
Bm:={P^∈Im}B\_{m}:=\{\widehat{P}\in I\_{m}\}.
Note that Z∈[−1,1]Z\in[-1,1], hence ZZ is integrable. Then

|  |  |  |
| --- | --- | --- |
|  | ECEM(θ;π)=𝔼[|𝔼[Z∣𝒢]|]≤𝔼[|Z|].\mathrm{ECE}\_{M}(\theta;\pi)=\mathbb{E}\Big[\,\big|\,\mathbb{E}[Z\mid\mathcal{G}]\,\big|\,\Big]\;\leq\;\mathbb{E}[|Z|]. |  |

(Empirical).
For any dataset 𝒟\mathcal{D},

|  |  |  |
| --- | --- | --- |
|  | ECEM​(θ;𝒟)≤1n​∑i=1n|𝟏​{Y^i=yi}−P^i|.\mathrm{ECE}\_{M}(\theta;\mathcal{D})\;\leq\;\frac{1}{n}\sum\_{i=1}^{n}\big|\mathbf{1}\{\widehat{Y}\_{i}=y\_{i}\}-\widehat{P}\_{i}\big|. |  |

###### Proof.

Population.
For each bin BmB\_{m} with ℙ​(Bm)>0\mathbb{P}(B\_{m})>0,

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[Z∣Bm]=ℙ​(Y^=Y∣Bm)−𝔼​[P^∣Bm]=acc​(Bm)−conf​(Bm),\mathbb{E}[Z\mid B\_{m}]=\mathbb{P}(\widehat{Y}=Y\mid B\_{m})-\mathbb{E}[\widehat{P}\mid B\_{m}]=\mathrm{acc}(B\_{m})-\mathrm{conf}(B\_{m}), |  |

and we set 𝔼​[Z∣Bm]:=0\mathbb{E}[Z\mid B\_{m}]:=0 when ℙ​(Bm)=0\mathbb{P}(B\_{m})=0.
Therefore,

|  |  |  |
| --- | --- | --- |
|  | ECEM(θ;π)=∑m=1Mℙ(Bm)|𝔼[Z∣Bm]|=𝔼[|𝔼[Z∣𝒢]|].\mathrm{ECE}\_{M}(\theta;\pi)=\sum\_{m=1}^{M}\mathbb{P}(B\_{m})\,\big|\mathbb{E}[Z\mid B\_{m}]\big|=\mathbb{E}\Big[\,\big|\,\mathbb{E}[Z\mid\mathcal{G}]\,\big|\,\Big]. |  |

By Jensen’s inequality for the convex function u↦|u|u\mapsto|u|,

|  |  |  |
| --- | --- | --- |
|  | 𝔼[|𝔼[Z∣𝒢]|]≤𝔼[𝔼[|Z|∣𝒢]]=𝔼[|Z|].\mathbb{E}\Big[\,\big|\,\mathbb{E}[Z\mid\mathcal{G}]\,\big|\,\Big]\leq\mathbb{E}\big[\mathbb{E}[|Z|\mid\mathcal{G}]\big]=\mathbb{E}[|Z|]. |  |

Empirical.
For each bin BmB\_{m} with |Bm|>0|B\_{m}|>0, let Zi:=𝟏​{Y^i=yi}−P^iZ\_{i}:=\mathbf{1}\{\widehat{Y}\_{i}=y\_{i}\}-\widehat{P}\_{i}.
Then

|  |  |  |
| --- | --- | --- |
|  | |acc​(Bm)−conf​(Bm)|=|1|Bm|​∑i∈BmZi|≤1|Bm|​∑i∈Bm|Zi|\big|\mathrm{acc}(B\_{m})-\mathrm{conf}(B\_{m})\big|=\left|\frac{1}{|B\_{m}|}\sum\_{i\in B\_{m}}Z\_{i}\right|\leq\frac{1}{|B\_{m}|}\sum\_{i\in B\_{m}}|Z\_{i}| |  |

by the triangle inequality. Multiplying by |Bm|/n|B\_{m}|/n and summing over mm yields the claim.
∎

###### Lemma B.2 (Correctness–confidence gap is controlled by the true-class probability).

For any (x,y)(x,y) and θ\theta, where y^​(x)\widehat{y}(x) and P^​(x)\widehat{P}(x) are defined as above,

|  |  |  |
| --- | --- | --- |
|  | |𝟏​{y^​(x)=y}−P^​(x)|≤ 1−pθ​(x)y.\big|\mathbf{1}\{\widehat{y}(x)=y\}-\widehat{P}(x)\big|\;\leq\;1-p\_{\theta}(x)\_{y}. |  |

###### Proof.

Let y^=y^​(x)\widehat{y}=\widehat{y}(x) and P^=P^​(x)=maxk⁡pθ​(x)k=pθ​(x)y^\widehat{P}=\widehat{P}(x)=\max\_{k}p\_{\theta}(x)\_{k}=p\_{\theta}(x)\_{\widehat{y}}.
If y^=y\widehat{y}=y, then |𝟏​{y^=y}−P^|=|1−py|=1−py|\mathbf{1}\{\widehat{y}=y\}-\widehat{P}|=|1-p\_{y}|=1-p\_{y}.
If y^≠y\widehat{y}\neq y, then |𝟏​{y^=y}−P^|=P^=py^≤∑j≠ypj=1−py|\mathbf{1}\{\widehat{y}=y\}-\widehat{P}|=\widehat{P}=p\_{\widehat{y}}\leq\sum\_{j\neq y}p\_{j}=1-p\_{y},
since py^p\_{\widehat{y}} is one of the nonnegative summands in ∑j≠ypj\sum\_{j\neq y}p\_{j}.
∎

###### Lemma B.3 (Softmax tail bound: 1−py1-p\_{y} is exponentially controlled by the true margin).

Let p=softmax​(z)∈ΔK−1p=\mathrm{softmax}(z)\in\Delta^{K-1} and fix a label yy.
Define the true margin m:=zy−maxj≠y⁡zjm:=z\_{y}-\max\_{j\neq y}z\_{j}.
Then

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1−py≤∑j≠yezj−zy≤(K−1)​e−m.1-p\_{y}\;\leq\;\sum\_{j\neq y}e^{z\_{j}-z\_{y}}\;\leq\;(K-1)e^{-m}. |  | (4) |

Moreover,

|  |  |  |  |
| --- | --- | --- | --- |
|  | e−m1+(K−1)​e−m≤ 1−py.\frac{e^{-m}}{1+(K-1)e^{-m}}\;\leq\;1-p\_{y}. |  | (5) |

In particular, if m≥0m\geq 0 then

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1K​e−m≤ 1−py≤(K−1)​e−m.\frac{1}{K}e^{-m}\;\leq\;1-p\_{y}\;\leq\;(K-1)e^{-m}. |  | (6) |

###### Proof.

Write

|  |  |  |
| --- | --- | --- |
|  | py=ezy∑k=1Kezk=11+∑j≠yezj−zy,1−py=∑j≠yezj−zy1+∑j≠yezj−zy.p\_{y}=\frac{e^{z\_{y}}}{\sum\_{k=1}^{K}e^{z\_{k}}}=\frac{1}{1+\sum\_{j\neq y}e^{z\_{j}-z\_{y}}},\qquad 1-p\_{y}=\frac{\sum\_{j\neq y}e^{z\_{j}-z\_{y}}}{1+\sum\_{j\neq y}e^{z\_{j}-z\_{y}}}. |  |

Let S:=∑j≠yezj−zy≥0S:=\sum\_{j\neq y}e^{z\_{j}-z\_{y}}\geq 0. Then 1−py=S/(1+S)≤S1-p\_{y}=S/(1+S)\leq S, proving the first inequality in ([4](#A2.E4 "Equation 4 ‣ Lemma B.3 (Softmax tail bound: 1-𝑝_𝑦 is exponentially controlled by the true margin). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")).
For each j≠yj\neq y, zj−zy≤maxk≠y⁡zk−zy=−mz\_{j}-z\_{y}\leq\max\_{k\neq y}z\_{k}-z\_{y}=-m, hence ezj−zy≤e−me^{z\_{j}-z\_{y}}\leq e^{-m} and S≤(K−1)​e−mS\leq(K-1)e^{-m},
proving the second inequality in ([4](#A2.E4 "Equation 4 ‣ Lemma B.3 (Softmax tail bound: 1-𝑝_𝑦 is exponentially controlled by the true margin). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")).
For ([5](#A2.E5 "Equation 5 ‣ Lemma B.3 (Softmax tail bound: 1-𝑝_𝑦 is exponentially controlled by the true margin). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")), pick j⋆∈arg⁡maxj≠y⁡zjj^{\star}\in\arg\max\_{j\neq y}z\_{j} so that zj⋆−zy=−mz\_{j^{\star}}-z\_{y}=-m and hence S≥e−mS\geq e^{-m}.
Therefore

|  |  |  |
| --- | --- | --- |
|  | 1−py=S1+S≥e−m1+S≥e−m1+(K−1)​e−m,1-p\_{y}=\frac{S}{1+S}\geq\frac{e^{-m}}{1+S}\geq\frac{e^{-m}}{1+(K-1)e^{-m}}, |  |

where the last step uses S≤(K−1)​e−mS\leq(K-1)e^{-m}.
If m≥0m\geq 0 then e−m≤1e^{-m}\leq 1 and thus 1+(K−1)​e−m≤1+(K−1)=K1+(K-1)e^{-m}\leq 1+(K-1)=K, giving ([6](#A2.E6 "Equation 6 ‣ Lemma B.3 (Softmax tail bound: 1-𝑝_𝑦 is exponentially controlled by the true margin). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")).
∎

###### Lemma B.4 (Cross-entropy logit Hessian top eigenvalue is controlled by 1−pmax1-p\_{\max}).

Let p∈ΔK−1p\in\Delta^{K-1} and define Hz​(p):=diag​(p)−p​p⊤H\_{z}(p):=\mathrm{diag}(p)-pp^{\top}.
Then

|  |  |  |
| --- | --- | --- |
|  | λmax​(Hz​(p))≤ 2​(1−pmax),pmax:=maxk⁡pk.\lambda\_{\max}\big(H\_{z}(p)\big)\;\leq\;2\bigl(1-p\_{\max}\bigr),\qquad p\_{\max}:=\max\_{k}p\_{k}. |  |

###### Proof.

We use Gershgorin’s circle theorem for symmetric matrices.
Write A:=Hz​(p)A:=H\_{z}(p), so that for each ii,

|  |  |  |
| --- | --- | --- |
|  | Ai​i=pi​(1−pi),Ai​j=−pi​pj(i≠j).A\_{ii}=p\_{i}(1-p\_{i}),\qquad A\_{ij}=-p\_{i}p\_{j}\quad(i\neq j). |  |

Let Ri:=∑j≠i|Ai​j|=∑j≠ipi​pj=pi​(1−pi)R\_{i}:=\sum\_{j\neq i}|A\_{ij}|=\sum\_{j\neq i}p\_{i}p\_{j}=p\_{i}(1-p\_{i}).
Gershgorin implies every eigenvalue λ\lambda of AA lies in at least one interval

|  |  |  |
| --- | --- | --- |
|  | λ∈[Ai​i−Ri,Ai​i+Ri]=[0, 2​pi​(1−pi)]for some ​i.\lambda\in[A\_{ii}-R\_{i},\,A\_{ii}+R\_{i}]=[0,\,2p\_{i}(1-p\_{i})]\quad\text{for some }i. |  |

Hence

|  |  |  |
| --- | --- | --- |
|  | λmax​(A)≤maxi⁡2​pi​(1−pi).\lambda\_{\max}(A)\leq\max\_{i}2p\_{i}(1-p\_{i}). |  |

Now fix k⋆∈arg⁡maxk⁡pkk^{\star}\in\arg\max\_{k}p\_{k} so that pk⋆=pmaxp\_{k^{\star}}=p\_{\max}.
If i=k⋆i=k^{\star}, then pi​(1−pi)=pmax​(1−pmax)≤1−pmaxp\_{i}(1-p\_{i})=p\_{\max}(1-p\_{\max})\leq 1-p\_{\max}.
If i≠k⋆i\neq k^{\star}, then pi≤1−pmaxp\_{i}\leq 1-p\_{\max} and pi​(1−pi)≤pi≤1−pmaxp\_{i}(1-p\_{i})\leq p\_{i}\leq 1-p\_{\max}.
Therefore maxi⁡pi​(1−pi)≤1−pmax\max\_{i}p\_{i}(1-p\_{i})\leq 1-p\_{\max} and consequently

|  |  |  |
| --- | --- | --- |
|  | λmax​(Hz​(p))≤2​(1−pmax),\lambda\_{\max}(H\_{z}(p))\leq 2(1-p\_{\max}), |  |

as claimed.
∎

###### Lemma B.5 (Robust margin comparisons (trivial upper bound; Lipschitz lower bound)).

For all (x,y)(x,y),

|  |  |  |  |
| --- | --- | --- | --- |
|  | mε,θ​(x,y)≤mθ​(x,y)⟹e−mθ​(x,y)≤e−mε,θ​(x,y).m\_{\varepsilon,\theta}(x,y)\leq m\_{\theta}(x,y)\qquad\Longrightarrow\qquad e^{-m\_{\theta}(x,y)}\leq e^{-m\_{\varepsilon,\theta}(x,y)}. |  | (7) |

If moreover there exists Lm​(x,y)∈[0,∞)L\_{m}(x,y)\in[0,\infty) such that

|  |  |  |
| --- | --- | --- |
|  | |mθ​(x+δ,y)−mθ​(x,y)|≤Lm​(x,y)​‖δ‖∀‖δ‖≤ε,|m\_{\theta}(x+\delta,y)-m\_{\theta}(x,y)|\leq L\_{m}(x,y)\,\|\delta\|\qquad\forall\,\|\delta\|\leq\varepsilon, |  |

then

|  |  |  |  |
| --- | --- | --- | --- |
|  | mε,θ​(x,y)≥mθ​(x,y)−ε​Lm​(x,y)⟹e−mθ​(x,y)≥e−ε​Lm​(x,y)​e−mε,θ​(x,y).m\_{\varepsilon,\theta}(x,y)\geq m\_{\theta}(x,y)-\varepsilon L\_{m}(x,y)\qquad\Longrightarrow\qquad e^{-m\_{\theta}(x,y)}\geq e^{-\varepsilon L\_{m}(x,y)}\,e^{-m\_{\varepsilon,\theta}(x,y)}. |  | (8) |

###### Proof.

Trivial robust-vs-clean comparison.
By definition of the infimum and because δ=0\delta=0 is feasible, we have

|  |  |  |
| --- | --- | --- |
|  | mε,θ​(x,y)=inf‖δ‖≤εmθ​(x+δ,y)≤mθ​(x,y).m\_{\varepsilon,\theta}(x,y)=\inf\_{\|\delta\|\leq\varepsilon}m\_{\theta}(x+\delta,y)\leq m\_{\theta}(x,y). |  |

Since the map t↦e−tt\mapsto e^{-t} is *decreasing*, this implies

|  |  |  |
| --- | --- | --- |
|  | e−mθ​(x,y)≤e−mε,θ​(x,y),e^{-m\_{\theta}(x,y)}\leq e^{-m\_{\varepsilon,\theta}(x,y)}, |  |

which is ([7](#A2.E7 "Equation 7 ‣ Lemma B.5 (Robust margin comparisons (trivial upper bound; Lipschitz lower bound)). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")).

Lipschitz lower bound.
Assume the stated local Lipschitz condition at (x,y)(x,y). Then for any ‖δ‖≤ε\|\delta\|\leq\varepsilon,

|  |  |  |
| --- | --- | --- |
|  | mθ​(x+δ,y)≥mθ​(x,y)−Lm​(x,y)​‖δ‖≥mθ​(x,y)−ε​Lm​(x,y).m\_{\theta}(x+\delta,y)\geq m\_{\theta}(x,y)-L\_{m}(x,y)\,\|\delta\|\geq m\_{\theta}(x,y)-\varepsilon L\_{m}(x,y). |  |

Taking the infimum over all ‖δ‖≤ε\|\delta\|\leq\varepsilon yields

|  |  |  |
| --- | --- | --- |
|  | mε,θ​(x,y)≥mθ​(x,y)−ε​Lm​(x,y),m\_{\varepsilon,\theta}(x,y)\geq m\_{\theta}(x,y)-\varepsilon L\_{m}(x,y), |  |

equivalently

|  |  |  |
| --- | --- | --- |
|  | mθ​(x,y)≤mε,θ​(x,y)+ε​Lm​(x,y).m\_{\theta}(x,y)\leq m\_{\varepsilon,\theta}(x,y)+\varepsilon L\_{m}(x,y). |  |

Multiply by −1-1 (which flips the inequality) to get

|  |  |  |
| --- | --- | --- |
|  | −mθ​(x,y)≥−mε,θ​(x,y)−ε​Lm​(x,y),-m\_{\theta}(x,y)\geq-m\_{\varepsilon,\theta}(x,y)-\varepsilon L\_{m}(x,y), |  |

and exponentiate to obtain

|  |  |  |
| --- | --- | --- |
|  | e−mθ​(x,y)≥e−mε,θ​(x,y)​e−ε​Lm​(x,y),e^{-m\_{\theta}(x,y)}\geq e^{-m\_{\varepsilon,\theta}(x,y)}\,e^{-\varepsilon L\_{m}(x,y)}, |  |

which is ([8](#A2.E8 "Equation 8 ‣ Lemma B.5 (Robust margin comparisons (trivial upper bound; Lipschitz lower bound)). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")).
∎

###### Remark B.6 (Label-free GN bound via predicted margin).

Because Hz​(pθ​(X))H\_{z}(p\_{\theta}(X)) depends only on XX, one can avoid the label YY in the GN bound.
Let y^​(x)∈arg⁡maxk⁡zθ​(x)k\widehat{y}(x)\in\arg\max\_{k}z\_{\theta}(x)\_{k} (with the deterministic tie-break rule) and define the *predicted margin*

|  |  |  |
| --- | --- | --- |
|  | m^θ​(x):=zθ​(x)y^​(x)−maxj≠y^​(x)⁡zθ​(x)j≥0.\widehat{m}\_{\theta}(x):=z\_{\theta}(x)\_{\widehat{y}(x)}-\max\_{j\neq\widehat{y}(x)}z\_{\theta}(x)\_{j}\;\geq 0. |  |

Applying Lemma [B.3](#A2.Thmtheorem3 "Lemma B.3 (Softmax tail bound: 1-𝑝_𝑦 is exponentially controlled by the true margin). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") with y=y^​(x)y=\widehat{y}(x) yields

|  |  |  |
| --- | --- | --- |
|  | 1−pmax​(x)=1−pθ​(x)y^​(x)≤(K−1)​e−m^θ​(x).1-p\_{\max}(x)=1-p\_{\theta}(x)\_{\widehat{y}(x)}\leq(K-1)e^{-\widehat{m}\_{\theta}(x)}. |  |

Combining with Lemma [B.4](#A2.Thmtheorem4 "Lemma B.4 (Cross-entropy logit Hessian top eigenvalue is controlled by 1-𝑝ₘₐₓ). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") gives

|  |  |  |
| --- | --- | --- |
|  | λmax​(Hz​(pθ​(x)))≤2​(1−pmax​(x))≤2​(K−1)​e−m^θ​(x).\lambda\_{\max}\!\big(H\_{z}(p\_{\theta}(x))\big)\leq 2\bigl(1-p\_{\max}(x)\bigr)\leq 2(K-1)e^{-\widehat{m}\_{\theta}(x)}. |  |

cConsequently, under ‖Jθ​(X)‖op≤CJ\|J\_{\theta}(X)\|\_{\mathrm{op}}\leq C\_{J} π\pi-a.s.,

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;π))≤2​CJ2​(K−1)​𝔼​[e−m^θ​(X)].\lambda\_{\max}\!\big(H\_{\mathrm{GN}}(\theta;\pi)\big)\leq 2C\_{J}^{2}(K-1)\,\mathbb{E}\big[e^{-\widehat{m}\_{\theta}(X)}\big]. |  |

This can be substantially tighter than bounds routing through YY when the model is confidently incorrect.

### B.3 Rigorous restatement of the main theorems

#### Notation alignment with Section [4](#S4 "4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").

Fix a robust radius ε>0\varepsilon>0.
To match the main-text notation, we use

|  |  |  |
| --- | --- | --- |
|  | Q​(θ;π):=𝔼(X,Y)∼π​[e−mε,θ​(X,Y)]andQ𝒟​(θ):=1n​∑i=1ne−mθ​(xi,yi).Q(\theta;\pi)\;:=\;\mathbb{E}\_{(X,Y)\sim\pi}\!\big[e^{-m\_{\varepsilon,\theta}(X,Y)}\big]\qquad\text{and}\qquad Q\_{\mathcal{D}}(\theta)\;:=\;\frac{1}{n}\sum\_{i=1}^{n}e^{-m\_{\theta}(x\_{i},y\_{i})}. |  |

When π\pi (or 𝒟\mathcal{D}) is clear from context, we may drop it from the notation.
For comparison with alternative functionals used in some intermediate lemmas, note that
Q​(θ;π)Q(\theta;\pi) coincides with the quantity previously denoted Ψε0​(θ;π)\Psi\_{\varepsilon}^{0}(\theta;\pi),
and Q𝒟​(θ)Q\_{\mathcal{D}}(\theta) coincides with the quantity previously denoted μ​(θ;𝒟)\mu(\theta;\mathcal{D}).
If a pointwise margin Lipschitz constant Lm​(⋅,⋅)L\_{m}(\cdot,\cdot) is available, we also define the (generally looser) population functional

|  |  |  |
| --- | --- | --- |
|  | Q+​(θ;π):=𝔼(X,Y)∼π​[eε​Lm​(X,Y)​e−mε,θ​(X,Y)],Q^{+}(\theta;\pi)\;:=\;\mathbb{E}\_{(X,Y)\sim\pi}\!\big[e^{\varepsilon L\_{m}(X,Y)}\,e^{-m\_{\varepsilon,\theta}(X,Y)}\big], |  |

and the finite-sample robust moments

|  |  |  |
| --- | --- | --- |
|  | Qε,𝒟0​(θ):=1n​∑i=1ne−mε,θ​(xi,yi),Qε,𝒟−​(θ):=1n​∑i=1ne−ε​Lm​(xi,yi)​e−mε,θ​(xi,yi).Q^{0}\_{\varepsilon,\mathcal{D}}(\theta)\;:=\;\frac{1}{n}\sum\_{i=1}^{n}e^{-m\_{\varepsilon,\theta}(x\_{i},y\_{i})},\qquad Q^{-}\_{\varepsilon,\mathcal{D}}(\theta)\;:=\;\frac{1}{n}\sum\_{i=1}^{n}e^{-\varepsilon L\_{m}(x\_{i},y\_{i})}\,e^{-m\_{\varepsilon,\theta}(x\_{i},y\_{i})}. |  |

The proofs appear in Subsections [B.4](#A2.SS4 "B.4 Proof of Theorem 4.1 ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") and [B.5](#A2.SS5 "B.5 Proof of Theorem 4.2 ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").

Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1 (Overlap regime: robust-margin upper bounds). ‣ 4.1 Regime I: overlap-dominated (non-separable) behavior ‣ 4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") (Overlap regime: simultaneous robust-margin upper bounds).
  
Let π\pi be any distribution on 𝒳×{1,…,K}\mathcal{X}\times\{1,\dots,K\} and let θ\theta be any parameter vector.

#### (i) Calibration upper bound.

For the population binned calibration error ECEM​(θ;π)\mathrm{ECE}\_{M}(\theta;\pi),

|  |  |  |
| --- | --- | --- |
|  | ECEM​(θ;π)≤(K−1)​𝔼​[e−mθ​(X,Y)]≤(K−1)​Q​(θ;π).\mathrm{ECE}\_{M}(\theta;\pi)\;\leq\;(K-1)\,\mathbb{E}\big[e^{-m\_{\theta}(X,Y)}\big]\;\leq\;(K-1)\,Q(\theta;\pi). |  |

If LmL\_{m} is defined, then also ECEM​(θ;π)≤(K−1)​Q+​(θ;π)\mathrm{ECE}\_{M}(\theta;\pi)\leq(K-1)\,Q^{+}(\theta;\pi), but this is never tighter than the
Q​(θ;π)Q(\theta;\pi) bound since Q+​(θ;π)≥Q​(θ;π)Q^{+}(\theta;\pi)\geq Q(\theta;\pi).

#### (ii) Gauss–Newton curvature (top eigenvalue) upper bound.

Assume additionally that the logit Jacobian is uniformly bounded in operator norm,

|  |  |  |
| --- | --- | --- |
|  | ‖Jθ​(X)‖op≤CJπ-a.s.\|J\_{\theta}(X)\|\_{\mathrm{op}}\leq C\_{J}\qquad\text{$\pi$-a.s.} |  |

Then the population Gauss–Newton matrix

|  |  |  |
| --- | --- | --- |
|  | HGN(θ;π):=𝔼(X,Y)∼π[Jθ(X)⊤Hz(pθ(X))Jθ(X)]H\_{\mathrm{GN}}(\theta;\pi):=\mathbb{E}\_{(X,Y)\sim\pi}\!\big[J\_{\theta}(X)^{\top}H\_{z}(p\_{\theta}(X))J\_{\theta}(X)\big] |  |

satisfies

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;π))≤ 2​CJ2​(K−1)​𝔼​[e−mθ​(X,Y)]≤ 2​CJ2​(K−1)​Q​(θ;π).\lambda\_{\max}\!\big(H\_{\mathrm{GN}}(\theta;\pi)\big)\;\leq\;2C\_{J}^{2}\,(K-1)\,\mathbb{E}\big[e^{-m\_{\theta}(X,Y)}\big]\;\leq\;2C\_{J}^{2}\,(K-1)\,Q(\theta;\pi). |  |

If LmL\_{m} is defined, then also
λmax​(HGN​(θ;π))≤2​CJ2​(K−1)​Q+​(θ;π)\lambda\_{\max}(H\_{\mathrm{GN}}(\theta;\pi))\leq 2C\_{J}^{2}\,(K-1)\,Q^{+}(\theta;\pi), again a looser bound than the one via Q​(θ;π)Q(\theta;\pi).

#### (iii) What the bound can (and cannot) certify.

If along a training trajectory {θt}\{\theta\_{t}\} the robust moment Q​(θt;π)Q(\theta\_{t};\pi) fails to converge to 0,
then the bounds in (i)–(ii) do not certify that ECEM​(θt;π)→0\mathrm{ECE}\_{M}(\theta\_{t};\pi)\to 0 or
λmax​(HGN​(θt;π))→0\lambda\_{\max}(H\_{\mathrm{GN}}(\theta\_{t};\pi))\to 0.

#### (iv) Remarks (label dependence and trivial clamping).

* •

  Label dependence vs. label-free curvature.
  λmax​(HGN​(θ;π))\lambda\_{\max}(H\_{\mathrm{GN}}(\theta;\pi)) depends only on the marginal law of XX (since Hz​(pθ​(X))H\_{z}(p\_{\theta}(X)) is label-free),
  whereas the bound above routes through YY via mθ​(X,Y)m\_{\theta}(X,Y) (equivalently 1−pθ​(X)Y1-p\_{\theta}(X)\_{Y}). This is valid but can be loose,
  especially when the model is confidently wrong.
  A label-free alternative (via the predicted margin) is given in Remark [B.6](#A2.Thmtheorem6 "Remark B.6 (Label-free GN bound via predicted margin). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").
* •

  Bounds may exceed 11. Since ECEM​(θ;π)∈[0,1]\mathrm{ECE}\_{M}(\theta;\pi)\in[0,1], any upper bound UU can be trivially tightened to min⁡{1,U}\min\{1,U\}.

Theorem [4.2](#S4.Thmtheorem2 "Theorem 4.2 (Interpolating regime: two-sided ECE–margin control and coupling to 𝜆ₘₐₓ). ‣ 4.2 Regime II: Interpolating (separable) behavior on the training set ‣ 4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") (Interpolating regime: two-sided ECE control and coupling to λmax\lambda\_{\max}).
  
Assume γ​(θ;𝒟)>0\gamma(\theta;\mathcal{D})>0, i.e. the training set 𝒟={(xi,yi)}i=1n\mathcal{D}=\{(x\_{i},y\_{i})\}\_{i=1}^{n} is correctly classified with strictly positive *true* margin.
(Strictness ensures Y^i=yi\widehat{Y}\_{i}=y\_{i} without tie-breaking subtleties.)

#### (i) Two-sided control of in-sample ECE by the exponential margin moment.

|  |  |  |
| --- | --- | --- |
|  | 1K​Q𝒟​(θ)≤ECEM​(θ;𝒟)≤(K−1)​Q𝒟​(θ)≤(K−1)​e−γ​(θ;𝒟).\frac{1}{K}\,Q\_{\mathcal{D}}(\theta)\;\leq\;\mathrm{ECE}\_{M}(\theta;\mathcal{D})\;\leq\;(K-1)\,Q\_{\mathcal{D}}(\theta)\;\leq\;(K-1)\,e^{-\gamma(\theta;\mathcal{D})}. |  |

(As always, one may clamp the upper bound by min⁡{1,⋅}\min\{1,\cdot\}.)

#### (ii) In-sample GN curvature bound in terms of the same moment.

Assume additionally that ‖Jθ​(xi)‖op≤CJ\|J\_{\theta}(x\_{i})\|\_{\mathrm{op}}\leq C\_{J} for all i=1,…,ni=1,\dots,n. Then

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;𝒟))≤ 2​CJ2​(K−1)​Q𝒟​(θ)≤ 2​CJ2​K​(K−1)​ECEM​(θ;𝒟).\lambda\_{\max}\!\big(H\_{\mathrm{GN}}(\theta;\mathcal{D})\big)\;\leq\;2C\_{J}^{2}\,(K-1)\,Q\_{\mathcal{D}}(\theta)\;\leq\;2C\_{J}^{2}\,K(K-1)\,\mathrm{ECE}\_{M}(\theta;\mathcal{D}). |  |

#### (iii) Consequence: in the interpolating regime, curvature and ECE are forced to co-vary.

Under the same assumptions,

|  |  |  |
| --- | --- | --- |
|  | ECEM​(θ;𝒟)≥λmax​(HGN​(θ;𝒟))2​CJ2​K​(K−1).\mathrm{ECE}\_{M}(\theta;\mathcal{D})\;\geq\;\frac{\lambda\_{\max}(H\_{\mathrm{GN}}(\theta;\mathcal{D}))}{2C\_{J}^{2}\,K(K-1)}. |  |

Thus, once the training set is correctly classified and Jacobians remain bounded, large GN curvature cannot occur without large in-sample ECE.

#### (iv) Robust-margin variant (optional; requires local Lipschitzness at (xi,yi)(x\_{i},y\_{i})).

Assume moreover that for each ii there exists Lm​(xi,yi)∈[0,∞)L\_{m}(x\_{i},y\_{i})\in[0,\infty) such that

|  |  |  |
| --- | --- | --- |
|  | |mθ​(xi+δ,yi)−mθ​(xi,yi)|≤Lm​(xi,yi)​‖δ‖∀‖δ‖≤ε.|m\_{\theta}(x\_{i}+\delta,y\_{i})-m\_{\theta}(x\_{i},y\_{i})|\leq L\_{m}(x\_{i},y\_{i})\,\|\delta\|\qquad\forall\ \|\delta\|\leq\varepsilon. |  |

Then, with the robust moments Qε,𝒟0​(θ)Q^{0}\_{\varepsilon,\mathcal{D}}(\theta) and Qε,𝒟−​(θ)Q^{-}\_{\varepsilon,\mathcal{D}}(\theta) defined above,

|  |  |  |
| --- | --- | --- |
|  | 1K​Qε,𝒟−​(θ)≤ECEM​(θ;𝒟)≤(K−1)​Qε,𝒟0​(θ),\frac{1}{K}\,Q^{-}\_{\varepsilon,\mathcal{D}}(\theta)\;\leq\;\mathrm{ECE}\_{M}(\theta;\mathcal{D})\;\leq\;(K-1)\,Q^{0}\_{\varepsilon,\mathcal{D}}(\theta), |  |

and, under maxi⁡‖Jθ​(xi)‖op≤CJ\max\_{i}\|J\_{\theta}(x\_{i})\|\_{\mathrm{op}}\leq C\_{J},

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;𝒟))≤ 2​CJ2​(K−1)​Qε,𝒟0​(θ).\lambda\_{\max}\!\big(H\_{\mathrm{GN}}(\theta;\mathcal{D})\big)\;\leq\;2C\_{J}^{2}\,(K-1)\,Q^{0}\_{\varepsilon,\mathcal{D}}(\theta). |  |

#### (v) Remark (binning irrelevance under perfect accuracy).

Under γ​(θ;𝒟)>0\gamma(\theta;\mathcal{D})>0, every nonempty bin has empirical accuracy 11,
so ECEM​(θ;𝒟)\mathrm{ECE}\_{M}(\theta;\mathcal{D}) reduces to the *mean misconfidence* and becomes independent of the choice of bins.

### B.4 Proof of Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1 (Overlap regime: robust-margin upper bounds). ‣ 4.1 Regime I: overlap-dominated (non-separable) behavior ‣ 4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")

###### Proof of Theorem 4.1.

(i) Calibration bound.
By Lemma [B.1](#A2.Thmtheorem1 "Lemma B.1 (ECE is bounded by the mean absolute correctness–confidence gap). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") (population version),

|  |  |  |
| --- | --- | --- |
|  | ECEM​(θ;π)≤𝔼(X,Y)∼π​[|𝟏​{Y^=Y}−P^|].\mathrm{ECE}\_{M}(\theta;\pi)\leq\mathbb{E}\_{(X,Y)\sim\pi}\Big[\big|\mathbf{1}\{\widehat{Y}=Y\}-\widehat{P}\big|\Big]. |  |

By Lemma [B.2](#A2.Thmtheorem2 "Lemma B.2 (Correctness–confidence gap is controlled by the true-class probability). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"),

|  |  |  |
| --- | --- | --- |
|  | |𝟏​{Y^=Y}−P^|≤1−pθ​(X)Y,\big|\mathbf{1}\{\widehat{Y}=Y\}-\widehat{P}\big|\leq 1-p\_{\theta}(X)\_{Y}, |  |

hence

|  |  |  |
| --- | --- | --- |
|  | ECEM​(θ;π)≤𝔼(X,Y)∼π​[1−pθ​(X)Y].\mathrm{ECE}\_{M}(\theta;\pi)\leq\mathbb{E}\_{(X,Y)\sim\pi}\big[1-p\_{\theta}(X)\_{Y}\big]. |  |

Applying Lemma [B.3](#A2.Thmtheorem3 "Lemma B.3 (Softmax tail bound: 1-𝑝_𝑦 is exponentially controlled by the true margin). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") with z=zθ​(X)z=z\_{\theta}(X) and y=Yy=Y yields

|  |  |  |
| --- | --- | --- |
|  | 1−pθ​(X)Y≤(K−1)​e−mθ​(X,Y)π​-a.s.,1-p\_{\theta}(X)\_{Y}\leq(K-1)e^{-m\_{\theta}(X,Y)}\qquad\pi\text{-a.s.}, |  |

so

|  |  |  |
| --- | --- | --- |
|  | ECEM​(θ;π)≤(K−1)​𝔼(X,Y)∼π​[e−mθ​(X,Y)].\mathrm{ECE}\_{M}(\theta;\pi)\leq(K-1)\,\mathbb{E}\_{(X,Y)\sim\pi}\big[e^{-m\_{\theta}(X,Y)}\big]. |  |

Finally, by Lemma [B.5](#A2.Thmtheorem5 "Lemma B.5 (Robust margin comparisons (trivial upper bound; Lipschitz lower bound)). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") (the trivial robust-vs-clean comparison),

|  |  |  |
| --- | --- | --- |
|  | e−mθ​(X,Y)≤e−mε,θ​(X,Y)π​-a.s.,e^{-m\_{\theta}(X,Y)}\leq e^{-m\_{\varepsilon,\theta}(X,Y)}\qquad\pi\text{-a.s.}, |  |

and therefore

|  |  |  |
| --- | --- | --- |
|  | ECEM​(θ;π)≤(K−1)​𝔼(X,Y)∼π​[e−mε,θ​(X,Y)]=(K−1)​Q​(θ;π).\mathrm{ECE}\_{M}(\theta;\pi)\leq(K-1)\,\mathbb{E}\_{(X,Y)\sim\pi}\big[e^{-m\_{\varepsilon,\theta}(X,Y)}\big]=(K-1)\,Q(\theta;\pi). |  |

If the local Lipschitz condition in Lemma [B.5](#A2.Thmtheorem5 "Lemma B.5 (Robust margin comparisons (trivial upper bound; Lipschitz lower bound)). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") holds so that Q+​(θ;π)Q^{+}(\theta;\pi) is defined, then
Q​(θ;π)≤Q+​(θ;π)Q(\theta;\pi)\leq Q^{+}(\theta;\pi) since eε​Lm​(X,Y)≥1e^{\varepsilon L\_{m}(X,Y)}\geq 1.

(ii) GN curvature bound.
Define the random PSD matrix

|  |  |  |
| --- | --- | --- |
|  | A​(X):=Jθ​(X)⊤​Hz​(pθ​(X))​Jθ​(X)⪰0.A(X):=J\_{\theta}(X)^{\top}H\_{z}(p\_{\theta}(X))J\_{\theta}(X)\succeq 0. |  |

Then HGN​(θ;π)=𝔼(X,Y)∼π​[A​(X)]H\_{\mathrm{GN}}(\theta;\pi)=\mathbb{E}\_{(X,Y)\sim\pi}[A(X)].
Since λmax\lambda\_{\max} is convex on the PSD cone (equivalently, by the variational characterization),

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;π))=λmax​(𝔼​[A​(X)])≤𝔼​[λmax​(A​(X))].\lambda\_{\max}\big(H\_{\mathrm{GN}}(\theta;\pi)\big)=\lambda\_{\max}\!\big(\mathbb{E}[A(X)]\big)\leq\mathbb{E}\big[\lambda\_{\max}(A(X))\big]. |  |

For each realization XX,

|  |  |  |
| --- | --- | --- |
|  | λmax​(A​(X))≤‖Jθ​(X)‖op2​λmax​(Hz​(pθ​(X))).\lambda\_{\max}(A(X))\leq\|J\_{\theta}(X)\|\_{\mathrm{op}}^{2}\,\lambda\_{\max}\big(H\_{z}(p\_{\theta}(X))\big). |  |

Under ‖Jθ​(X)‖op≤CJ\|J\_{\theta}(X)\|\_{\mathrm{op}}\leq C\_{J} π\pi-a.s.,

|  |  |  |
| --- | --- | --- |
|  | λmax​(A​(X))≤CJ2​λmax​(Hz​(pθ​(X))).\lambda\_{\max}(A(X))\leq C\_{J}^{2}\,\lambda\_{\max}\big(H\_{z}(p\_{\theta}(X))\big). |  |

By Lemma [B.4](#A2.Thmtheorem4 "Lemma B.4 (Cross-entropy logit Hessian top eigenvalue is controlled by 1-𝑝ₘₐₓ). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"),

|  |  |  |
| --- | --- | --- |
|  | λmax​(Hz​(pθ​(X)))≤2​(1−pmax​(X)),pmax​(X):=maxk⁡pθ​(X)k.\lambda\_{\max}(H\_{z}(p\_{\theta}(X)))\leq 2(1-p\_{\max}(X)),\qquad p\_{\max}(X):=\max\_{k}p\_{\theta}(X)\_{k}. |  |

Since pmax​(X)≥pθ​(X)Yp\_{\max}(X)\geq p\_{\theta}(X)\_{Y}, we have 1−pmax​(X)≤1−pθ​(X)Y1-p\_{\max}(X)\leq 1-p\_{\theta}(X)\_{Y}, hence

|  |  |  |
| --- | --- | --- |
|  | λmax​(Hz​(pθ​(X)))≤2​(1−pθ​(X)Y)≤2​(K−1)​e−mθ​(X,Y)π​-a.s.\lambda\_{\max}(H\_{z}(p\_{\theta}(X)))\leq 2\bigl(1-p\_{\theta}(X)\_{Y}\bigr)\leq 2(K-1)e^{-m\_{\theta}(X,Y)}\qquad\pi\text{-a.s.} |  |

Therefore,

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;π))≤2​CJ2​(K−1)​𝔼(X,Y)∼π​[e−mθ​(X,Y)]≤2​CJ2​(K−1)​𝔼(X,Y)∼π​[e−mε,θ​(X,Y)]=2​CJ2​(K−1)​Q​(θ;π),\lambda\_{\max}\big(H\_{\mathrm{GN}}(\theta;\pi)\big)\leq 2C\_{J}^{2}(K-1)\,\mathbb{E}\_{(X,Y)\sim\pi}\big[e^{-m\_{\theta}(X,Y)}\big]\leq 2C\_{J}^{2}(K-1)\,\mathbb{E}\_{(X,Y)\sim\pi}\big[e^{-m\_{\varepsilon,\theta}(X,Y)}\big]=2C\_{J}^{2}(K-1)\,Q(\theta;\pi), |  |

where the last inequality uses Lemma [B.5](#A2.Thmtheorem5 "Lemma B.5 (Robust margin comparisons (trivial upper bound; Lipschitz lower bound)). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").
If the local Lipschitz condition in Lemma [B.5](#A2.Thmtheorem5 "Lemma B.5 (Robust margin comparisons (trivial upper bound; Lipschitz lower bound)). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") holds, then also
Q​(θ;π)≤Q+​(θ;π)Q(\theta;\pi)\leq Q^{+}(\theta;\pi).

(iii) Certification statement.
Immediate from (i)–(ii): if Q​(θt;π)↛0Q(\theta\_{t};\pi)\not\to 0 (or likewise Q+​(θt;π)↛0Q^{+}(\theta\_{t};\pi)\not\to 0),
then the corresponding right-hand sides do not converge to 0 and therefore cannot certify
ECEM​(θt;π)→0\mathrm{ECE}\_{M}(\theta\_{t};\pi)\to 0 nor λmax​(HGN​(θt;π))→0\lambda\_{\max}(H\_{\mathrm{GN}}(\theta\_{t};\pi))\to 0.
∎

### B.5 Proof of Theorem [4.2](#S4.Thmtheorem2 "Theorem 4.2 (Interpolating regime: two-sided ECE–margin control and coupling to 𝜆ₘₐₓ). ‣ 4.2 Regime II: Interpolating (separable) behavior on the training set ‣ 4 Curvature and Calibration in the Separable and Non-separable Regimes ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")

###### Proof of Theorem 4.2.

Assume γ​(θ;𝒟)>0\gamma(\theta;\mathcal{D})>0, i.e. mθ​(xi,yi)>0m\_{\theta}(x\_{i},y\_{i})>0 for all ii.
Hence Y^i=yi\widehat{Y}\_{i}=y\_{i} for all ii (no tie-breaking occurs).

(i) Two-sided ECE–moment bounds.
Because Y^i=yi\widehat{Y}\_{i}=y\_{i}, every nonempty bin BmB\_{m} has empirical accuracy acc​(Bm)=1\mathrm{acc}(B\_{m})=1.
Therefore, for each nonempty bin,

|  |  |  |
| --- | --- | --- |
|  | |1−conf​(Bm)|=1−conf​(Bm)since ​conf​(Bm)∈[0,1].\bigl|1-\mathrm{conf}(B\_{m})\bigr|=1-\mathrm{conf}(B\_{m})\qquad\text{since }\mathrm{conf}(B\_{m})\in[0,1]. |  |

Hence

|  |  |  |
| --- | --- | --- |
|  | ECEM​(θ;𝒟)=∑m=1M|Bm|n​(1−conf​(Bm))=1−1n​∑i=1nP^i.\mathrm{ECE}\_{M}(\theta;\mathcal{D})=\sum\_{m=1}^{M}\frac{|B\_{m}|}{n}\,\Bigl(1-\mathrm{conf}(B\_{m})\Bigr)=1-\frac{1}{n}\sum\_{i=1}^{n}\widehat{P}\_{i}. |  |

Since P^i=maxk⁡pθ​(xi)k=pθ​(xi)Y^i\widehat{P}\_{i}=\max\_{k}p\_{\theta}(x\_{i})\_{k}=p\_{\theta}(x\_{i})\_{\widehat{Y}\_{i}} and Y^i=yi\widehat{Y}\_{i}=y\_{i}, we have
P^i=pθ​(xi)yi\widehat{P}\_{i}=p\_{\theta}(x\_{i})\_{y\_{i}}, and thus

|  |  |  |  |
| --- | --- | --- | --- |
|  | ECEM​(θ;𝒟)=1n​∑i=1n(1−pθ​(xi)yi).\mathrm{ECE}\_{M}(\theta;\mathcal{D})=\frac{1}{n}\sum\_{i=1}^{n}\bigl(1-p\_{\theta}(x\_{i})\_{y\_{i}}\bigr). |  | (9) |

Recalling Q𝒟​(θ):=1n​∑i=1ne−mθ​(xi,yi)Q\_{\mathcal{D}}(\theta):=\frac{1}{n}\sum\_{i=1}^{n}e^{-m\_{\theta}(x\_{i},y\_{i})}, Lemma [B.3](#A2.Thmtheorem3 "Lemma B.3 (Softmax tail bound: 1-𝑝_𝑦 is exponentially controlled by the true margin). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")
implies (since mθ​(xi,yi)≥0m\_{\theta}(x\_{i},y\_{i})\geq 0 for all ii) that

|  |  |  |
| --- | --- | --- |
|  | 1K​e−mθ​(xi,yi)≤1−pθ​(xi)yi≤(K−1)​e−mθ​(xi,yi).\frac{1}{K}\,e^{-m\_{\theta}(x\_{i},y\_{i})}\leq 1-p\_{\theta}(x\_{i})\_{y\_{i}}\leq(K-1)\,e^{-m\_{\theta}(x\_{i},y\_{i})}. |  |

Averaging over ii and using ([9](#A2.E9 "Equation 9 ‣ Proof of Theorem 4.2. ‣ B.5 Proof of Theorem 4.2 ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature")) yields

|  |  |  |
| --- | --- | --- |
|  | 1K​Q𝒟​(θ)≤ECEM​(θ;𝒟)≤(K−1)​Q𝒟​(θ).\frac{1}{K}\,Q\_{\mathcal{D}}(\theta)\leq\mathrm{ECE}\_{M}(\theta;\mathcal{D})\leq(K-1)\,Q\_{\mathcal{D}}(\theta). |  |

Finally, mθ​(xi,yi)≥γ​(θ;𝒟)m\_{\theta}(x\_{i},y\_{i})\geq\gamma(\theta;\mathcal{D}) implies Q𝒟​(θ)≤e−γ​(θ;𝒟)Q\_{\mathcal{D}}(\theta)\leq e^{-\gamma(\theta;\mathcal{D})}.

(ii) GN curvature bound.
For each ii, define Hz,i:=Hz​(pθ​(xi))H\_{z,i}:=H\_{z}(p\_{\theta}(x\_{i})) and Ji:=Jθ​(xi)J\_{i}:=J\_{\theta}(x\_{i}).
Then

|  |  |  |
| --- | --- | --- |
|  | HGN​(θ;𝒟)=1n​∑i=1nJi⊤​Hz,i​Ji⪰0.H\_{\mathrm{GN}}(\theta;\mathcal{D})=\frac{1}{n}\sum\_{i=1}^{n}J\_{i}^{\top}H\_{z,i}J\_{i}\succeq 0. |  |

Since λmax\lambda\_{\max} is convex on the PSD cone,

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;𝒟))≤1n​∑i=1nλmax​(Ji⊤​Hz,i​Ji)≤1n​∑i=1n‖Ji‖op2​λmax​(Hz,i).\lambda\_{\max}\big(H\_{\mathrm{GN}}(\theta;\mathcal{D})\big)\leq\frac{1}{n}\sum\_{i=1}^{n}\lambda\_{\max}(J\_{i}^{\top}H\_{z,i}J\_{i})\leq\frac{1}{n}\sum\_{i=1}^{n}\|J\_{i}\|\_{\mathrm{op}}^{2}\,\lambda\_{\max}(H\_{z,i}). |  |

Under ‖Ji‖op≤CJ\|J\_{i}\|\_{\mathrm{op}}\leq C\_{J} for all ii,

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;𝒟))≤CJ2⋅1n​∑i=1nλmax​(Hz,i).\lambda\_{\max}\big(H\_{\mathrm{GN}}(\theta;\mathcal{D})\big)\leq C\_{J}^{2}\cdot\frac{1}{n}\sum\_{i=1}^{n}\lambda\_{\max}(H\_{z,i}). |  |

By Lemma [B.4](#A2.Thmtheorem4 "Lemma B.4 (Cross-entropy logit Hessian top eigenvalue is controlled by 1-𝑝ₘₐₓ). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"), λmax​(Hz,i)≤2​(1−pmax,i)\lambda\_{\max}(H\_{z,i})\leq 2(1-p\_{\max,i}).
Since Y^i=yi\widehat{Y}\_{i}=y\_{i}, we have pmax,i=pθ​(xi)yip\_{\max,i}=p\_{\theta}(x\_{i})\_{y\_{i}}, hence by Lemma [B.3](#A2.Thmtheorem3 "Lemma B.3 (Softmax tail bound: 1-𝑝_𝑦 is exponentially controlled by the true margin). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"),

|  |  |  |
| --- | --- | --- |
|  | λmax​(Hz,i)≤2​(1−pθ​(xi)yi)≤2​(K−1)​e−mθ​(xi,yi).\lambda\_{\max}(H\_{z,i})\leq 2\bigl(1-p\_{\theta}(x\_{i})\_{y\_{i}}\bigr)\leq 2(K-1)e^{-m\_{\theta}(x\_{i},y\_{i})}. |  |

Therefore

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;𝒟))≤2​CJ2​(K−1)⋅1n​∑i=1ne−mθ​(xi,yi)=2​CJ2​(K−1)​Q𝒟​(θ).\lambda\_{\max}\big(H\_{\mathrm{GN}}(\theta;\mathcal{D})\big)\leq 2C\_{J}^{2}(K-1)\cdot\frac{1}{n}\sum\_{i=1}^{n}e^{-m\_{\theta}(x\_{i},y\_{i})}=2C\_{J}^{2}(K-1)\,Q\_{\mathcal{D}}(\theta). |  |

(iii) Coupling to ECEM\mathrm{ECE}\_{M} (rearranged lower bound).
Combining the bound in (ii) with ECEM​(θ;𝒟)≥1K​Q𝒟​(θ)\mathrm{ECE}\_{M}(\theta;\mathcal{D})\geq\frac{1}{K}Q\_{\mathcal{D}}(\theta) from (i) yields

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;𝒟))≤2​CJ2​K​(K−1)​ECEM​(θ;𝒟),equivalentlyECEM​(θ;𝒟)≥λmax​(HGN​(θ;𝒟))2​CJ2​K​(K−1).\lambda\_{\max}\big(H\_{\mathrm{GN}}(\theta;\mathcal{D})\big)\leq 2C\_{J}^{2}K(K-1)\,\mathrm{ECE}\_{M}(\theta;\mathcal{D}),\qquad\text{equivalently}\qquad\mathrm{ECE}\_{M}(\theta;\mathcal{D})\geq\frac{\lambda\_{\max}(H\_{\mathrm{GN}}(\theta;\mathcal{D}))}{2C\_{J}^{2}K(K-1)}. |  |

(iv) Robust-margin variant.
Assume that for each (xi,yi)(x\_{i},y\_{i}) there exists Lm​(xi,yi)∈[0,∞)L\_{m}(x\_{i},y\_{i})\in[0,\infty) such that

|  |  |  |
| --- | --- | --- |
|  | |mθ​(xi+δ,yi)−mθ​(xi,yi)|≤Lm​(xi,yi)​‖δ‖∀‖δ‖≤ε.|m\_{\theta}(x\_{i}+\delta,y\_{i})-m\_{\theta}(x\_{i},y\_{i})|\leq L\_{m}(x\_{i},y\_{i})\|\delta\|\qquad\forall\|\delta\|\leq\varepsilon. |  |

Define the robust moments (as in Appendix E.3)

|  |  |  |
| --- | --- | --- |
|  | Qε,𝒟0​(θ):=1n​∑i=1ne−mε,θ​(xi,yi),Qε,𝒟−​(θ):=1n​∑i=1ne−ε​Lm​(xi,yi)​e−mε,θ​(xi,yi).Q^{0}\_{\varepsilon,\mathcal{D}}(\theta):=\frac{1}{n}\sum\_{i=1}^{n}e^{-m\_{\varepsilon,\theta}(x\_{i},y\_{i})},\qquad Q^{-}\_{\varepsilon,\mathcal{D}}(\theta):=\frac{1}{n}\sum\_{i=1}^{n}e^{-\varepsilon L\_{m}(x\_{i},y\_{i})}e^{-m\_{\varepsilon,\theta}(x\_{i},y\_{i})}. |  |

By Lemma [B.5](#A2.Thmtheorem5 "Lemma B.5 (Robust margin comparisons (trivial upper bound; Lipschitz lower bound)). ‣ B.2 Core lemmas ‣ Appendix B Proofs for Section 4 ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"),

|  |  |  |
| --- | --- | --- |
|  | e−mθ​(xi,yi)≥e−ε​Lm​(xi,yi)​e−mε,θ​(xi,yi)ande−mθ​(xi,yi)≤e−mε,θ​(xi,yi).e^{-m\_{\theta}(x\_{i},y\_{i})}\geq e^{-\varepsilon L\_{m}(x\_{i},y\_{i})}e^{-m\_{\varepsilon,\theta}(x\_{i},y\_{i})}\qquad\text{and}\qquad e^{-m\_{\theta}(x\_{i},y\_{i})}\leq e^{-m\_{\varepsilon,\theta}(x\_{i},y\_{i})}. |  |

Insert these bounds into 1K​Q𝒟​(θ)≤ECEM​(θ;𝒟)≤(K−1)​Q𝒟​(θ)\frac{1}{K}Q\_{\mathcal{D}}(\theta)\leq\mathrm{ECE}\_{M}(\theta;\mathcal{D})\leq(K-1)Q\_{\mathcal{D}}(\theta) to obtain

|  |  |  |
| --- | --- | --- |
|  | 1K​Qε,𝒟−​(θ)≤ECEM​(θ;𝒟)≤(K−1)​Qε,𝒟0​(θ).\frac{1}{K}Q^{-}\_{\varepsilon,\mathcal{D}}(\theta)\leq\mathrm{ECE}\_{M}(\theta;\mathcal{D})\leq(K-1)Q^{0}\_{\varepsilon,\mathcal{D}}(\theta). |  |

For λmax​(HGN​(θ;𝒟))\lambda\_{\max}(H\_{\mathrm{GN}}(\theta;\mathcal{D})), repeat the argument in (ii) and apply
e−mθ​(xi,yi)≤e−mε,θ​(xi,yi)e^{-m\_{\theta}(x\_{i},y\_{i})}\leq e^{-m\_{\varepsilon,\theta}(x\_{i},y\_{i})} in the final step to get

|  |  |  |
| --- | --- | --- |
|  | λmax​(HGN​(θ;𝒟))≤2​CJ2​(K−1)​Qε,𝒟0​(θ).\lambda\_{\max}\big(H\_{\mathrm{GN}}(\theta;\mathcal{D})\big)\leq 2C\_{J}^{2}(K-1)\,Q^{0}\_{\varepsilon,\mathcal{D}}(\theta). |  |

∎

## Appendix C Additional Sharpness–Calibration Experiments

### C.1 Sharpness–Calibration Correlation Analysis

We present detailed training dynamics for each optimizer on CIFAR-10 and CIFAR-100, showing the co-evolution of loss, accuracy, ECE, margin, and sharpness throughout training. Since computing the full Hessian eigenvalue is expensive, these experiments use an MLP with a 5K/5K train/validation split. For each dataset, a scatter summary visualizes the temporal coupling across optimizers in a single view; per-optimizer figures report training (left) and validation (right) metrics across learning rates.

![Refer to caption](/html/2604.20614/assets/x3.png)


Figure A1: ECE vs. GN sharpness trajectories (CIFAR-10). Each curve traces the joint evolution of ECE and GN sharpness (λmax\lambda\_{\max}) across training steps for one optimizer and learning rate, with a filled circle marking the first training step and a cross (×\times) the last; color encodes learning rate. Trajectories lie near the diagonal, visualizing the temporal coupling between the two quantities.



![Refer to caption](/html/2604.20614/assets/x4.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x5.png)


(b) Validation metrics

Figure A2: Gradient Descent (GD). Training dynamics (loss, accuracy, ECE, margin, sharpness) across four learning rates on CIFAR-10; training (left) and validation (right).



![Refer to caption](/html/2604.20614/assets/x6.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x7.png)


(b) Validation metrics

Figure A3: Stochastic Gradient Descent (SGD). Training dynamics (loss, accuracy, ECE, margin, sharpness) across four learning rates on CIFAR-10; training (left) and validation (right).



![Refer to caption](/html/2604.20614/assets/x8.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x9.png)


(b) Validation metrics

Figure A4: Sharpness-Aware Minimization (SAM). Training dynamics (loss, accuracy, ECE, margin, sharpness) across four learning rates on CIFAR-10; training (left) and validation (right).



![Refer to caption](/html/2604.20614/assets/x10.png)


(a) Training metrics

![Refer to caption]()


(b) Validation metrics

Figure A5: Muon. Training dynamics (loss, accuracy, ECE, margin, sharpness) across four learning rates on CIFAR-10; training (left) and validation (right).



![Refer to caption](/html/2604.20614/assets/x12.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x13.png)


(b) Validation metrics

Figure A6: AdamW. Training dynamics (loss, accuracy, ECE, margin, sharpness) across four learning rates on CIFAR-10; training (left) and validation (right).



![Refer to caption](/html/2604.20614/assets/x14.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x15.png)


(b) Validation metrics

Figure A7: BulkSGD. Training dynamics (loss, accuracy, ECE, margin, sharpness) across four learning rates on CIFAR-10; training (left) and validation (right).

![Refer to caption](/html/2604.20614/assets/x16.png)


Figure A8: ECE vs. GN sharpness trajectories (CIFAR-100). Same format as Figure [A1](#A3.F1 "Figure A1 ‣ C.1 Sharpness–Calibration Correlation Analysis ‣ Appendix C Additional Sharpness–Calibration Experiments ‣ Too Sharp, Too Sure: When Calibration Follows Curvature").



![Refer to caption](/html/2604.20614/assets/x17.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x18.png)


(b) Validation metrics

Figure A9: Gradient Descent (GD) — CIFAR-100. Training dynamics (loss, accuracy, ECE, margin, sharpness) across learning rates on CIFAR-100; training (left) and validation (right).



![Refer to caption](/html/2604.20614/assets/x19.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x20.png)


(b) Validation metrics

Figure A10: Stochastic Gradient Descent (SGD) — CIFAR-100. Training dynamics (loss, accuracy, ECE, margin, sharpness) across learning rates on CIFAR-100; training (left) and validation (right).



![Refer to caption](/html/2604.20614/assets/x21.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x22.png)


(b) Validation metrics

Figure A11: Sharpness-Aware Minimization (SAM) — CIFAR-100. Training dynamics (loss, accuracy, ECE, margin, sharpness) across learning rates on CIFAR-100; training (left) and validation (right).



![Refer to caption](/html/2604.20614/assets/x23.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x24.png)


(b) Validation metrics

Figure A12: Muon — CIFAR-100. Training dynamics (loss, accuracy, ECE, margin, sharpness) across learning rates on CIFAR-100; training (left) and validation (right).



![Refer to caption](/html/2604.20614/assets/x25.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x26.png)


(b) Validation metrics

Figure A13: AdamW — CIFAR-100. Training dynamics (loss, accuracy, ECE, margin, sharpness) across learning rates on CIFAR-100; training (left) and validation (right).



![Refer to caption](/html/2604.20614/assets/x27.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x28.png)


(b) Validation metrics

Figure A14: BulkSGD — CIFAR-100. Training dynamics (loss, accuracy, ECE, margin, sharpness) across learning rates on CIFAR-100; training (left) and validation (right).

### C.2 Optimizer Details: SAM, Muon, and BulkSGD

There is literature to support the notion that SAM may lead to improved calibration metrics, specifically that SAM act as an implicit regularizer and therefore prevents overfitting during training (Tan et al., [2026](#bib.bib48)). At every step, SAM solves

|  |  |  |
| --- | --- | --- |
|  | minw⁡max‖ϵ‖≤ρ⁡L​(w+ϵ)\min\_{w}\max\_{||\epsilon||\leq\rho}L(w+\epsilon) |  |

which explicitly penalizes the sharpness of the Hessian and leads to convergence to flatter minima (Zhou et al., [2025](#bib.bib57)). We train networks using SAM to test the first hypothesis, looking to confirm that flat minima lead to lower calibration error.

To test the second hypothesis, we apply optimizers that explicitly suppress the contribution of eigenvectors associated with directions of steep descent, through Muon and BulkSGD. Muon rescales the gradient components at each update, so all directions contribute with comparable magnitude. This means directions of steep descent are clamped, while the flatter directions are amplified (Jordan et al., [2024](#bib.bib19)). Another method to suppress directions of steepest descent is using BulkSGD, which at each step projects the gradient to the space orthogonal to the subspace spanned by the top eigenvectors. We try projecting out the top eigenvector, as well as the top three and five eigenvectors (Song et al., [2025](#bib.bib44)). We note that with BulkSGD, we entirely omit the directions of steepest descent, while with Muon we still allow small updates to be made in those directions.

For BulkSGD, training suffers from high levels of instability depending on the number of dominant eigenvectors that are projected out. We observe that training loss is still minimized over 100,000 steps, however the trajectory features steep oscillations. Similarly, sharpness explodes to values in the thousands, which has not been previously observed with other optimizers. This could be due to the fact that, with the dominant eigenvectors projected out, the gradient continues to remain in areas of high curvature, without following the directions of steepest descent.

## Appendix D CalMO: Extended Results

### D.1 Per-Optimizer Training Dynamics

To evaluate out-of-sample calibration, we train ResNet-20 on CIFAR-10 using the full dataset (45K training / 5K validation split). We compare standard cross-entropy loss against CalMO. Each plot shows accuracy,
ECE, the margin functional Q​(θ)=𝔼​[e−m]Q(\theta)=\mathbb{E}[e^{-m}], and loss. For each optimizer, we report both training metrics (left) and validation metrics (right). Lines show
mean over 3 seeds; shaded regions indicate ±1\pm 1 standard deviation (multiplicative for log-scale plots).

![Refer to caption](/html/2604.20614/assets/x29.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x30.png)


(b) Validation metrics

Figure A15: SGD: CE vs. CalMO. Training dynamics on ResNet-20/CIFAR-10; training (left) and validation (right), mean ±1\pm 1 std over 3 seeds.



![Refer to caption](/html/2604.20614/assets/x31.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x32.png)


(b) Validation metrics

Figure A16: AdamW: CE vs. CalMO. Training dynamics on ResNet-20/CIFAR-10; training (left) and validation (right), mean ±1\pm 1 std over 3 seeds.



![Refer to caption](/html/2604.20614/assets/x33.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x34.png)


(b) Validation metrics

Figure A17: Muon: CE vs. CalMO. Training dynamics on ResNet-20/CIFAR-10; training (left) and validation (right), mean ±1\pm 1 std over 3 seeds.



![Refer to caption](/html/2604.20614/assets/x35.png)


(a) Training metrics

![Refer to caption](/html/2604.20614/assets/x36.png)


(b) Validation metrics

Figure A18: SAM: CE vs. CalMO. Training dynamics on ResNet-20/CIFAR-10; training (left) and validation (right), mean ±1\pm 1 std over 3 seeds.

### D.2 Train–Test Calibration Gap

Table [A1](#A4.T1 "Table A1 ‣ D.2 Train–Test Calibration Gap ‣ Appendix D CalMO: Extended Results ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") extends Table [2](#S5.T2 "Table 2 ‣ Performance. ‣ 5.2 Empirical Results with CalMO ‣ 5 From Margin Theory to Calibrated Training ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") with training accuracy and training ECE. All methods reach near-perfect training accuracy and near-zero training ECE after interpolation, confirming that the differences observed on the test set reflect generalization of calibration rather than training dynamics.

| Optimizer | Loss | Train Acc (%) | Train ECE | Test Acc (%) ↑\uparrow | Test ECE ↓\downarrow |
| --- | --- | --- | --- | --- | --- |
| SGD | CE | 100.0±0.0100.0\pm 0.0 | 0.000±0.0000.000\pm 0.000 | 75.2±0.975.2\pm 0.9 | 0.081±0.0170.081\pm 0.017 |
| CalMO | 100.0±0.0100.0\pm 0.0 | 0.004±0.0000.004\pm 0.000 | 80.1±1.180.1\pm 1.1 | 0.056±0.0010.056\pm 0.001 |
| AdamW | CE | 100.0±0.0100.0\pm 0.0 | 0.000±0.0000.000\pm 0.000 | 80.7±0.380.7\pm 0.3 | 0.061±0.0050.061\pm 0.005 |
| CalMO | 100.0±0.0100.0\pm 0.0 | 0.004±0.0000.004\pm 0.000 | 83.2±0.983.2\pm 0.9 | 0.045±0.0050.045\pm 0.005 |
| SAM | CE | 100.0±0.0100.0\pm 0.0 | 0.006±0.0020.006\pm 0.002 | 85.0±0.285.0\pm 0.2 | 0.020±0.0040.020\pm 0.004 |
| CalMO | 100.0±0.0100.0\pm 0.0 | 0.014±0.0010.014\pm 0.001 | 85.2±0.385.2\pm 0.3 | 0.017±0.0040.017\pm 0.004 |
| Muon | CE | 99.5±0.099.5\pm 0.0 | 0.003±0.0010.003\pm 0.001 | 80.3±0.380.3\pm 0.3 | 0.065±0.0130.065\pm 0.013 |
| CalMO | 99.9±0.099.9\pm 0.0 | 0.008±0.0000.008\pm 0.000 | 81.7±0.781.7\pm 0.7 | 0.019±0.0020.019\pm 0.002 |

Table A1: Train–test calibration gap. CE vs. CalMO on ResNet-20/CIFAR-10; extends Table [2](#S5.T2 "Table 2 ‣ Performance. ‣ 5.2 Empirical Results with CalMO ‣ 5 From Margin Theory to Calibrated Training ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") with training metrics.

### D.3 Benchmark vs. Intrinsic Calibration Methods

| Optimizer | Method | Acc (%) | ECE ↓\downarrow |
| --- | --- | --- | --- |
| SGD | CE | 75.2±1.275.2\pm 1.2 | 0.081±0.0210.081\pm 0.021 |
| Label Smooth. | 81.0±1.0\mathbf{81.0\pm 1.0} | 0.082±0.0090.082\pm 0.009 |
| Focal Loss | 78.0±2.978.0\pm 2.9 | 0.029±0.018\mathbf{0.029\pm 0.018} |
| CalMO | 80.1±1.480.1\pm 1.4 | 0.056±0.0010.056\pm 0.001 |
| AdamW | CE | 80.7±0.480.7\pm 0.4 | 0.061±0.0070.061\pm 0.007 |
| Label Smooth. | 85.8±0.1\mathbf{85.8\pm 0.1} | 0.059±0.0020.059\pm 0.002 |
| Focal Loss | 80.5±1.380.5\pm 1.3 | 0.042±0.019\mathbf{0.042\pm 0.019} |
| CalMO | 83.2±1.183.2\pm 1.1 | 0.045±0.0070.045\pm 0.007 |
| Muon | CE | 80.3±0.380.3\pm 0.3 | 0.065±0.0160.065\pm 0.016 |
| Label Smooth. | 82.4±0.7\mathbf{82.4\pm 0.7} | 0.045±0.0030.045\pm 0.003 |
| Focal Loss | 78.7±0.578.7\pm 0.5 | 0.025±0.0170.025\pm 0.017 |
| CalMO | 81.7±0.981.7\pm 0.9 | 0.019±0.002\mathbf{0.019\pm 0.002} |
| SAM | CE | 85.0±0.285.0\pm 0.2 | 0.020±0.0040.020\pm 0.004 |
| Label Smooth. | 86.2±0.5\mathbf{86.2\pm 0.5} | 0.067±0.0040.067\pm 0.004 |
| Focal Loss | 84.0±1.084.0\pm 1.0 | 0.089±0.0100.089\pm 0.010 |
| CalMO | 85.8±0.385.8\pm 0.3 | 0.017±0.005\mathbf{0.017\pm 0.005} |

Table A2: Accuracy and ECE at the best-validation step for CE, label smoothing, focal loss, and CalMO on ResNet-20/CIFAR-10 (90/10 train/validation split).

Table [A2](#A4.T2 "Table A2 ‣ D.3 Benchmark vs. Intrinsic Calibration Methods ‣ Appendix D CalMO: Extended Results ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") compares CalMO to commonly used intrinsic calibration methods—label smoothing and focal loss—across optimizers on CIFAR-10 with ResNet-20 with a 90/10 train/validation split. For each model, we compute accuracy and ECE at the training step that minimizes loss on the validation set. The results highlight that the effectiveness of calibration interventions is optimizer-dependent. For SGD and AdamW, focal loss or label smoothing often achieve the lowest ECE, consistent with prior observations that these methods implicitly regularize confidence. In contrast, for Muon, CalMO yields the largest reduction in ECE while maintaining competitive accuracy. Similarly for SAM, the flatness and robustness terms alone do not control ECE; their combination leads to the lowest miscalibration while maintaining a high accuracy. Across optimizers, CalMO tends to strike a favorable balance between predictive accuracy and calibration error, avoiding the larger accuracy–ECE trade-offs exhibited by some single-mechanism baselines.

## Appendix E Extension to Mean Squared Error

### E.1 GD and SGD Experiments

We rerun the experimental setup of Section [3](#S3 "3 The Coupling Between Calibration and Sharpness ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") with mean squared error (MSE) loss in place of cross-entropy, on 5000 CIFAR-10 training samples. Figures [A19](#A5.F19 "Figure A19 ‣ E.1 GD and SGD Experiments ‣ Appendix E Extension to Mean Squared Error ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") and [A20](#A5.F20 "Figure A20 ‣ E.1 GD and SGD Experiments ‣ Appendix E Extension to Mean Squared Error ‣ Too Sharp, Too Sure: When Calibration Follows Curvature") report the training dynamics for gradient descent and stochastic gradient descent, respectively.

![Refer to caption](/html/2604.20614/assets/img/other/mse_gd.png)


Figure A19: CIFAR-10 | Optimizer: Gradient Descent | Loss: Mean Squared Error

![Refer to caption](/html/2604.20614/assets/img/other/mse_sgd.png)


Figure A20: CIFAR-10 | Optimizer: Stochastic Gradient Descent | Loss: Mean Squared Error

The results show that MSE is extremely miscalibrated, resulting in severely underconfident models as evidenced by the reliability diagrams. This is a consequence of the fact that MSE is not a proper scoring rule.

For LMSEL\_{\mathrm{MSE}}, treating zθ​(xi)z\_{\theta}(x\_{i}) as free variables, the unique minimizer for a
single example is

|  |  |  |
| --- | --- | --- |
|  | zk⋆​(xi)={1,k=yi,0,k≠yi.z\_{k}^{\star}(x\_{i})=\begin{cases}1,&k=y\_{i},\\[3.0pt] 0,&k\neq y\_{i}.\end{cases} |  |

Thus the loss penalizes pushing zθ​(xi)yiz\_{\theta}(x\_{i})\_{y\_{i}} above 11 or the other logits below 0. However, when using the logits in the softmax before the ECE computation, this finite target pattern leads to underconfidence. In fact, at the optimum, the confidence is

|  |  |  |
| --- | --- | --- |
|  | P^​(xi)=pθ​(xi)yi=e1e1+(K−1)​e0=ee+K−1≈0.23for ​K=10.\widehat{P}(x\_{i})=p\_{\theta}(x\_{i})\_{y\_{i}}=\frac{e^{1}}{e^{1}+(K-1)e^{0}}=\frac{e}{e+K-1}\approx 0.23\quad\text{for }K=10. |  |

Consequently, in regimes where training accuracy is close to 11 but logits are near this finite pattern, the
model is systematically underconfident on the training set (accuracy ≈1\approx 1 vs. confidence
≈0.23\approx 0.23 in the main bin), and the training ECE remains large instead of decaying towards zero as in
the CE case.

### E.2 Asymptotics of MSE

For MSE,

|  |  |  |
| --- | --- | --- |
|  | Hz,iMSE​(θ)=∇zi2LMSE​(zi,yi)=2K​IK,H^{\mathrm{MSE}}\_{z,i}(\theta)\;=\;\nabla\_{z\_{i}}^{2}L\_{\mathrm{MSE}}(z\_{i},y\_{i})\;=\;\frac{2}{K}I\_{K}, |  |

so the logit-level Hessian is constant and does not depend on the predicted probabilities pθ​(xi)p\_{\theta}(x\_{i}).
Hence, unlike CE, the Gauss–Newton curvature does not attenuate as the model improves its fit:

|  |  |  |
| --- | --- | --- |
|  | HGNMSE​(θ)=1n​∑iJi​(θ)⊤​Hz,iMSE​Ji​(θ)=2K⋅1n​∑iJi​(θ)⊤​Ji​(θ),H\_{\mathrm{GN}}^{\mathrm{MSE}}(\theta)=\frac{1}{n}\sum\_{i}J\_{i}(\theta)^{\top}H^{\mathrm{MSE}}\_{z,i}J\_{i}(\theta)=\frac{2}{K}\cdot\frac{1}{n}\sum\_{i}J\_{i}(\theta)^{\top}J\_{i}(\theta), |  |

so the eigenvalues of HGNMSE​(θ)H\_{\mathrm{GN}}^{\mathrm{MSE}}(\theta) are governed entirely by the Jacobians Ji​(θ)J\_{i}(\theta), with no probability-dependent factor diag​(p)−p​p⊤\mathrm{diag}(p)-pp^{\top} to drive curvature toward zero. Combined with the underconfidence analysis of Appendix [E.1](#A5.SS1 "E.1 GD and SGD Experiments ‣ Appendix E Extension to Mean Squared Error ‣ Too Sharp, Too Sure: When Calibration Follows Curvature"), this explains why neither sharpness nor training ECE collapse under MSE in the interpolation regime, in contrast with the CE case.

[◄](/html/2604.20613)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2604.20614)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2604.20614)
[View original  
on arXiv](https://arxiv.org/abs/2604.20614)[►](/html/2604.20615)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue May 5 18:02:09 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
