---
arxiv: '2403.19196'
authors:
- Jeffrey Näf, Julie Josse Inria, PreMeDICaL Team, University of Montpellier
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: What Is a Good Imputation Under MAR Missingness?
url: https://arxiv.org/abs/2403.19196
year: 2024
---

[2403.19196] What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.















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



# What Is a Good Imputation Under MAR Missingness?111We thank Giulia Marchello for providing the code for GAIN and MIWAE.

Jeffrey Näf, Julie Josse
  
Inria, PreMeDICaL Team, University of Montpellier

###### Abstract

Missing values pose a persistent challenge in modern data science. Consequently, there is an ever-growing number of publications introducing new imputation methods in various fields. The present paper attempts to take a step back and provide a more systematic analysis: Starting from an in-depth discussion of the Missing at Random (MAR) condition for nonparametric imputation, we first develop an identification result, showing that the widely used Multiple Imputation by Chained Equations (MICE) approach indeed identifies the right conditional distributions. This result, together with two illuminating examples, allows us to propose four essential properties a successful MICE imputation method should meet, thus enabling a more principled evaluation of existing methods and more targeted development of new methods. In particular, we introduce a new method that meets 3 out of the 4 criteria. We then discuss and refine ways to rank imputation methods, even in the challenging setting when the true underlying values are not available. The result is a powerful, easy-to-use scoring algorithm to rank missing value imputations under MAR missingness.

#### Keywords:

imputation, missing at random, distributional prediction, proper scores

## 1 Introduction

In this paper, we study general-purpose (multiple) imputation of missing data sets. That is, instead of imputing for a specific estimation goal or target, we focus on imputations that can be used in a second step for a wide variety of analyses. Developing such imputation methods is still an area of active research, as is benchmarking imputations. To categorize the wealth of imputation methods, one usually differentiates between joint modeling methods that impute the data using one (implicit or explicit) model and the fully conditional specification (FCS) where a different model for each dimension is trained (van Buuren,, [2007](#bib.bib39), [2018](#bib.bib40)). Examples of joint modeling include using parametric distributions (Schafer,, [1997](#bib.bib31)), and more recently, Generative Adversarial Network (GAN)-based (Yoon et al., ([2018](#bib.bib45)); Deng et al., ([2022](#bib.bib5)); Fang and Bao, ([2023](#bib.bib9))) and Variational Autoencoder (VAE)-based methods (Mattei and Frellsen, ([2019](#bib.bib20)); Nazábal et al., ([2020](#bib.bib25)); Qiu et al., ([2020](#bib.bib27)); Yuan et al., ([2021](#bib.bib46))). Another set of examples are methods that use a sequential approach to joint modeling, whereby, for a certain ordering of variables, the joint distribution is specified through a sequence of conditional distributions, see e.g., Ibrahim et al., ([1999](#bib.bib13)); Lee and Mitra, ([2016](#bib.bib15)); Xu et al., ([2016](#bib.bib44)); Murray, ([2018](#bib.bib23)) among others. The most prominent example of FCS is the Multiple Imputation by Chained Equations (MICE) methodology (van Buuren and Groothuis-Oudshoorn,, [2011](#bib.bib41)). While there has been recent progress providing results for MAR imputation in the case of GAN-based methods (Deng et al., ([2022](#bib.bib5)); Fang and Bao, ([2023](#bib.bib9))), such results appear to lack for the FCS approach. Indeed, while some papers claim that imputation is possible under MAR using a methodology such as MICE, without providing a source, Fang and Bao, ([2023](#bib.bib9)) claims MICE can only be used to impute MCAR data.

This paper provides new insights into this research by, among other things, proving that the FCS approach identifies the right distributions under MAR in a population setting and providing a list of desirable properties a successful regression method should meet for FCS. We address three questions: First, is imputation under MAR possible with the FCS approach? Formally, we study whether the conditional distribution needed to impute a missing value is identifiable from the data. Since we do not specify a parametrization and in particular, do not assume that the parameters of the missingness mechanism and the distribution of the data are distinct, this is not clear in general as we will demonstrate using the so-called pattern-mixture model (PMM) representation of missingness (Little, ([1993](#bib.bib17))). We then show that it is nonetheless the case that the imputation distribution is identifiable, allowing for nonparametric imputations in MAR settings using the FCS approach. Our identification result, though simple, appears to be stronger than what exists already. It shows that imputation with the FCS approach is feasible in principle. In particular, we compare the MAR condition we use to stronger conditions used in the context of GAN-based imputation methods in Deng et al., ([2022](#bib.bib5)) and Fang and Bao, ([2023](#bib.bib9)). Second, what properties should the ideal imputation method have? We first illustrate that, despite this identification result, MAR imputation can be extremely challenging. For instance, we consider a simple two-dimensional MAR example with two patterns with widely varying distributions of the observed variable. Based on these insights we develop four properties a successful imputation method should meet in a FCS/MICE framework. In short, a successful imputation method under MAR needs to be a distributional regression method that is able to deal with *covariate shifts*. We discuss existing methods that meet some of these criteria and introduce a new method, denoted “mice-DRF”. Third, given MAR missingness how can one generally find the best imputation for a given data set? This question is independent of whether the FCS or generative approach has been used and has not been addressed at all until very recently. The first important contribution towards solving this problem was made in Näf et al., ([2023](#bib.bib26)) who define the concept of Imputation Scores (I-Scores) to rank imputations. These scores are called “proper” if their population versions rank the imputation methods highest that imputes from the correct conditional distributions. We follow their argument in this paper that imputation is a distributional prediction task and needs to be evaluated as such. In particular, when comparing imputation methods, even under purely academic scenarios where the true underlying values are available, one should refrain from using measures such as the Root Mean Squared Error (RMSE), as already pointed out previously (van Buuren,, [2018](#bib.bib40); Hong and Lynn,, [2020](#bib.bib12); Näf et al.,, [2023](#bib.bib26)). Measures like RMSE favor methods that impute conditional means instead of draws from the conditional distribution. This artificially strengthens the dependence between variables and leads to severe biases in parameter estimates and uncertainty quantification. Instead, an imputation method should draw from the conditional distribution of missing given observed, which might include values in the tail of the distribution. Currently, imputation methods are largely benchmarked and evaluated based on measuring the RMSE between the imputed and the underlying true values, see e.g., Waljee et al., ([2013](#bib.bib42)); Anil Jadhav and Ramanathan, ([2019](#bib.bib1)); Bertsimas et al., ([2018](#bib.bib2)); Stekhoven and Bühlmann, ([2011](#bib.bib34)); Nazábal et al., ([2020](#bib.bib25)); Qiu et al., ([2020](#bib.bib27)); Jäger et al., ([2021](#bib.bib14)); Yoon et al., ([2018](#bib.bib45)); Dong et al., ([2021](#bib.bib6)) and many others.

Instead, we advocate to use a distributional metric or score (Gneiting and Raftery,, [2007](#bib.bib10)) between actual and imputed data sets when the true values are available. For instance, we propose to evaluate imputation methods by calculating the energy distance (Székely,, [2003](#bib.bib35)) between real and imputed datasets. In the more realistic scenario when true values are not available, we advocate using proper I-Scores, as in Näf et al., ([2023](#bib.bib26)). However, we show that the score developed in Näf et al., ([2023](#bib.bib26)) is only proper under a condition much stronger than MAR and instead define a score that is indeed proper under MAR while also more computationally efficient and easier to implement.

The remainder of the article is organized as follows. The remainder of this section introduces notation and related work. In Section [2](#S2 "2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), we discuss the MAR condition and imputation in more detail with two illuminating examples and present our identification result. We then use these insights to present recommendations for imputation methods, including four properties the ideal imputation method should meet in Section [3](#S3 "3 Requirements for Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). Section [4](#S4 "4 Assessing Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") then turns to the question of how to evaluate imputation methods and presents a new proper I-Score. Finally, we illustrate the main points of this paper in a three empirical examples in Section [5](#S5 "5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). Code to replicate the experiments and to use the new scoring methodology can be found in <https://github.com/JeffNaef/MARimputation>.

### 1.1 Notation

We assume an underlying probability space (Ω,𝒜,ℙ)Ω𝒜ℙ(\Omega,\mathcal{A},{\mathbb{P}}) on which all random elements are defined. Throughout, we take 𝒫𝒫\mathcal{P} to be a collection of probability measures on ℝdsuperscriptℝ𝑑{\mathbb{R}}^{d}, dominated by some σ𝜎\sigma-finite measure μ𝜇\mu. We denote the (unobserved) complete data distribution by P∗∈𝒫superscript𝑃𝒫P^{\*}\in\mathcal{P} and by P𝑃P the actually observed distribution with missing values. We assume that P𝑃P (P∗superscript𝑃P^{\*}) has a density p𝑝p (p∗superscript𝑝p^{\*}). We take X𝑋X (X∗superscript𝑋X^{\*}) to be the random vector with distribution P𝑃P (P∗superscript𝑃P^{\*}) and let xisubscript𝑥𝑖x\_{i} (xi∗subscriptsuperscript𝑥𝑖x^{\*}\_{i}), i=1,…,n𝑖

1…𝑛i=1,\ldots,n, be realizations of an i.i.d. copy of the random vector X𝑋X (X∗superscript𝑋X^{\*}). Similarly, M𝑀M is the random vector in {0,1}dsuperscript01𝑑\{0,1\}^{d}, encoding the missingness pattern of X𝑋X, with realization m𝑚m, whereby for j=1,…,d𝑗

1…𝑑j=1,\ldots,d, mj=0subscript𝑚𝑗0m\_{j}=0 means that variable j𝑗j is observed, while mj=1subscript𝑚𝑗1m\_{j}=1 means it is missing. For instance, the observation (NA,x2,x3)NAsubscript𝑥2subscript𝑥3(\texttt{NA},x\_{2},x\_{3}) corresponds to the pattern (1,0,0)100(1,0,0). We denote the support of X𝑋X as 𝒳⊂ℝd𝒳superscriptℝ𝑑\mathcal{X}\subset{\mathbb{R}}^{d} M𝑀M as ℳ⊂{0,1}dℳsuperscript01𝑑\mathcal{M}\subset\{0,1\}^{d}.

To denote assumptions on the missingness mechanism, we use a notation along the lines of Seaman et al., ([2013](#bib.bib32)). For each realization m𝑚m of the missingness random vector M𝑀M we define with o​(X,m):=(Xj)j∈{1,…,d}:mj=0assign𝑜𝑋𝑚subscriptsubscript𝑋𝑗:𝑗1…𝑑subscript𝑚𝑗0o(X,m):=(X\_{j})\_{j\in\{1,\ldots,d\}:m\_{j}=0} the observed part of X𝑋X according to m𝑚m and with oc​(X,m):=(Xj)j∈{1,…,d}:mj=1assignsuperscript𝑜𝑐𝑋𝑚subscriptsubscript𝑋𝑗:𝑗1…𝑑subscript𝑚𝑗1o^{c}(X,m):=(X\_{j})\_{j\in\{1,\ldots,d\}:m\_{j}=1} the corresponding missing part. Note that this operation only filters the corresponding elements of X𝑋X according to m𝑚m, regardless of whether or not these elements are actually missing or not. For instance, we might consider the unobserved part oc​(X,m)superscript𝑜𝑐𝑋𝑚o^{c}(X,m) according to m𝑚m for the fully observed X𝑋X, that is X∼P|M=𝟎similar-to𝑋conditional𝑃𝑀0X\sim P|M=\mathbf{0}, where 𝟎0\mathbf{0} denotes the vector of zeros of length d𝑑d.

As in Näf et al., ([2023](#bib.bib26)), we define ℋP⊂𝒫subscriptℋ𝑃𝒫\mathcal{H}\_{P}\subset\mathcal{P} to be the set of imputation distributions compatible with P𝑃P, that is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℋP:={ℋ∈𝒫:\displaystyle\mathcal{H}\_{P}:=\{{\mathcal{H}}\in\mathcal{P}: | H​ admits density ​h​ and ​h​(o​(x,m)|M=m)=p​(o​(x,m)|M=m)𝐻 admits density ℎ and ℎconditional𝑜𝑥𝑚𝑀𝑚𝑝conditional𝑜𝑥𝑚𝑀𝑚\displaystyle\ H\text{ admits density }h\text{ and }h(o(x,m)|M=m)=p(o(x,m)|M=m) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | for all m∈ℳ},\displaystyle\ \text{ for all }m\in\mathcal{M}\}, |  | (1.1) |

where as above for a pattern m𝑚m, o​(x,m)=(xj)j∈{1,…,d}:mj=0𝑜𝑥𝑚subscriptsubscript𝑥𝑗:𝑗1…𝑑subscript𝑚𝑗0o(x,m)=(x\_{j})\_{j\in\{1,\ldots,d\}:m\_{j}=0} subsets the observed elements of x𝑥x according to m𝑚m, while oc​(x,m)=(xj)j∈{1,…,d}:mj=1,superscript𝑜𝑐𝑥𝑚subscriptsubscript𝑥𝑗

:𝑗1…𝑑subscript𝑚𝑗1o^{c}(x,m)=(x\_{j})\_{j\in\{1,\ldots,d\}:m\_{j}=1,} subsets the missing elements222Note that while hℎh and p𝑝p are densities on ℝdsuperscriptℝ𝑑{\mathbb{R}}^{d}, notation is slightly abused by using expressions such as h​(o​(x,m)|M=m)ℎconditional𝑜𝑥𝑚𝑀𝑚h(o(x,m)|M=m) and p​(o​(x,m)|M=m)𝑝conditional𝑜𝑥𝑚𝑀𝑚p(o(x,m)|M=m), which are densities on ℝ|{j:mj=0}|superscriptℝconditional-set𝑗subscript𝑚𝑗0{\mathbb{R}}^{|\{j:m\_{j}=0\}|}.. Clearly, P∗∈ℋPsuperscript𝑃subscriptℋ𝑃P^{\*}\in\mathcal{H}\_{P}, so that the true distribution P∗superscript𝑃P^{\*} can be seen as an imputation.

### 1.2 Contributions

Inspired by the discussion in Molenberghs et al., ([2008](#bib.bib22)); Näf et al., ([2023](#bib.bib26)), we study the MAR condition under the framework of pattern-mixture models (PMMs) introduced in Little, ([1993](#bib.bib17)), which we argue is more natural for imputation. Overall, we present four main contributions: First, we thoroughly analyse different MAR conditions through the lens of imputation. Crucially, we do not follow the traditional assumption that the distribution of X𝑋X is parametrized by a vector θ𝜃\theta and the distribution of M∣Xconditional𝑀𝑋M\mid X by a distinct vector ϕitalic-ϕ\phi. This removes the question of parameters of interest and allows to study general-purpose nonparametric imputation. Second, we provide an identification result for the FCS approach under the weakest MAR assumption. As the result concerns the identification of conditional distributions in a MAR setting, it can also be applied to the sequential approach of joint modeling as we show in a corollary. Third, based on the previous two contributions we discuss four essential properties a successful imputation method needs to meet in the FCS/MICE framework under MAR. We moreover discuss methods that approximately meet most of these criteria, including a new methodology which we refer to as mice-DRF. As an added benefit, this new methodology is able to impute a block of several variables at once, potentially reducing the heavy computational burden of MICE in high dimensions. We provide an implementation of this imputation method, based on the mice package, as part of our freely available code. Fourth, we discuss the evaluation of imputation methods and show that the Imputation Score developed in Näf et al., ([2023](#bib.bib26)) is not proper under MAR and build a new easy-to-use score with propriety under MAR. The new score is simple to implement and remarkably accurate, even in challenging examples, as we demonstrate empirically in Section [5](#S5 "5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). Though the score needs a set of fully observed variables to be provably proper, we also discuss an alternative version of the score that empirically works well even when all variables have missing values. Throughout, we provide a discussion of imputation under MAR that connects several threads of literature and also make the connection to classical ignorability in a likelihood framework.

### 1.3 Related Work

Though the literature on missingness is vast, the results and discussions presented in this paper are new to the best of our knowledge. Most papers discussing MAR add the additional assumption that the distribution of X𝑋X and M∣Xconditional𝑀𝑋M\mid X are parametrized by two distinct sets of parameters as mentioned above, leading to the classical ignorability result of Rubin, ([1976](#bib.bib30)). This simplifies the analysis and generally avoids the issues we discuss here. For instance, while the FCS and, in particular, the MICE approach has been studied theoretically (Little and Rubin,, [1986](#bib.bib18); Liu et al.,, [2014](#bib.bib19); Zhu and Raghunathan,, [2015](#bib.bib47)) under this ignorability, the problems of identification in this general setting appear to not have been discussed before. Instead, these papers generally focus on the challenging problem of potential incompatibility of the conditional models and analyze the convergence and asymptotic properties of the FCS iterations. Our aim is in a sense much simpler, as we want to answer the question of whether the right conditional distributions are identifiable under MAR when no assumption on the parametrization is placed.

As the paper views missingness through the lens of pattern-mixture models of Little, ([1993](#bib.bib17)), the conceptually closest papers to ours are those based on the Generative Adversarial Network (GAN) approach: Both Deng et al., ([2022](#bib.bib5)); Fang and Bao, ([2023](#bib.bib9)) make use of the PMM view in their proofs, without explicitly mentioning this, as does the original GAIN paper of Yoon et al., ([2018](#bib.bib45)). We essentially provide a similar identification result for the FCS or sequential approach under MAR as Deng et al., ([2022](#bib.bib5)) provide for their GAN-based approach. Despite the simplicity of our identification result, it appears to be stronger than what exists in the literature. For instance, the identification results in Deng et al., ([2022](#bib.bib5)); Fang and Bao, ([2023](#bib.bib9)) for GAN-based methods rely on stronger MAR conditions, as shown below. Similarly, Tian, ([2017](#bib.bib37)) claims the full distribution is recoverable under MAR, but uses a conditional independence condition that is much stronger than the MAR condition we consider. Indeed, graph-based papers concerned with recoverability usually assume variables that are always observed and formulate MAR as conditional independence statements, see e.g Doretti et al., ([2018](#bib.bib8)). This is much stronger than the traditional MAR condition of Rubin, ([1976](#bib.bib30)).
To the best of our knowledge, we are also the first to propose a list of properties an imputation method in the FCS framework should have, based on a thorough analysis of the MAR condition. This list complements existing guidelines on general imputation methods with a different focus, see e.g., Murray, ([2018](#bib.bib23), Section 4). Finally, when considering the evaluation of imputation methods, we build upon the arguments in Näf et al., ([2023](#bib.bib26)) but heavily improve their score to develop a score that is truly proper under MAR, in the sense that it provably ranks the best imputation method highest in a population setting.

## 2 Sequential Imputation under MAR

In the following, we first define MAR properly, following Rubin, ([1976](#bib.bib30)); Seaman et al., ([2013](#bib.bib32)); Mealli and Rubin, ([2015](#bib.bib21)), and analyze several different MAR conditions relevant to our discussion. The different definitions considered here are summarized in Table [1](#S2.T1 "Table 1 ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). Crucially, we do not assume anything about parametrization and instead purely focus on statements about conditional distributions. Using two examples illustrating these definitions, we show that this “nonparametric” view on MAR leads to nontrivial identification problems due to potential distribution shifts. We then present our identification result showing that identification is nonetheless possible in a population setting if one learns the conditional distribution using all available patterns. Finally, we return to the parametrized distribution case and contrast our findings with classical ignorability results in a likelihood framework.

### 2.1 MAR Definitions

| Selection Model: ℙ​(M=m∣x)​p∗​(x)ℙ𝑀conditional𝑚𝑥superscript𝑝𝑥{\mathbb{P}}(M=m\mid x)p^{\*}(x) | Pattern Mixture Model: p∗​(x∣M=m)​ℙ​(M=m)superscript𝑝conditional𝑥𝑀𝑚ℙ𝑀𝑚p^{\*}(x\mid M=m){\mathbb{P}}(M=m) |
| --- | --- |
| ℙ​(M=m|x)=ℙ​(M=m|x~)​ for all ​m∈ℳℙ𝑀conditional𝑚𝑥ℙ𝑀conditional𝑚~𝑥 for all 𝑚ℳ{\mathbb{P}}(M=m|x)={\mathbb{P}}(M=m|\tilde{x})\text{ for all }m\in\mathcal{M} | p∗​(oc​(x,m)∣o​(x,m),M=m)=p∗​(oc​(x,m)∣o​(x,m))superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚𝑜𝑥𝑚p^{\*}(o^{c}(x,m)\mid o(x,m),M=m)=p^{\*}(o^{c}(x,m)\mid o(x,m)) |
| and ​x,x~​ such that ​o​(x,m)=o​(x~,m)  and 𝑥~𝑥 such that 𝑜𝑥𝑚 𝑜~𝑥𝑚\text{ and }x,\tilde{x}\text{ such that }o(x,m)=o(\tilde{x},m) ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) | for all ​m∈ℳ,x∈𝒳formulae-sequencefor all 𝑚ℳ𝑥𝒳\text{ for all }m\in\mathcal{M},x\in\mathcal{X} ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) |
| ℙ​(M=m|x)=ℙ​(M=m|o​(x,m))ℙ𝑀conditional𝑚𝑥ℙ𝑀conditional𝑚𝑜𝑥𝑚{\mathbb{P}}(M=m|x)={\mathbb{P}}(M=m|o(x,m)) |  |
| for all ​m∈ℳ,x∈𝒳formulae-sequencefor all 𝑚ℳ𝑥𝒳\text{ for all }m\in\mathcal{M},x\in\mathcal{X} ([SM-MAR II](#S2.Ex4 "In Definition 2.2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) |  |
|  | p∗​(oc​(x,m)∣o​(x,m),M=m′)=p∗​(oc​(x,m)∣o​(x,m))superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀superscript𝑚′superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚𝑜𝑥𝑚p^{\*}(o^{c}(x,m)\mid o(x,m),M=m^{\prime})=p^{\*}(o^{c}(x,m)\mid o(x,m)) |
|  | for ​m′=m​ or ​m′=0for superscript𝑚′𝑚 or superscript𝑚′0\text{ for }m^{\prime}=m\text{ or }m^{\prime}=0 and all x∈𝒳𝑥𝒳x\in\mathcal{X} ([EMAR](#S2.Ex26 "In Definition 2.6. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) |
|  | p∗​(oc​(x,m)∣o​(x,m),M=m′)=p∗​(oc​(x,m)∣o​(x,m))superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀superscript𝑚′superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚𝑜𝑥𝑚p^{\*}(o^{c}(x,m)\mid o(x,m),M=m^{\prime})=p^{\*}(o^{c}(x,m)\mid o(x,m)) |
|  | for all ​m∈ℳ,m′∈ℳformulae-sequencefor all 𝑚ℳsuperscript𝑚′ℳ\text{ for all }m\in\mathcal{M},m^{\prime}\in\mathcal{M}, x∈𝒳𝑥𝒳x\in\mathcal{X} ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) |
| ℙ​(M=m|x)=ℙ​(M=m)ℙ𝑀conditional𝑚𝑥ℙ𝑀𝑚{\mathbb{P}}(M=m|x)={\mathbb{P}}(M=m) | p∗​(x∣M=m)=p∗​(x∣M=m′)=p∗​(x)superscript𝑝conditional𝑥𝑀𝑚superscript𝑝conditional𝑥𝑀superscript𝑚′superscript𝑝𝑥p^{\*}(x\mid M=m)=p^{\*}(x\mid M=m^{\prime})=p^{\*}(x) |
| for all ​m∈ℳfor all 𝑚ℳ\text{ for all }m\in\mathcal{M}, x∈𝒳𝑥𝒳x\in\mathcal{X} (SM-MCAR) | for all ​m∈ℳ,m′∈ℳformulae-sequencefor all 𝑚ℳsuperscript𝑚′ℳ\text{ for all }m\in\mathcal{M},m^{\prime}\in\mathcal{M}, x∈𝒳𝑥𝒳x\in\mathcal{X} ([PMM-MCAR](#S2.Ex11 "In Definition 2.5. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) |

Table 1: Summary of the different MAR conditions discussed in this paper, when available both in the selection model and the pattern-mixture model. The conditions are ordered from weakest (top) to strongest (bottom). Conditions on the same level are equivalent.

We first properly define what we mean by MAR in the framework of the so-called *selection model* (SM, Little, ([1993](#bib.bib17))). In this framework, the joint distribution of X𝑋X and M𝑀M is factored as,

|  |  |  |
| --- | --- | --- |
|  | p∗​(x,M=m)=ℙ​(M=m∣x)​p∗​(x).superscript𝑝  𝑥𝑀 𝑚ℙ𝑀conditional𝑚𝑥superscript𝑝𝑥p^{\*}(x,M=m)={\mathbb{P}}(M=m\mid x)p^{\*}(x). |  |

Through this view MAR is defined as:

###### Definition 2.1.

The missingness mechanism is missing at random (MAR) if

|  |  |  |
| --- | --- | --- |
|  | ℙ​(M=m|x)=ℙ​(M=m|x~)​ for all ​m∈ℳℙ𝑀conditional𝑚𝑥ℙ𝑀conditional𝑚~𝑥 for all 𝑚ℳ\displaystyle{\mathbb{P}}(M=m|x)={\mathbb{P}}(M=m|\tilde{x})\text{ for all }m\in\mathcal{M} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | and ​x,x~​ such that ​o​(x,m)=o​(x~,m).  and 𝑥~𝑥 such that 𝑜𝑥𝑚 𝑜~𝑥𝑚\displaystyle\text{ and }x,\tilde{x}\text{ such that }o(x,m)=o(\tilde{x},m). |  | (SM-MAR) |

This is sometimes referred to as “Always Missing at Random”, see e.g., Mealli and Rubin, ([2015](#bib.bib21)); Deng et al., ([2022](#bib.bib5)). One can also weaken this requirement to be true only for the data and patterns that are actually observed, which is usually referred to as Realized MAR (RMAR). The arguments in this paper go through with slight modification, also in the case of RMAR, thus we focus on ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) for simplicity. An alternative way to define MAR is

###### Definition 2.2.

The missingness mechanism is missing at random (MAR) if

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℙ​(M=m|x)=ℙ​(M=m|o​(x,m))​ for all ​m∈ℳ,x∈𝒳.formulae-sequenceℙ𝑀conditional𝑚𝑥ℙ𝑀conditional𝑚𝑜𝑥𝑚 for all 𝑚ℳ𝑥𝒳\displaystyle{\mathbb{P}}(M=m|x)={\mathbb{P}}(M=m|o(x,m))\text{ for all }m\in\mathcal{M},x\in\mathcal{X}. |  | (SM-MAR II) |

This is the definition used for instance in Molenberghs et al., ([2008](#bib.bib22)). Note that o​(x,m)𝑜𝑥𝑚o(x,m) is different for each m𝑚m, and thus neither ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) nor ([SM-MAR II](#S2.Ex4 "In Definition 2.2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) are statements about conditional independence as remarked in Mealli and Rubin, ([2015](#bib.bib21)). Nonetheless, ([SM-MAR II](#S2.Ex4 "In Definition 2.2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) is very intuitive: For any value of m𝑚m, we assume that the probability of this value occurring only depends on the observed part of x𝑥x. We show below that both definitions are indeed equivalent.

Considering instead the *pattern-mixture model* (PMM) framework (Little,, [1993](#bib.bib17)), we observe

|  |  |  |
| --- | --- | --- |
|  | p∗​(x,M=m)=p∗​(x∣M=m)​ℙ​(M=m).superscript𝑝  𝑥𝑀 𝑚superscript𝑝conditional𝑥𝑀𝑚ℙ𝑀𝑚p^{\*}(x,M=m)=p^{\*}(x\mid M=m){\mathbb{P}}(M=m). |  |

This view emphasizes that the data we observe in X𝑋X are masked data from a vector X∗∣Mconditionalsuperscript𝑋𝑀X^{\*}\mid M and in particular, when learning quantities from one pattern, we have to be careful when changing to another, as distributions can change from pattern to pattern. A typical example is the Gaussian pattern-mixture model, whereby

|  |  |  |
| --- | --- | --- |
|  | X∗∣M=m∼N​(μm∣Σm),conditionalsuperscript𝑋𝑀𝑚similar-to𝑁conditionalsubscript𝜇𝑚subscriptΣ𝑚\displaystyle X^{\*}\mid M=m\sim N(\mu\_{m}\mid\Sigma\_{m}), |  |

so that the distribution in each pattern might follow a different Gaussian distribution. It is well-known (Little,, [1993](#bib.bib17)), that the parameters of a pattern-mixture model are generally not identifiable without restrictions on how the distributions are allowed to change. Thus an immediate question becomes how the MAR condition constrains these distributions. This was answered in Molenberghs et al., ([2008](#bib.bib22)). We first give a new definition for better readability:

###### Definition 2.3.

The missingness mechanism is missing at random (MAR) if

|  |  |  |
| --- | --- | --- |
|  | p∗​(oc​(x,m)∣o​(x,m),M=m)=p∗​(oc​(x,m)∣o​(x,m))superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚𝑜𝑥𝑚\displaystyle p^{\*}(o^{c}(x,m)\mid o(x,m),M=m)=p^{\*}(o^{c}(x,m)\mid o(x,m)) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | for all ​m∈ℳ,x∈𝒳.formulae-sequencefor all 𝑚ℳ𝑥𝒳\displaystyle\text{ for all }m\in\mathcal{M},x\in\mathcal{X}. |  | (PMM-MAR) |

###### Proposition 2.1 (Molenberghs et al., ([2008](#bib.bib22))).

Condition ([SM-MAR II](#S2.Ex4 "In Definition 2.2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) is equivalent to ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")).

###### Corollary 2.1.

Condition ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) is equivalent to ([SM-MAR II](#S2.Ex4 "In Definition 2.2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) and both are equivalent to ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")).

###### Remark.

The proof starting from ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) is taken from Näf et al., ([2023](#bib.bib26)), though they wrongly thought to have proven equivalence to the stronger condition ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) below.

To understand this condition, and how weak it in fact is, it makes sense to first consider a stronger, but more intuitive condition:

###### Definition 2.4.

The missingness mechanism is conditionally independent MAR (CIMAR) if

|  |  |  |
| --- | --- | --- |
|  | p∗​(oc​(x,m)∣o​(x,m),M=m′)=p∗​(oc​(x,m)∣o​(x,m))superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀superscript𝑚′superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚𝑜𝑥𝑚\displaystyle p^{\*}(o^{c}(x,m)\mid o(x,m),M=m^{\prime})=p^{\*}(o^{c}(x,m)\mid o(x,m)) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | for all ​m∈ℳ,m′∈ℳ,x∈𝒳.formulae-sequencefor all 𝑚ℳformulae-sequencesuperscript𝑚′ℳ𝑥𝒳\displaystyle\text{ for all }m\in\mathcal{M},m^{\prime}\in\mathcal{M},x\in\mathcal{X}. |  | (CIMAR) |

![Refer to caption](/html/2403.19196/assets/illustration1.png)


Figure 1: Illustration: 𝐗∗superscript𝐗\mathbf{X}^{\*} is the assumed underlying full data, 𝐌𝐌\mathbf{M} is the vector of missing indicators and 𝐗𝐗\mathbf{X} arises when 𝐌𝐌\mathbf{M} is applied to 𝐗∗superscript𝐗\mathbf{X}^{\*}. Thus each row of 𝐗/𝐗∗𝐗superscript𝐗\mathbf{X}/\mathbf{X}^{\*} is an observation under a different pattern. Under condition ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")), the distribution of X1,X2∣X3

subscript𝑋1conditionalsubscript𝑋2subscript𝑋3X\_{1},X\_{2}\mid X\_{3} is not allowed to change when moving from one pattern to another, though the marginal distribution of X3subscript𝑋3X\_{3} is allowed to change. In contrast, under MCAR ([PMM-MCAR](#S2.Ex11 "In Definition 2.5. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")), no change is allowed. Under MAR ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) the only constraint is that the distribution of X1,X2∣X3

subscript𝑋1conditionalsubscript𝑋2subscript𝑋3X\_{1},X\_{2}\mid X\_{3} in the third pattern is the same as the unconditional one.

This is a conditional independence statement, namely that oc​(X,M)∣o​(X,M)conditionalsuperscript𝑜𝑐𝑋𝑀𝑜𝑋𝑀o^{c}(X,M)\mid o(X,M) is independent of M′superscript𝑀′M^{\prime}. That is, no matter what pattern m′superscript𝑚′m^{\prime} is considered, the distribution of oc​(X,M)∣o​(X,M)conditionalsuperscript𝑜𝑐𝑋𝑀𝑜𝑋𝑀o^{c}(X,M)\mid o(X,M) remains the same. As such, ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) allows to learn the distribution of oc​(X,m)∣o​(X,m)conditionalsuperscript𝑜𝑐𝑋𝑚𝑜𝑋𝑚o^{c}(X,m)\mid o(X,m) from any pattern m′superscript𝑚′m^{\prime}. It in turn is still weaker than MCAR however, which requires that

###### Definition 2.5.

The missingness mechanism is missing completely at random (MCAR), if

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | p∗​(x∣M=m)=p∗​(x∣M=m′)=p∗​(x)superscript𝑝conditional𝑥𝑀𝑚superscript𝑝conditional𝑥𝑀superscript𝑚′superscript𝑝𝑥\displaystyle p^{\*}(x\mid M=m)=p^{\*}(x\mid M=m^{\prime})=p^{\*}(x) | for all ​m∈ℳ,m′∈ℳ,x∈𝒳.formulae-sequencefor all 𝑚ℳformulae-sequencesuperscript𝑚′ℳ𝑥𝒳\displaystyle\text{ for all }m\in\mathcal{M},m^{\prime}\in\mathcal{M},x\in\mathcal{X}. |  | (PMM-MCAR) |

Figure [1](#S2.F1 "Figure 1 ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") illustrates these different conditions in a small example.

Under ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")), the observed variables can widely change their distribution from pattern to pattern, as shown in the following example:

###### Example 1.

Consider the following Gaussian mixture model for two patterns m1=(0,0)subscript𝑚100m\_{1}=(0,0) and m2=(1,0)subscript𝑚210m\_{2}=(1,0):

|  |  |  |  |
| --- | --- | --- | --- |
|  | (X1,X2)∣M=m1conditionalsubscript𝑋1subscript𝑋2𝑀subscript𝑚1\displaystyle(X\_{1},X\_{2})\mid M=m\_{1} | ∼N​((00),(2111))similar-toabsent𝑁matrix00matrix2111\displaystyle\sim N\left(\begin{pmatrix}0\\ 0\end{pmatrix},\begin{pmatrix}2&1\\ 1&1\end{pmatrix}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | (X1,X2)∣M=m2conditionalsubscript𝑋1subscript𝑋2𝑀subscript𝑚2\displaystyle(X\_{1},X\_{2})\mid M=m\_{2} | ∼N​((55),(2111)).similar-toabsent𝑁matrix55matrix2111\displaystyle\sim N\left(\begin{pmatrix}5\\ 5\end{pmatrix},\begin{pmatrix}2&1\\ 1&1\end{pmatrix}\right). |  |

For both patterns, the conditional distribution of X1subscript𝑋1X\_{1} given X2subscript𝑋2X\_{2} is given as

|  |  |  |
| --- | --- | --- |
|  | p​(x1∣x2,M=m1)=p​(x1∣x2,M=m2)=N​(x2,1)​(x1),𝑝conditionalsubscript𝑥1  subscript𝑥2𝑀subscript𝑚1𝑝conditionalsubscript𝑥1  subscript𝑥2𝑀subscript𝑚2𝑁subscript𝑥21subscript𝑥1\displaystyle p(x\_{1}\mid x\_{2},M=m\_{1})=p(x\_{1}\mid x\_{2},M=m\_{2})=N(x\_{2},1)(x\_{1}), |  |

where N​(x2,1)​(x1)𝑁subscript𝑥21subscript𝑥1N(x\_{2},1)(x\_{1}) is the univariate Gaussian density with mean x2subscript𝑥2x\_{2} and variance 1 evaluated at x1subscript𝑥1x\_{1}. We first verify that the condition in ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) holds:

|  |  |  |  |
| --- | --- | --- | --- |
|  | p∗​(x1∣x2)superscript𝑝conditionalsubscript𝑥1subscript𝑥2\displaystyle p^{\*}(x\_{1}\mid x\_{2}) | =P​(M=m1)​p∗​(x1,x2∣M=m1)+P​(M=m2)​p∗​(x1,x2∣M=m2)P​(M=m1)​p∗​(x2∣M=m1)+P​(M=m2)​p∗​(x2∣M=m2)absent𝑃𝑀subscript𝑚1superscript𝑝  subscript𝑥1conditionalsubscript𝑥2𝑀 subscript𝑚1𝑃𝑀subscript𝑚2superscript𝑝  subscript𝑥1conditionalsubscript𝑥2𝑀 subscript𝑚2𝑃𝑀subscript𝑚1superscript𝑝conditionalsubscript𝑥2𝑀subscript𝑚1𝑃𝑀subscript𝑚2superscript𝑝conditionalsubscript𝑥2𝑀subscript𝑚2\displaystyle=\frac{P(M=m\_{1})p^{\*}(x\_{1},x\_{2}\mid M=m\_{1})+P(M=m\_{2})p^{\*}(x\_{1},x\_{2}\mid M=m\_{2})}{P(M=m\_{1})p^{\*}(x\_{2}\mid M=m\_{1})+P(M=m\_{2})p^{\*}(x\_{2}\mid M=m\_{2})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(P​(M=m1)​p∗​(x2∣M=m1)+P​(M=m2)​p∗​(x2∣M=m2))​p∗​(x1∣x2,M=m2)P​(M=m1)​p∗​(x2∣M=m1)+P​(M=m2)​p∗​(x2∣M=m2)absent𝑃𝑀subscript𝑚1superscript𝑝conditionalsubscript𝑥2𝑀subscript𝑚1𝑃𝑀subscript𝑚2superscript𝑝conditionalsubscript𝑥2𝑀subscript𝑚2superscript𝑝conditionalsubscript𝑥1  subscript𝑥2𝑀subscript𝑚2𝑃𝑀subscript𝑚1superscript𝑝conditionalsubscript𝑥2𝑀subscript𝑚1𝑃𝑀subscript𝑚2superscript𝑝conditionalsubscript𝑥2𝑀subscript𝑚2\displaystyle=\frac{\left(P(M=m\_{1})p^{\*}(x\_{2}\mid M=m\_{1})+P(M=m\_{2})p^{\*}(x\_{2}\mid M=m\_{2})\right)p^{\*}(x\_{1}\mid x\_{2},M=m\_{2})}{P(M=m\_{1})p^{\*}(x\_{2}\mid M=m\_{1})+P(M=m\_{2})p^{\*}(x\_{2}\mid M=m\_{2})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =p∗​(x1∣x2,M=m2)absentsuperscript𝑝conditionalsubscript𝑥1  subscript𝑥2𝑀subscript𝑚2\displaystyle=p^{\*}(x\_{1}\mid x\_{2},M=m\_{2}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =p∗​(x1∣x2,M=m1).absentsuperscript𝑝conditionalsubscript𝑥1  subscript𝑥2𝑀subscript𝑚1\displaystyle=p^{\*}(x\_{1}\mid x\_{2},M=m\_{1}). |  |

However, the distribution of X2subscript𝑋2X\_{2} in pattern m1subscript𝑚1m\_{1} (N​(0,1)𝑁01N(0,1)) is heavily shifted compared to pattern m2subscript𝑚2m\_{2} (N​(5,1)𝑁51N(5,1)). Section [3](#S3 "3 Requirements for Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") demonstrates how different imputation methods struggle to deal with this shift in distribution on simulated data.

In the above example, we only have 2 patterns and thus ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) and ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) turn out to be equivalent and both hold in this example. However, an example with 3 patterns shows that ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) is strictly weaker than ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")):

###### Example 2.

Consider

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐗=(x1,1x1,2x1,3x2,1N​Ax2,3N​Ax3,2x3,3),𝐌=(000010100)=(m1m2m3).formulae-sequence𝐗matrixsubscript𝑥  11subscript𝑥  12subscript𝑥  13subscript𝑥  21𝑁𝐴subscript𝑥  23𝑁𝐴subscript𝑥  32subscript𝑥  33𝐌matrix000010100matrixsubscript𝑚1subscript𝑚2subscript𝑚3\displaystyle\mathbf{X}=\begin{pmatrix}x\_{1,1}&x\_{1,2}&x\_{1,3}\\ x\_{2,1}&NA&x\_{2,3}\\ NA&x\_{3,2}&x\_{3,3}\\ \end{pmatrix},\mathbf{M}=\begin{pmatrix}0&0&0\\ 0&1&0\\ 1&0&0\\ \end{pmatrix}=\begin{pmatrix}m\_{1}\\ m\_{2}\\ m\_{3}\end{pmatrix}. |  | (2.1) |

whereby (X1,X2,X3)subscript𝑋1subscript𝑋2subscript𝑋3(X\_{1},X\_{2},X\_{3}) are independently uniformly distributed on [0,1]01[0,1]. We further specify that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℙ​(M=m1∣x)ℙ𝑀conditionalsubscript𝑚1𝑥\displaystyle{\mathbb{P}}(M=m\_{1}\mid x) | =ℙ​(M=m1∣x1)=x1/3absentℙ𝑀conditionalsubscript𝑚1subscript𝑥1subscript𝑥13\displaystyle={\mathbb{P}}(M=m\_{1}\mid x\_{1})=x\_{1}/3 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ℙ​(M=m2∣x)ℙ𝑀conditionalsubscript𝑚2𝑥\displaystyle{\mathbb{P}}(M=m\_{2}\mid x) | =ℙ​(M=m2∣x1)=2/3−x1/3absentℙ𝑀conditionalsubscript𝑚2subscript𝑥123subscript𝑥13\displaystyle={\mathbb{P}}(M=m\_{2}\mid x\_{1})=2/3-x\_{1}/3 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ℙ​(M=m3∣x)ℙ𝑀conditionalsubscript𝑚3𝑥\displaystyle{\mathbb{P}}(M=m\_{3}\mid x) | =ℙ​(M=m3)=1/3.absentℙ𝑀subscript𝑚313\displaystyle={\mathbb{P}}(M=m\_{3})=1/3. |  |

It can be checked that these are valid distributions, as in particular, ∑mℙ​(M=m)=1subscript𝑚ℙ𝑀𝑚1\sum\_{m}{\mathbb{P}}(M=m)=1 and ∑mℙ​(M=m∣xj)=1subscript𝑚ℙ𝑀conditional𝑚subscript𝑥𝑗1\sum\_{m}{\mathbb{P}}(M=m\mid x\_{j})=1 for j=1,…,3𝑗

1…3j=1,\ldots,3. Moreover, ℙ​(M=m∣x)=ℙ​(M=m∣o​(x,m))ℙ𝑀conditional𝑚𝑥ℙ𝑀conditional𝑚𝑜𝑥𝑚{\mathbb{P}}(M=m\mid x)={\mathbb{P}}(M=m\mid o(x,m)) and thus the MAR condition ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) holds. In particular, for variable x1subscript𝑥1x\_{1} in pattern m3subscript𝑚3m\_{3}, it holds that

|  |  |  |
| --- | --- | --- |
|  | p∗​(x1∣x2,x3,M=m3)=p∗​(x1∣x2,x3).superscript𝑝conditionalsubscript𝑥1  subscript𝑥2subscript𝑥3𝑀subscript𝑚3superscript𝑝conditionalsubscript𝑥1  subscript𝑥2subscript𝑥3\displaystyle p^{\*}(x\_{1}\mid x\_{2},x\_{3},M=m\_{3})=p^{\*}(x\_{1}\mid x\_{2},x\_{3}). |  |

However, if we consider x1subscript𝑥1x\_{1} given (x2,x3)subscript𝑥2subscript𝑥3(x\_{2},x\_{3}) in the first pattern, we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | p∗​(x1∣x2,x3,M=m1)superscript𝑝conditionalsubscript𝑥1  subscript𝑥2subscript𝑥3𝑀subscript𝑚1\displaystyle p^{\*}(x\_{1}\mid x\_{2},x\_{3},M=m\_{1}) | =ℙ​(M=m1∣x1,x2,x3)ℙ​(M=m1∣x2,x3)​p∗​(x1∣x2,x3)absentℙ𝑀conditionalsubscript𝑚1  subscript𝑥1subscript𝑥2subscript𝑥3ℙ𝑀conditionalsubscript𝑚1  subscript𝑥2subscript𝑥3superscript𝑝conditionalsubscript𝑥1  subscript𝑥2subscript𝑥3\displaystyle=\frac{{\mathbb{P}}(M=m\_{1}\mid x\_{1},x\_{2},x\_{3})}{{\mathbb{P}}(M=m\_{1}\mid x\_{2},x\_{3})}p^{\*}(x\_{1}\mid x\_{2},x\_{3}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =x1​p∗​(x1∣x2,x3),absentsubscript𝑥1superscript𝑝conditionalsubscript𝑥1  subscript𝑥2subscript𝑥3\displaystyle=x\_{1}p^{\*}(x\_{1}\mid x\_{2},x\_{3}), |  |

showing that ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) does not hold. Figure [2](#S2.F2 "Figure 2 ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") illustrates this behavior: It shows the distribution of X1subscript𝑋1X\_{1} in different patterns. As the distribution of (X2,X3)subscript𝑋2subscript𝑋3(X\_{2},X\_{3}) in the different patterns is always the same, this directly illustrates the change in the conditional distribution of X1∣X2,X3conditionalsubscript𝑋1

subscript𝑋2subscript𝑋3X\_{1}\mid X\_{2},X\_{3} when changing from pattern m1subscript𝑚1m\_{1} to pattern m3subscript𝑚3m\_{3}. The key is thus that ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) still allows for a change in the conditional distributions over different patterns. That is the distribution X1∣X2,X3conditionalsubscript𝑋1

subscript𝑋2subscript𝑋3X\_{1}\mid X\_{2},X\_{3} in pattern m1subscript𝑚1m\_{1} is different in the above example from the distribution X1∣X2,X3conditionalsubscript𝑋1

subscript𝑋2subscript𝑋3X\_{1}\mid X\_{2},X\_{3} in pattern m3subscript𝑚3m\_{3}. All that is required is that the distribution X1∣X2,X3conditionalsubscript𝑋1

subscript𝑋2subscript𝑋3X\_{1}\mid X\_{2},X\_{3} in pattern m3subscript𝑚3m\_{3} corresponds to the unconditional one.

![Refer to caption](/html/2403.19196/assets/Example1.png)


Figure 2: Illustration of Example [2](#Thmexample2 "Example 2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). Left: Distribution we would like to impute X1∣M=m3conditionalsubscript𝑋1𝑀subscript𝑚3X\_{1}\mid M=m\_{3}. Middle: Distribution of X1subscript𝑋1X\_{1} in the fully observed pattern (X1∣M=m1)conditionalsubscript𝑋1𝑀subscript𝑚1(X\_{1}\mid M=m\_{1}). Right: Distribution of all patterns for which X1subscript𝑋1X\_{1} is observed (Mixture of the distribution of X1subscript𝑋1X\_{1} in patterns m1subscript𝑚1m\_{1} and m2subscript𝑚2m\_{2}).

Thus we have just shown that

###### Proposition 2.2.

MCAR ([PMM-MCAR](#S2.Ex11 "In Definition 2.5. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) is strictly stronger than ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) which is strictly stronger than ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")).

Another important MAR condition is the extended MAR condition:

###### Definition 2.6.

The missingness mechanism is extended missing at random (EMAR), if

|  |  |  |
| --- | --- | --- |
|  | p∗​(oc​(x,m)∣o​(x,m),M=m′)=p∗​(oc​(x,m)∣o​(x,m))superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀superscript𝑚′superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚𝑜𝑥𝑚\displaystyle p^{\*}(o^{c}(x,m)\mid o(x,m),M=m^{\prime})=p^{\*}(o^{c}(x,m)\mid o(x,m)) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | for ​m′=m​ or ​m′=0, for all ​x∈𝒳.formulae-sequencefor superscript𝑚′𝑚 or superscript𝑚′0 for all 𝑥𝒳\displaystyle\text{ for }m^{\prime}=m\text{ or }m^{\prime}=0,\text{ for all }x\in\mathcal{X}. |  | (EMAR) |

This is clearly stronger than ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) and weaker than ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). Moreover, it is a useful condition as it allows to learn any conditional distribution of missing given observed from the fully observed pattern.

### 2.2 FCS in MAR

The previous discussion illustrates that from ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) alone, it is not clear whether learning a distribution in one pattern allows to impute values in the other pattern. However, in Example [2](#Thmexample2 "Example 2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") while P​(M=m1∣x1)𝑃𝑀conditionalsubscript𝑚1subscript𝑥1P(M=m\_{1}\mid x\_{1}) depends on x1subscript𝑥1x\_{1}, P​(M=m1∣x1)+P​(M=m2∣x1)𝑃𝑀conditionalsubscript𝑚1subscript𝑥1𝑃𝑀conditionalsubscript𝑚2subscript𝑥1P(M=m\_{1}\mid x\_{1})+P(M=m\_{2}\mid x\_{1}) does not. This is the key property in the proof of identification under MAR as it implies that p∗​(x1∣x2,x3)superscript𝑝conditionalsubscript𝑥1

subscript𝑥2subscript𝑥3p^{\*}(x\_{1}\mid x\_{2},x\_{3}) needed for imputation can be identified if *all* patterns for which x1subscript𝑥1x\_{1} are observed are considered. We detail this now.

The goal of the FCS in general and the MICE approach in particular is to impute by iteratively drawing for all j∈{1,…,d}𝑗1…𝑑j\in\{1,\ldots,d\} and t≥1𝑡1t\geq 1,

|  |  |  |
| --- | --- | --- |
|  | xj(t+1)∼p∗​(xj∣x−j(t)),similar-tosuperscriptsubscript𝑥𝑗𝑡1superscript𝑝conditionalsubscript𝑥𝑗superscriptsubscript𝑥𝑗𝑡\displaystyle x\_{j}^{(t+1)}\sim p^{\*}(x\_{j}\mid x\_{-j}^{(t)}), |  |

whereby x−j(t)={xl(t)}l≠jsuperscriptsubscript𝑥𝑗𝑡subscriptsuperscriptsubscript𝑥𝑙𝑡𝑙𝑗x\_{-j}^{(t)}=\{x\_{l}^{(t)}\}\_{l\neq j} are the imputed and observed values of all other variables except j𝑗j at the t𝑡tth iteration. Doing this repeatedly leads to a Gibbs sampler that converges under quite mild conditions (Little and Rubin, ([1986](#bib.bib18), Chapter 10.2.4.)). Naturally, if one does not have access to the true distribution p∗superscript𝑝p^{\*} and estimates the conditional model nonparametrically, this is a very complicated problem to analyze theoretically. Here we focus on a very simple question: If x−jsubscript𝑥𝑗x\_{-j} has already been imputed by the correct distribution, that is we have access to the true underlying (d−1)−limit-from𝑑1(d-1)-variate distribution p∗​(x−j)superscript𝑝subscript𝑥𝑗p^{\*}(x\_{-j}), can we successfully impute xjsubscript𝑥𝑗x\_{j} by only looking at the patterns where xjsubscript𝑥𝑗x\_{j} is observed? This view connects to Example [2](#Thmexample2 "Example 2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") and avoids any question of convergence of the Gibbs sampler to focus purely on identification.

Let in the following,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lj={m∈ℳ:xj∈o​(x,m)},subscript𝐿𝑗conditional-set𝑚ℳsubscript𝑥𝑗𝑜𝑥𝑚\displaystyle L\_{j}=\{m\in\mathcal{M}:x\_{j}\in o(x,m)\}, |  | (2.2) |

be the set of patterns in which xjsubscript𝑥𝑗x\_{j} is observed. The best action one can do in this case is to draw from the distribution,

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | h∗​(xj∣x−j)superscriptℎconditionalsubscript𝑥𝑗subscript𝑥𝑗\displaystyle h^{\*}(x\_{j}\mid x\_{-j}) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =∑m∈Ljℙ​(M=m)∑m∈Ljp∗​(x−j∣M=m)​ℙ​(M=m)​p∗​(x∣M=m),absentsubscript𝑚subscript𝐿𝑗ℙ𝑀𝑚subscript𝑚subscript𝐿𝑗superscript𝑝conditionalsubscript𝑥𝑗𝑀𝑚ℙ𝑀𝑚superscript𝑝conditional𝑥𝑀𝑚\displaystyle=\sum\_{m\in L\_{j}}\frac{{\mathbb{P}}(M=m)}{\sum\_{m\in L\_{j}}p^{\*}(x\_{-j}\mid M=m){\mathbb{P}}(M=m)}p^{\*}(x\mid M=m), |  | (2.3) |

which is the conditional distribution of Xj∣X−jconditionalsubscript𝑋𝑗subscript𝑋𝑗X\_{j}\mid X\_{-j} *learned from all patterns in which xjsubscript𝑥𝑗x\_{j} is observed*. Owing to the above example, the question is whether under MAR, h∗​(xj∣x−j)superscriptℎconditionalsubscript𝑥𝑗subscript𝑥𝑗h^{\*}(x\_{j}\mid x\_{-j}) is indeed the same as p∗​(xj∣x−j)superscript𝑝conditionalsubscript𝑥𝑗subscript𝑥𝑗p^{\*}(x\_{j}\mid x\_{-j});

###### Proposition 2.3.

Assume MAR in ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) holds. Then for h∗​(xj∣x−j)superscriptℎconditionalsubscript𝑥𝑗subscript𝑥𝑗h^{\*}(x\_{j}\mid x\_{-j}) as in ([2.2](#S2.Ex28 "2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")),

|  |  |  |  |
| --- | --- | --- | --- |
|  | h∗​(xj∣x−j)=p∗​(xj∣x−j),superscriptℎconditionalsubscript𝑥𝑗subscript𝑥𝑗superscript𝑝conditionalsubscript𝑥𝑗subscript𝑥𝑗\displaystyle h^{\*}(x\_{j}\mid x\_{-j})=p^{\*}(x\_{j}\mid x\_{-j}), |  | (2.4) |

for all x−jsubscript𝑥𝑗x\_{-j} with p∗​(x−j)>0superscript𝑝subscript𝑥𝑗0p^{\*}(x\_{-j})>0.

This shows that the desired distribution is indeed recoverable in principle from all available patterns. Intuitively at Xjsubscript𝑋𝑗X\_{j}, one can reduce the |ℳ|ℳ|\mathcal{M}| patterns to two, one where Xjsubscript𝑋𝑗X\_{j} is missing, and one where it is observed. Though these two aggregated patterns are mixtures of several patterns m∈ℳ𝑚ℳm\in\mathcal{M}, it can be shown that the MAR condition implies that both aggregated patterns have the same conditional distribution Xj∣X−jconditionalsubscript𝑋𝑗subscript𝑋𝑗X\_{j}\mid X\_{-j}, thus allowing to identify the right conditional distribution in the pattern where Xjsubscript𝑋𝑗X\_{j} is observed.

Even with perfect estimation, conditioning on X−jsubscript𝑋𝑗X\_{-j} would require iteration over several imputations, as mentioned above. To make the result more tangible, we can study the following simplified procedure that avoids iteration entirely: Assume in the following that one variable is fully observed. That is the possible pattern in ℳℳ\mathcal{M} all share one zero, or

|  |  |  |  |
| --- | --- | --- | --- |
|  | O={l:ml=0​ for all ​m∈ℳ},𝑂conditional-set𝑙subscript𝑚𝑙0 for all 𝑚ℳ\displaystyle O=\{l:m\_{l}=0\text{ for all }m\in\mathcal{M}\}, |  | (2.5) |

is not empty. Without loss of generality, we assume that this fully observed variable is the p𝑝pth one. Then for j∈{1,…,p−1}𝑗1…𝑝1j\in\{1,\ldots,p-1\}, let Ljsubscript𝐿𝑗L\_{j} be defined as in ([2.2](#S2.E2 "In 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). We then impute by drawing observations from

|  |  |  |
| --- | --- | --- |
|  | h∗​(xj∣xj+1,…,xp)superscriptℎconditionalsubscript𝑥𝑗  subscript𝑥𝑗1…subscript𝑥𝑝\displaystyle h^{\*}(x\_{j}\mid x\_{j+1},\ldots,x\_{p}) |  |
|  |  |  |
| --- | --- | --- |
|  | =∑m∈Ljℙ​(M=m)∑m∈Ljp∗​(xj+1,…,xp∣M=m)​ℙ​(M=m)​p∗​(xj,xj+1,…,xp∣M=m),absentsubscript𝑚subscript𝐿𝑗ℙ𝑀𝑚subscript𝑚subscript𝐿𝑗superscript𝑝  subscript𝑥𝑗1…conditionalsubscript𝑥𝑝𝑀 𝑚ℙ𝑀𝑚superscript𝑝  subscript𝑥𝑗subscript𝑥𝑗1…conditionalsubscript𝑥𝑝𝑀 𝑚\displaystyle=\sum\_{m\in L\_{j}}\frac{{\mathbb{P}}(M=m)}{\sum\_{m\in L\_{j}}p^{\*}(x\_{j+1},\ldots,x\_{p}\mid M=m){\mathbb{P}}(M=m)}p^{\*}(x\_{j},x\_{j+1},\ldots,x\_{p}\mid M=m), |  |

which is the conditional distribution Xj∣Xj+1,…,Xpconditionalsubscript𝑋𝑗

subscript𝑋𝑗1…subscript𝑋𝑝X\_{j}\mid X\_{j+1},\ldots,X\_{p} learned from all patterns mlsubscript𝑚𝑙m\_{l}, l∈Lj𝑙subscript𝐿𝑗l\in L\_{j}. This in fact corresponds to the sequential approach to joint modeling, see e.g., Murray, ([2018](#bib.bib23)) and the references therein. We denote the resulting distribution of the fully imputed data as H∗superscript𝐻H^{\*} with density h∗superscriptℎh^{\*}. For this simplified imputation approach it holds that:

###### Corollary 2.2.

Assume MAR in ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) holds and that O𝑂O in ([2.5](#S2.E5 "In 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) is not empty. Then H∗∈ℋPsuperscript𝐻subscriptℋ𝑃H^{\*}\in\mathcal{H}\_{P} has

|  |  |  |  |
| --- | --- | --- | --- |
|  | h∗​(x)=p∗​(x), for all ​x.superscriptℎ𝑥  superscript𝑝𝑥 for all 𝑥\displaystyle h^{\*}(x)=p^{\*}(x),\text{ for all }x. |  | (2.6) |

Proposition [2.3](#S2.Thmproposition3 "Proposition 2.3. ‣ 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") and Corollary [2.2](#S2.Thmcorollary2 "Corollary 2.2. ‣ 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") show that sequential imputation with an algorithm that can perfectly learn the distribution under MAR is indeed identified, in the sense that we are able to learn the true conditional distribution needed to impute a missing value. The key for the proof is that (1) all available patterns are used to learn a distribution of xj∣xj+1,…,xpconditionalsubscript𝑥𝑗

subscript𝑥𝑗1…subscript𝑥𝑝x\_{j}\mid x\_{j+1},\ldots,x\_{p}, (2) use of ([SM-MAR II](#S2.Ex4 "In Definition 2.2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")), which is equivalent to ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")), and (3) that the conditional distributions P​(M=m∣x)𝑃𝑀conditional𝑚𝑥P(M=m\mid x) still need to sum to 1 over all values of m𝑚m.

###### Remark.

In particular, Proposition [2.3](#S2.Thmproposition3 "Proposition 2.3. ‣ 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") and Corollary [2.2](#S2.Thmcorollary2 "Corollary 2.2. ‣ 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") show that the FCS approach can identify the right conditional distributions under a weaker condition than GAN-based approaches. Deng et al., ([2022](#bib.bib5)) show that their GAN architecture is able to impute missingness under EMAR, ([EMAR](#S2.Ex26 "In Definition 2.6. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). This condition allows to learn a distribution from the fully observed pattern and is thus strictly stronger than ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). Similarly, Fang and Bao, ([2023](#bib.bib9)) show that their GAN-based method can identify the conditional distribution of missing given observed data. However, while they claim this shows identification under MAR, the condition they present in Section 3.2. is actually stronger and more akin to ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")).

From the above, it can be seen that we can in general not just simply learn the conditional distributions from the fully observed data and then impute the missing variables. Instead, we need to consider *all* patterns wherein a variable xjsubscript𝑥𝑗x\_{j} is observed to be able to impute it. We now want to highlight why this discussion of distribution shifts under MAR may not be relevant for Maximum Likelihood Estimation (MLE).

### 2.3 Ignorability in Maximum Likelihood Estimation

In the context of MLE, it has long been established (Rubin,, [1976](#bib.bib30)) that the missing mechanism is ignorable under MAR and an additional condition. This additional condition is critical for our discussion. To formalize this assume p∗superscript𝑝p^{\*} is parametrized by a vector θ𝜃\theta. Moreover, assume the conditional distribution of M∣xconditional𝑀𝑥M\mid x is parametrized by ϕitalic-ϕ\phi. Then we can rewrite the MAR definition in ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) slightly, as in Rubin, ([1976](#bib.bib30)); Mealli and Rubin, ([2015](#bib.bib21)):

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ℙϕ​(M=m|x)=ℙϕ​(M=m|x~)​ for all ​m∈ℳsubscriptℙitalic-ϕ𝑀conditional𝑚𝑥subscriptℙitalic-ϕ𝑀conditional𝑚~𝑥 for all 𝑚ℳ\displaystyle{\mathbb{P}}\_{\phi}(M=m|x)={\mathbb{P}}\_{\phi}(M=m|\tilde{x})\text{ for all }m\in\mathcal{M} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | and ​x,x~​ such that ​o​(x,m)=o​(x~,m).  and 𝑥~𝑥 such that 𝑜𝑥𝑚 𝑜~𝑥𝑚\displaystyle\text{ and }x,\tilde{x}\text{ such that }o(x,m)=o(\tilde{x},m). |  | (2.7) |

As so far, ϕitalic-ϕ\phi and θ𝜃\theta are not restricted to be finite-dimensional, this could in principle be assumed without loss of generality, such that ([2.3](#S2.Ex31 "2.3 Ignorability in Maximum Likelihood Estimation ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) is indeed the same as condition ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). In the following, we will assume for simplicity that θ𝜃\theta is finite-dimensional. Let ΩθsubscriptΩ𝜃\Omega\_{\theta} be the space of θ𝜃\theta, ΩϕsubscriptΩitalic-ϕ\Omega\_{\phi} the space of possible ϕitalic-ϕ\phi and Ωθ,ϕsubscriptΩ

𝜃italic-ϕ\Omega\_{\theta,\phi} the joint space of the parameters. The crucial additional condition is that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ωθ,ϕ=Ωθ×Ωϕ.subscriptΩ  𝜃italic-ϕsubscriptΩ𝜃subscriptΩitalic-ϕ\displaystyle\Omega\_{\theta,\phi}=\Omega\_{\theta}\times\Omega\_{\phi}. |  | (2.8) |

This just means that ϕitalic-ϕ\phi is distinct from θ𝜃\theta, so that ℙϕ​(M=m|x)subscriptℙitalic-ϕ𝑀conditional𝑚𝑥{\mathbb{P}}\_{\phi}(M=m|x) does not depend on θ𝜃\theta (Rubin,, [1976](#bib.bib30); Seaman et al.,, [2013](#bib.bib32); Mealli and Rubin,, [2015](#bib.bib21)). In this case, we can rederive the classical ignorability result for MAR in a likelihood context: Consider the likelihood for a pattern m𝑚m,

|  |  |  |
| --- | --- | --- |
|  | ℒ​(θ;o​(x,m))=pθ,ϕ∗​(o​(x,m),M=m)=∫pθ,ϕ∗​(x,M=m)​𝑑oc​(x,m).ℒ  𝜃𝑜𝑥𝑚superscriptsubscript𝑝  𝜃italic-ϕ  𝑜𝑥𝑚𝑀 𝑚superscriptsubscript𝑝  𝜃italic-ϕ  𝑥𝑀 𝑚differential-dsuperscript𝑜𝑐𝑥𝑚\mathcal{L}(\theta;o(x,m))=p\_{\theta,\phi}^{\*}(o(x,m),M=m)=\int p\_{\theta,\phi}^{\*}(x,M=m)do^{c}(x,m). |  |

That is, ℒ​(θ;o​(x,m))ℒ

𝜃𝑜𝑥𝑚\mathcal{L}(\theta;o(x,m)) is the joint density of the observed values with respect to pattern m𝑚m, and M=m𝑀𝑚M=m, seen as a function of θ𝜃\theta. Under ([2.3](#S2.Ex31 "2.3 Ignorability in Maximum Likelihood Estimation ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) it can be checked that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∫pθ,ϕ∗​(x,M=m)​𝑑oc​(x,m)superscriptsubscript𝑝  𝜃italic-ϕ  𝑥𝑀 𝑚differential-dsuperscript𝑜𝑐𝑥𝑚\displaystyle\int p\_{\theta,\phi}^{\*}(x,M=m)do^{c}(x,m) | =ℙϕ​(M=m∣o​(x,m))​pθ∗​(o​(x,m))absentsubscriptℙitalic-ϕ𝑀conditional𝑚𝑜𝑥𝑚superscriptsubscript𝑝𝜃𝑜𝑥𝑚\displaystyle={\mathbb{P}}\_{\phi}(M=m\mid o(x,m))p\_{\theta}^{\*}(o(x,m)) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =c​(o​(x,m))​pθ∗​(o​(x,m)),absent𝑐𝑜𝑥𝑚superscriptsubscript𝑝𝜃𝑜𝑥𝑚\displaystyle=c(o(x,m))p\_{\theta}^{\*}(o(x,m)), |  |

with c​(o​(x,m))𝑐𝑜𝑥𝑚c(o(x,m)) not depending on θ𝜃\theta. Consequently, it is possible to ignore the missingness mechanism (and potential distribution shifts) in a likelihood setting due to (a) the assumption of distinct parameters θ,ϕ

𝜃italic-ϕ\theta,\phi ([2.8](#S2.E8 "In 2.3 Ignorability in Maximum Likelihood Estimation ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) and (b) the nature of maximum likelihood. In particular, even though the distribution pθ,ϕ∗​(o​(x,m),M=m)superscriptsubscript𝑝

𝜃italic-ϕ

𝑜𝑥𝑚𝑀
𝑚p\_{\theta,\phi}^{\*}(o(x,m),M=m) is not the same as the pθ∗​(o​(x,m))superscriptsubscript𝑝𝜃𝑜𝑥𝑚p\_{\theta}^{\*}(o(x,m)), it is *essentially* the same from an MLE perspective: We can therefore simply maximize pθ∗​(o​(x,m))superscriptsubscript𝑝𝜃𝑜𝑥𝑚p\_{\theta}^{\*}(o(x,m)) over θ𝜃\theta to get the MLE. Whether this ignorability holds under MAR is a question of *parametrization*, as we illustrate in Example [2](#Thmexample2 "Example 2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."):

###### Example 3 (Example [2](#Thmexample2 "Example 2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") Continued).

Consider again the setting of Example [2](#Thmexample2 "Example 2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), that is:

|  |  |  |
| --- | --- | --- |
|  | 𝐗=(x1,1x1,2x1,3x2,1N​Ax2,3N​Ax3,2x3,3),𝐌=(000010100)=(m1m2m3).formulae-sequence𝐗matrixsubscript𝑥  11subscript𝑥  12subscript𝑥  13subscript𝑥  21𝑁𝐴subscript𝑥  23𝑁𝐴subscript𝑥  32subscript𝑥  33𝐌matrix000010100matrixsubscript𝑚1subscript𝑚2subscript𝑚3\displaystyle\mathbf{X}=\begin{pmatrix}x\_{1,1}&x\_{1,2}&x\_{1,3}\\ x\_{2,1}&NA&x\_{2,3}\\ NA&x\_{3,2}&x\_{3,3}\\ \end{pmatrix},\mathbf{M}=\begin{pmatrix}0&0&0\\ 0&1&0\\ 1&0&0\\ \end{pmatrix}=\begin{pmatrix}m\_{1}\\ m\_{2}\\ m\_{3}\end{pmatrix}. |  |

whereby (X1,X2,X3)subscript𝑋1subscript𝑋2subscript𝑋3(X\_{1},X\_{2},X\_{3}) are uniformly distributed on [0,1]01[0,1] and

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℙ​(M=m1∣x)ℙ𝑀conditionalsubscript𝑚1𝑥\displaystyle{\mathbb{P}}(M=m\_{1}\mid x) | =ℙ​(M=m1∣x1)=x1/3absentℙ𝑀conditionalsubscript𝑚1subscript𝑥1subscript𝑥13\displaystyle={\mathbb{P}}(M=m\_{1}\mid x\_{1})=x\_{1}/3 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ℙ​(M=m2∣x)ℙ𝑀conditionalsubscript𝑚2𝑥\displaystyle{\mathbb{P}}(M=m\_{2}\mid x) | =ℙ​(M=m2∣x1)=2/3−x1/3absentℙ𝑀conditionalsubscript𝑚2subscript𝑥123subscript𝑥13\displaystyle={\mathbb{P}}(M=m\_{2}\mid x\_{1})=2/3-x\_{1}/3 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ℙ​(M=m3∣x)ℙ𝑀conditionalsubscript𝑚3𝑥\displaystyle{\mathbb{P}}(M=m\_{3}\mid x) | =ℙ​(M=m3)=1/3.absentℙ𝑀subscript𝑚313\displaystyle={\mathbb{P}}(M=m\_{3})=1/3. |  |

Now assume that the parameter of interest is the upper boundary of x1subscript𝑥1x\_{1}, such that X1subscript𝑋1X\_{1} is uniform on [0,θ]0𝜃[0,\theta]. As ℙ​(M=mi∣x)ℙ𝑀conditionalsubscript𝑚𝑖𝑥{\mathbb{P}}(M=m\_{i}\mid x) does not change, it follows that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | pθ,ϕ∗​(x1,x2,x3,M=m1)=ℙ​(M=m1∣x1)​pθ​(x1,x2,x3)=x13​pθ∗​(x1,x2,x3).subscriptsuperscript𝑝  𝜃italic-ϕ  subscript𝑥1subscript𝑥2subscript𝑥3𝑀 subscript𝑚1ℙ𝑀conditionalsubscript𝑚1subscript𝑥1subscript𝑝𝜃subscript𝑥1subscript𝑥2subscript𝑥3subscript𝑥13subscriptsuperscript𝑝𝜃subscript𝑥1subscript𝑥2subscript𝑥3\displaystyle p^{\*}\_{\theta,\phi}(x\_{1},x\_{2},x\_{3},M=m\_{1})={\mathbb{P}}(M=m\_{1}\mid x\_{1})p\_{\theta}(x\_{1},x\_{2},x\_{3})=\frac{x\_{1}}{3}p^{\*}\_{\theta}(x\_{1},x\_{2},x\_{3}). |  | (2.9) |

Thus for optimization purposes, maximizing pθ,ϕ∗​(x1,x2,x3,M=m1)subscriptsuperscript𝑝

𝜃italic-ϕ

subscript𝑥1subscript𝑥2subscript𝑥3𝑀
subscript𝑚1p^{\*}\_{\theta,\phi}(x\_{1},x\_{2},x\_{3},M=m\_{1}) over θ𝜃\theta is equivalent to maximizing pθ∗​(x1,x2,x3)subscriptsuperscript𝑝𝜃subscript𝑥1subscript𝑥2subscript𝑥3p^{\*}\_{\theta}(x\_{1},x\_{2},x\_{3}) over θ𝜃\theta. In particular, being able to identify θ𝜃\theta allows to identify pθ∗​(x1∣x2,x3)subscriptsuperscript𝑝𝜃conditionalsubscript𝑥1

subscript𝑥2subscript𝑥3p^{\*}\_{\theta}(x\_{1}\mid x\_{2},x\_{3}) and thus to impute x1subscript𝑥1x\_{1}. This, despite the fact that

|  |  |  |
| --- | --- | --- |
|  | pθ∗​(x1,x2,x3,M=m1)=x13​pθ∗​(x1,x2,x3)≠pθ∗​(x1,x2,x3).superscriptsubscript𝑝𝜃  subscript𝑥1subscript𝑥2subscript𝑥3𝑀 subscript𝑚1subscript𝑥13superscriptsubscript𝑝𝜃subscript𝑥1subscript𝑥2subscript𝑥3superscriptsubscript𝑝𝜃subscript𝑥1subscript𝑥2subscript𝑥3\displaystyle p\_{\theta}^{\*}(x\_{1},x\_{2},x\_{3},M=m\_{1})=\frac{x\_{1}}{3}p\_{\theta}^{\*}(x\_{1},x\_{2},x\_{3})\neq p\_{\theta}^{\*}(x\_{1},x\_{2},x\_{3}). |  |

Having obtained θ𝜃\theta, it is then possible to impute X1subscript𝑋1X\_{1} in the third patterns by drawing from pθ∗​(x1∣x2,x3)superscriptsubscript𝑝𝜃conditionalsubscript𝑥1

subscript𝑥2subscript𝑥3p\_{\theta}^{\*}(x\_{1}\mid x\_{2},x\_{3}). However, notice that this is not the same as saying that θ𝜃\theta can be recovered from only looking at the first pattern m1subscript𝑚1m\_{1}. Indeed in this case:

|  |  |  |  |
| --- | --- | --- | --- |
|  | pθ,ϕ∗​(x1,x2,x3∣M=m1)=ℙ​(M=m1∣x1)ℙ​(M=m1)​pθ​(x1,x2,x3)=x1θ​pθ​(x1,x2,x3),subscriptsuperscript𝑝  𝜃italic-ϕ  subscript𝑥1subscript𝑥2conditionalsubscript𝑥3𝑀 subscript𝑚1ℙ𝑀conditionalsubscript𝑚1subscript𝑥1ℙ𝑀subscript𝑚1subscript𝑝𝜃subscript𝑥1subscript𝑥2subscript𝑥3subscript𝑥1𝜃subscript𝑝𝜃subscript𝑥1subscript𝑥2subscript𝑥3\displaystyle p^{\*}\_{\theta,\phi}(x\_{1},x\_{2},x\_{3}\mid M=m\_{1})=\frac{{\mathbb{P}}(M=m\_{1}\mid x\_{1})}{{\mathbb{P}}(M=m\_{1})}p\_{\theta}(x\_{1},x\_{2},x\_{3})=\frac{x\_{1}}{\theta}p\_{\theta}(x\_{1},x\_{2},x\_{3}), |  | (2.10) |

as P​(M=m1)=θ/3𝑃𝑀subscript𝑚1𝜃3P(M=m\_{1})=\theta/3. Thus maximizing pθ,ϕ∗​(x1,x2,x3∣M=m1)subscriptsuperscript𝑝

𝜃italic-ϕ

subscript𝑥1subscript𝑥2conditionalsubscript𝑥3𝑀
subscript𝑚1p^{\*}\_{\theta,\phi}(x\_{1},x\_{2},x\_{3}\mid M=m\_{1}) is not equivalent to maximizing pθ∗​(x1,x2,x3)subscriptsuperscript𝑝𝜃subscript𝑥1subscript𝑥2subscript𝑥3p^{\*}\_{\theta}(x\_{1},x\_{2},x\_{3}). On the flipside, if one changes ℙ​(M=m1∣x1)ℙ𝑀conditionalsubscript𝑚1subscript𝑥1{\mathbb{P}}(M=m\_{1}\mid x\_{1}) to x1/3​θsubscript𝑥13𝜃x\_{1}/3\theta, violating ([2.8](#S2.E8 "In 2.3 Ignorability in Maximum Likelihood Estimation ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")), maximizing pθ,ϕ​(x1,x2,x3,M=m1)subscript𝑝

𝜃italic-ϕ

subscript𝑥1subscript𝑥2subscript𝑥3𝑀
subscript𝑚1p\_{\theta,\phi}(x\_{1},x\_{2},x\_{3},M=m\_{1}) will not recover θ𝜃\theta.

## 3 Requirements for Imputation Methods

We have seen that both conditional as well as marginal distribution shifts can occur for different patterns under MAR. However, conditional shifts can be disregarded when using a sequential approach (i.e. MICE), as, for a variable Xjsubscript𝑋𝑗X\_{j}, considering all patterns m𝑚m in which Xjsubscript𝑋𝑗X\_{j} is missing identifies the right conditional distribution. Nonetheless, marginal distribution shifts, such as in Example [1](#Thmexample1 "Example 1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), can still occur. In particular, a successful imputation method needs to be able to deal with distributional shifts in the observed variables. Moreover, in practice, an estimation method in the FCS framework should be able to estimate the potentially complex distribution of Xj∣X−jconditionalsubscript𝑋𝑗subscript𝑋𝑗X\_{j}\mid X\_{-j} as accurately as possible.

The above considerations thus suggest desirable properties an imputation method should meet in an FCS framework: It should

* (1)

  be a distributional regression method,
* (2)

  be able to capture nonlinearities and interactions in the data,
* (3)

  be fast to fit,
* (4)

  be able to deal with distributional shifts in the observed variables.

Moreover, a helpful property to allow to use the FCS approach in high dimensions is if

* (5)

  the method is able to deal with multivariate responses.

missForest which was shown to be extremely successful in terms of RMSE in various benchmarking analyses, only meets (2) and (3). In particular, random forest imputation such as missForest was deemed more successful than GAN-based methods in Jäger et al., ([2021](#bib.bib14)). However, a more appropriate scoring will very likely reverse these insights, ranking GAIN and similar methods higher than missForest.333Though our experiments in Appendix [B](#A2 "Appendix B Comparison of MICE to GAIN and MIWAE ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") suggest otherwise. On the other hand, Wang et al., ([2022](#bib.bib43)) which provides a careful benchmarking of imputation methods more in line with this paper, finds that mice-cart and mice-RF (Burgette and Reiter,, [2010](#bib.bib3); Doove et al.,, [2014](#bib.bib7)) are more successful than GAIN. These methods use one or several trees respectively, but sample from the leaves to obtain the imputation, approximating draws from the conditional distribution to approximate (1). Similarly, Näf et al., ([2023](#bib.bib26)) find mice-cart/RF to be extremely successful imputation methods. As such, they could be combining the best of both worlds; inheriting the accuracy of missForest, while providing draws from the conditional distribution. However, they are ultimately not designed for the task of distributional regression. Thus a forest-based distributional method such as DRF of Ćevid et al., ([2022](#bib.bib4)) might even attain better results and indeed meets (1)–(3). Moreover, DRF is designed to handle multivariate outputs and thus also meets (5). This makes the method accessible to high-dimensional datasets, as MICE can be used in blocks as described in van Buuren, ([2018](#bib.bib40), Chapter 4.7). We implemented this option in our new mice-DRF R function. Thus if d=1000𝑑1000d=1000, one might define blocks of size 100 and in each pass, train DRF by regression a 100 variables on the remaining 900. This would reduce the number of passes in each iteration from 1000 to 10. We thus implement the following routine in mice: For each j𝑗j, fit a DRF regressing the observed xi,jsubscript𝑥

𝑖𝑗x\_{i,j} onto xi,−jsubscript𝑥

𝑖𝑗x\_{i,-j} to obtain an estimate of the conditional distribution, given by forest-induced weights. For each unobserved xi,jsubscript𝑥

𝑖𝑗x\_{i,j}, we predict the weights based on xi,−jsubscript𝑥

𝑖𝑗x\_{i,-j} and draw from the observed set according to those weights. This is essentially the mice-RF implementation described in Doove et al., ([2014](#bib.bib7)), with the traditional Random Forest exchanged by the Distributional Random Forest.

However, as a forest-based method, DRF still generalizes poorly outside of the training set, i.e. Requirement (4) is not met. Figure [3](#S3.F3 "Figure 3 ‣ 3 Requirements for Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") illustrates the behavior of different imputation strategies for Example [1](#Thmexample1 "Example 1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). First, the Gaussian imputation simply fits a regression in pattern m1subscript𝑚1m\_{1} and then draws from a conditional Gaussian distribution given the estimated parameters. As such it is the ideal method in this setting and serves as an illustration that the data can be correctly imputed. For the nonparametric methods, DRF, as a distributional method, performs better than mice-RF. However, it still fails to deal with the covariate shift, centering around 2, when it should center around 5.

Thus, while previous analysis indicates that forest-based methods such as mice-cart, mice-RF, and likely also mice-DRF might be some of the most successful methods currently available, and in particular will likely beat GAN-based methods such as GAIN, finding an imputation method that (approximately) meets (1)–(5) is still an open problem.

Finally, the above list overlaps with and complements the three points mentioned in Murray, ([2018](#bib.bib23), Section 4) for general imputation methods:

* (1’)

  Imputations should reflect uncertainty about missing values and about the imputation
  model.
* (2’)

  Imputation models should generally include as many variables as possible.
* (3’)

  Imputation models should be as flexible as possible.

The first part of (1’) corresponds to (1) of our list; instead of providing the *best* value for imputation, one should draw from the right conditional distribution to impute, such that the underlying distribution is replicated. To reiterate this, Figure [4](#S3.F4 "Figure 4 ‣ 3 Requirements for Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") shows a small example. However, as we note in Section [6](#S6 "6 Discussion ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), the second part, that the uncertainty of the imputation model should be considered as well, is not met by the imputation methods we present here and is an open problem for nonparametric imputation. While this gets less consequential in large samples, this additional uncertainty is needed for reliable uncertainty quantification with multiple imputation. Point (2’) is not relevant to our discussion, while (3’) coincides with (2) above.

![Refer to caption](/html/2403.19196/assets/Example2.png)


Figure 3: The true distribution against a draw from different imputation procedures for imputing X1subscript𝑋1X\_{1} in Example [1](#Thmexample1 "Example 1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.").

![Refer to caption](/html/2403.19196/assets/MotivationExample.png)


Figure 4: 5000 observations of a bivariate Gaussian Example with around 50% MCAR missing values in X1subscript𝑋1X\_{1}. Left: Imputation by fitting a regression model and imputing the prediction, Right: Imputation by fitting a regression model and imputing by drawing from a conditional Gaussian distribution. Parameters calculated with the regression imputation tend to have a large bias, more so than if only complete-case analysis is used.

## 4 Assessing Imputation Methods

We now turn to the question of how to find the best out of several imputations. First, the above discussion suggests that in academic scenarios, where the true underlying values are available, distributional distances or scores should be used to evaluate imputation methods. We will in the following use the (negative) energy distance between imputed and real data:

|  |  |  |
| --- | --- | --- |
|  | d​(H,P∗)=2​𝔼​[‖X−Y‖ℝd]−𝔼​[‖X−X′‖ℝd]−𝔼​[‖Y−Y′‖ℝd],𝑑𝐻superscript𝑃2𝔼delimited-[]subscriptnorm𝑋𝑌superscriptℝ𝑑𝔼delimited-[]subscriptnorm𝑋superscript𝑋′superscriptℝ𝑑𝔼delimited-[]subscriptnorm𝑌superscript𝑌′superscriptℝ𝑑\displaystyle d(H,P^{\*})=2{\mathbb{E}}[\|X-Y\|\_{{\mathbb{R}}^{d}}]-{\mathbb{E}}[\|X-X^{\prime}\|\_{{\mathbb{R}}^{d}}]-{\mathbb{E}}[\|Y-Y^{\prime}\|\_{{\mathbb{R}}^{d}}], |  |

where ∥⋅∥ℝd\|\cdot\|\_{{\mathbb{R}}^{d}} is the Euclidean metric on ℝdsuperscriptℝ𝑑{\mathbb{R}}^{d}, X∼Hsimilar-to𝑋𝐻X\sim H, Y∼P∗similar-to𝑌superscript𝑃Y\sim P^{\*} and X′,Y′

superscript𝑋′superscript𝑌′X^{\prime},Y^{\prime} are independent copies of X𝑋X and Y𝑌Y. The energy distance is directly related to the energy score (Gneiting and Raftery,, [2007](#bib.bib10); Gneiting et al.,, [2008](#bib.bib11)):

|  |  |  |  |
| --- | --- | --- | --- |
|  | e​s​(H,y)=12​𝔼​[‖X−X′‖ℝd]−𝔼​[‖X−y‖ℝd],𝑒𝑠𝐻𝑦12𝔼delimited-[]subscriptnorm𝑋superscript𝑋′superscriptℝ𝑑𝔼delimited-[]subscriptnorm𝑋𝑦superscriptℝ𝑑\displaystyle es(H,y)=\frac{1}{2}{\mathbb{E}}[\|X-X^{\prime}\|\_{{\mathbb{R}}^{d}}]-{\mathbb{E}}[\|X-y\|\_{{\mathbb{R}}^{d}}], |  | (4.1) |

where X∼Hsimilar-to𝑋𝐻X\sim H and X′∼Hsimilar-tosuperscript𝑋′𝐻X^{\prime}\sim H is an independent copy. Let S​(H,P∗)=𝔼​[e​s​(H,Y)]𝑆𝐻superscript𝑃𝔼delimited-[]𝑒𝑠𝐻𝑌S(H,P^{\*})={\mathbb{E}}[es(H,Y)], where the expectation is taken over Y∼P∗similar-to𝑌superscript𝑃Y\sim P^{\*}. Gneiting and Raftery, ([2007](#bib.bib10)) showed that

|  |  |  |  |
| --- | --- | --- | --- |
|  | S​(H,P∗)≤S​(P∗,P∗),𝑆𝐻superscript𝑃𝑆superscript𝑃superscript𝑃\displaystyle S(H,P^{\*})\leq S(P^{\*},P^{\*}), |  | (4.2) |

i.e. S𝑆S is proper in the traditional sense. That is, if we predict the distribution P∗superscript𝑃P^{\*}, and the “test data” y𝑦y are indeed drawn from P∗superscript𝑃P^{\*}, taking the average over a “large” number of y𝑦y will lead to the maximal value. We will make use of the energy score to create a reliable ranking method when the underlying data are not available. To this end, we consider the I-Scores framework of Näf et al., ([2023](#bib.bib26)):

###### Definition 4.1 (Definition 4.1 in Näf et al., ([2023](#bib.bib26))).

A real-valued function SN​A​(H,P)subscript𝑆𝑁𝐴𝐻𝑃S\_{NA}(H,P)
is a proper I-Score iff

|  |  |  |
| --- | --- | --- |
|  | SN​A​(H,P)≤SN​A​(P∗,P),subscript𝑆𝑁𝐴𝐻𝑃subscript𝑆𝑁𝐴superscript𝑃𝑃S\_{NA}(H,P)\leq S\_{NA}(P^{\*},P), |  |

for any imputation distribution H∈ℋP𝐻subscriptℋ𝑃H\in\mathcal{H}\_{P}. It is strictly proper iff the inequality is strict for H≠P∗𝐻superscript𝑃H\neq P^{\*}.

The key is that we would like to score P∗superscript𝑃P^{\*}, when only samples from P𝑃P are available.

Näf et al., ([2023](#bib.bib26)) developed a first I-Score using Density Ratios and random projections A⊂{1,…,d}𝐴1…𝑑A\subset\{1,\ldots,d\}. The score was shown to be proper under ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). In Appendix [C.1](#A3.SS1 "C.1 DR-I-Score is not proper under MAR ‣ Appendix C Proofs and Additional Results ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") we show that it is however not proper under MAR. Following the arguments in this paper, we now develop a score that is not only easier to use but also proper under MAR, without any projections. However, it necessitates that there is at least one variable that is always observed, or that O𝑂O defined in ([2.5](#S2.E5 "In 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) is not empty. Adapting the proof of Proposition [2.3](#S2.Thmproposition3 "Proposition 2.3. ‣ 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), the perfect imputation method learns the distribution,

|  |  |  |
| --- | --- | --- |
|  | h∗​(xj∣xO)superscriptℎconditionalsubscript𝑥𝑗subscript𝑥𝑂\displaystyle h^{\*}(x\_{j}\mid x\_{O}) |  |
|  |  |  |
| --- | --- | --- |
|  | =∑m∈Ljℙ​(M=m)∑m∈Ljp∗​(xO∣M=m)​ℙ​(M=m)​p∗​(xj,xO∣M=m),absentsubscript𝑚subscript𝐿𝑗ℙ𝑀𝑚subscript𝑚subscript𝐿𝑗superscript𝑝conditionalsubscript𝑥𝑂𝑀𝑚ℙ𝑀𝑚superscript𝑝  subscript𝑥𝑗conditionalsubscript𝑥𝑂𝑀 𝑚\displaystyle=\sum\_{m\in L\_{j}}\frac{{\mathbb{P}}(M=m)}{\sum\_{m\in L\_{j}}p^{\*}(x\_{O}\mid M=m){\mathbb{P}}(M=m)}p^{\*}(x\_{j},x\_{O}\mid M=m), |  |

which is simply the conditional distribution of xj∣xOconditionalsubscript𝑥𝑗subscript𝑥𝑂x\_{j}\mid x\_{O} learned from all patterns in which xjsubscript𝑥𝑗x\_{j} is not missing. Consequently, 𝔼​[e​s​(HXj∣xO,Y)],𝔼delimited-[]𝑒𝑠subscript𝐻conditionalsubscript𝑋𝑗subscript𝑥𝑂𝑌{\mathbb{E}}[es(H\_{X\_{j}\mid x\_{O}},Y)],
with the integration taken over Y∼HXj∣xO∗similar-to𝑌subscriptsuperscript𝐻conditionalsubscript𝑋𝑗subscript𝑥𝑂Y\sim H^{\*}\_{X\_{j}\mid x\_{O}}, is maximal when h​(xj∣xO)=h∗​(xj∣xO)ℎconditionalsubscript𝑥𝑗subscript𝑥𝑂superscriptℎconditionalsubscript𝑥𝑗subscript𝑥𝑂h(x\_{j}\mid x\_{O})=h^{\*}(x\_{j}\mid x\_{O}) by propriety of the energy score. We then define the score of variable j𝑗j as

|  |  |  |  |
| --- | --- | --- | --- |
|  | SN​Aj​(H,P)=𝔼​[𝔼​[e​s​(HXj∣XO,Y)]],subscriptsuperscript𝑆𝑗𝑁𝐴𝐻𝑃𝔼delimited-[]𝔼delimited-[]𝑒𝑠subscript𝐻conditionalsubscript𝑋𝑗subscript𝑋𝑂𝑌\displaystyle S^{j}\_{NA}(H,P)={\mathbb{E}}[{\mathbb{E}}[es(H\_{X\_{j}\mid X\_{O}},Y)]], |  | (4.3) |

where the outer expectation is taken over XO∼PO∗similar-tosubscript𝑋𝑂subscriptsuperscript𝑃𝑂X\_{O}\sim P^{\*}\_{O}, the distribution of all fully observed variables. Usually, in the scoring literature, one only considers the inner expectation, even though in practice “scores are reported as averages over comparable sets of probabilistic forecasts” (Gneiting et al.,, [2008](#bib.bib11), page 222). We thus also consider the outer expectation to model the different test points. Finally, the full score is given as

|  |  |  |
| --- | --- | --- |
|  | SN​Ae​s​(H,P)=1|Oc|​∑j∈OcSN​Aj​(H,P),superscriptsubscript𝑆𝑁𝐴𝑒𝑠𝐻𝑃1superscript𝑂𝑐subscript𝑗superscript𝑂𝑐superscriptsubscript𝑆𝑁𝐴𝑗𝐻𝑃\displaystyle S\_{NA}^{es}(H,P)=\frac{1}{|O^{c}|}\sum\_{j\in O^{c}}S\_{NA}^{j}(H,P), |  |

whereby Ocsuperscript𝑂𝑐O^{c} is the complement of O𝑂O, i.e. the set of all variables with at least one missing element. Since by Proposition [2.3](#S2.Thmproposition3 "Proposition 2.3. ‣ 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), h∗​(xj∣x−j)=p∗​(xj∣x−j)superscriptℎconditionalsubscript𝑥𝑗subscript𝑥𝑗superscript𝑝conditionalsubscript𝑥𝑗subscript𝑥𝑗h^{\*}(x\_{j}\mid x\_{-j})=p^{\*}(x\_{j}\mid x\_{-j}), for all x−jsubscript𝑥𝑗x\_{-j} with p−j∗​(x−j)>0subscriptsuperscript𝑝𝑗subscript𝑥𝑗0p^{\*}\_{-j}(x\_{-j})>0, we obtain:

###### Proposition 4.1.

Assume MAR in ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) holds and that O𝑂O is not empty. Then SN​Ae​s​(H,P)superscriptsubscript𝑆𝑁𝐴𝑒𝑠𝐻𝑃S\_{NA}^{es}(H,P) is a proper I-Score.

In practice, we propose the following two approximations to this approach: Consider a dimension j∉O𝑗𝑂j\notin O and recall that Ljsubscript𝐿𝑗L\_{j} collects all patterns m𝑚m, such that mj=0subscript𝑚𝑗0m\_{j}=0. For each observed (xi,j)mi∈Ljsubscriptsubscript𝑥

𝑖𝑗subscript𝑚𝑖subscript𝐿𝑗(x\_{i,j})\_{m\_{i}\in L\_{j}}, we assume to have a sample of N𝑁N points, say (X~l(i))superscriptsubscript~𝑋𝑙𝑖(\tilde{X}\_{l}^{(i)}), l=1,…,N𝑙

1…𝑁l=1,\ldots,N, approximately generated from HXj∣xi,Osubscript𝐻conditionalsubscript𝑋𝑗subscript𝑥

𝑖𝑂H\_{X\_{j}\mid x\_{i,O}}. This can be used to estimate SN​Aj​(H,P)subscriptsuperscript𝑆𝑗𝑁𝐴𝐻𝑃S^{j}\_{NA}(H,P), as

|  |  |  |  |
| --- | --- | --- | --- |
|  | S^N​Aj​(H,P)=1|i:mi∈Lj|​∑i:mi∈Lj(12​N2​∑l=1N∑ℓ=1N‖X~l(i)−X~ℓ(i)‖ℝ−1N​∑l=1N‖X~l(i)−xi,j‖ℝ),\displaystyle\hat{S}^{j}\_{NA}(H,P)=\frac{1}{|{i:m\_{i}\in L\_{j}}|}\sum\_{{i:m\_{i}\in L\_{j}}}\left(\frac{1}{2N^{2}}\sum\_{l=1}^{N}\sum\_{\ell=1}^{N}\|\tilde{X}\_{l}^{(i)}-\tilde{X}\_{\ell}^{(i)}\|\_{{\mathbb{R}}}-\frac{1}{N}\sum\_{l=1}^{N}\|\tilde{X}\_{l}^{(i)}-x\_{i,j}\|\_{{\mathbb{R}}}\right), |  | (4.4) |

as in Gneiting et al., ([2008](#bib.bib11), Equation (7)). Thus the observed points of Xjsubscript𝑋𝑗X\_{j} act as the “test points” for the predicted distribution HXj∣XOsubscript𝐻conditionalsubscript𝑋𝑗subscript𝑋𝑂H\_{X\_{j}\mid X\_{O}}. The final score is then given as

|  |  |  |  |
| --- | --- | --- | --- |
|  | S^N​Ae​s​(H,P)=1|Oc|​∑j∈OcS^N​Aj​(H,P).superscriptsubscript^𝑆𝑁𝐴𝑒𝑠𝐻𝑃1superscript𝑂𝑐subscript𝑗superscript𝑂𝑐superscriptsubscript^𝑆𝑁𝐴𝑗𝐻𝑃\displaystyle\hat{S}\_{NA}^{es}(H,P)=\frac{1}{|O^{c}|}\sum\_{j\in O^{c}}\hat{S}\_{NA}^{j}(H,P). |  | (4.5) |

![Refer to caption](/html/2403.19196/assets/scoreillustrationO.png)


Figure 5: Conceptual illustration of the score approximation.

###### Remark.

Formally, always observed observations are needed to ensure the test points xi,jsubscript𝑥

𝑖𝑗x\_{i,j} are truly sampled from h∗​(xj∣xO)superscriptℎconditionalsubscript𝑥𝑗subscript𝑥𝑂h^{\*}(x\_{j}\mid x\_{O}), which in turn is equal to p∗​(xj∣xO)superscript𝑝conditionalsubscript𝑥𝑗subscript𝑥𝑂p^{\*}(x\_{j}\mid x\_{O}). While these points are observed, their marginal distribution is fixed to p∗​(xj)superscript𝑝subscript𝑥𝑗p^{\*}(x\_{j}), but since x−jsubscript𝑥𝑗x\_{-j} is imputed and thus drawn from H−jsubscript𝐻𝑗H\_{-j}, relative to the imputed point xi,−jsubscript𝑥

𝑖𝑗x\_{i,-j}, the test point xi,jsubscript𝑥

𝑖𝑗x\_{i,j} might not be sampled from the right distribution p∗​(xj∣x−j)superscript𝑝conditionalsubscript𝑥𝑗subscript𝑥𝑗p^{\*}(x\_{j}\mid x\_{-j}). Appendix [A](#A1 "Appendix A Score Version without Fully Observed Data ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") presents an informal argument, indicating that in general, the score might not be proper if all variables xi,−jsubscript𝑥

𝑖𝑗x\_{i,-j} instead of xi,Osubscript𝑥

𝑖𝑂x\_{i,O} are used. Nonetheless, another version of the score is presented there, based on X−jsubscript𝑋𝑗X\_{-j} instead of XOsubscript𝑋𝑂X\_{O}. Though it remains an open problem under which conditions this score can be proven to be proper, it works remarkably well empirically.

We now detail how we obtain (X~l(i))superscriptsubscript~𝑋𝑙𝑖(\tilde{X}\_{l}^{(i)}), l=1,…,N𝑙

1…𝑁l=1,\ldots,N. Given an imputed data set and the imputation function itself, we subset and concatenate the imputed points and observed points xi,jsubscript𝑥

𝑖𝑗x\_{i,j} of j𝑗j and the fully observed points xi,Osubscript𝑥

𝑖𝑂x\_{i,O}, i=1,…,n𝑖

1…𝑛i=1,\ldots,n. In this new data set, we keep the imputed points, that is all Xi,jsubscript𝑋

𝑖𝑗X\_{i,j} with mi∈Ljcsubscript𝑚𝑖superscriptsubscript𝐿𝑗𝑐m\_{i}\in L\_{j}^{c} are still drawn from H𝐻H, while we set the *observed* observations of Xjsubscript𝑋𝑗X\_{j} to missing, i.e. Xi,j=NAsubscript𝑋

𝑖𝑗NAX\_{i,j}=\texttt{NA} for i𝑖i with mi∈Ljsubscript𝑚𝑖subscript𝐿𝑗m\_{i}\in L\_{j}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | (NA(xi,O)mi∈Lj(xi,j)mi∈Ljc(xi,O)mi∈Ljc)matrixNAsubscriptsubscript𝑥  𝑖𝑂subscript𝑚𝑖subscript𝐿𝑗subscriptsubscript𝑥  𝑖𝑗subscript𝑚𝑖superscriptsubscript𝐿𝑗𝑐subscriptsubscript𝑥  𝑖𝑂subscript𝑚𝑖superscriptsubscript𝐿𝑗𝑐\displaystyle\begin{pmatrix}\texttt{NA}&(x\_{i,O})\_{m\_{i}\in L\_{j}}\\ (x\_{i,j})\_{m\_{i}\in L\_{j}^{c}}&(x\_{i,O})\_{m\_{i}\in L\_{j}^{c}}\end{pmatrix} |  | (4.6) |

Then we approximate the sampling from HXj∣xi,Osubscript𝐻conditionalsubscript𝑋𝑗subscript𝑥

𝑖𝑂H\_{X\_{j}\mid x\_{i,O}} in two ways:

* (1)

  Regress (xi,j)mi∈Ljcsubscriptsubscript𝑥
  𝑖𝑗subscript𝑚𝑖superscriptsubscript𝐿𝑗𝑐(x\_{i,j})\_{m\_{i}\in L\_{j}^{c}} onto (xi,O)mi∈Ljcsubscriptsubscript𝑥
  𝑖𝑂subscript𝑚𝑖superscriptsubscript𝐿𝑗𝑐(x\_{i,O})\_{m\_{i}\in L\_{j}^{c}} in ([4.6](#S4.E6 "In 4 Assessing Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) using DRF. Then for each test point xi,Osubscript𝑥
  𝑖𝑂x\_{i,O}, mi∈Ljsubscript𝑚𝑖subscript𝐿𝑗m\_{i}\in L\_{j}, sample N𝑁N times from the estimated conditional distribution obtained from DRF.
* (2)

  Impute the NA values in ([4.6](#S4.E6 "In 4 Assessing Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) with H𝐻H, N𝑁N times.

We refer to the first approach as drf-I-Score, and to the second as m𝑚m-I-Score. The idea in both cases is to use XOsubscript𝑋𝑂X\_{O} and the imputation of Xjsubscript𝑋𝑗X\_{j} to generate a sample from the distribution HXj∣XOsubscript𝐻conditionalsubscript𝑋𝑗subscript𝑋𝑂H\_{X\_{j}\mid X\_{O}} for *points that are already observed*. Note that while the drf-I-Score does this by utilizing the sampling of DRF, the m𝑚m-I-Score uses the ability of the imputation method itself to generate samples. Thus, while the drf-I-Score can be averaged over several imputations to score multiple imputations, the m𝑚m-I-Score scores multiple imputation naturally.

As a downside, the m𝑚m-I-Score can be computationally demanding, as N𝑁N should be chosen high, say at least 50 to give an accurate score. This would be infeasible for realistic dimensions if the full data set had to be imputed. However note that in step (2), by construction only one variable has missing values, while all the others are observed. This means that only one pass is needed to impute, which essentially corresponds to fitting the chosen model (e.g., RF) once.

## 5 Empirical Study

The goal of this section is to illustrate the concepts discussed in this paper on both simulated and real data, including the performance of the new score. We employ the FCS methods discussed above, namely mice-cart and the new mice-DRF, missForest, as well as regression and Gaussian imputations used in the previous section. Both fit a regression to the observed data to obtain the regression parameters. The regression imputation then simply imputes by predicting from the linear regression model, while Gaussian imputation uses the prediction as the mean of a Gaussian distribution from which it draws imputed values. However, in the following, we will follow the naming guideline of the R-package mice (van Buuren and Groothuis-Oudshoorn,, [2011](#bib.bib41)) and refer to the regression imputation as mice-norm.predict and to the Gaussian imputation as mice-norm.nob. If a method requires the specification of parameters, we use the default values. To evaluate the imputation methods we calculate the (negative) energy distance between the true and imputed data sets, using the energy R-package (Rizzo and Szekely,, [2022](#bib.bib29)). As this “score” is able to access the true underlying values, we will refer to it as the full information score. We compare the orderings of the full information score with the drf- and m𝑚m-I-Score, which do not have access to the values underlying the missing values. The only hyperparameter to choose in this case is the number of samples N𝑁N, which we will set to N=100𝑁100N=100. Finally, though we focus on FCS imputation methods here, we also add a comparison to GAIN (Yoon et al.,, [2018](#bib.bib45)) and MIWAE (Mattei and Frellsen,, [2019](#bib.bib20)), in terms of the full information score in the Appendix [B](#A2 "Appendix B Comparison of MICE to GAIN and MIWAE ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.").

The three examples considered in this section, as well as the analysis in Appendices [A](#A1 "Appendix A Score Version without Fully Observed Data ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), [B](#A2 "Appendix B Comparison of MICE to GAIN and MIWAE ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") and further tests that are not shown here, indicate that:

* (I)

  For the methods and data sets considered here mice-DRF and mice-cart are the most promising methods. This aligns with the findings in Wang et al., ([2022](#bib.bib43)); Näf et al., ([2023](#bib.bib26)). In particular, they tend to perform stronger than missForest, GAIN, and MIWAE.
* (II)

  However, none of the methods is able to reliably deal with distributional shifts and nonlinearity, showing once again that better imputation methods need to be found.
* (III)

  Contrary to our speculation in Section [3](#S3 "3 Requirements for Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), Appendix [B](#A2 "Appendix B Comparison of MICE to GAIN and MIWAE ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") indicates that GAIN and MIWAE do not beat missForest, even in terms of negative energy distance. However, it remains to be seen whether this changes for higher dimensional data sets.
* (IV)

  The ordering of the m𝑚m-I-Score is quite sensible and similar to the one of the full information score, even in the first challenging distributional shift example in Section [5.2](#S5.SS2 "5.2 Gaussian Mixture Model ‣ 5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). If differences arise, it is often because the m𝑚m-I-Score penalizes methods that cannot produce multiple imputations. Given the discussion in this paper, this might be desirable. An exception is the third example in Section [5.3](#S5.SS3 "5.3 Mixture Model with Nonlinear Relationships ‣ 5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") where none of the methods perform well. Here the scores disagree quite heavily.
* (V)

  Remarkably, the score using the full data X−jsubscript𝑋𝑗X\_{-j} in Appendix [A](#A1 "Appendix A Score Version without Fully Observed Data ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") appears to work as well as the one using XOsubscript𝑋𝑂X\_{O}.

### 5.1 Air Quality Data

We start with the air quality data set obtained from <https://github.com/lorismichel/drf/tree/master/applications/air_data/data/datasets/air_data_benchmark2.Rdata>. This is a preprocessed version of the data set that was originally obtained from the website of the Environmental Protection Agency website (<https://aqs.epa.gov/aqsweb/airdata/download_files.html>). For a detailed description of the data set, we refer to Ćevid et al., ([2022](#bib.bib4), Appendix C.1). The data set contains a total of 50’000 observations with 11 dimensions.

The goal of this example is to consider a real dataset with MAR missing values generated with an established procedure. We use the “ampute” function of the mice package (van Buuren and Groothuis-Oudshoorn,, [2011](#bib.bib41)) to introduce MAR missingness into the first four numerical variables. The ampute function presents a flexible way of introducing missingness according to a desired mechanism, based on Rianne Margaretha Schouten and Vink, ([2018](#bib.bib28)). We specify the 4 patterns

|  |  |  |  |
| --- | --- | --- | --- |
|  | m1subscript𝑚1\displaystyle m\_{1} | =(1,0,0,0,…,0)absent1000…0\displaystyle=(1,0,0,0,\ldots,0) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | m2subscript𝑚2\displaystyle m\_{2} | =(0,1,0,0,…,0)absent0100…0\displaystyle=(0,1,0,0,\ldots,0) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | m3subscript𝑚3\displaystyle m\_{3} | =(0,0,1,0,…,0)absent0010…0\displaystyle=(0,0,1,0,\ldots,0) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | m4subscript𝑚4\displaystyle m\_{4} | =(0,0,0,1,…,0),absent0001…0\displaystyle=(0,0,0,1,\ldots,0), |  |

and the ampute function to generate missingness according to these patterns.

The wealth of data allows us to redraw a data set of 2’000 observations B=10𝐵10B=10 times to get an idea of the variation of our scores. That is, we redraw the data randomly B𝐵B times and generate the missingness mechanism using the ampute function. Figure [6](#S5.F6 "Figure 6 ‣ 5.1 Air Quality Data ‣ 5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") shows the drf- and m𝑚m-I-Scores (obtained without using the true underlying values), while Figure [7](#S5.F7 "Figure 7 ‣ 5.1 Air Quality Data ‣ 5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") shows the negative energy distance between imputed and true data set. The ordering of the scores is remarkably similar, showing mice-cart and mice-DRF first and mice-norm.predict last. This makes sense as mice-norm.predict neither draws from the conditional distribution nor is it able to deal with the apparent nonlinearities in the data. In contrast, missForest scores higher, though interestingly the scores are not in complete agreement. While both the full information score and drf-I-Score put it in third place, the m𝑚m-I-Score puts it just above mice-norm.predict. This might be due to the fact that missForest, while predicting instead of drawing from a conditional distribution, still models the nonlinearities in the data relatively well, a feat the Gaussian-based norm.nob cannot achieve. However, the m𝑚m-I-Score punishes the inability of missForest to draw samples more severely and thus puts it lower than the other two scores. Given the discussion in this paper, one might argue that the low ordering of missForest of the m𝑚m-I-Score is more accurate in this example.

![Refer to caption](/html/2403.19196/assets/Application_1_Scores_withO.png)


Figure 6: Scores for the air quality data example. Top: DRF-Score over 10 iterations. Bottom: m𝑚m-I-Score over 10 iterations.

![Refer to caption](/html/2403.19196/assets/Application_1_Energy_Score_withO.png)


Figure 7: Negative Energy Distance for the air quality data example, calculated with full data.

### 5.2 Gaussian Mixture Model

![Refer to caption](/html/2403.19196/assets/Application_2_Scores_withO.png)


Figure 8: Scores for the Gaussian mixture model with distribution shift. Top: DRF-Score over 10 iterations. Bottom: m𝑚m-I-Score over 10 iterations.

![Refer to caption](/html/2403.19196/assets/Application_2_Energy_Score_withO.png)


Figure 9: Negative Energy Distance for the Gaussian mixture model with distribution shift, calculated with full data.

We next turn to a Gaussian Mixture model to be able to put more emphasis on distribution shifts under MAR. In particular, we will simulate the distribution shift of Example [1](#Thmexample1 "Example 1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") in a larger setting. We take d=6𝑑6d=6 and 3 patterns,

|  |  |  |  |
| --- | --- | --- | --- |
|  | m1subscript𝑚1\displaystyle m\_{1} | =(1,0,0,0,0,0)absent100000\displaystyle=(1,0,0,0,0,0) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | m2subscript𝑚2\displaystyle m\_{2} | =(0,1,0,0,0,0)absent010000\displaystyle=(0,1,0,0,0,0) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | m3subscript𝑚3\displaystyle m\_{3} | =(0,0,1,0,0,0)absent001000\displaystyle=(0,0,1,0,0,0) |  |

The last three columns of fully observed variables are all drawn from three-dimensional Gaussians with randomly generated mean and covariance. For instance, for the first pattern, the mean (rounded) is given as (3,3,4)334(3,3,4), while for the second it is given as (−4,−3,−5)435(-4,-3,-5). Thus each pattern can have quite different parameters. To preserve MAR, the (potentially unobserved) first three columns are built as

|  |  |  |
| --- | --- | --- |
|  | XOc=𝐁​XO+(ε1ε2ε3),subscript𝑋superscript𝑂𝑐𝐁subscript𝑋𝑂matrixsubscript𝜀1subscript𝜀2subscript𝜀3\displaystyle X\_{O^{c}}=\mathbf{B}X\_{O}+\begin{pmatrix}\varepsilon\_{1}\\ \varepsilon\_{2}\\ \varepsilon\_{3}\end{pmatrix}, |  |

where 𝐁𝐁\mathbf{B} is a 3×3333\times 3 matrix of coefficients, (ε1,ε2,ε3)subscript𝜀1subscript𝜀2subscript𝜀3(\varepsilon\_{1},\varepsilon\_{2},\varepsilon\_{3}) are independent standard Gaussian random errors and O={4,5,6}𝑂456O=\{4,5,6\} is again the index of fully observed values.
This is a somewhat different example than the one before. Now the data is Gaussian with linear relationships, but there is a strong distribution shift between the different patterns. However, this distributional shift only stems from the observed variables, leaving the conditional distributions of missing given observed unchanged, as in Example [1](#Thmexample1 "Example 1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). Consequently, it can be shown that the missingness mechanism meets ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) and is thus MAR.

In this example, the ability to generalize is important, while the ability to model nonlinear relationships is not. Indeed, we note that P∗superscript𝑃P^{\*} corresponds to the Gaussian imputation (mice-norm.nob) with the (unknown) true parameters. As such, a proper score should rank mice-norm.nob highest. In contrast, the forest-based scores should have the worst performance here, as they are not able to deal with the distribution shift. On the other hand, they might still be deemed better than mice-norm.predict, which only imputes the regression prediction. Results for the drf- and m𝑚m-I-Score are given in Figure [8](#S5.F8 "Figure 8 ‣ 5.2 Gaussian Mixture Model ‣ 5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), while Figure [9](#S5.F9 "Figure 9 ‣ 5.2 Gaussian Mixture Model ‣ 5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") shows the full information score. While the full information and m𝑚m-I-Score behave as expected, with mice-norm.nob and mice-norm.predict in first and second place, and the forest-based methods last, the inability of DRF to meaningfully extrapolate beyond the sample points severely biases the drf-I-Score. Thus, it wrongly scores the forest-based methods highest. In contrast, despite the challenging setting, the m𝑚m-I-Score still provides a very sensible ordering. An interesting difference between the m𝑚m-I-Score and the full information score is that DRF and missForest are reversed in the two. However, this again makes sense as missForest gets more severely punished when it creates N𝑁N imputations with very limited variation. In fact, in this sense, the score, without having access to the true data, might actually give a more accurate picture of the correct ordering.

### 5.3 Mixture Model with Nonlinear Relationships

We now turn to a more complex version of the model in Section [5.2](#S5.SS2 "5.2 Gaussian Mixture Model ‣ 5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") to add nonlinear relationships to the distributional shifts. This final example should indicate that the search for successful imputation methods is by no means completed.

Using the same missingness pattern, and Gaussian variables XOsubscript𝑋𝑂X\_{O} we use a nonlinear function f𝑓f for the conditional distribution:

|  |  |  |
| --- | --- | --- |
|  | XOc=𝐁​f​(XO)+(ε1ε2ε3),subscript𝑋superscript𝑂𝑐𝐁𝑓subscript𝑋𝑂matrixsubscript𝜀1subscript𝜀2subscript𝜀3\displaystyle X\_{O^{c}}=\mathbf{B}f(X\_{O})+\begin{pmatrix}\varepsilon\_{1}\\ \varepsilon\_{2}\\ \varepsilon\_{3}\end{pmatrix}, |  |

with

|  |  |  |
| --- | --- | --- |
|  | f​(x1,x2,x3)=(x3​sin⁡(x1​x2),x2⋅𝟏​{x2>0},arctan⁡(x1)​arctan⁡(x2)).𝑓subscript𝑥1subscript𝑥2subscript𝑥3subscript𝑥3subscript𝑥1subscript𝑥2⋅subscript𝑥21subscript𝑥20subscript𝑥1subscript𝑥2f(x\_{1},x\_{2},x\_{3})=(x\_{3}\sin(x\_{1}x\_{2}),x\_{2}\cdot\mathbf{1}\{x\_{2}>0\},\arctan(x\_{1})\arctan(x\_{2})). |  |

This introduces highly nonlinear relationships between the elements of XOcsubscript𝑋superscript𝑂𝑐X\_{O^{c}} and XOsubscript𝑋𝑂X\_{O}, though the conditional distribution of XOc∣XOconditionalsubscript𝑋superscript𝑂𝑐subscript𝑋𝑂X\_{O^{c}}\mid X\_{O} is still Gaussian and the missingness mechanism is CIMAR. In this example, the ability to generalize is important, and so is the ability to model nonlinear relationships. Accordingly, this is a very difficult example and the ordering of the scores is quite different. In particular, they do not agree on the best two methods, though they all rank mice-DRF high. This serves to illustrate, that while at least the m𝑚m-I-Score should be able to identify the “ideal” imputation, there is no guarantee for what happens when all imputations are bad. The disagreement of the scores should thus be seen as more of a testament that none of the methods perform well than a sign that the scores themselves are flawed.

![Refer to caption](/html/2403.19196/assets/Application_4_Scores_withO.png)


Figure 10: Scores for the nonlinear mixture model with distribution shift. Top: DRF-Score over 10 iterations. Bottom: m𝑚m-I-Score over 10 iterations.

![Refer to caption](/html/2403.19196/assets/Application_4_Energy_Score_withO.png)


Figure 11: Negative Energy Distance for the nonlinear mixture model with distribution shift, calculated with full data.

## 6 Discussion

This paper attempted to give a more systematic discussion of MAR imputation. We analyse the MAR condition in detail for imputation and, based on this analysis, propose four essential properties an ideal imputation method should meet, as well as a principled way of ranking imputation methods.

An important message of the paper is that RMSE is not a sensible way of evaluating imputations. Dropping RMSE as an evaluation method likely has important implications. For instance, the recommendation of papers to use single imputation methods such as k-NN imputation (Anil Jadhav and Ramanathan,, [2019](#bib.bib1)) or missForest (Waljee et al.,, [2013](#bib.bib42); Tang and Ishwaran,, [2017](#bib.bib36)) appears to rest entirely on the use of RMSE. Even well-designed paper benchmarking imputation methods such as Jäger et al., ([2021](#bib.bib14)) use RMSE. Nonetheless, there appear to be only a handful of recent papers that at least consider different evaluation methods, for instance, Muzellec et al., ([2020](#bib.bib24)); Hong and Lynn, ([2020](#bib.bib12)); Wang et al., ([2022](#bib.bib43)). Indeed, the problems of RMSE and its recommendations appear to be being rediscovered in different fields. For instance, recently Hong and Lynn, ([2020](#bib.bib12)) again demonstrated empirically that, while missForest achieves the smallest RMSE, parameters attained from linear regression are severely biased. Similarly, Wang et al., ([2022](#bib.bib43)) discusses some problems with using RMSE in the machine learning literature. In contrast, GAN-based approaches recognize the objective of drawing imputations from the respective conditional distributions and naturally use the pattern-mixture modeling approach. However, despite having the right objective, these papers again use RMSE to compare the imputation quality of their method to competitors.

A second important message is that the problem of imputation is by no means solved. Though there is a set of promising imputation methods with mice-cart, mice-RF, and mice-DRF that will likely work well in a wide range of settings, there is room for improvement, especially concerning the ability to deal with covariate shifts. In particular, Section [5.3](#S5.SS3 "5.3 Mixture Model with Nonlinear Relationships ‣ 5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") shows an example with distribution shifts and nonlinear relationships for which all methods fail. The m𝑚m-I-Score developed here can help to identify the right distribution, though this might not be helpful if all imputations are sufficiently bad. Appendix [B](#A2 "Appendix B Comparison of MICE to GAIN and MIWAE ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") demonstrates that modern joint modeling approaches do not fare better in this example. In fact, contrary to what we theorized in Section [3](#S3 "3 Requirements for Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), on these low-dimensional data sets GAIN and MIWAE are outperformed by missForest, even in terms of energy distance.

We believe the paper touches on a few crucial issues that have not been discussed before. However, it also has several shortcomings. The m𝑚m-I-Score, while promising, needs a set of fully observed variables, at least theoretically. In addition, the performance of mice-cart stands out, even when compared to mice-DRF. It remains an open question why the performance of mice-cart is so strong and whether a systematic benchmarking of imputation methods over a wider array of data sets can confirm the empirical findings in this paper. In general, a much more comprehensive empirical evaluation of both the new score and the forest-based imputation methods is needed. Finally, when talking about multiple imputation, we note that none of the studied nonparametric methods is able to include *model uncertainty*. However this would technically be needed for correct uncertainty quantification with multiple imputation, see e.g., Murray, ([2018](#bib.bib23)). Though both mice-rf of Doove et al., ([2014](#bib.bib7)) and the new mice-DRF attempt to account for model uncertainty using several trees, this is only a heuristic solution. Moreover, the scores developed in this paper are unable to account for this and will instead likely place methods that include model uncertainty lower than those that do not, which in turn could explain the success of mice-cart in terms of these scores.

Finally, we discussed some challenging MAR conditions, particularly using the Gaussian Mixture Model. However, we did not discuss how *likely* such MAR settings may be. Intuitively, it appears that distributional shifts under MAR should be quite common. Consider an example with two variables, X1subscript𝑋1X\_{1} being income, and X2subscript𝑋2X\_{2} being age. Moreover, assume a missing mechanism for the income X1subscript𝑋1X\_{1}, whereby X1subscript𝑋1X\_{1} tends to be missing whenever age is “high”. Thus the probability of income (X1subscript𝑋1X\_{1}) being missing depends entirely on the value of age (X2subscript𝑋2X\_{2}), which is always observed. This is a textbook MAR example with two patterns, one where both variables are fully observed (m1subscript𝑚1m\_{1}) and a second (m2subscript𝑚2m\_{2}), wherein X1subscript𝑋1X\_{1} is missing. Despite the simplicity of this example, if we assume that higher age is related to higher income, there is a clear shift in the distribution of income and age when moving from one pattern to the other. In pattern m2subscript𝑚2m\_{2}, where income is missing, values of both the observed age and the (unobserved) income tend to be higher. It thus appears intuitive that the combination of distributional shifts and nonlinear relationships is widespread in real data. At the same time, the success of forest-based methods such as missForest and mice-cart in benchmark papers suggests that current ways of introducing MAR might not produce enough distribution shifts in general. For instance, Näf et al., ([2023](#bib.bib26)) analyzed a range of data sets using the standard MAR mechanism of the ampute function implementing the procedure of Rianne Margaretha Schouten and Vink, ([2018](#bib.bib28)), as we did in Section [5.1](#S5.SS1 "5.1 Air Quality Data ‣ 5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). Though their score is not proper under MAR, as shown in Appendix [C.1](#A3.SS1 "C.1 DR-I-Score is not proper under MAR ‣ Appendix C Proofs and Additional Results ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), their analysis also showed mice-cart consistently in first place. Thus, tweaking the approach of Rianne Margaretha Schouten and Vink, ([2018](#bib.bib28)) to produce MAR data with distribution shifts, might be an avenue for further research.

## Appendix A Score Version without Fully Observed Data

We first informally discuss the problems that arise when instead of the set of fully observed variables, we use all remaining variables X−j∼H−jsimilar-tosubscript𝑋𝑗subscript𝐻𝑗X\_{-j}\sim H\_{-j} in the score defined in Section [4](#S4 "4 Assessing Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). We note that, while the distribution of the observed test points Xjsubscript𝑋𝑗X\_{j} is fixed to p∗​(xj)superscript𝑝subscript𝑥𝑗p^{\*}(x\_{j}), it holds that

|  |  |  |  |
| --- | --- | --- | --- |
|  | p∗​(xj)superscript𝑝subscript𝑥𝑗\displaystyle p^{\*}(x\_{j}) | =∫p∗​(xj∣x−j)​p∗​(x−j)​𝑑x−jabsentsuperscript𝑝conditionalsubscript𝑥𝑗subscript𝑥𝑗superscript𝑝subscript𝑥𝑗differential-dsubscript𝑥𝑗\displaystyle=\int p^{\*}(x\_{j}\mid x\_{-j})p^{\*}(x\_{-j})dx\_{-j} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∫h​(xj∣x−j)​h​(x−j)​𝑑x−j.absentℎconditionalsubscript𝑥𝑗subscript𝑥𝑗ℎsubscript𝑥𝑗differential-dsubscript𝑥𝑗\displaystyle=\int h(x\_{j}\mid x\_{-j})h(x\_{-j})dx\_{-j}. |  |

If H−jsubscript𝐻𝑗H\_{-j} is different from P−j∗subscriptsuperscript𝑃𝑗P^{\*}\_{-j}, then the two conditional distribution will in general be different as well. This means with X−j∼H−jsimilar-tosubscript𝑋𝑗subscript𝐻𝑗X\_{-j}\sim H\_{-j} it is not clear, whether Xj∣X−jconditionalsubscript𝑋𝑗subscript𝑋𝑗X\_{j}\mid X\_{-j} has the desired distribution (p∗​(xj∣x−j)superscript𝑝conditionalsubscript𝑥𝑗subscript𝑥𝑗p^{\*}(x\_{j}\mid x\_{-j})). This problem is numerically evident when the original imputation is used for H−jsubscript𝐻𝑗H\_{-j}. For instance, simply adapting the m𝑚m-I-Score by using X−jsubscript𝑋𝑗X\_{-j} instead of XOsubscript𝑋𝑂X\_{O} tends to score the regression imputation higher than the Gaussian imputation in the example in Section [5.2](#S5.SS2 "5.2 Gaussian Mixture Model ‣ 5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."). To alleviate this, we instead generate X−j∼H−jsimilar-tosubscript𝑋𝑗subscript𝐻𝑗X\_{-j}\sim H\_{-j} independently, i.e., we impute the d−1𝑑1d-1 dimensional dataset without Xjsubscript𝑋𝑗X\_{j} and keep the original imputation of Xjsubscript𝑋𝑗X\_{j}. That is, the only difference to the score of Section [4](#S4 "4 Assessing Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") is that, given an imputed data set and the imputation function itself, we first generate a new draw X1,−j,…,Xn,−j

subscript𝑋

1𝑗…subscript𝑋

𝑛𝑗X\_{1,-j},\ldots,X\_{n,-j} from H−jsubscript𝐻𝑗H\_{-j} by imputing all variables except Xjsubscript𝑋𝑗X\_{j}. Then we proceed as before: For Xjsubscript𝑋𝑗X\_{j} we keep the imputed points, that is all xi,jsubscript𝑥

𝑖𝑗x\_{i,j} with mi∈Ljcsubscript𝑚𝑖superscriptsubscript𝐿𝑗𝑐m\_{i}\in L\_{j}^{c} are still drawn from the original imputation, while we set the *observed* observations of Xjsubscript𝑋𝑗X\_{j} to missing, i.e. xi,j=NAsubscript𝑥

𝑖𝑗NAx\_{i,j}=\texttt{NA} for i𝑖i with mi∈Ljsubscript𝑚𝑖subscript𝐿𝑗m\_{i}\in L\_{j}. Then we concatenate

|  |  |  |
| --- | --- | --- |
|  | (NA(xi,−j)mi∈Lj(xi,j)mi∈Ljc(xi,−j)mi∈Ljc,)matrixNAsubscriptsubscript𝑥  𝑖𝑗subscript𝑚𝑖subscript𝐿𝑗subscriptsubscript𝑥  𝑖𝑗subscript𝑚𝑖superscriptsubscript𝐿𝑗𝑐subscriptsubscript𝑥  𝑖𝑗subscript𝑚𝑖superscriptsubscript𝐿𝑗𝑐\displaystyle\begin{pmatrix}\texttt{NA}&(x\_{i,-j})\_{m\_{i}\in L\_{j}}\\ (x\_{i,j})\_{m\_{i}\in L\_{j}^{c}}&(x\_{i,-j})\_{m\_{i}\in L\_{j}^{c}},\end{pmatrix} |  |

and approximate the sampling from HXj∣xi,−jsubscript𝐻conditionalsubscript𝑋𝑗subscript𝑥

𝑖𝑗H\_{X\_{j}\mid x\_{i,-j}} in two ways:

* (1)

  Regress (xi,j)mi∈Ljcsubscriptsubscript𝑥
  𝑖𝑗subscript𝑚𝑖superscriptsubscript𝐿𝑗𝑐(x\_{i,j})\_{m\_{i}\in L\_{j}^{c}} onto (xi,−j)mi∈Ljcsubscriptsubscript𝑥
  𝑖𝑗subscript𝑚𝑖superscriptsubscript𝐿𝑗𝑐(x\_{i,-j})\_{m\_{i}\in L\_{j}^{c}} using DRF. Then for each test point xi,−jsubscript𝑥
  𝑖𝑗x\_{i,-j}, mi∈Ljsubscript𝑚𝑖subscript𝐿𝑗m\_{i}\in L\_{j}, sample N𝑁N times from the estimated conditional distribution obtained from DRF.
* (2)

  Impute the NA values with H𝐻H, N𝑁N times.

The j𝑗jth score is then given as in ([4.4](#S4.E4 "In 4 Assessing Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")), but now with the N𝑁N points generated approximately from HXj∣X−jsubscript𝐻conditionalsubscript𝑋𝑗subscript𝑋𝑗H\_{X\_{j}\mid X\_{-j}}. The idea in both cases is to use an imputation of X−jsubscript𝑋𝑗X\_{-j} and the initial imputation of Xjsubscript𝑋𝑗X\_{j} to generate a sample from the distribution HXj∣X−jsubscript𝐻conditionalsubscript𝑋𝑗subscript𝑋𝑗H\_{X\_{j}\mid X\_{-j}} for points that are already observed. This approximation is clearly not perfect to obtain a sample from HXj∣X−jsubscript𝐻conditionalsubscript𝑋𝑗subscript𝑋𝑗H\_{X\_{j}\mid X\_{-j}} but repeating the experiments in Section [5](#S5 "5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), Figures [12](#A1.F12 "Figure 12 ‣ Appendix A Score Version without Fully Observed Data ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")–[14](#A1.F14 "Figure 14 ‣ Appendix A Score Version without Fully Observed Data ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") show that the two scores closely follow their counterparts in Section [5](#S5 "5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.").

###### Remark.

We note that compared to the score in Section [4](#S4 "4 Assessing Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), we now also need to obtain a sample from H−jsubscript𝐻𝑗H\_{-j}. Thus a (d−1)𝑑1(d-1)-variate data set has to be imputed for each j𝑗j, which can be computationally challenging when d𝑑d is large. This could be solved by using random or predefined *projections* A𝐴A, as in Näf et al., ([2023](#bib.bib26)), thus reducing the dimensionality. This will not hurt propriety but might diminish the power to detect differences between the methods. In fact, the score in Section [4](#S4 "4 Assessing Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") might be seen as an example of this with A=O𝐴𝑂A=O.

![Refer to caption](/html/2403.19196/assets/Application_1_Scores.png)


Figure 12: Scores for the air quality data example. Top: DRF-Score over 10 iterations. Bottom: m𝑚m-I-Score over 10 iterations.

![Refer to caption](/html/2403.19196/assets/Application_2_Scores.png)


Figure 13: Scores for the Gaussian mixture model with distribution shift. Top: DRF-Score over 10 iterations. Bottom: m𝑚m-I-Score over 10 iterations.

![Refer to caption](/html/2403.19196/assets/Application_3_Scores.png)


Figure 14: Scores for the nonlinear mixture model with distribution shift. Top: DRF-Score over 10 iterations. Bottom: m𝑚m-I-Score over 10 iterations.

## Appendix B Comparison of MICE to GAIN and MIWAE

Here we use the negative energy distance advocated in the main text (i.e. the “full information score”) to compare the performance of the MICE methods used in Section [5](#S5 "5 Empirical Study ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") to the joint modeling methods GAIN and MIWAE. The code for GAIN was taken from the original Github repository <https://github.com/jsyoon0823/GAIN>, while the implementation of MIWAE was obtained from <https://github.com/nbip/notMIWAE/blob/master/MIWAE.py>. As both were coded in Python, the R package reticulate (Ushey et al.,, [2024](#bib.bib38)) was used to embed the code into R.

Figures [15](#A2.F15 "Figure 15 ‣ Appendix B Comparison of MICE to GAIN and MIWAE ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") – [17](#A2.F17 "Figure 17 ‣ Appendix B Comparison of MICE to GAIN and MIWAE ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.") show the results. Overall these two methods cannot compete with MICE and usually are scored last, except in the Gaussian example with distribution shift (Figure [16](#A2.F16 "Figure 16 ‣ Appendix B Comparison of MICE to GAIN and MIWAE ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) where MIWAE performs about the same as mice-cart and mice-DRF. However, we gave MIWAE a somewhat unfair advantage: We standardized the data in Application 1, as otherwise the implementation broke down, but did not do this for Applications 2 and 3. In practice, one would likely always standardize the data, given the numerical problems one faces otherwise, and this would have led to a lower ranking of MIWAE. Interestingly, this experiment does not confirm our suspicion in Section [3](#S3 "3 Requirements for Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."); GAIN and MIWAE tend to perform worse than missForest, even in terms of the energy distance. To analyze this further, we additionally consider a larger data set, the spambase data set of Lichman, ([2013](#bib.bib16)), with more missing values. Specifically, we consider a simple MCAR mechanism whereby each variable is missing randomly such that we have around 20% of missingness in total. This dataset has dimension d=57𝑑57d=57 and n=4601𝑛4601n=4601 observations and was used to show that GAIN performs better than other imputation methods in Yoon et al., ([2018](#bib.bib45)). The combination of high frequency of missing values, and relatively high dimension and number of observations, means imputation with MICE takes considerably longer than in the two examples before. In particular, on a desktop computer, imputation times ranged from three minutes for mice-cart up to 29 minutes for missForest obtained from the missForest R-package (Stekhoven,, [2022](#bib.bib33)), mice-DRF needed 9 minutes, a computation time that can however be halved with around the same accuracy when defining blocks of size 2 and imputing once per block as described above. The result, shown in Figure [18](#A2.F18 "Figure 18 ‣ Appendix B Comparison of MICE to GAIN and MIWAE ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), remains the same however, GAIN and MIWAE perform worse than even missForest in terms of energy distance, while missForest in turn is largely outperformed by mice-DRF and mice-cart.

All in all this small analysis provides a further hint that, at least for data sets of small or moderate dimensions, modern joint modeling methods such as GAIN and MIWAE cannot compete with FCS.

![Refer to caption](/html/2403.19196/assets/Application_1_Energy_Score_withGAINMIWAE.png)


Figure 15: Negative Energy Distance for the air quality data example with GAIN and MIWAE, calculated with full data.

![Refer to caption](/html/2403.19196/assets/Application_2_Energy_Score_withGAINMIWAE.png)


Figure 16: Negative Energy Distance for the Gaussian mixture model with distribution shift with GAIN and MIWAE, calculated with full data.

![Refer to caption](/html/2403.19196/assets/Application_3_Energy_Score_withGAINMIWAE.png)


Figure 17: Negative Energy Distance for the nonlinear mixture model with distribution shift with GAIN and MIWAE, calculated with full data.

![Refer to caption](/html/2403.19196/assets/Application_4_Energy_Score_withGAINMIWAE.png)


Figure 18: Negative Energy Distance for the spam data example with GAIN and MIWAE, calculated with full data.

## Appendix C Proofs and Additional Results

In this section, we provide additional results and collect the proofs of the results not shown in the main paper. We start by showing that the score developed in Näf et al., ([2023](#bib.bib26)) is not proper under ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")).

### C.1 DR-I-Score is not proper under MAR

Here we show that the Density Ratio I-Score of Näf et al., ([2023](#bib.bib26)) is not proper under MAR. Define the Kullback-Leibler divergence (KL divergence) between two distributions P,Q∈𝒫

𝑃𝑄
𝒫P,Q\in\mathcal{P} on ℝdsuperscriptℝ𝑑{\mathbb{R}}^{d} with densities p𝑝p, q𝑞q

|  |  |  |
| --- | --- | --- |
|  | DK​L(p∣∣q):=∫p(x)log(p​(x)q​(x))dμ(x).D\_{KL}(p\mid\mid q):=\int p(x)\log\left(\frac{p(x)}{q(x)}\right)d\mu(x). |  |

Näf et al., ([2023](#bib.bib26)) developed a proper I-Score using the KL divergence estimated by a classifier in conjunction with random projections A⊂{1,…,p}𝐴1…𝑝A\subset\{1,\ldots,p\}. The projections were done as a way to obtain more observations of each pattern. They proved that the population version of their score is a proper I-Score if condition ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) holds *each projection A𝐴A*. Even without considering any projections, i.e. A={1,…,d}𝐴1…𝑑A=\{1,\ldots,d\}, this is a stronger condition than ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")), as was shown above. In particular, in Example [2](#Thmexample2 "Example 2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), their score will not be proper. Since the score is defined using a pattern-by-pattern comparison, when H=P∗𝐻superscript𝑃H=P^{\*} it will compare p∗​(x1∣x2,x3)​p∗​(x2,x3)superscript𝑝conditionalsubscript𝑥1

subscript𝑥2subscript𝑥3superscript𝑝subscript𝑥2subscript𝑥3p^{\*}(x\_{1}\mid x\_{2},x\_{3})p^{\*}(x\_{2},x\_{3}) (third pattern) to

|  |  |  |
| --- | --- | --- |
|  | p∗​(x1∣x2,x3,M=m1)​p∗​(x2,x3)=x1​p∗​(x1∣x2,x3)​p∗​(x2,x3),superscript𝑝conditionalsubscript𝑥1  subscript𝑥2subscript𝑥3𝑀subscript𝑚1superscript𝑝subscript𝑥2subscript𝑥3subscript𝑥1superscript𝑝conditionalsubscript𝑥1  subscript𝑥2subscript𝑥3superscript𝑝subscript𝑥2subscript𝑥3\displaystyle p^{\*}(x\_{1}\mid x\_{2},x\_{3},M=m\_{1})p^{\*}(x\_{2},x\_{3})=x\_{1}p^{\*}(x\_{1}\mid x\_{2},x\_{3})p^{\*}(x\_{2},x\_{3}), |  |

in the second pattern. Thus, while we would like to score the imputation p∗​(x1∣x2,x3)superscript𝑝conditionalsubscript𝑥1

subscript𝑥2subscript𝑥3p^{\*}(x\_{1}\mid x\_{2},x\_{3}) highest, imputing by h​(x1∣x2,x3)=x1​p∗​(x1∣x2,x3)ℎconditionalsubscript𝑥1

subscript𝑥2subscript𝑥3subscript𝑥1superscript𝑝conditionalsubscript𝑥1

subscript𝑥2subscript𝑥3h(x\_{1}\mid x\_{2},x\_{3})=x\_{1}p^{\*}(x\_{1}\mid x\_{2},x\_{3}) will lead to a score value of exactly zero, while

|  |  |  |
| --- | --- | --- |
|  | DK​L(p∗∣∣p∗)=∫p∗(x1,x2,x3)log(1x1)dμ(x1,x2,x3)>0.D\_{KL}(p^{\*}\mid\mid p^{\*})=\int p^{\*}(x\_{1},x\_{2},x\_{3})\log\left(\frac{1}{x\_{1}}\right)d\mu(x\_{1},x\_{2},x\_{3})>0. |  |

Thus we have just shown that

###### Proposition C.1.

The I-Score defined in Näf et al., ([2023](#bib.bib26)) is not proper if ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) holds, but not ([CIMAR](#S2.Ex10 "In Definition 2.4. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")).

### C.2 Proofs

See [2.1](#S2.Thmcorollary1 "Corollary 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")

###### Proof.

We start by reformulating ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")), for any x,x~

𝑥~𝑥x,\tilde{x} such that o​(x,m)=o​(x~,m)𝑜𝑥𝑚𝑜~𝑥𝑚o(x,m)=o(\tilde{x},m),

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ℙ​(M=m|x)=ℙ​(M=m|x~)⇔⇔ℙ𝑀conditional𝑚𝑥ℙ𝑀conditional𝑚~𝑥absent\displaystyle{\mathbb{P}}(M=m|x)={\mathbb{P}}(M=m|\tilde{x})\Leftrightarrow |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | p∗​(x|M=m)​ℙ​(M=m)p∗​(x)=p∗​(x~|M=m)​ℙ​(M=m)p∗​(x~)⇔⇔superscript𝑝conditional𝑥𝑀𝑚ℙ𝑀𝑚superscript𝑝𝑥superscript𝑝conditional~𝑥𝑀𝑚ℙ𝑀𝑚superscript𝑝~𝑥absent\displaystyle\frac{p^{\*}(x|M=m){\mathbb{P}}(M=m)}{p^{\*}(x)}=\frac{p^{\*}(\tilde{x}|M=m){\mathbb{P}}(M=m)}{{p^{\*}(\tilde{x})}}\Leftrightarrow |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | p∗​(o​(x,m),oc​(x,m)∣M=m)p∗​(o​(x~,m),oc​(x~,m)∣M=m)=p∗​(o​(x,m),oc​(x,m))p∗​(o​(x~,m),oc​(x~,m))⇔⇔superscript𝑝  𝑜𝑥𝑚conditionalsuperscript𝑜𝑐𝑥𝑚𝑀 𝑚superscript𝑝  𝑜~𝑥𝑚conditionalsuperscript𝑜𝑐~𝑥𝑚𝑀 𝑚superscript𝑝𝑜𝑥𝑚superscript𝑜𝑐𝑥𝑚superscript𝑝𝑜~𝑥𝑚superscript𝑜𝑐~𝑥𝑚absent\displaystyle\frac{p^{\*}(o(x,m),o^{c}(x,m)\mid M=m)}{p^{\*}(o(\tilde{x},m),o^{c}(\tilde{x},m)\mid M=m)}=\frac{p^{\*}(o(x,m),o^{c}(x,m))}{p^{\*}(o(\tilde{x},m),o^{c}(\tilde{x},m))}\Leftrightarrow |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | p∗​(oc​(x,m)∣o​(x,m),M=m)p∗​(oc​(x,m)∣o​(x,m))=p∗​(oc​(x~,m)∣o​(x,m),M=m)p∗​(oc​(x~,m)∣o​(x,m))⇔⇔superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚𝑜𝑥𝑚superscript𝑝conditionalsuperscript𝑜𝑐~𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscript𝑝conditionalsuperscript𝑜𝑐~𝑥𝑚𝑜𝑥𝑚absent\displaystyle\frac{p^{\*}(o^{c}(x,m)\mid o(x,m),M=m)}{p^{\*}(o^{c}(x,m)\mid o(x,m))}=\frac{p^{\*}(o^{c}(\tilde{x},m)\mid o(x,m),M=m)}{p^{\*}(o^{c}(\tilde{x},m)\mid o(x,m))}\Leftrightarrow |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | p∗​(oc​(x,m)∣o​(x,m),M=m)=p∗​(oc​(x~,m)∣o​(x,m),M=m)p∗​(oc​(x~,m)∣o​(x,m))​p∗​(oc​(x,m)∣o​(x,m))superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscript𝑝conditionalsuperscript𝑜𝑐~𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscript𝑝conditionalsuperscript𝑜𝑐~𝑥𝑚𝑜𝑥𝑚superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚𝑜𝑥𝑚\displaystyle p^{\*}(o^{c}(x,m)\mid o(x,m),M=m)=\frac{p^{\*}(o^{c}(\tilde{x},m)\mid o(x,m),M=m)}{p^{\*}(o^{c}(\tilde{x},m)\mid o(x,m))}p^{\*}(o^{c}(x,m)\mid o(x,m)) |  | (C.1) |

Integrating ([C.2](#A3.Ex4 "Proof. ‣ C.2 Proofs ‣ Appendix C Proofs and Additional Results ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) with respect to the missing part of x𝑥x, oc​(x,m)superscript𝑜𝑐𝑥𝑚o^{c}(x,m), only shows that

|  |  |  |
| --- | --- | --- |
|  | p∗​(oc​(x~,m)∣o​(x,m),M=m)p∗​(oc​(x~,m)∣o​(x,m))=1,superscript𝑝conditionalsuperscript𝑜𝑐~𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscript𝑝conditionalsuperscript𝑜𝑐~𝑥𝑚𝑜𝑥𝑚1\frac{p^{\*}(o^{c}(\tilde{x},m)\mid o(x,m),M=m)}{p^{\*}(o^{c}(\tilde{x},m)\mid o(x,m))}=1, |  |

and thus also ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). This shows that ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) and ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) are equivalent. Molenberghs et al., ([2008](#bib.bib22)) show that ([SM-MAR II](#S2.Ex4 "In Definition 2.2. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) is also equivalent to ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")), proving the result.
∎

See [2.3](#S2.Thmproposition3 "Proposition 2.3. ‣ 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")

###### Proof.

Let in the following Ljsubscript𝐿𝑗L\_{j} be defined as in ([2.2](#S2.E2 "In 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). We assume that Ljsubscript𝐿𝑗L\_{j} is not empty. As all previous variables have been imputed and xjsubscript𝑥𝑗x\_{j} is observed, it is thus possible to identify the full distribution p∗​(x∣M=m)superscript𝑝conditional𝑥𝑀𝑚p^{\*}(x\mid M=m) for all m∈Lj𝑚subscript𝐿𝑗m\in L\_{j}. Thus, we learn the mixture of joint distributions

|  |  |  |  |
| --- | --- | --- | --- |
|  | h∗​(xj,x−j)superscriptℎsubscript𝑥𝑗subscript𝑥𝑗\displaystyle h^{\*}(x\_{j},x\_{-j}) | =1C​∑m∈Ljℙ​(M=m)⋅p∗​(x∣M=m)absent1𝐶subscript𝑚subscript𝐿𝑗⋅ℙ𝑀𝑚superscript𝑝conditional𝑥𝑀𝑚\displaystyle=\frac{1}{C}\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m)\cdot p^{\*}(x\mid M=m) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1C​∑m∈Ljℙ​(M=m∣x)⋅p∗​(x),absent1𝐶subscript𝑚subscript𝐿𝑗⋅ℙ𝑀conditional𝑚𝑥superscript𝑝𝑥\displaystyle=\frac{1}{C}\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x)\cdot p^{\*}(x), |  |

where C𝐶C is a constant such that h∗​(xj,x−j)superscriptℎsubscript𝑥𝑗subscript𝑥𝑗h^{\*}(x\_{j},x\_{-j}) integrates to 1. Integrating h∗​(xj,x−j)superscriptℎsubscript𝑥𝑗subscript𝑥𝑗h^{\*}(x\_{j},x\_{-j}) over xjsubscript𝑥𝑗x\_{j}, we obtain similarly

|  |  |  |
| --- | --- | --- |
|  | h∗​(x−j)=1C​∑m∈Ljℙ​(M=m∣x−j)⋅p∗​(x−j)superscriptℎsubscript𝑥𝑗1𝐶subscript𝑚subscript𝐿𝑗⋅ℙ𝑀conditional𝑚subscript𝑥𝑗superscript𝑝subscript𝑥𝑗\displaystyle h^{\*}(x\_{-j})=\frac{1}{C}\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{-j})\cdot p^{\*}(x\_{-j}) |  |

Thus in fact:

|  |  |  |  |
| --- | --- | --- | --- |
|  | h∗​(xj∣x−j)superscriptℎconditionalsubscript𝑥𝑗subscript𝑥𝑗\displaystyle h^{\*}(x\_{j}\mid x\_{-j}) | =h∗​(xj,x−j)h∗​(x−j)absentsuperscriptℎsubscript𝑥𝑗subscript𝑥𝑗superscriptℎsubscript𝑥𝑗\displaystyle=\frac{h^{\*}(x\_{j},x\_{-j})}{h^{\*}(x\_{-j})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑m∈Ljℙ​(M=m∣x)⋅p∗​(x)∑m∈Ljℙ​(M=m∣x−j)⋅p∗​(x−j)absentsubscript𝑚subscript𝐿𝑗⋅ℙ𝑀conditional𝑚𝑥superscript𝑝𝑥subscript𝑚subscript𝐿𝑗⋅ℙ𝑀conditional𝑚subscript𝑥𝑗superscript𝑝subscript𝑥𝑗\displaystyle=\frac{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x)\cdot p^{\*}(x)}{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{-j})\cdot p^{\*}(x\_{-j})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =p∗​(xj∣x−j)​∑m∈Ljℙ​(M=m∣x)∑m∈Ljℙ​(M=m∣x−j).absentsuperscript𝑝conditionalsubscript𝑥𝑗subscript𝑥𝑗subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚𝑥subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚subscript𝑥𝑗\displaystyle=p^{\*}(x\_{j}\mid x\_{-j})\frac{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x)}{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{-j})}. |  |

It only remains to show that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑m∈Ljℙ​(M=m∣x)∑m∈Ljℙ​(M=m∣x−j)=1.subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚𝑥subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚subscript𝑥𝑗1\displaystyle\frac{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x)}{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{-j})}=1. |  | (C.2) |

Indeed, we note that for any m∈Ljc𝑚superscriptsubscript𝐿𝑗𝑐m\in L\_{j}^{c},

|  |  |  |
| --- | --- | --- |
|  | ℙ​(M=m∣x)=ℙ​(M=m∣x−j),ℙ𝑀conditional𝑚𝑥ℙ𝑀conditional𝑚subscript𝑥𝑗\displaystyle{\mathbb{P}}(M=m\mid x)={\mathbb{P}}(M=m\mid x\_{-j}), |  |

by ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). Consequently,

|  |  |  |
| --- | --- | --- |
|  | 1=∑m∈Ljℙ​(M=m∣x)+∑m∈Ljcℙ​(M=m∣x−j),1subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚𝑥subscript𝑚superscriptsubscript𝐿𝑗𝑐ℙ𝑀conditional𝑚subscript𝑥𝑗\displaystyle 1=\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x)+\sum\_{m\in L\_{j}^{c}}{\mathbb{P}}(M=m\mid x\_{-j}), |  |

so that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑m∈Ljℙ​(M=m∣x)subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚𝑥\displaystyle\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x) | =1−∑m∈Ljcℙ​(M=m∣x−j)absent1subscript𝑚superscriptsubscript𝐿𝑗𝑐ℙ𝑀conditional𝑚subscript𝑥𝑗\displaystyle=1-\sum\_{m\in L\_{j}^{c}}{\mathbb{P}}(M=m\mid x\_{-j}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑m∈Ljℙ​(M=m∣x−j),absentsubscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚subscript𝑥𝑗\displaystyle=\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{-j}), |  |

and thus ([C.2](#A3.E2 "In Proof. ‣ C.2 Proofs ‣ Appendix C Proofs and Additional Results ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) indeed holds.
∎

See [2.2](#S2.Thmcorollary2 "Corollary 2.2. ‣ 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")

###### Proof.

By construction, H∗∈ℋPsuperscript𝐻subscriptℋ𝑃H^{\*}\in\mathcal{H}\_{P}. Assume that we are at step j∈{1,…,p}𝑗1…𝑝j\in\{1,\ldots,p\} of our imputation. That is, all variables xi,lsubscript𝑥

𝑖𝑙x\_{i,l}, i=1,…,n𝑖

1…𝑛i=1,\ldots,n, l>j𝑙𝑗l>j have successfully be imputed with a draw from p∗​(xl∣xl+1,…,xp,M=m)superscript𝑝conditionalsubscript𝑥𝑙

subscript𝑥𝑙1…subscript𝑥𝑝𝑀𝑚p^{\*}(x\_{l}\mid x\_{l+1},\ldots,x\_{p},M=m). Let in the following Ljsubscript𝐿𝑗L\_{j} be defined as in ([2.2](#S2.E2 "In 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). We first note that any pattern m∈Ljc𝑚superscriptsubscript𝐿𝑗𝑐m\in L\_{j}^{c} ( where xj∈oc​(x,m)subscript𝑥𝑗superscript𝑜𝑐𝑥𝑚x\_{j}\in o^{c}(x,m)) has

|  |  |  |  |
| --- | --- | --- | --- |
|  | p∗​(oc​(x,m)∣o​(x,m),M=m)=p∗​(oc​(x,m)∣o​(x,m)),superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚𝑜𝑥𝑚\displaystyle p^{\*}(o^{c}(x,m)\mid o(x,m),M=m)=p^{\*}(o^{c}(x,m)\mid o(x,m)), |  | (C.3) |

by ([PMM-MAR](#S2.Ex8 "In Definition 2.3. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). Integrating both sides, this means that for any A⊂{x1,…,xp}𝐴subscript𝑥1…subscript𝑥𝑝A\subset\{x\_{1},\ldots,x\_{p}\},

|  |  |  |
| --- | --- | --- |
|  | p∗​(A∩oc​(x,m)∣o​(x,m),M=m)=p∗​(A∩oc​(x,m)∣o​(x,m)).superscript𝑝𝐴conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscript𝑝𝐴conditionalsuperscript𝑜𝑐𝑥𝑚𝑜𝑥𝑚\displaystyle p^{\*}(A\cap o^{c}(x,m)\mid o(x,m),M=m)=p^{\*}(A\cap o^{c}(x,m)\mid o(x,m)). |  |

Thus the correct imputation distribution for this pattern m𝑚m is given by

|  |  |  |
| --- | --- | --- |
|  | p∗​(xj∣xj+1,…,xp,M=m)superscript𝑝conditionalsubscript𝑥𝑗  subscript𝑥𝑗1…subscript𝑥𝑝𝑀𝑚\displaystyle p^{\*}(x\_{j}\mid x\_{j+1},\ldots,x\_{p},M=m) |  |
|  |  |  |
| --- | --- | --- |
|  | =p∗​(xj∣({xj+1,…,xp}∩oc​(x,m))∪({xj+1,…,xp}∩o​(x,m)),M=m)absentsuperscript𝑝conditionalsubscript𝑥𝑗  subscript𝑥𝑗1…subscript𝑥𝑝superscript𝑜𝑐𝑥𝑚subscript𝑥𝑗1…subscript𝑥𝑝𝑜𝑥𝑚𝑀𝑚\displaystyle=p^{\*}(x\_{j}\mid(\{x\_{j+1},\ldots,x\_{p}\}\cap o^{c}(x,m))\cup(\{x\_{j+1},\ldots,x\_{p}\}\cap o(x,m)),M=m) |  |
|  |  |  |
| --- | --- | --- |
|  | =p∗​({xj,…,xp}∩oc​(x,m)∣{xj+1,…,xp}∩o​(x,m),M=m)p∗​({xj+1,…,xp}∩oc​(x,m)∣{xj+1,…,xp}∩o​(x,m),M=m)absentsuperscript𝑝subscript𝑥𝑗…subscript𝑥𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  subscript𝑥𝑗1…subscript𝑥𝑝𝑜𝑥𝑚𝑀𝑚superscript𝑝subscript𝑥𝑗1…subscript𝑥𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  subscript𝑥𝑗1…subscript𝑥𝑝𝑜𝑥𝑚𝑀𝑚\displaystyle=\frac{p^{\*}(\{x\_{j},\ldots,x\_{p}\}\cap o^{c}(x,m)\mid\{x\_{j+1},\ldots,x\_{p}\}\cap o(x,m),M=m)}{p^{\*}(\{x\_{j+1},\ldots,x\_{p}\}\cap o^{c}(x,m)\mid\{x\_{j+1},\ldots,x\_{p}\}\cap o(x,m),M=m)} |  |
|  |  |  |
| --- | --- | --- |
|  | =p∗​({xj,…,xp}∩oc​(x,m)∣{xj+1,…,xp}∩o​(x,m))p∗​({xj+1,…,xp}∩oc​(x,m)∣{xj+1,…,xp}∩o​(x,m))absentsuperscript𝑝subscript𝑥𝑗…subscript𝑥𝑝conditionalsuperscript𝑜𝑐𝑥𝑚subscript𝑥𝑗1…subscript𝑥𝑝𝑜𝑥𝑚superscript𝑝subscript𝑥𝑗1…subscript𝑥𝑝conditionalsuperscript𝑜𝑐𝑥𝑚subscript𝑥𝑗1…subscript𝑥𝑝𝑜𝑥𝑚\displaystyle=\frac{p^{\*}(\{x\_{j},\ldots,x\_{p}\}\cap o^{c}(x,m)\mid\{x\_{j+1},\ldots,x\_{p}\}\cap o(x,m))}{p^{\*}(\{x\_{j+1},\ldots,x\_{p}\}\cap o^{c}(x,m)\mid\{x\_{j+1},\ldots,x\_{p}\}\cap o(x,m))} |  |
|  |  |  |
| --- | --- | --- |
|  | =p∗​(xj∣xj+1,…,xp).absentsuperscript𝑝conditionalsubscript𝑥𝑗  subscript𝑥𝑗1…subscript𝑥𝑝\displaystyle=p^{\*}(x\_{j}\mid x\_{j+1},\ldots,x\_{p}). |  |

Thus we need to learn p∗​(xj∣xj+1,…,xp)superscript𝑝conditionalsubscript𝑥𝑗

subscript𝑥𝑗1…subscript𝑥𝑝p^{\*}(x\_{j}\mid x\_{j+1},\ldots,x\_{p}) to successfully impute all patterns m𝑚m where xjsubscript𝑥𝑗x\_{j} is not observed. We assume that Ljsubscript𝐿𝑗L\_{j} is not empty for any j𝑗j. As all previous variables have been imputed and xjsubscript𝑥𝑗x\_{j} is observed, it is thus possible to learn the full distribution p∗​(xj,xj+1,…,xp∣M=m)superscript𝑝

subscript𝑥𝑗subscript𝑥𝑗1…conditionalsubscript𝑥𝑝𝑀
𝑚p^{\*}(x\_{j},x\_{j+1},\ldots,x\_{p}\mid M=m) for all m∈Lj𝑚subscript𝐿𝑗m\in L\_{j}. With the same arguments as in the proof of Proposition [2.3](#S2.Thmproposition3 "Proposition 2.3. ‣ 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), we then obtain that

|  |  |  |
| --- | --- | --- |
|  | h∗​(xj∣xj+1,…,xp)=p∗​(xj∣xj+1,…,xp)superscriptℎconditionalsubscript𝑥𝑗  subscript𝑥𝑗1…subscript𝑥𝑝superscript𝑝conditionalsubscript𝑥𝑗  subscript𝑥𝑗1…subscript𝑥𝑝\displaystyle h^{\*}(x\_{j}\mid x\_{j+1},\ldots,x\_{p})=p^{\*}(x\_{j}\mid x\_{j+1},\ldots,x\_{p}) |  |

Thus we have shown that the learned (imputation) distribution is indeed the correct one. It then also holds that

|  |  |  |  |
| --- | --- | --- | --- |
|  | h∗​(x)superscriptℎ𝑥\displaystyle h^{\*}(x) | =∑m∈ℳℙ​(M=m)​h∗​(oc​(x,m)∣o​(x,m),M=m)​h∗​(o​(x,m)∣M=m)absentsubscript𝑚ℳℙ𝑀𝑚superscriptℎconditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscriptℎconditional𝑜𝑥𝑚𝑀𝑚\displaystyle=\sum\_{m\in\mathcal{M}}{\mathbb{P}}(M=m)h^{\*}(o^{c}(x,m)\mid o(x,m),M=m)h^{\*}(o(x,m)\mid M=m) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑m∈ℳℙ​(M=m)​p∗​(oc​(x,m)∣o​(x,m),M=m)​p∗​(o​(x,m)∣M=m)absentsubscript𝑚ℳℙ𝑀𝑚superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚  𝑜𝑥𝑚𝑀𝑚superscript𝑝conditional𝑜𝑥𝑚𝑀𝑚\displaystyle=\sum\_{m\in\mathcal{M}}{\mathbb{P}}(M=m)p^{\*}(o^{c}(x,m)\mid o(x,m),M=m)p^{\*}(o(x,m)\mid M=m) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =p∗​(x),absentsuperscript𝑝𝑥\displaystyle=p^{\*}(x), |  |

whereby h∗​(o​(x,m)∣M=m)=p∗​(o​(x,m)∣M=m)superscriptℎconditional𝑜𝑥𝑚𝑀𝑚superscript𝑝conditional𝑜𝑥𝑚𝑀𝑚h^{\*}(o(x,m)\mid M=m)=p^{\*}(o(x,m)\mid M=m) by assumption and h∗​(oc​(x,m)∣o​(x,m),M=m)=p∗​(oc​(x,m)∣o​(x,m),M=m)superscriptℎconditionalsuperscript𝑜𝑐𝑥𝑚

𝑜𝑥𝑚𝑀𝑚superscript𝑝conditionalsuperscript𝑜𝑐𝑥𝑚

𝑜𝑥𝑚𝑀𝑚h^{\*}(o^{c}(x,m)\mid o(x,m),M=m)=p^{\*}(o^{c}(x,m)\mid o(x,m),M=m) as shown above.
∎

See [4.1](#S4.Thmproposition1 "Proposition 4.1. ‣ 4 Assessing Imputation Methods ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")

###### Proof.

We show that for each j𝑗j,

|  |  |  |
| --- | --- | --- |
|  | SN​Ae​s​(H,P)≤SN​Ae​s​(P∗,P)superscriptsubscript𝑆𝑁𝐴𝑒𝑠𝐻𝑃superscriptsubscript𝑆𝑁𝐴𝑒𝑠superscript𝑃𝑃S\_{NA}^{es}(H,P)\leq S\_{NA}^{es}(P^{\*},P) |  |

holds. Indeed, by propriety of the energy score 𝔼​[e​s​(HXj∣xO,Y)]≤𝔼​[e​s​(HXj∣xO∗,Y)]𝔼delimited-[]𝑒𝑠subscript𝐻conditionalsubscript𝑋𝑗subscript𝑥𝑂𝑌𝔼delimited-[]𝑒𝑠subscriptsuperscript𝐻conditionalsubscript𝑋𝑗subscript𝑥𝑂𝑌{\mathbb{E}}[es(H\_{X\_{j}\mid x\_{O}},Y)]\leq{\mathbb{E}}[es(H^{\*}\_{X\_{j}\mid x\_{O}},Y)], when Y∼HXj∣xO∗similar-to𝑌subscriptsuperscript𝐻conditionalsubscript𝑋𝑗subscript𝑥𝑂Y\sim H^{\*}\_{X\_{j}\mid x\_{O}}. Taking expectations on both sides shows that

|  |  |  |  |
| --- | --- | --- | --- |
|  | SN​Ae​s​(H,P)=𝔼​[𝔼​[e​s​(HXj∣XO,Y)]]≤𝔼​[𝔼​[e​s​(HXj∣XO∗,Y)]].superscriptsubscript𝑆𝑁𝐴𝑒𝑠𝐻𝑃𝔼delimited-[]𝔼delimited-[]𝑒𝑠subscript𝐻conditionalsubscript𝑋𝑗subscript𝑋𝑂𝑌𝔼delimited-[]𝔼delimited-[]𝑒𝑠subscriptsuperscript𝐻conditionalsubscript𝑋𝑗subscript𝑋𝑂𝑌\displaystyle S\_{NA}^{es}(H,P)={\mathbb{E}}[{\mathbb{E}}[es(H\_{X\_{j}\mid X\_{O}},Y)]]\leq{\mathbb{E}}[{\mathbb{E}}[es(H^{\*}\_{X\_{j}\mid X\_{O}},Y)]]. |  | (C.4) |

Moreover, similar to Proposition [2.3](#S2.Thmproposition3 "Proposition 2.3. ‣ 2.2 FCS in MAR ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE."), it can be shown that 𝔼​[e​s​(HXj∣xO∗,Y)]=𝔼​[e​s​(PXj∣xO∗,Y)]𝔼delimited-[]𝑒𝑠subscriptsuperscript𝐻conditionalsubscript𝑋𝑗subscript𝑥𝑂𝑌𝔼delimited-[]𝑒𝑠subscriptsuperscript𝑃conditionalsubscript𝑋𝑗subscript𝑥𝑂𝑌{\mathbb{E}}[es(H^{\*}\_{X\_{j}\mid x\_{O}},Y)]={\mathbb{E}}[es(P^{\*}\_{X\_{j}\mid x\_{O}},Y)]. We repeat the argument here for completeness: First

|  |  |  |  |
| --- | --- | --- | --- |
|  | h∗​(xj∣xO)superscriptℎconditionalsubscript𝑥𝑗subscript𝑥𝑂\displaystyle h^{\*}(x\_{j}\mid x\_{O}) | =h∗​(xj,xO)h∗​(xO)absentsuperscriptℎsubscript𝑥𝑗subscript𝑥𝑂superscriptℎsubscript𝑥𝑂\displaystyle=\frac{h^{\*}(x\_{j},x\_{O})}{h^{\*}(x\_{O})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑m∈Ljℙ​(M=m∣xj,xO)⋅p∗​(xj,xO)∑m∈Ljℙ​(M=m∣xO)⋅p∗​(xO)absentsubscript𝑚subscript𝐿𝑗⋅ℙ𝑀conditional𝑚  subscript𝑥𝑗subscript𝑥𝑂superscript𝑝subscript𝑥𝑗subscript𝑥𝑂subscript𝑚subscript𝐿𝑗⋅ℙ𝑀conditional𝑚subscript𝑥𝑂superscript𝑝subscript𝑥𝑂\displaystyle=\frac{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{j},x\_{O})\cdot p^{\*}(x\_{j},x\_{O})}{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{O})\cdot p^{\*}(x\_{O})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =p∗​(xj∣xO)​∑m∈Ljℙ​(M=m∣xj,xO)∑m∈Ljℙ​(M=m∣xO).absentsuperscript𝑝conditionalsubscript𝑥𝑗subscript𝑥𝑂subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚  subscript𝑥𝑗subscript𝑥𝑂subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚subscript𝑥𝑂\displaystyle=p^{\*}(x\_{j}\mid x\_{O})\frac{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{j},x\_{O})}{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{O})}. |  |

It only remains to show that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑m∈Ljℙ​(M=m∣xj,xO)∑m∈Ljℙ​(M=m∣xO)=1.subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚  subscript𝑥𝑗subscript𝑥𝑂subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚subscript𝑥𝑂1\displaystyle\frac{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{j},x\_{O})}{\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{O})}=1. |  | (C.5) |

Indeed, we note that for any m∈Ljc𝑚superscriptsubscript𝐿𝑗𝑐m\in L\_{j}^{c},

|  |  |  |
| --- | --- | --- |
|  | ℙ​(M=m∣xj,xO)=ℙ​(M=m∣xO),ℙ𝑀conditional𝑚  subscript𝑥𝑗subscript𝑥𝑂ℙ𝑀conditional𝑚subscript𝑥𝑂\displaystyle{\mathbb{P}}(M=m\mid x\_{j},x\_{O})={\mathbb{P}}(M=m\mid x\_{O}), |  |

by ([SM-MAR](#S2.Ex3 "In Definition 2.1. ‣ 2.1 MAR Definitions ‣ 2 Sequential Imputation under MAR ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")). Consequently,

|  |  |  |
| --- | --- | --- |
|  | 1=∑m∈Ljℙ​(M=m∣xj,xO)+∑m∈Ljcℙ​(M=m∣xO),1subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚  subscript𝑥𝑗subscript𝑥𝑂subscript𝑚superscriptsubscript𝐿𝑗𝑐ℙ𝑀conditional𝑚subscript𝑥𝑂\displaystyle 1=\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{j},x\_{O})+\sum\_{m\in L\_{j}^{c}}{\mathbb{P}}(M=m\mid x\_{O}), |  |

so that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑m∈Ljℙ​(M=m∣xj,xO)subscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚  subscript𝑥𝑗subscript𝑥𝑂\displaystyle\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{j},x\_{O}) | =1−∑m∈Ljcℙ​(M=m∣xO)absent1subscript𝑚superscriptsubscript𝐿𝑗𝑐ℙ𝑀conditional𝑚subscript𝑥𝑂\displaystyle=1-\sum\_{m\in L\_{j}^{c}}{\mathbb{P}}(M=m\mid x\_{O}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑m∈Ljℙ​(M=m∣xO).absentsubscript𝑚subscript𝐿𝑗ℙ𝑀conditional𝑚subscript𝑥𝑂\displaystyle=\sum\_{m\in L\_{j}}{\mathbb{P}}(M=m\mid x\_{O}). |  |

It follows that,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[𝔼​[e​s​(HXj∣XO∗,Y)]]=𝔼​[𝔼​[e​s​(PXj∣XO∗,Y)]]=SN​Ae​s​(P∗,P).𝔼delimited-[]𝔼delimited-[]𝑒𝑠subscriptsuperscript𝐻conditionalsubscript𝑋𝑗subscript𝑋𝑂𝑌𝔼delimited-[]𝔼delimited-[]𝑒𝑠subscriptsuperscript𝑃conditionalsubscript𝑋𝑗subscript𝑋𝑂𝑌superscriptsubscript𝑆𝑁𝐴𝑒𝑠superscript𝑃𝑃\displaystyle{\mathbb{E}}[{\mathbb{E}}[es(H^{\*}\_{X\_{j}\mid X\_{O}},Y)]]={\mathbb{E}}[{\mathbb{E}}[es(P^{\*}\_{X\_{j}\mid X\_{O}},Y)]]=S\_{NA}^{es}(P^{\*},P). |  | (C.6) |

Combining ([C.4](#A3.E4 "In Proof. ‣ C.2 Proofs ‣ Appendix C Proofs and Additional Results ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) and ([C.6](#A3.E6 "In Proof. ‣ C.2 Proofs ‣ Appendix C Proofs and Additional Results ‣ What Is a Good Imputation Under MAR Missingness?1footnote 11footnote 1We thank Giulia Marchello for providing the code for GAIN and MIWAE.")) gives the result.
∎

## References

* Anil Jadhav and Ramanathan, (2019)

  Anil Jadhav, D. P. and Ramanathan, K. (2019).
  Comparison of performance of data imputation methods for numeric
  dataset.
  Applied Artificial Intelligence, 33(10):913–933.
* Bertsimas et al., (2018)

  Bertsimas, D., Pawlowski, C., and Zhuo, Y. D. (2018).
  From predictive methods to missing data imputation: An optimization
  approach.
  Journal of Machine Learning Research, 18(196):1–39.
* Burgette and Reiter, (2010)

  Burgette, L. F. and Reiter, J. P. (2010).
  Multiple Imputation for Missing Data via Sequential Regression
  Trees.
  American Journal of Epidemiology, 172(9):1070–1076.
* Ćevid et al., (2022)

  Ćevid, D., Michel, L., Näf, J., Meinshausen, N., and Bühlmann, P.
  (2022).
  Distributional random forests: Heterogeneity adjustment and
  multivariate distributional regression.
  Journal of Machine Learning Research, 23(333):1–79.
* Deng et al., (2022)

  Deng, G., Han, C., and Matteson, D. S. (2022).
  Extended missing data imputation via GANs for ranking applications.
  Data Mining and Knowledge Discovery, 36(4):1498–1520.
* Dong et al., (2021)

  Dong, W., Fong, D. Y. T., Yoon, J.-s., Wan, E. Y. F., Bedford, L. E., Tang, E.
  H. M., and Lam, C. L. K. (2021).
  Generative adversarial networks for imputing missing data for big
  data clinical research.
  BMC Medical Research Methodology, 21(1):78.
* Doove et al., (2014)

  Doove, L., Van Buuren, S., and Dusseldorp, E. (2014).
  Recursive partitioning for missing data imputation in the presence of
  interaction effects.
  Computational Statistics & Data Analysis, 72:92–104.
* Doretti et al., (2018)

  Doretti, M., Geneletti, S., and Stanghellini, E. (2018).
  Missing data: A unified taxonomy guided by conditional independence.
  International Statistical Review, 86(2):189–204.
* Fang and Bao, (2023)

  Fang, F. and Bao, S. (2023).
  FragmGAN: Generative adversarial nets for fragmentary data
  imputation and prediction.
  Statistical Theory and Related Fields, 0(0):1–14.
* Gneiting and Raftery, (2007)

  Gneiting, T. and Raftery, A. E. (2007).
  Strictly proper scoring rules, prediction, and estimation.
  Journal of the American Statistical Association,
  102(477):359–378.
* Gneiting et al., (2008)

  Gneiting, T., Stanberry, L. I., Grimit, E. P., Held, L., and Johnson, N. A.
  (2008).
  Assessing probabilistic forecasts of multivariate quantities, with an
  application to ensemble predictions of surface winds.
  TEST, 17(2):211–235.
* Hong and Lynn, (2020)

  Hong, S. and Lynn, H. S. (2020).
  Accuracy of random-forest-based imputation of missing data in the
  presence of non-normality, non-linearity, and interaction.
  BMC Medical Research Methodology, 20(1):199.
* Ibrahim et al., (1999)

  Ibrahim, J. G., Lipsitz, S. R., and Chen, M.-H. (1999).
  Missing covariates in generalized linear models when the missing data
  mechanism is non-ignorable.
  Journal of the Royal Statistical Society: Series B (Statistical
  Methodology), 61(1):173–190.
* Jäger et al., (2021)

  Jäger, S., Allhorn, A., and Bießmann, F. (2021).
  A benchmark for data imputation methods.
  Frontiers in Big Data, 4.
* Lee and Mitra, (2016)

  Lee, M. C. and Mitra, R. (2016).
  Multiply imputing missing values in data sets with mixed measurement
  scales using a sequence of generalised linear models.
  Computational Statistics & Data Analysis, 95:24–38.
* Lichman, (2013)

  Lichman, M. (2013).
  UCI machine learning repository.
* Little, (1993)

  Little, R. J. A. (1993).
  Pattern-mixture models for multivariate incomplete data.
  Journal of the American Statistical Association,
  88(421):125–134.
* Little and Rubin, (1986)

  Little, R. J. A. and Rubin, D. B. (1986).
  Statistical Analysis with Missing Data.
  John Wiley & Sons, Inc.
* Liu et al., (2014)

  Liu, J., Gelman, A., Hill, J., Su, Y.-S., and Kropko, J. (2014).
  On the stationary distribution of iterative imputations.
  Biometrika, (1):155–173.
* Mattei and Frellsen, (2019)

  Mattei, P.-A. and Frellsen, J. (2019).
  MIWAE: Deep generative modelling and imputation of incomplete data
  sets.
  In Proceedings of the 36th International Conference on Machine
  Learning, volume 97 of Proceedings of Machine Learning Research, pages
  4413–4423.
* Mealli and Rubin, (2015)

  Mealli, F. and Rubin, D. B. (2015).
  Clarifying missing at random and related definitions, and
  implications when coupled with exchangeability.
  Biometrika, 102(4):995–1000.
* Molenberghs et al., (2008)

  Molenberghs, G., Beunckens, C., Sotto, C., and Kenward, M. G. (2008).
  Every missingness not at random model has a missingness at random
  counterpart with equal fit.
  Journal of the Royal Statistical Society. Series B (Statistical
  Methodology), 70(2):371–388.
* Murray, (2018)

  Murray, J. S. (2018).
  Multiple imputation: A review of practical and theoretical findings.
  Statistical Science, 33(2):142 – 159.
* Muzellec et al., (2020)

  Muzellec, B., Josse, J., Boyer, C., and Cuturi, M. (2020).
  Missing data imputation using optimal transport.
  In Proceedings of the 37th International Conference on Machine
  Learning, volume 119 of Proceedings of Machine Learning Research,
  pages 7130–7140.
* Nazábal et al., (2020)

  Nazábal, A., Olmos, P. M., Ghahramani, Z., and Valera, I. (2020).
  Handling incomplete heterogeneous data using VAEs.
  Pattern Recognition, 107:107501.
* Näf et al., (2023)

  Näf, J., Spohn, M.-L., Michel, L., and Meinshausen, N. (2023).
  Imputation scores.
  The Annals of Applied Statistics, 17(3):2452 – 2472.
* Qiu et al., (2020)

  Qiu, Y. L., Zheng, H., and Gevaert, O. (2020).
  Genomic data imputation with variational auto-encoders.
  GigaScience, 9(8):giaa082.
* Rianne Margaretha Schouten and Vink, (2018)

  Rianne Margaretha Schouten, P. L. and Vink, G. (2018).
  Generating missing values for simulation purposes: a multivariate
  amputation procedure.
  Journal of Statistical Computation and Simulation,
  88(15):2909–2930.
* Rizzo and Szekely, (2022)

  Rizzo, M. and Szekely, G. (2022).
  energy: E-Statistics: Multivariate Inference via the Energy of
  Data.
  R package version 1.7-11.
* Rubin, (1976)

  Rubin, D. B. (1976).
  Inference and missing data.
  Biometrika, 63(3):581–592.
* Schafer, (1997)

  Schafer, J. L. (1997).
  Analysis of incomplete multivariate data.
  Chapman and Hall/CRC.
* Seaman et al., (2013)

  Seaman, S., Galati, J., Jackson, D., and Carlin, J. (2013).
  What is meant by “Missing at Random”?
  Statistical Science, 28(2):257–268.
* Stekhoven, (2022)

  Stekhoven, D. J. (2022).
  missForest: Nonparametric Missing Value Imputation using Random
  Forest.
  R package version 1.5.
* Stekhoven and Bühlmann, (2011)

  Stekhoven, D. J. and Bühlmann, P. (2011).
  MissForest—non-parametric missing value imputation for mixed-type
  data.
  Bioinformatics, 28(1):112–118.
* Székely, (2003)

  Székely, G. J. (2003).
  E-statistics: the energy of statistical samples.
  Technical Report 05, Bowling Green State University, Department of
  Mathematics and Statistics.
* Tang and Ishwaran, (2017)

  Tang, F. and Ishwaran, H. (2017).
  Random forest missing data algorithms.
  Stat Anal Data Min, 10(6):363–377.
* Tian, (2017)

  Tian, J. (2017).
  Recovering probability distributions from missing data.
  In Proceedings of the Ninth Asian Conference on Machine
  Learning, volume 77 of Proceedings of Machine Learning Research, pages
  574–589.
* Ushey et al., (2024)

  Ushey, K., Allaire, J., and Tang, Y. (2024).
  reticulate: Interface to ’Python’.
  R package version 1.35.0, https://github.com/rstudio/reticulate.
* van Buuren, (2007)

  van Buuren, S. (2007).
  Multiple imputation of discrete and continuous data by fully
  conditional specification.
  Stat Methods Med Res, 16(3):219–242.
* van Buuren, (2018)

  van Buuren, S. (2018).
  Flexible Imputation of Missing Data. Second Edition.
  Chapman & Hall/CRC Press.
* van Buuren and Groothuis-Oudshoorn, (2011)

  van Buuren, S. and Groothuis-Oudshoorn, K. (2011).
  mice: Multivariate imputation by chained equations in R.
  Journal of Statistical Software, 45(3):1–67.
* Waljee et al., (2013)

  Waljee, A. K., Mukherjee, A., Singal, A. G., Zhang, Y., Warren, J., Balis, U.,
  Marrero, J., Zhu, J., and Higgins, P. D. (2013).
  Comparison of imputation methods for missing laboratory data in
  medicine.
  BMJ open, 3(8):e002847.
* Wang et al., (2022)

  Wang, Z., Akande, O., Poulos, J., and Li, F. (2022).
  Are deep learning models superior for missing data imputation in
  surveys? Evidence from an empirical comparison.
  Survey Methodology, 48(2).
* Xu et al., (2016)

  Xu, D., Daniels, M. J., and Winterstein, A. G. (2016).
  Sequential BART for imputation of missing covariates.
  Biostatistics, 17(3):589–602.
* Yoon et al., (2018)

  Yoon, J., Jordon, J., and van der Schaar, M. (2018).
  GAIN: Missing data imputation using generative adversarial nets.
  In Proceedings of the 35th International Conference on Machine
  Learning, volume 80 of Proceedings of Machine Learning Research, pages
  5689–5698.
* Yuan et al., (2021)

  Yuan, Y., Shen, Y., Wang, J., Liu, Y., and Zhang, L. (2021).
  VAEM: a deep generative model for heterogeneous mixed type data.
  In Advances in Neural Information Processing Systems 34, pages
  4044–4054.
* Zhu and Raghunathan, (2015)

  Zhu, J. and Raghunathan, T. E. (2015).
  Convergence properties of a sequential regression multiple imputation
  algorithm.
  Journal of the American Statistical Association,
  110(511):1112–1124.

[◄](/html/2403.19195)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2403.19196)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2403.19196)
[View original  
on arXiv](https://arxiv.org/abs/2403.19196)[►](/html/2403.19197)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Apr 5 15:13:35 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
