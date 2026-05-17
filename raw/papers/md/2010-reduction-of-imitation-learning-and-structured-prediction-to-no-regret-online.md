---
arxiv: '1011.0686'
authors:
- Stephane Ross
- Geoffrey J. Gordon
- J. Andrew Bagnell
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online
  Learning
url: https://arxiv.org/abs/1011.0686
year: 2010
---

[1011.0686] 1 INTRODUCTION

else if(!window.matchMedia) { return false; }
else if(window.matchMedia("(prefers-color-scheme: dark)").matches) {
theme = "dark"; }
if (theme=="dark") {
document.documentElement.setAttribute("data-theme", "dark");
} else {
document.documentElement.setAttribute("data-theme", "light"); } }

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

\newindex

todotodotndTodo List

A Reduction of Imitation Learning and Structured Prediction
  
to No-Regret Online Learning

Stéphane Ross
                      

Geoffrey J. Gordon
                      

J. Andrew Bagnell

Robotics Institute

Carnegie Mellon University

Pittsburgh, PA 15213, USA

stephaneross@cmu.edu
                      

Machine Learning Department

Carnegie Mellon University

Pittsburgh, PA 15213, USA

ggordon@cs.cmu.edu
                      

Robotics Institute

Carnegie Mellon University

Pittsburgh, PA 15213, USA

dbagnell@ri.cmu.edu

###### Abstract

Sequential prediction problems such as imitation learning, where future observations depend on previous predictions (actions), violate the common i.i.d. assumptions made in statistical learning. This leads to poor performance in theory and often in practice. Some recent approaches (Daumé III et al., [2009](#bib.bib7); Ross and Bagnell, [2010](#bib.bib17)) provide stronger guarantees in this setting, but remain somewhat unsatisfactory as they train either non-stationary or stochastic policies and require a large number of iterations. In this paper, we propose a new iterative algorithm, which trains a stationary deterministic policy, that can be seen as a no regret algorithm in an online learning setting. We show that any such no regret algorithm, combined with additional reduction assumptions, must find a policy with good performance under the distribution of observations it induces in such sequential settings. We demonstrate that this new approach outperforms previous approaches on two challenging imitation learning problems and a benchmark sequence labeling problem.

## 1 INTRODUCTION

Sequence Prediction problems arise commonly in practice. For instance, most robotic systems must be able to predict/make a sequence of actions given a sequence of observations revealed to them over time. In complex robotic systems where standard control methods fail, we must often resort to learning a controller that can make such predictions. Imitation learning techniques, where expert demonstrations of good behavior are used to learn a controller, have proven very useful in practice and have led to state-of-the art performance in a variety of applications (Schaal, [1999](#bib.bib18); Abbeel and Ng, [2004](#bib.bib1); Ratliff et al., [2006](#bib.bib13); Silver et al., [2008](#bib.bib19); Argall et al., [2009](#bib.bib2); Chernova and Veloso, [2009](#bib.bib6); Ross and Bagnell, [2010](#bib.bib17)). A typical approach to imitation learning is to train a classifier or regressor to predict an expert’s behavior given training data of the encountered observations (input) and actions (output) performed by the expert. However since the learner’s prediction affects future input observations/states during execution of the learned policy, this violate the crucial i.i.d. assumption made by most statistical learning approaches.

Ignoring this issue leads to poor performance both in theory and practice (Ross and Bagnell, [2010](#bib.bib17)). In particular, a classifier that makes a mistake with probability ϵitalic-ϵ\epsilon under the distribution of states/observations encountered by the expert can make as many as T2​ϵsuperscript𝑇2italic-ϵT^{2}\epsilon mistakes in expectation over T𝑇T-steps under the distribution of states the classifier itself induces (Ross and Bagnell, [2010](#bib.bib17)). Intuitively this is because as soon as the learner makes a mistake, it may encounter completely different observations than those under expert demonstration, leading to a compounding of errors.

Recent approaches (Ross and Bagnell, [2010](#bib.bib17)) can guarantee an expected number of mistakes linear (or nearly so) in the task horizon T𝑇T and error ϵitalic-ϵ\epsilon by training over several iterations and allowing the learner to influence the input states where expert demonstration is provided (through execution of its own controls in the system). One approach (Ross and Bagnell, [2010](#bib.bib17)) learns a non-stationary policy by training a different policy for each time step in sequence, starting from the first step. Unfortunately this is impractical when T𝑇T is large or ill-defined. Another approach called SMILe (Ross and Bagnell, [2010](#bib.bib17)), similar to SEARN (Daumé III et al., [2009](#bib.bib7)) and CPI (Kakade and Langford, [2002](#bib.bib10)), trains a stationary stochastic policy (a finite mixture of policies) by adding a new policy to the mixture at each iteration of training. However this may be unsatisfactory for practical applications as some policies in the mixture are worse than others and the learned controller may be unstable.

We propose a new meta-algorithm for imitation learning which learns a stationary deterministic policy guaranteed to perform well under its induced distribution of states (number of mistakes/costs that grows linearly in T𝑇T and classification cost ϵitalic-ϵ\epsilon). We take a reduction-based approach (Beygelzimer et al., [2005](#bib.bib3)) that enables reusing existing supervised learning algorithms.
Our approach is simple to implement, has no free parameters except the supervised learning algorithm sub-routine, and requires a
number of iterations that scales nearly linearly with the effective horizon of the problem. It naturally handles continuous as well as discrete
predictions. Our approach is closely related to no regret online learning algorithms (Cesa-Bianchi et al., [2004](#bib.bib5); Hazan et al., [2006](#bib.bib8); Kakade and Shalev-Shwartz, [2008](#bib.bib11)) (in particular *Follow-The-Leader*) but better leverages the expert in our setting. Additionally, we show that any no-regret learner can be used in a particular fashion to learn a policy that achieves similar guarantees.

We begin by establishing our notation and setting, discuss related work, and then present the DAgger (Dataset Aggregation) method. We analyze this approach using a no-regret and a reduction approach (Beygelzimer et al., [2005](#bib.bib3)). Beyond the reduction analysis, we consider the sample complexity of our approach using online-to-batch (Cesa-Bianchi et al., [2004](#bib.bib5)) techniques. We demonstrate DAgger is scalable and outperforms previous approaches in practice on two challenging imitation learning problems: 1) learning to steer a car in a 3D racing game (*Super Tux Kart*) and 2) and learning to play *Super Mario Bros.*, given input image features and corresponding actions by a human expert and near-optimal planner respectively. Following Daumé III et al. ([2009](#bib.bib7)) in treating structured prediction as a degenerate imitation learning problem, we apply DAgger to the OCR (Taskar et al., [2003](#bib.bib20)) benchmark prediction problem achieving results competitive with the state-of-the-art (Taskar et al., [2003](#bib.bib20); Ratliff et al., [2007](#bib.bib14); Daumé III et al., [2009](#bib.bib7)) using only single-pass, greedy prediction.

## 2 PRELIMINARIES

We begin by introducing notation relevant to our setting. We denote by ΠΠ\Pi the class of policies the learner is considering and T𝑇T the task horizon. For any policy π𝜋\pi, we let dπtsubscriptsuperscript𝑑𝑡𝜋d^{t}\_{\pi} denote the distribution of states at time t𝑡t if the learner executed policy π𝜋\pi from time step 111 to t−1𝑡1t-1. Furthermore, we denote dπ=1T​∑t=1Tdπtsubscript𝑑𝜋1𝑇superscriptsubscript𝑡1𝑇subscriptsuperscript𝑑𝑡𝜋d\_{\pi}=\frac{1}{T}\sum\_{t=1}^{T}d^{t}\_{\pi} the average distribution of states if we follow policy π𝜋\pi for T𝑇T steps. Given a state s𝑠s, we denote C​(s,a)𝐶𝑠𝑎C(s,a) the expected immediate cost of performing action a𝑎a in state s𝑠s for the task we are considering and denote Cπ​(s)=𝔼a∼π​(s)​[C​(s,a)]subscript𝐶𝜋𝑠subscript𝔼similar-to𝑎𝜋𝑠delimited-[]𝐶𝑠𝑎C\_{\pi}(s)=\mathbb{E}\_{a\sim\pi(s)}[C(s,a)] the expected immediate cost of π𝜋\pi in s𝑠s. We assume C𝐶C is bounded in [0,1]01[0,1]. The total cost of executing policy π𝜋\pi for T𝑇T-steps (*i.e.*, the cost-to-go) is denoted J​(π)=∑t=1T𝔼s∼dπt​[Cπ​(s)]=T​𝔼s∼dπ​[Cπ​(s)]𝐽𝜋superscriptsubscript𝑡1𝑇subscript𝔼similar-to𝑠subscriptsuperscript𝑑𝑡𝜋delimited-[]subscript𝐶𝜋𝑠𝑇subscript𝔼similar-to𝑠subscript𝑑𝜋delimited-[]subscript𝐶𝜋𝑠J(\pi)=\sum\_{t=1}^{T}\mathbb{E}\_{s\sim d^{t}\_{\pi}}[C\_{\pi}(s)]=T\mathbb{E}\_{s\sim d\_{\pi}}[C\_{\pi}(s)].

In imitation learning, we may not necessarily know or observe true costs C​(s,a)𝐶𝑠𝑎C(s,a) for the particular task. Instead, we observe expert demonstrations and seek to bound J​(π)𝐽𝜋J(\pi) for any cost function C𝐶C based on how well π𝜋\pi mimics the expert’s policy π∗superscript𝜋\pi^{\*}. Denote ℓℓ\ell the observed surrogate loss function we minimize instead of C𝐶C. For instance ℓ​(s,π)ℓ𝑠𝜋\ell(s,\pi) may be the expected 0-1 loss of π𝜋\pi with respect to π∗superscript𝜋\pi^{\*} in state s𝑠s, or a squared/hinge loss of π𝜋\pi with respect to π∗superscript𝜋\pi^{\*} in s𝑠s. Importantly, in many instances, C𝐶C and ℓℓ\ell may be the same function– for instance, if we are interested in optimizing the learner’s ability to predict the actions chosen by an expert.

Our goal is to find a policy π^^𝜋\hat{\pi} which minimizes the observed surrogate loss under its induced distribution of states, i.e.:

|  |  |  |  |
| --- | --- | --- | --- |
|  | π^=arg​minπ∈Π⁡𝔼s∼dπ​[ℓ​(s,π)]^𝜋subscriptargmin𝜋Πsubscript𝔼similar-to𝑠subscript𝑑𝜋delimited-[]ℓ𝑠𝜋\hat{\pi}=\operatorname\*{arg\,min}\_{\pi\in\Pi}\mathbb{E}\_{s\sim d\_{\pi}}[\ell(s,\pi)] |  | (1) |

As system dynamics are assumed both unknown and complex, we cannot compute dπsubscript𝑑𝜋d\_{\pi} and can only sample it by executing π𝜋\pi in the system. Hence this is a non-i.i.d. supervised learning problem due to the dependence of the input distribution on the policy π𝜋\pi itself. The interaction between
policy and the resulting distribution makes optimization difficult as it results in a non-convex objective even if the loss ℓ​(s,⋅)ℓ𝑠⋅\ell(s,\cdot{}) is convex in π𝜋\pi for all states s𝑠s. We now briefly review previous approaches and their guarantees.

### 2.1 Supervised Approach to Imitation

The traditional approach to imitation learning ignores the change in distribution and simply trains a policy π𝜋\pi that performs well under the distribution of states encountered by the expert dπ∗subscript𝑑superscript𝜋d\_{\pi^{\*}}. This can be achieved using any standard supervised learning algorithm. It finds the policy π^s​u​psubscript^𝜋𝑠𝑢𝑝\hat{\pi}\_{sup}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | π^s​u​p=arg​minπ∈Π⁡𝔼s∼dπ∗​[ℓ​(s,π)]subscript^𝜋𝑠𝑢𝑝subscriptargmin𝜋Πsubscript𝔼similar-to𝑠subscript𝑑superscript𝜋delimited-[]ℓ𝑠𝜋\hat{\pi}\_{sup}=\operatorname\*{arg\,min}\_{\pi\in\Pi}\mathbb{E}\_{s\sim d\_{\pi^{\*}}}[\ell(s,\pi)] |  | (2) |

Assuming ℓ​(s,π)ℓ𝑠𝜋\ell(s,\pi) is the 0-1 loss (or upper bound on the 0-1 loss) implies the following performance guarantee with respect to any task cost function C𝐶C bounded in [0,1]01[0,1]:

###### Theorem 2.1.

(Ross and Bagnell, [2010](#bib.bib17)) Let 𝔼s∼dπ∗​[ℓ​(s,π)]=ϵsubscript𝔼similar-to𝑠subscript𝑑superscript𝜋delimited-[]ℓ𝑠𝜋italic-ϵ\mathbb{E}\_{s\sim d\_{\pi^{\*}}}[\ell(s,\pi)]=\epsilon, then J​(π)≤J​(π∗)+T2​ϵ𝐽𝜋𝐽superscript𝜋superscript𝑇2italic-ϵJ(\pi)\leq J(\pi^{\*})+T^{2}\epsilon.

###### Proof.

Follows from result in Ross and Bagnell ([2010](#bib.bib17)) since ϵitalic-ϵ\epsilon is an upper bound on the 0-1 loss of π𝜋\pi in dπ∗subscript𝑑superscript𝜋d\_{\pi^{\*}}.
∎

Note that this bound is tight, i.e. there exist problems such that a policy π𝜋\pi with ϵitalic-ϵ\epsilon 0-1 loss on dπ∗subscript𝑑superscript𝜋d\_{\pi^{\*}} can incur extra cost that grows quadratically in T𝑇T. Kääriäinen ([2006](#bib.bib9)) demonstrated this in a sequence prediction setting111In their example, an error rate of ϵ>0italic-ϵ0\epsilon>0 when trained to predict the next output in sequence with the previous correct output as input can lead to an expected number of mistakes of T2−1−(1−2​ϵ)T+14​ϵ+12𝑇21superscript12italic-ϵ𝑇14italic-ϵ12\frac{T}{2}-\frac{1-(1-2\epsilon)^{T+1}}{4\epsilon}+\frac{1}{2} over sequences of length T𝑇T at test time. This is bounded by T2​ϵsuperscript𝑇2italic-ϵT^{2}\epsilon and behaves as Θ​(T2​ϵ)Θsuperscript𝑇2italic-ϵ\Theta(T^{2}\epsilon) for small ϵitalic-ϵ\epsilon. and Ross and Bagnell ([2010](#bib.bib17)) provided an imitation learning example where J​(π^s​u​p)=(1−ϵ​T)​J​(π∗)+T2​ϵ𝐽subscript^𝜋𝑠𝑢𝑝1italic-ϵ𝑇𝐽superscript𝜋superscript𝑇2italic-ϵJ(\hat{\pi}\_{sup})=(1-\epsilon T)J(\pi^{\*})+T^{2}\epsilon. Hence the traditional supervised learning approach has poor performance guarantees due to the quadratic growth in T𝑇T. Instead we would prefer approaches that can guarantee growth linear or near-linear in T𝑇T and ϵitalic-ϵ\epsilon. The following two approaches from Ross and Bagnell ([2010](#bib.bib17)) achieve this on some classes of imitation learning problems, including all those where surrogate loss ℓℓ\ell upper bounds C𝐶C.

### 2.2 Forward Training

The forward training algorithm introduced by Ross and Bagnell ([2010](#bib.bib17)) trains a non-stationary policy (one policy πtsubscript𝜋𝑡\pi\_{t} for each time step t𝑡t) iteratively over T𝑇T iterations, where at iteration t𝑡t, πtsubscript𝜋𝑡\pi\_{t} is trained to mimic π∗superscript𝜋\pi^{\*} on the distribution of states at time t𝑡t induced by the previously trained policies π1,π2,…,πt−1

subscript𝜋1subscript𝜋2…subscript𝜋𝑡1\pi\_{1},\pi\_{2},\dots,\pi\_{t-1}. By doing so, πtsubscript𝜋𝑡\pi\_{t} is trained on the actual distribution of states it will encounter during execution of the learned policy. Hence the forward algorithm guarantees that the expected loss under the distribution of states induced by the learned policy matches the average loss during training, and hence improves performance.

We here provide a theorem slightly more general than the one provided by Ross and Bagnell ([2010](#bib.bib17)) that applies to any policy π𝜋\pi that can guarantee ϵitalic-ϵ\epsilon surrogate loss under its own distribution of states. This will be useful to bound the performance of our new approach presented in Section [3](#S3 "3 DATASET AGGREGATION").

Let Qtπ′​(s,π)subscriptsuperscript𝑄superscript𝜋′𝑡𝑠𝜋Q^{\pi^{\prime}}\_{t}(s,\pi) denote the t𝑡t-step cost of executing π𝜋\pi in initial state s𝑠s and then following policy π′superscript𝜋′\pi^{\prime} and assume ℓ​(s,π)ℓ𝑠𝜋\ell(s,\pi) is the 0-1 loss (or an upper bound on the 0-1 loss), then we have the following performance guarantee with respect to any task cost function C𝐶C bounded in [0,1]01[0,1]:

###### Theorem 2.2.

Let π𝜋\pi be such that 𝔼s∼dπ​[ℓ​(s,π)]=ϵsubscript𝔼similar-to𝑠subscript𝑑𝜋delimited-[]ℓ𝑠𝜋italic-ϵ\mathbb{E}\_{s\sim d\_{\pi}}[\ell(s,\pi)]=\epsilon, and QT−t+1π∗​(s,a)−QT−t+1π∗​(s,π∗)≤usubscriptsuperscript𝑄superscript𝜋𝑇𝑡1𝑠𝑎subscriptsuperscript𝑄superscript𝜋𝑇𝑡1𝑠superscript𝜋𝑢Q^{\pi^{\*}}\_{T-t+1}(s,a)-Q^{\pi^{\*}}\_{T-t+1}(s,\pi^{\*})\leq u for all action a𝑎a, t∈{1,2,…,T}𝑡12…𝑇t\in\{1,2,\dots,T\}, dπt​(s)>0subscriptsuperscript𝑑𝑡𝜋𝑠0d^{t}\_{\pi}(s)>0, then J​(π)≤J​(π∗)+u​T​ϵ𝐽𝜋𝐽superscript𝜋𝑢𝑇italic-ϵJ(\pi)\leq J(\pi^{\*})+uT\epsilon.

###### Proof.

We here follow a similar proof to Ross and Bagnell ([2010](#bib.bib17)). Given our policy π𝜋\pi, consider the policy π1:tsubscript𝜋:1𝑡\pi\_{1:t}, which executes π𝜋\pi in the first t𝑡t-steps and then execute the expert π∗superscript𝜋\pi^{\*}. Then

|  |  |  |
| --- | --- | --- |
|  | J​(π)=J​(π∗)+∑t=0T−1[J​(π1:T−t)−J​(π1:T−t−1)]=J​(π∗)+∑t=1T𝔼s∼dπt​[QT−t+1π∗​(s,π)−QT−t+1π∗​(s,π∗)]≤J​(π∗)+u​∑t=1T𝔼s∼dπt​[ℓ​(s,π)]=J​(π∗)+u​T​ϵ𝐽𝜋𝐽superscript𝜋superscriptsubscript𝑡0𝑇1delimited-[]𝐽subscript𝜋:1𝑇𝑡𝐽subscript𝜋:1𝑇𝑡1𝐽superscript𝜋superscriptsubscript𝑡1𝑇subscript𝔼similar-to𝑠subscriptsuperscript𝑑𝑡𝜋delimited-[]subscriptsuperscript𝑄superscript𝜋𝑇𝑡1𝑠𝜋subscriptsuperscript𝑄superscript𝜋𝑇𝑡1𝑠superscript𝜋𝐽superscript𝜋𝑢superscriptsubscript𝑡1𝑇subscript𝔼similar-to𝑠subscriptsuperscript𝑑𝑡𝜋delimited-[]ℓ𝑠𝜋𝐽superscript𝜋𝑢𝑇italic-ϵ\begin{array}[]{rl}\lx@intercol J(\pi)\hfil\lx@intercol\\ =&J(\pi^{\*})+\sum\_{t=0}^{T-1}[J(\pi\_{1:T-t})-J(\pi\_{1:T-t-1})]\\ =&J(\pi^{\*})+\sum\_{t=1}^{T}\mathbb{E}\_{s\sim d^{t}\_{\pi}}[Q^{\pi^{\*}}\_{T-t+1}(s,\pi)-Q^{\pi^{\*}}\_{T-t+1}(s,\pi^{\*})]\\ \leq&J(\pi^{\*})+u\sum\_{t=1}^{T}\mathbb{E}\_{s\sim d^{t}\_{\pi}}[\ell(s,\pi)]\\ =&J(\pi^{\*})+uT\epsilon\\ \end{array} |  |

The inequality follows from the fact that ℓ​(s,π)ℓ𝑠𝜋\ell(s,\pi) upper bounds the 0-1 loss, and hence the probability π𝜋\pi and π∗superscript𝜋\pi^{\*} pick different actions in s𝑠s; when they pick different actions, the increase in cost-to-go ≤uabsent𝑢\leq u.
∎

In the worst case, u𝑢u could be O​(T)𝑂𝑇O(T) and the forward algorithm wouldn’t provide any improvement over the traditional supervised learning approach. However, in many cases u𝑢u is O​(1)𝑂1O(1) or sub-linear in T𝑇T and the forward algorithm leads to improved performance. For instance if C𝐶C is the 0-1 loss with respect to the expert, then u≤1𝑢1u\leq 1. Additionally if π∗superscript𝜋\pi^{\*} is able to recover from mistakes made by π𝜋\pi, in the sense that within a few steps, π∗superscript𝜋\pi^{\*} is back in a distribution of states that is close to what π∗superscript𝜋\pi^{\*} would be in if π∗superscript𝜋\pi^{\*} had been executed initially instead of π𝜋\pi, then u𝑢u will be O​(1)𝑂1O(1). 222This is the case for instance in Markov Desision Processes (MDPs) when the Markov Chain defined by the system dynamics and policy π∗superscript𝜋\pi^{\*} is rapidly mixing. In particular, if it is α𝛼\alpha-mixing with exponential decay rate δ𝛿\delta then u𝑢u is O​(11−exp⁡(−δ))𝑂11𝛿O(\frac{1}{1-\exp(-\delta)}). A drawback of the forward algorithm is that it is impractical when T𝑇T is large (or undefined) as we must train T𝑇T different policies sequentially and cannot stop the algorithm before we complete all T𝑇T iterations. Hence it can not be applied to most real-world applications.

### 2.3 Stochastic Mixing Iterative Learning

SMILe, proposed by Ross and Bagnell ([2010](#bib.bib17)), alleviates this problem and can be applied in practice when T𝑇T is large or undefined by adopting an approach similar to SEARN (Daumé III et al., [2009](#bib.bib7)) where a stochastic stationary policy is trained over several iterations. Initially SMILe starts with a policy π0subscript𝜋0\pi\_{0} which always queries and executes the expert’s action choice. At iteration n𝑛n, a policy π^nsubscript^𝜋𝑛\hat{\pi}\_{n} is trained to mimic the expert under the distribution of trajectories πn−1subscript𝜋𝑛1\pi\_{n-1} induces and then updates πn=πn−1+α​(1−α)n−1​(π^n−π0)subscript𝜋𝑛subscript𝜋𝑛1𝛼superscript1𝛼𝑛1subscript^𝜋𝑛subscript𝜋0\pi\_{n}=\pi\_{n-1}+\alpha(1-\alpha)^{n-1}(\hat{\pi}\_{n}-\pi\_{0}). This update is interpreted as adding probability α​(1−α)n−1𝛼superscript1𝛼𝑛1\alpha(1-\alpha)^{n-1} to executing policy π^nsubscript^𝜋𝑛\hat{\pi}\_{n} at any step and removing probability α​(1−α)n−1𝛼superscript1𝛼𝑛1\alpha(1-\alpha)^{n-1} of executing the queried expert’s action. At iteration n𝑛n, πnsubscript𝜋𝑛\pi\_{n} is a mixture of n𝑛n policies and the probability of using the queried expert’s action is (1−α)nsuperscript1𝛼𝑛(1-\alpha)^{n}. We can stop the algorithm at any iteration N𝑁N by returning the re-normalized policy π~N=πN−(1−α)N​π01−(1−α)Nsubscript~𝜋𝑁subscript𝜋𝑁superscript1𝛼𝑁subscript𝜋01superscript1𝛼𝑁\tilde{\pi}\_{N}=\frac{\pi\_{N}-(1-\alpha)^{N}\pi\_{0}}{1-(1-\alpha)^{N}} which doesn’t query the expert anymore. Ross and Bagnell ([2010](#bib.bib17)) showed that choosing α𝛼\alpha in O​(1T2)𝑂1superscript𝑇2O(\frac{1}{T^{2}}) and N𝑁N in O​(T2​log⁡T)𝑂superscript𝑇2𝑇O(T^{2}\log T) guarantees near-linear regret in T𝑇T and ϵitalic-ϵ\epsilon for some class of problems.

## 3 DATASET AGGREGATION

We now present DAgger (Dataset Aggregation), an iterative algorithm that trains a deterministic policy that achieves good performance guarantees under its induced distribution of states.

In its simplest form, the algorithm proceeds as follows. At the first iteration, it uses the expert’s policy to gather a dataset of trajectories 𝒟𝒟\mathcal{D} and train a policy π^2subscript^𝜋2\hat{\pi}\_{2} that best mimics the expert on those trajectories. Then at iteration n𝑛n, it uses π^nsubscript^𝜋𝑛\hat{\pi}\_{n} to collect more trajectories and adds those trajectories to the dataset 𝒟𝒟\mathcal{D}. The next policy π^n+1subscript^𝜋𝑛1\hat{\pi}\_{n+1} is the policy that best mimics the expert on the whole dataset 𝒟𝒟\mathcal{D}. In other words, DAgger proceeds by collecting a dataset at each iteration under the current policy and trains the next policy under the aggregate of all collected datasets. The intuition behind this algorithm is that over the iterations, we are building up the set of inputs that the learned policy is likely to encounter during its execution based on previous experience (training iterations). This algorithm can be interpreted as a *Follow-The-Leader* algorithm in that at iteration n𝑛n we pick the best policy π^n+1subscript^𝜋𝑛1\hat{\pi}\_{n+1} in hindsight, i.e. under all trajectories seen so far over the iterations.

To better leverage the presence of the expert in our imitation learning setting, we optionally allow the algorithm to use a modified policy πi=βi​π∗+(1−βi)​π^isubscript𝜋𝑖subscript𝛽𝑖superscript𝜋1subscript𝛽𝑖subscript^𝜋𝑖\pi\_{i}=\beta\_{i}\pi^{\*}+(1-\beta\_{i})\hat{\pi}\_{i} at iteration i𝑖i that queries the expert to choose controls a fraction of the time while collecting the next dataset.
This is often desirable in practice as the first few policies, with relatively few datapoints, may make many more mistakes and visit states that are irrelevant as the policy improves.

We will typically use β1=1subscript𝛽11\beta\_{1}=1 so that we do not have to specify an initial policy π^1subscript^𝜋1\hat{\pi}\_{1} before getting data from the expert’s behavior. Then we could choose βi=pi−1subscript𝛽𝑖superscript𝑝𝑖1\beta\_{i}=p^{i-1} to have a probability of using the expert that decays exponentially as in SMILe and SEARN. We show below the only requirement is that {βi}subscript𝛽𝑖\{\beta\_{i}\} be a sequence such that β¯N=1N​∑i=1Nβi→0subscript¯𝛽𝑁1𝑁superscriptsubscript𝑖1𝑁subscript𝛽𝑖→0\overline{\beta}\_{N}=\frac{1}{N}\sum\_{i=1}^{N}\beta\_{i}\rightarrow 0 as N→∞→𝑁N\rightarrow\infty. The simple, parameter-free version of the algorithm described above is the special case βi=I​(i=1)subscript𝛽𝑖𝐼𝑖1\beta\_{i}=I(i=1) for I𝐼I the indicator function, which often performs best in practice (see Section [5](#S5 "5 EXPERIMENTS")). The general DAgger algorithm is detailed in Algorithm [3.1](#S3.alg1 "Algorithm 3.1 ‣ 3 DATASET AGGREGATION").

Initialize 𝒟←∅←𝒟\mathcal{D}\leftarrow\emptyset.

Initialize π^1subscript^𝜋1\hat{\pi}\_{1} to any policy in ΠΠ\Pi.

for i=1𝑖1i=1 to N𝑁N do

Let πi=βi​π∗+(1−βi)​π^isubscript𝜋𝑖subscript𝛽𝑖superscript𝜋1subscript𝛽𝑖subscript^𝜋𝑖\pi\_{i}=\beta\_{i}\pi^{\*}+(1-\beta\_{i})\hat{\pi}\_{i}.

Sample T𝑇T-step trajectories using πisubscript𝜋𝑖\pi\_{i}.

Get dataset 𝒟i={(s,π∗​(s))}subscript𝒟𝑖𝑠superscript𝜋𝑠\mathcal{D}\_{i}=\{(s,\pi^{\*}(s))\} of visited states by πisubscript𝜋𝑖\pi\_{i} and actions given by expert.

Aggregate datasets: 𝒟←𝒟​⋃𝒟i←𝒟𝒟subscript𝒟𝑖\mathcal{D}\leftarrow\mathcal{D}\bigcup\mathcal{D}\_{i}.

Train classifier π^i+1subscript^𝜋𝑖1\hat{\pi}\_{i+1} on 𝒟𝒟\mathcal{D}.

end for

Return best π^isubscript^𝜋𝑖\hat{\pi}\_{i} on validation.

Algorithm 3.1  DAgger Algorithm.

The main result of our analysis in the next section is the following guarantee for DAgger. Let π1:Nsubscript𝜋:1𝑁\pi\_{1:N} denote the sequence of policies π1,π2,…,πN

subscript𝜋1subscript𝜋2…subscript𝜋𝑁\pi\_{1},\pi\_{2},\dots,\pi\_{N}. Assume ℓℓ\ell is strongly convex and bounded over ΠΠ\Pi. Suppose βi≤(1−α)i−1subscript𝛽𝑖superscript1𝛼𝑖1\beta\_{i}\leq(1-\alpha)^{i-1} for all i𝑖i for some constant α𝛼\alpha independent of T𝑇T. Let ϵN=minπ∈Π⁡1N​∑i=1N𝔼s∼dπi​[ℓ​(s,π)]subscriptitalic-ϵ𝑁subscript𝜋Π1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝑑subscript𝜋𝑖delimited-[]ℓ𝑠𝜋\epsilon\_{N}=\min\_{\pi\in\Pi}\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim d\_{\pi\_{i}}}[\ell(s,\pi)] be the true loss of the best policy in hindsight. Then the following holds in the infinite sample case (infinite number of sample trajectories at each iteration):

###### Theorem 3.1.

For DAgger, if N𝑁N is O~​(T)~𝑂𝑇\tilde{O}(T) there exists a policy π^∈π^1:N^𝜋subscript^𝜋:1𝑁\hat{\pi}\in\hat{\pi}\_{1:N} s.t. 𝔼s∼dπ^​[ℓ​(s,π^)]≤ϵN+O​(1/T)subscript𝔼similar-to𝑠subscript𝑑^𝜋delimited-[]ℓ𝑠^𝜋subscriptitalic-ϵ𝑁𝑂1𝑇\mathbb{E}\_{s\sim d\_{\hat{\pi}}}[\ell(s,\hat{\pi})]\leq\epsilon\_{N}+O(1/T)

In particular, this holds for the policy π^=arg​minπ∈π^1:N⁡𝔼s∼dπ​[ℓ​(s,π)]^𝜋subscriptargmin𝜋subscript^𝜋:1𝑁subscript𝔼similar-to𝑠subscript𝑑𝜋delimited-[]ℓ𝑠𝜋\hat{\pi}=\operatorname\*{arg\,min}\_{\pi\in\hat{\pi}\_{1:N}}\mathbb{E}\_{s\sim d\_{\pi}}[\ell(s,\pi)]. 333It is not necessary to find the best policy in the sequence that minimizes the loss under its distribution; the same guarantee holds for the policy which uniformly randomly picks one policy in the sequence π^1:Nsubscript^𝜋:1𝑁\hat{\pi}\_{1:N} and executes that policy for T𝑇T steps.
If the task cost function C𝐶C corresponds to (or is upper bounded by) the surrogate loss ℓℓ\ell then this bound tells us directly that J​(π^)≤T​ϵN+O​(1)𝐽^𝜋𝑇subscriptitalic-ϵ𝑁𝑂1J(\hat{\pi})\leq T\epsilon\_{N}+O(1). For arbitrary task cost function C𝐶C, then if ℓℓ\ell is an upper bound on the 0-1 loss with respect to π∗superscript𝜋\pi^{\*}, combining this result with Theorem [2.2](#S2.Thmtheorem2 "Theorem 2.2. ‣ 2.2 Forward Training ‣ 2 PRELIMINARIES") yields that:

###### Theorem 3.2.

For DAgger, if N𝑁N is O~​(u​T)~𝑂𝑢𝑇\tilde{O}(uT) there exists a policy π^∈π^1:N^𝜋subscript^𝜋:1𝑁\hat{\pi}\in\hat{\pi}\_{1:N} s.t. J​(π^)≤J​(π∗)+u​T​ϵN+O​(1)𝐽^𝜋𝐽superscript𝜋𝑢𝑇subscriptitalic-ϵ𝑁𝑂1J(\hat{\pi})\leq J(\pi^{\*})+uT\epsilon\_{N}+O(1).

##### Finite Sample Results

In the finite sample case, suppose we sample m𝑚m trajectories with πisubscript𝜋𝑖\pi\_{i} at each iteration i𝑖i, and denote this dataset Disubscript𝐷𝑖D\_{i}. Let ϵ^N=minπ∈Π⁡1N​∑i=1N𝔼s∼Di​[ℓ​(s,π)]subscript^italic-ϵ𝑁subscript𝜋Π1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝐷𝑖delimited-[]ℓ𝑠𝜋\hat{\epsilon}\_{N}=\min\_{\pi\in\Pi}\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim D\_{i}}[\ell(s,\pi)] be the training loss of the best policy on the sampled trajectories, then using Azuma-Hoeffding’s inequality leads to the following guarantee:

###### Theorem 3.3.

For DAgger, if N𝑁N is O​(T2​log⁡(1/δ))𝑂superscript𝑇21𝛿O(T^{2}\log(1/\delta)) and m𝑚m is O​(1)𝑂1O(1) then with probability at least 1−δ1𝛿1-\delta there exists a policy π^∈π^1:N^𝜋subscript^𝜋:1𝑁\hat{\pi}\in\hat{\pi}\_{1:N} s.t. 𝔼s∼dπ^​[ℓ​(s,π^)]≤ϵ^N+O​(1/T)subscript𝔼similar-to𝑠subscript𝑑^𝜋delimited-[]ℓ𝑠^𝜋subscript^italic-ϵ𝑁𝑂1𝑇\mathbb{E}\_{s\sim d\_{\hat{\pi}}}[\ell(s,\hat{\pi})]\leq\hat{\epsilon}\_{N}+O(1/T)

A more refined analysis taking advantage of the strong convexity of the loss function (Kakade and Tewari, [2009](#bib.bib12)) may lead to tighter generalization bounds that require N𝑁N only of order O~​(T​log⁡(1/δ))~𝑂𝑇1𝛿\tilde{O}(T\log(1/\delta)). Similarly:

###### Theorem 3.4.

For DAgger, if N𝑁N is O​(u2​T2​log⁡(1/δ))𝑂superscript𝑢2superscript𝑇21𝛿O(u^{2}T^{2}\log(1/\delta)) and m𝑚m is O​(1)𝑂1O(1) then with probability at least 1−δ1𝛿1-\delta there exists a policy π^∈π^1:N^𝜋subscript^𝜋:1𝑁\hat{\pi}\in\hat{\pi}\_{1:N} s.t. J​(π^)≤J​(π∗)+u​T​ϵ^N+O​(1)𝐽^𝜋𝐽superscript𝜋𝑢𝑇subscript^italic-ϵ𝑁𝑂1J(\hat{\pi})\leq J(\pi^{\*})+uT\hat{\epsilon}\_{N}+O(1).

## 4 THEORETICAL ANALYSIS

The theoretical analysis of DAgger only relies on the no-regret property of the underlying *Follow-The-Leader* algorithm on strongly convex losses (Kakade and Tewari, [2009](#bib.bib12)) which picks the sequence of policies π^1:Nsubscript^𝜋:1𝑁\hat{\pi}\_{1:N}. Hence the presented results also hold for *any* other no regret online learning algorithm we would apply to our imitation learning setting. In particular, we can consider the results here a reduction of imitation learning to no-regret online learning where we treat mini-batches of trajectories under a single policy as a single online-learning example. We first briefly review concepts of online learning and no regret that will be used for this analysis.

### 4.1 Online Learning

In online learning, an algorithm must provide a policy πnsubscript𝜋𝑛\pi\_{n} at iteration n𝑛n which incurs a loss ℓn​(πn)subscriptℓ𝑛subscript𝜋𝑛\ell\_{n}(\pi\_{n}). After observing this loss, the algorithm can provide a different policy πn+1subscript𝜋𝑛1\pi\_{n+1} for the next iteration which will incur loss ℓn+1​(πn+1)subscriptℓ𝑛1subscript𝜋𝑛1\ell\_{n+1}(\pi\_{n+1}). The loss functions ℓn+1subscriptℓ𝑛1\ell\_{n+1} may vary in an unknown or even adversarial fashion over time. A no-regret algorithm is an algorithm that produces a sequence of policies π1,π2,…,πN

subscript𝜋1subscript𝜋2…subscript𝜋𝑁\pi\_{1},\pi\_{2},\dots,\pi\_{N} such that the average regret with respect to the best policy in hindsight goes to 0 as N𝑁N goes to ∞\infty:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1N​∑i=1Nℓi​(πi)−minπ∈Π⁡1N​∑i=1Nℓi​(π)≤γN1𝑁superscriptsubscript𝑖1𝑁subscriptℓ𝑖subscript𝜋𝑖subscript𝜋Π1𝑁superscriptsubscript𝑖1𝑁subscriptℓ𝑖𝜋subscript𝛾𝑁\frac{1}{N}\sum\_{i=1}^{N}\ell\_{i}(\pi\_{i})-\min\_{\pi\in\Pi}\frac{1}{N}\sum\_{i=1}^{N}\ell\_{i}(\pi)\leq\gamma\_{N} |  | (3) |

for limN→∞γN=0subscript→𝑁subscript𝛾𝑁0\lim\_{N\rightarrow\infty}\gamma\_{N}=0. Many no-regret algorithms guarantee that γNsubscript𝛾𝑁\gamma\_{N} is O~​(1N)~𝑂1𝑁\tilde{O}(\frac{1}{N}) (e.g. when ℓℓ\ell is strongly convex) (Hazan et al., [2006](#bib.bib8); Kakade and Shalev-Shwartz, [2008](#bib.bib11); Kakade and Tewari, [2009](#bib.bib12)).

### 4.2 No Regret Algorithms Guarantees

Now we show that no-regret algorithms can be used to find a policy which has good performance guarantees under its own distribution of states in our imitation learning setting. To do so, we must choose the loss functions to be the loss under the distribution of states of the current policy chosen by the online algorithm: ℓi​(π)=𝔼s∼dπi​[ℓ​(s,π)]subscriptℓ𝑖𝜋subscript𝔼similar-to𝑠subscript𝑑subscript𝜋𝑖delimited-[]ℓ𝑠𝜋\ell\_{i}(\pi)=\mathbb{E}\_{s\sim d\_{\pi\_{i}}}[\ell(s,\pi)].

For our analysis of DAgger, we need to bound the total variation distance between the distribution of states encountered by π^isubscript^𝜋𝑖\hat{\pi}\_{i} and πisubscript𝜋𝑖\pi\_{i},
which continues to call the expert. The following lemma is useful:

###### Lemma 4.1.

‖dπi−dπ^i‖1≤2​T​βisubscriptnormsubscript𝑑subscript𝜋𝑖subscript𝑑subscript^𝜋𝑖12𝑇subscript𝛽𝑖||d\_{\pi\_{i}}-d\_{\hat{\pi}\_{i}}||\_{1}\leq 2T\beta\_{i}.

###### Proof.

Let d𝑑d the distribution of states over T𝑇T steps conditioned on πisubscript𝜋𝑖\pi\_{i} picking π∗superscript𝜋\pi^{\*} at least once over T𝑇T steps. Since πisubscript𝜋𝑖\pi\_{i} always executes π^isubscript^𝜋𝑖\hat{\pi}\_{i} over T𝑇T steps with probability (1−βi)Tsuperscript1subscript𝛽𝑖𝑇(1-\beta\_{i})^{T} we have dπi=(1−βi)T​dπ^i+(1−(1−βi)T)​dsubscript𝑑subscript𝜋𝑖superscript1subscript𝛽𝑖𝑇subscript𝑑subscript^𝜋𝑖1superscript1subscript𝛽𝑖𝑇𝑑d\_{\pi\_{i}}=(1-\beta\_{i})^{T}d\_{\hat{\pi}\_{i}}+(1-(1-\beta\_{i})^{T})d. Thus

|  |  |  |
| --- | --- | --- |
|  | ‖dπi−dπ^i‖1=(1−(1−βi)T)​‖d−dπ^i‖1≤2​(1−(1−βi)T)≤2​T​βisubscriptnormsubscript𝑑subscript𝜋𝑖subscript𝑑subscript^𝜋𝑖11superscript1subscript𝛽𝑖𝑇subscriptnorm𝑑subscript𝑑subscript^𝜋𝑖121superscript1subscript𝛽𝑖𝑇2𝑇subscript𝛽𝑖\begin{array}[]{rl}\lx@intercol||d\_{\pi\_{i}}-d\_{\hat{\pi}\_{i}}||\_{1}\hfil\lx@intercol\\ =&(1-(1-\beta\_{i})^{T})||d-d\_{\hat{\pi}\_{i}}||\_{1}\\ \leq&2(1-(1-\beta\_{i})^{T})\\ \leq&2T\beta\_{i}\\ \end{array} |  |

The last inequality follows from the fact that (1−β)T≥1−β​Tsuperscript1𝛽𝑇1𝛽𝑇(1-\beta)^{T}\geq 1-\beta T for any β∈[0,1]𝛽01\beta\in[0,1].
∎

This is only better than the trivial bound ‖dπi−dπ^i‖1≤2subscriptnormsubscript𝑑subscript𝜋𝑖subscript𝑑subscript^𝜋𝑖12||d\_{\pi\_{i}}-d\_{\hat{\pi}\_{i}}||\_{1}\leq 2 for βi≤1Tsubscript𝛽𝑖1𝑇\beta\_{i}\leq\frac{1}{T}. Assume βisubscript𝛽𝑖\beta\_{i} is non-increasing and define nβsubscript𝑛𝛽n\_{\beta} the largest n≤N𝑛𝑁n\leq N such that βn>1Tsubscript𝛽𝑛1𝑇\beta\_{n}>\frac{1}{T}. Let ϵN=minπ∈Π⁡1N​∑i=1N𝔼s∼dπi​[ℓ​(s,π)]subscriptitalic-ϵ𝑁subscript𝜋Π1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝑑subscript𝜋𝑖delimited-[]ℓ𝑠𝜋\epsilon\_{N}=\min\_{\pi\in\Pi}\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim d\_{\pi\_{i}}}[\ell(s,\pi)] the loss of the best policy in hindsight after N𝑁N iterations and let ℓmaxsubscriptℓ\ell\_{\max} be an upper bound on the loss, i.e. ℓi​(s,π^i)≤ℓmaxsubscriptℓ𝑖𝑠subscript^𝜋𝑖subscriptℓ\ell\_{i}(s,\hat{\pi}\_{i})\leq\ell\_{\max} for all policies π^isubscript^𝜋𝑖\hat{\pi}\_{i}, and state s𝑠s such that dπ^i​(s)>0subscript𝑑subscript^𝜋𝑖𝑠0d\_{\hat{\pi}\_{i}}(s)>0. We have the following:

###### Theorem 4.1.

For DAgger, there exists a policy π^∈π^1:N^𝜋subscript^𝜋:1𝑁\hat{\pi}\in\hat{\pi}\_{1:N} s.t. 𝔼s∼dπ^​[ℓ​(s,π^)]≤ϵN+γN+2​ℓmaxN​[nβ+T​∑i=nβ+1Nβi]subscript𝔼similar-to𝑠subscript𝑑^𝜋delimited-[]ℓ𝑠^𝜋subscriptitalic-ϵ𝑁subscript𝛾𝑁2subscriptℓ𝑁delimited-[]subscript𝑛𝛽𝑇superscriptsubscript𝑖subscript𝑛𝛽1𝑁subscript𝛽𝑖\mathbb{E}\_{s\sim d\_{\hat{\pi}}}[\ell(s,\hat{\pi})]\leq\epsilon\_{N}+\gamma\_{N}+\frac{2\ell\_{\max}}{N}[n\_{\beta}+T\sum\_{i=n\_{\beta}+1}^{N}\beta\_{i}], for γNsubscript𝛾𝑁\gamma\_{N} the average regret of π^1:Nsubscript^𝜋:1𝑁\hat{\pi}\_{1:N}.

###### Proof.

The last lemma implies 𝔼s∼dπ^i​(ℓi​(s,π^i))≤𝔼s∼dπi​(ℓi​(s,π^i))+2​ℓmax​min⁡(1,T​βi)subscript𝔼similar-to𝑠subscript𝑑subscript^𝜋𝑖subscriptℓ𝑖𝑠subscript^𝜋𝑖subscript𝔼similar-to𝑠subscript𝑑subscript𝜋𝑖subscriptℓ𝑖𝑠subscript^𝜋𝑖2subscriptℓ1𝑇subscript𝛽𝑖\mathbb{E}\_{s\sim d\_{\hat{\pi}\_{i}}}(\ell\_{i}(s,\hat{\pi}\_{i}))\leq\mathbb{E}\_{s\sim d\_{\pi\_{i}}}(\ell\_{i}(s,\hat{\pi}\_{i}))+2\ell\_{\max}\min(1,T\beta\_{i}). Then:
minπ^∈π^1:N⁡𝔼s∼dπ^​[ℓ​(s,π^)]≤1N​∑i=1N𝔼s∼dπ^i​(ℓ​(s,π^i))≤1N​∑i=1N[𝔼s∼dπi​(ℓ​(s,π^i))+2​ℓmax​min⁡(1,T​βi)]≤γN+2​ℓmaxN​[nβ+T​∑i=nβ+1Nβi]+minπ∈Π​∑i=1Nℓi​(π)=γN+ϵN+2​ℓmaxN​[nβ+T​∑i=nβ+1Nβi]subscript^𝜋subscript^𝜋:1𝑁subscript𝔼similar-to𝑠subscript𝑑^𝜋delimited-[]ℓ𝑠^𝜋1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝑑subscript^𝜋𝑖ℓ𝑠subscript^𝜋𝑖1𝑁superscriptsubscript𝑖1𝑁delimited-[]subscript𝔼similar-to𝑠subscript𝑑subscript𝜋𝑖ℓ𝑠subscript^𝜋𝑖2subscriptℓ1𝑇subscript𝛽𝑖subscript𝛾𝑁2subscriptℓ𝑁delimited-[]subscript𝑛𝛽𝑇superscriptsubscript𝑖subscript𝑛𝛽1𝑁subscript𝛽𝑖subscript𝜋Πsuperscriptsubscript𝑖1𝑁subscriptℓ𝑖𝜋subscript𝛾𝑁subscriptitalic-ϵ𝑁2subscriptℓ𝑁delimited-[]subscript𝑛𝛽𝑇superscriptsubscript𝑖subscript𝑛𝛽1𝑁subscript𝛽𝑖\begin{array}[]{rl}\lx@intercol\min\_{\hat{\pi}\in\hat{\pi}\_{1:N}}\mathbb{E}\_{s\sim d\_{\hat{\pi}}}[\ell(s,\hat{\pi})]\hfil\lx@intercol\\
\leq&\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim d\_{\hat{\pi}\_{i}}}(\ell(s,\hat{\pi}\_{i}))\\
\leq&\frac{1}{N}\sum\_{i=1}^{N}[\mathbb{E}\_{s\sim d\_{\pi\_{i}}}(\ell(s,\hat{\pi}\_{i}))+2\ell\_{\max}\min(1,T\beta\_{i})]\\
\leq&\gamma\_{N}+\frac{2\ell\_{\max}}{N}[n\_{\beta}+T\sum\_{i=n\_{\beta}+1}^{N}\beta\_{i}]+\min\_{\pi\in\Pi}\sum\_{i=1}^{N}\ell\_{i}(\pi)\\
=&\gamma\_{N}+\epsilon\_{N}+\frac{2\ell\_{\max}}{N}[n\_{\beta}+T\sum\_{i=n\_{\beta}+1}^{N}\beta\_{i}]\\
\end{array}
∎

Under an error reduction assumption that for any input distribution, there is some policy π∈Π𝜋Π\pi\in\Pi that achieves surrogate loss of ϵitalic-ϵ\epsilon, this implies we are guaranteed to find a policy π^^𝜋\hat{\pi} which achieves ϵitalic-ϵ\epsilon surrogate loss under its own state distribution in the limit, provided β¯N→0→subscript¯𝛽𝑁0\overline{\beta}\_{N}\rightarrow 0. For instance, if we choose βisubscript𝛽𝑖\beta\_{i} to be of the form (1−α)i−1superscript1𝛼𝑖1(1-\alpha)^{i-1}, then 1N​[nβ+T​∑i=nβ+1Nβi]≤1N​α​[log⁡T+1]1𝑁delimited-[]subscript𝑛𝛽𝑇superscriptsubscript𝑖subscript𝑛𝛽1𝑁subscript𝛽𝑖1𝑁𝛼delimited-[]𝑇1\frac{1}{N}[n\_{\beta}+T\sum\_{i=n\_{\beta}+1}^{N}\beta\_{i}]\leq\frac{1}{N\alpha}[\log T+1] and this extra penalty becomes negligible for N𝑁N as O~​(T)~𝑂𝑇\tilde{O}(T). As we need at least O~​(T)~𝑂𝑇\tilde{O}(T) iterations to make γNsubscript𝛾𝑁\gamma\_{N} negligible, the number of iterations required by DAgger is similar to that required by any no-regret algorithm. Note that this is not as strong as the general error or regret reductions considered in (Beygelzimer et al., [2005](#bib.bib3); Ross and Bagnell, [2010](#bib.bib17); Daumé III et al., [2009](#bib.bib7)) which require only classification: we require a no-regret method or strongly convex surrogate loss function, a
stronger (albeit common) assumption.

##### Finite Sample Case:

The previous results hold if the online learning algorithm observes the infinite sample loss, i.e. the loss on the true distribution of trajectories induced by the current policy πisubscript𝜋𝑖\pi\_{i}. In practice however the algorithm would only observe its loss on a small sample of trajectories at each iteration. We wish to bound the true loss under its own distribution of the best policy in the sequence as a function of the regret on the finite sample of trajectories.

At each iteration i𝑖i, we assume the algorithm samples m𝑚m trajectories using πisubscript𝜋𝑖\pi\_{i} and then observes the loss ℓi​(π)=𝔼s∼Di​(ℓ​(s,π))subscriptℓ𝑖𝜋subscript𝔼similar-to𝑠subscript𝐷𝑖ℓ𝑠𝜋\ell\_{i}(\pi)=\mathbb{E}\_{s\sim D\_{i}}(\ell(s,\pi)), for Disubscript𝐷𝑖D\_{i} the dataset of those m𝑚m trajectories. The online learner guarantees 1N​∑i=1N𝔼s∼Di​(ℓ​(s,πi))−minπ∈Π⁡1N​∑i=1N𝔼s∼Di​(ℓ​(s,π))≤γN1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝐷𝑖ℓ𝑠subscript𝜋𝑖subscript𝜋Π1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝐷𝑖ℓ𝑠𝜋subscript𝛾𝑁\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim D\_{i}}(\ell(s,\pi\_{i}))-\min\_{\pi\in\Pi}\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim D\_{i}}(\ell(s,\pi))\leq\gamma\_{N}. Let ϵ^N=minπ∈Π⁡1N​∑i=1N𝔼s∼Di​[ℓ​(s,π)]subscript^italic-ϵ𝑁subscript𝜋Π1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝐷𝑖delimited-[]ℓ𝑠𝜋\hat{\epsilon}\_{N}=\min\_{\pi\in\Pi}\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim D\_{i}}[\ell(s,\pi)] the training loss of the best policy in hindsight. Following a similar analysis to Cesa-Bianchi et al. ([2004](#bib.bib5)), we obtain:

###### Theorem 4.2.

For DAgger, with probability at least 1−δ1𝛿1-\delta, there exists a policy π^∈π^1:N^𝜋subscript^𝜋:1𝑁\hat{\pi}\in\hat{\pi}\_{1:N} s.t. 𝔼s∼dπ^​[ℓ​(s,π^)]≤ϵ^N+γN+2​ℓmaxN​[nβ+T​∑i=nβ+1Nβi]+ℓmax​2​log⁡(1/δ)m​Nsubscript𝔼similar-to𝑠subscript𝑑^𝜋delimited-[]ℓ𝑠^𝜋subscript^italic-ϵ𝑁subscript𝛾𝑁2subscriptℓ𝑁delimited-[]subscript𝑛𝛽𝑇superscriptsubscript𝑖subscript𝑛𝛽1𝑁subscript𝛽𝑖subscriptℓ21𝛿𝑚𝑁\mathbb{E}\_{s\sim d\_{\hat{\pi}}}[\ell(s,\hat{\pi})]\leq\hat{\epsilon}\_{N}+\gamma\_{N}+\frac{2\ell\_{\max}}{N}[n\_{\beta}+T\sum\_{i=n\_{\beta}+1}^{N}\beta\_{i}]+\ell\_{\max}\sqrt{\frac{2\log(1/\delta)}{mN}}, for γNsubscript𝛾𝑁\gamma\_{N} the average regret of π^1:Nsubscript^𝜋:1𝑁\hat{\pi}\_{1:N}.

###### Proof.

Let Yi​jsubscript𝑌𝑖𝑗Y\_{ij} be the difference between the expected per step loss of π^isubscript^𝜋𝑖\hat{\pi}\_{i} under state distribution dπisubscript𝑑subscript𝜋𝑖d\_{\pi\_{i}} and the average per step loss of π^isubscript^𝜋𝑖\hat{\pi}\_{i} under the jt​hsuperscript𝑗𝑡ℎj^{th} sample trajectory with πisubscript𝜋𝑖\pi\_{i} at iteration i𝑖i. The random variables Yi​jsubscript𝑌𝑖𝑗Y\_{ij} over all i∈{1,2,…,N}𝑖12…𝑁i\in\{1,2,\dots,N\} and j∈{1,2,…,m}𝑗12…𝑚j\in\{1,2,\dots,m\} are all zero mean, bounded in [−ℓmax,ℓmax]subscriptℓsubscriptℓ[-\ell\_{\max},\ell\_{\max}] and form a martingale (considering the order Y11,Y12,…,Y1​m,Y21,…,YN​m

subscript𝑌11subscript𝑌12…subscript𝑌1𝑚subscript𝑌21…subscript𝑌𝑁𝑚Y\_{11},Y\_{12},\dots,Y\_{1m},Y\_{21},\dots,Y\_{Nm}). By Azuma-Hoeffding’s inequality 1m​N​∑i=1N∑j=1mYi​j≤ℓmax​2​log⁡(1/δ)m​N1𝑚𝑁superscriptsubscript𝑖1𝑁superscriptsubscript𝑗1𝑚subscript𝑌𝑖𝑗subscriptℓ21𝛿𝑚𝑁\frac{1}{mN}\sum\_{i=1}^{N}\sum\_{j=1}^{m}Y\_{ij}\leq\ell\_{\max}\sqrt{\frac{2\log(1/\delta)}{mN}} with probability at least 1−δ1𝛿1-\delta. Hence, we obtain that with probability at least 1−δ1𝛿1-\delta:

|  |  |  |
| --- | --- | --- |
|  | minπ^∈π^1:N⁡𝔼s∼dπ^​[ℓ​(s,π^)]≤1N​∑i=1N𝔼s∼dπ^i​[ℓ​(s,π^i)]≤1N​∑i=1N𝔼s∼dπi​[ℓ​(s,π^i)]+2​ℓmaxN​[nβ+T​∑i=nβ+1Nβi]=1N​∑i=1N𝔼s∼Di​[ℓ​(s,π^i)]+1m​N​∑i=1N∑j=1mYi​j+2​ℓmaxN​[nβ+T​∑i=nβ+1Nβi]≤1N​∑i=1N𝔼s∼Di​[ℓ​(s,π^i)]+ℓmax​2​log⁡(1/δ)m​N+2​ℓmaxN​[nβ+T​∑i=nβ+1Nβi]≤ϵ^N+γN+ℓmax​2​log⁡(1/δ)m​N+2​ℓmaxN​[nβ+T​∑i=nβ+1Nβi]subscript^𝜋subscript^𝜋:1𝑁subscript𝔼similar-to𝑠subscript𝑑^𝜋delimited-[]ℓ𝑠^𝜋1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝑑subscript^𝜋𝑖delimited-[]ℓ𝑠subscript^𝜋𝑖1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝑑subscript𝜋𝑖delimited-[]ℓ𝑠subscript^𝜋𝑖2subscriptℓ𝑁delimited-[]subscript𝑛𝛽𝑇superscriptsubscript𝑖subscript𝑛𝛽1𝑁subscript𝛽𝑖1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝐷𝑖delimited-[]ℓ𝑠subscript^𝜋𝑖1𝑚𝑁superscriptsubscript𝑖1𝑁superscriptsubscript𝑗1𝑚subscript𝑌𝑖𝑗missing-subexpression2subscriptℓ𝑁delimited-[]subscript𝑛𝛽𝑇superscriptsubscript𝑖subscript𝑛𝛽1𝑁subscript𝛽𝑖1𝑁superscriptsubscript𝑖1𝑁subscript𝔼similar-to𝑠subscript𝐷𝑖delimited-[]ℓ𝑠subscript^𝜋𝑖subscriptℓ21𝛿𝑚𝑁missing-subexpression2subscriptℓ𝑁delimited-[]subscript𝑛𝛽𝑇superscriptsubscript𝑖subscript𝑛𝛽1𝑁subscript𝛽𝑖subscript^italic-ϵ𝑁subscript𝛾𝑁subscriptℓ21𝛿𝑚𝑁2subscriptℓ𝑁delimited-[]subscript𝑛𝛽𝑇superscriptsubscript𝑖subscript𝑛𝛽1𝑁subscript𝛽𝑖\begin{array}[]{rl}\lx@intercol\min\_{\hat{\pi}\in\hat{\pi}\_{1:N}}\mathbb{E}\_{s\sim d\_{\hat{\pi}}}[\ell(s,\hat{\pi})]\hfil\lx@intercol\\ \leq&\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim d\_{\hat{\pi}\_{i}}}[\ell(s,\hat{\pi}\_{i})]\\ \leq&\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim d\_{\pi\_{i}}}[\ell(s,\hat{\pi}\_{i})]+\frac{2\ell\_{\max}}{N}[n\_{\beta}+T\sum\_{i=n\_{\beta}+1}^{N}\beta\_{i}]\\ =&\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim D\_{i}}[\ell(s,\hat{\pi}\_{i})]+\frac{1}{mN}\sum\_{i=1}^{N}\sum\_{j=1}^{m}Y\_{ij}\\ &+\frac{2\ell\_{\max}}{N}[n\_{\beta}+T\sum\_{i=n\_{\beta}+1}^{N}\beta\_{i}]\\ \leq&\frac{1}{N}\sum\_{i=1}^{N}\mathbb{E}\_{s\sim D\_{i}}[\ell(s,\hat{\pi}\_{i})]+\ell\_{\max}\sqrt{\frac{2\log(1/\delta)}{mN}}\\ &+\frac{2\ell\_{\max}}{N}[n\_{\beta}+T\sum\_{i=n\_{\beta}+1}^{N}\beta\_{i}]\\ \leq&\hat{\epsilon}\_{N}+\gamma\_{N}+\ell\_{\max}\sqrt{\frac{2\log(1/\delta)}{mN}}+\frac{2\ell\_{\max}}{N}[n\_{\beta}+T\sum\_{i=n\_{\beta}+1}^{N}\beta\_{i}]\\ \end{array} |  |

∎

The use of Azuma-Hoeffding’s inequality suggests we need N​m𝑁𝑚Nm in O​(T2​log⁡(1/δ))𝑂superscript𝑇21𝛿O(T^{2}\log(1/\delta)) for the generalization error to be O​(1/T)𝑂1𝑇O(1/T) and negligible over T𝑇T steps. Leveraging the strong convexity of ℓℓ\ell as in (Kakade and Tewari, [2009](#bib.bib12)) may lead to a tighter bound requiring only O​(T​log⁡(T/δ))𝑂𝑇𝑇𝛿O(T\log(T/\delta)) trajectories.

## 5 EXPERIMENTS

To demonstrate the efficacy and scalability of DAgger, we apply it to two challenging imitation learning problems and a sequence labeling task (handwriting recognition).

### 5.1 Super Tux Kart

Super Tux Kart is a 3D racing game similar to the popular Mario Kart. Our goal is to train the computer to steer the kart moving at fixed speed on a particular race track, based on the current game image features as input (see Figure [1](#S5.F1 "Figure 1 ‣ 5.1 Super Tux Kart ‣ 5 EXPERIMENTS")). A human expert is used to provide demonstrations of the correct steering (analog joystick value in [-1,1]) for each of the observed game images.

!(/html/1011.0686/assets/stk.png)

Figure 1: Image from Super Tux Kart’s Star Track.

For all methods, we use a linear controller as the base learner which updates the steering at 5Hz based on the vector of image features444Features x𝑥x: LAB color values of each pixel in a 25x19 resized image of the 800x600 image; output steering: y^=wT​x+b^𝑦superscript𝑤𝑇𝑥𝑏\hat{y}=w^{T}x+b where w𝑤w, b𝑏b minimizes ridge regression objective: L​(w,b)=1n​∑i=1n(wT​xi+b−yi)2+λ2​wT​w𝐿𝑤𝑏1𝑛superscriptsubscript𝑖1𝑛superscriptsuperscript𝑤𝑇subscript𝑥𝑖𝑏subscript𝑦𝑖2𝜆2superscript𝑤𝑇𝑤L(w,b)=\frac{1}{n}\sum\_{i=1}^{n}(w^{T}x\_{i}+b-y\_{i})^{2}+\frac{\lambda}{2}w^{T}w, for regularizer λ=10−3𝜆superscript103\lambda=10^{-3}..

We compare performance on a race track called Star Track. As this track floats in space, the kart can fall off the track at any point (the kart is repositioned at the center of the track when this occurs). We measure performance in terms of the average number of falls per lap. For SMILe and DAgger, we used 1 lap of training per iteration (∼similar-to\sim1000 data points) and run both methods for 20 iterations. For SMILe we choose parameter α=0.1𝛼0.1\alpha=0.1 as in Ross and Bagnell ([2010](#bib.bib17)), and for DAgger the parameter βi=I​(i=1)subscript𝛽𝑖𝐼𝑖1\beta\_{i}=I(i=1) for I𝐼I the indicator function. Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Super Tux Kart ‣ 5 EXPERIMENTS") shows 95% confidence intervals on the average falls per lap of each method after 1, 5, 10, 15 and 20 iterations as a function of the total number of training data collected.

!(/html/1011.0686/assets/x1.png)

Figure 2: Average falls/lap as a function of training data.

We first observe that with the baseline supervised approach where training always occurs under the expert’s trajectories that performance does not improve as more data is collected. This is because most of the training laps are all very similar and do not help the learner to learn how to recover from mistakes it makes. With SMILe we obtain some improvements but the policy after 20 iterations still falls off the track about twice per lap on average. This is in part due to the stochasticity of the policy which sometimes makes bad choices of actions. For DAgger, we were able to obtain a policy that never falls off the track after 15 iterations of training. Though even after 5 iterations, the policy we obtain almost never falls off the track and is significantly outperforming both SMILe and the baseline supervised approach. Furthermore, the policy obtained by DAgger is smoother and looks qualitatively better than the policy obtained with SMILe. A video available on YouTube (Ross, [2010a](#bib.bib15)) shows a qualitative comparison of the behavior obtained with each method.

### 5.2 Super Mario Bros.

Super Mario Bros. is a platform video game where the character, Mario, must move across each stage by avoiding being hit by enemies and falling into gaps, and before running out of time. We used the simulator from a recent Mario Bros. AI competition (Togelius and Karakovskiy, [2009](#bib.bib21)) which can randomly generate stages of varying difficulty (more difficult gaps and types of enemies). Our goal is to train the computer to play this game based on the current game image features as input (see Figure [3](#S5.F3 "Figure 3 ‣ 5.2 Super Mario Bros. ‣ 5 EXPERIMENTS")). Our expert in this scenario is a near-optimal planning algorithm that has full access to the game’s internal state and can simulate exactly the consequence of future actions. An action consists of 4 binary variables indicating which subset of buttons we should press in {{\{left,right,jump,speed}}\}.

!(/html/1011.0686/assets/mario.png)

Figure 3: Captured image from Super Mario Bros.

For all methods, we use 4 independent linear SVM as the base learner which update the 4 binary actions at 5Hz based on the vector of image features555For the input features x𝑥x: each image is discretized in a grid of 22x22 cells centered around Mario; 14 binary features describe each cell (types of ground, enemies, blocks and other special items); a history of those features over the last 4 images is used, in addition to other features describing the last 6 actions and the state of Mario (small,big,fire,touches ground), for a total of 27152 binary features (very sparse). The kt​hsuperscript𝑘𝑡ℎk^{th} output binary variable y^k=I​(wkT​x+bk>0)subscript^𝑦𝑘𝐼superscriptsubscript𝑤𝑘𝑇𝑥subscript𝑏𝑘0\hat{y}\_{k}=I(w\_{k}^{T}x+b\_{k}>0), where wk,bk

subscript𝑤𝑘subscript𝑏𝑘w\_{k},b\_{k} optimizes the SVM objective with regularizer λ=10−4𝜆superscript104\lambda=10^{-4} using stochastic gradient descent (Ratliff et al., [2007](#bib.bib14); Bottou, [2009](#bib.bib4))..

We compare performance in terms of the average distance travelled by Mario per stage before dying, running out of time or completing the stage, on randomly generated stages of difficulty 1 with a time limit of 60 seconds to complete the stage. The total distance of each stage varies but is around 4200-4300 on average, so performance can vary roughly in [0,4300]. Stages of difficulty 1 are fairly easy for an average human player but contain most types of enemies and gaps, except with fewer enemies and gaps than stages of harder difficulties. We compare performance of DAgger, SMILe and SEARN666We use the same cost-to-go approximation in Daumé III et al. ([2009](#bib.bib7)); in this case SMILe and SEARN differs only in how the weights in the mixture are updated at each iteration. to the supervised approach (Sup). With each approach we collect 5000 data points per iteration (each stage is about 150 data points if run to completion) and run the methods for 20 iterations. For SMILe we choose parameter α=0.1𝛼0.1\alpha=0.1 (Sm0.1) as in Ross and Bagnell ([2010](#bib.bib17)). For DAgger we obtain results with different choice of the parameter βisubscript𝛽𝑖\beta\_{i}: 1) βi=I​(i=1)subscript𝛽𝑖𝐼𝑖1\beta\_{i}=I(i=1) for I𝐼I the indicator function (D0); 2) βi=pi−1subscript𝛽𝑖superscript𝑝𝑖1\beta\_{i}=p^{i-1} for all values of p∈{0.1,0.2,…,0.9}𝑝0.10.2…0.9p\in\{0.1,0.2,\dots,0.9\}. We report the best results obtained with p=0.5𝑝0.5p=0.5 (D0.5). We also report the results with p=0.9𝑝0.9p=0.9 (D0.9) which shows the slower convergence of using the expert more frequently at later iterations. Similarly for SEARN, we obtain results with all choice of α𝛼\alpha in {0.1,0.2,…,1}0.10.2…1\{0.1,0.2,\dots,1\}. We report the best results obtained with α=0.4𝛼0.4\alpha=0.4 (Se0.4). We also report results with α=1.0𝛼1.0\alpha=1.0 (Se1), which shows the unstability of such a pure policy iteration approach. Figure [4](#S5.F4 "Figure 4 ‣ 5.2 Super Mario Bros. ‣ 5 EXPERIMENTS") shows 95% confidence intervals on the average distance travelled per stage at each iteration as a function of the total number of training data collected.

!(/html/1011.0686/assets/x2.png)

Figure 4: Average distance/stage as a function of data.

Again here we observe that with the supervised approach, performance stagnates as we collect more data from the expert demonstrations, as this does not help the particular errors the learned controller makes. In particular, a reason the supervised approach gets such a low score is that under the learned controller, Mario is often stuck at some location against an obstacle instead of jumping over it. Since the expert always jumps over obstacles at a significant distance away, the controller did not learn how to get unstuck in situations where it is right next to an obstacle. On the other hand, all the other iterative methods perform much better as they eventually learn to get unstuck in those situations by encountering them at the later iterations. Again in this experiment, DAgger outperforms SMILe, and also outperforms SEARN for all choice of α𝛼\alpha we considered. When using βi=0.9i−1subscript𝛽𝑖superscript0.9𝑖1\beta\_{i}=0.9^{i-1}, convergence is significantly slower could have benefited from more iterations as performance was still improving at the end of the 20 iterations. Choosing 0.5i−1superscript0.5𝑖10.5^{i-1} yields slightly better performance (3030) then with the indicator function (2980). This is potentially due to the large number of data generated where mario is stuck at the same location in the early iterations when using the indicator; whereas using the expert a small fraction of the time still allows to observe those locations but also unstucks mario and makes it collect a wider variety of useful data. A video available on YouTube (Ross, [2010b](#bib.bib16)) also shows a qualitative comparison of the behavior obtained with each method.

### 5.3 Handwriting Recognition

Finally, we demonstrate the efficacy of our approach on a structured prediction problem involving recognizing handwritten words given the sequence of images of each character in the word. We follow Daumé III et al. ([2009](#bib.bib7)) in adopting a view of structured prediction as a degenerate form of imitation learning where the system dynamics are deterministic and trivial in simply passing on earlier predictions made as inputs for future predictions.
We use the dataset of Taskar et al. ([2003](#bib.bib20)) which has been used extensively in the literature to compare several structured prediction approaches. This dataset contains roughly 6600 words (for a total of over 52000 characters) partitioned in 10 folds. We consider the large dataset experiment which consists of training on 9 folds and testing on 1 fold and repeating this over all folds. Performance is measured in terms of the character accuracy on the test folds.

We consider predicting the word by predicting each character in sequence in a left to right order, using the previously predicted character to help predict the next and a linear SVM777Each character is 8x16 binary pixels (128 input features); 26 binary features are used to encode the previously predicted letter in the word. We train the multiclass SVM using the all-pairs reduction to binary classification (Beygelzimer et al., [2005](#bib.bib3))., following the greedy SEARN approach in Daumé III et al. ([2009](#bib.bib7)). Here we compare our method to SMILe, as well as SEARN (using the same approximations used in Daumé III et al. ([2009](#bib.bib7))). We also compare these approaches to two baseline, a non-structured approach which simply predicts each character independently and the supervised training approach where training is conducted with the previous character always correctly labeled. Again we try all choice of α∈{0.1,0.2,…,1}𝛼0.10.2…1\alpha\in\{0.1,0.2,\dots,1\} for SEARN, and report results for α=0.1𝛼0.1\alpha=0.1, α=1𝛼1\alpha=1 (pure policy iteration) and the best α=0.8𝛼0.8\alpha=0.8, and run all approaches for 20 iterations. Figure [5](#S5.F5 "Figure 5 ‣ 5.3 Handwriting Recognition ‣ 5 EXPERIMENTS") shows the performance of each approach on the test folds after each iteration as a function of training data.

!(/html/1011.0686/assets/x3.png)

Figure 5: Character accuracy as a function of iteration.

The baseline result without structure achieves 82% character accuracy by just using an SVM that predicts each character independently. When adding the previous character feature, but training with always the previous character correctly labeled (supervised approach), performance increases up to 83.6%. Using DAgger increases performance further to 85.5%. Surprisingly, we observe SEARN with α=1𝛼1\alpha=1, which is a pure policy iteration approach performs very well on this experiment, similarly to the best α=0.8𝛼0.8\alpha=0.8 and DAgger. Because there is only a small part of the input that is influenced by the current policy (the previous predicted character feature) this makes this approach not as unstable as in general reinforcement/imitation learning problems (as we saw in the previous experiment). SEARN and SMILe with small α=0.1𝛼0.1\alpha=0.1 performs similarly but significantly worse than DAgger. Note that we chose the simplest (greedy, one-pass) decoding to illustrate the benefits of the DAGGER approach with respect to existing reductions. Similar techniques can be applied to multi-pass or beam-search decoding leading to results that are competitive with the state-of-the-art.

## 6 FUTURE WORK

We show that by batching over iterations of interaction with a system, no-regret methods, including the presented DAgger approach can provide
a learning reduction with strong performance guarantees in both imitation learning and structured prediction. In future work, we will consider more sophisticated strategies
than simple greedy forward decoding for structured prediction, as well as using base classifiers that rely on Inverse Optimal Control (Abbeel and Ng, [2004](#bib.bib1); Ratliff et al., [2006](#bib.bib13)) techniques to learn a cost function for a planner to aid prediction in imitation learning.
Further we believe techniques similar to those presented, by leveraging a cost-to-go estimate, may provide an understanding of the success of online methods for reinforcement
learning and suggest a similar data-aggregation method that can guarantee performance in such settings.

#### Acknowledgements

This work is supported by the ONR MURI grant N00014-09-1-1052, Reasoning in Reduced Information Spaces, and by the National Sciences and Engineering Research Council of Canada (NSERC).

## References

* Abbeel and Ng (2004)

  P. Abbeel and A. Y. Ng.
  Apprenticeship learning via inverse reinforcement learning.
  In *Proceedings of the 21st International Conference on Machine
  Learning (ICML)*, 2004.
* Argall et al. (2009)

  B. D. Argall, S. Chernova, M. Veloso, and B. Browning.
  A survey of robot learning from demonstration.
  *Robotics and Autonomous Systems*, 2009.
* Beygelzimer et al. (2005)

  A. Beygelzimer, V. Dani, T. Hayes, J. Langford, and B. Zadrozny.
  Error limiting reductions between classification tasks.
  In *Proceedings of the 22nd International Conference on Machine
  Learning (ICML)*, 2005.
* Bottou (2009)

  L. Bottou.
  sgd code, 2009.
  URL <http://www.leon.bottou.org/projects/sgd>.
* Cesa-Bianchi et al. (2004)

  N. Cesa-Bianchi, A. Conconi, and C. Gentile.
  On the generalization ability of on-line learning algorithms.
  2004.
* Chernova and Veloso (2009)

  S. Chernova and M. Veloso.
  Interactive policy learning through confidence-based autonomy.
  2009.
* Daumé III et al. (2009)

  H. Daumé III, J. Langford, and D. Marcu.
  Search-based structured prediction.
  *Machine Learning*, 2009.
* Hazan et al. (2006)

  E. Hazan, A. Kalai, S. Kale, and A. Agarwal.
  Logarithmic regret algorithms for online convex optimization.
  In *Proceedings of the 19th annual conference on Computational
  Learning Theory (COLT)*, 2006.
* Kääriäinen (2006)

  M. Kääriäinen.
  Lower bounds for reductions, 2006.
  Atomic Learning workshop.
* Kakade and Langford (2002)

  S. Kakade and J. Langford.
  Approximately optimal approximate reinforcement learning.
  In *Proceedings of the 19th International Conference on Machine
  Learning (ICML)*, 2002.
* Kakade and Shalev-Shwartz (2008)

  S. Kakade and S. Shalev-Shwartz.
  Mind the duality gap: Logarithmic regret algorithms for online
  optimization.
  In *Advances in Neural Information Processing Systems (NIPS)*,
  2008.
* Kakade and Tewari (2009)

  S. Kakade and A. Tewari.
  On the generalization ability of online strongly convex programming
  algorithms.
  In *Advances in Neural Information Processing Systems (NIPS)*,
  2009.
* Ratliff et al. (2006)

  N. Ratliff, D. Bradley, J. A. Bagnell, and J. Chestnutt.
  Boosting structured prediction for imitation learning.
  In *Advances in Neural Information Processing Systems (NIPS)*,
  2006.
* Ratliff et al. (2007)

  N. Ratliff, J. A. Bagnell, and M. Zinkevich.
  (Online) subgradient methods for structured prediction.
  In *Proceedings of the International Conference on Artificial
  Intelligence and Statistics (AISTATS)*, 2007.
* Ross (2010a)

  S. Ross.
  Comparison of imitation learning approaches on Super Tux Kart,
  2010a.
  URL <http://www.youtube.com/watch?v=V00npNnWzSU>.
* Ross (2010b)

  S. Ross.
  Comparison of imitation learning approaches on Super Mario
  Bros, 2010b.
  URL <http://www.youtube.com/watch?v=anOI0xZ3kGM>.
* Ross and Bagnell (2010)

  S. Ross and J. A. Bagnell.
  Efficient reductions for imitation learning.
  In *Proceedings of the 13th International Conference on
  Artificial Intelligence and Statistics (AISTATS)*, 2010.
* Schaal (1999)

  S. Schaal.
  Is imitation learning the route to humanoid robots?
  In *Trends in Cognitive Sciences*, 1999.
* Silver et al. (2008)

  D. Silver, J. A. Bagnell, and A. Stentz.
  High performance outdoor navigation from overhead data using
  imitation learning.
  In *Proceedings of Robotics Science and Systems (RSS)*, 2008.
* Taskar et al. (2003)

  B. Taskar, C. Guestrin, and D. Koller.
  Max margin markov networks.
  In *Advances in Neural Information Processing Systems (NIPS)*,
  2003.
* Togelius and Karakovskiy (2009)

  J. Togelius and S. Karakovskiy.
  Mario AI Competition, 2009.
  URL <http://julian.togelius.com/mariocompetition2009>.
