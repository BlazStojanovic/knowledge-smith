---
arxiv: '2408.15664'
authors:
- Lean Wang
- Huazuo Gao
- Chenggang Zhao
- Xu Sun
- Damai Dai
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts
url: https://arxiv.org/abs/2408.15664
year: 2024
---

[2408.15664] Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts














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



# Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts

Lean Wang1,211footnotemark: 1  , Huazuo Gao1, Chenggang Zhao1, Xu Sun222footnotemark: 2 , Damai Dai122footnotemark: 2
  
1DeepSeek-AI
  
2State Key Laboratory of Multimedia Information Processing,
  
School of Computer Science, Peking University
  
lean@pku.edu.cn, xusun@pku.edu.cn, damai.dai@deepseek.com

###### Abstract

For Mixture-of-Experts (MoE) models, an unbalanced expert load will lead to routing collapse or increased computational overhead.
Existing methods commonly employ an auxiliary loss to encourage load balance, but a large auxiliary loss will introduce non-negligible interference gradients into training and thus impair the model performance.
In order to control load balance while not producing undesired gradients during training, we propose Loss-Free Balancing, featured by an auxiliary-loss-free load balancing strategy.
To be specific, before the top-K routing decision, Loss-Free Balancing will first apply an expert-wise bias to the routing scores of each expert.
By dynamically updating the bias of each expert according to its recent load, Loss-Free Balancing can consistently maintain a balanced distribution of expert load.
In addition, since Loss-Free Balancing does not produce any interference gradients, it also elevates the upper bound of model performance gained from MoE training.
We validate the performance of Loss-Free Balancing on MoE models with up to 3B parameters trained on up to 200B tokens.
Experimental results show that Loss-Free Balancing achieves both better performance and better load balance compared with traditional auxiliary-loss-controlled load balancing strategies.

## 1 Introduction

Mixture-of-Experts (MoE) architectures have emerged as a promising solution for managing computational costs when scaling up parameters in large language models (LLMs).
Recent applications of MoE in Transformer-based models (Vaswani et al., [2017](#bib.bib9)) have led to successful attempts at scaling language models to substantial sizes (Shao et al., [2024](#bib.bib7); DeepSeek-AI et al., [2024](#bib.bib2); Dai et al., [2024](#bib.bib1); Fedus et al., [2021](#bib.bib3); Lepikhin et al., [2020](#bib.bib4)), resulting in remarkable performance improvements.
However, training MoE models always face the circumstance of load imbalance, which may result in routing collapse (Shazeer et al., [2017](#bib.bib8)) or increased computational overhead (Fedus et al., [2021](#bib.bib3); Lepikhin et al., [2020](#bib.bib4); Shazeer et al., [2017](#bib.bib8)).
In order to avoid imbalanced routing, existing methods (Fedus et al., [2021](#bib.bib3); Lepikhin et al., [2020](#bib.bib4)) commonly use an auxiliary loss to encourage balanced expert load.
Although the auxiliary loss can alleviate load imbalance during training, it also introduces undesired gradients that conflict with the language modeling objective.
These interference gradients will impair the model performance, so existing MoE methods always need to consider the trade-off between load balance and model performance.

In this paper, we propose Loss-Free Balancing, an auxiliary-loss-free load balancing strategy, aiming at maintaining control over expert load balance while not introducing interference gradients.
Loss-Free Balancing features an iterative process of token routing and bias updating.
As illustrated in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"), before the top-K routing decision of MoE, Loss-Free Balancing will first apply expert-wise biases to the original routing scores to produce biased gating scores, which determine the actual routing targets of each token during training.
These expert-wise biases will keep updating according to the expert load observed on recent training tokens, where the biases of heavy-load experts will be depressed and those of lite-load experts will be elevated.
Through this dynamic updating strategy, Loss-Free Balancing ensures that the biased gating scores can consistently lead to balanced routing results.
Compared with the auxiliary-loss-controlled load balancing strategies, Loss-Free Balancing does not introduce undesired gradients that disrupt the primary language modeling objective, so its training process is more noise-free and friendly.

In order to validate the performance of Loss-Free Balancing, we train MoE language models with 1B parameters on 100B tokens and 3B parameters on 200B tokens from scratch.
Experimental results demonstrate that Loss-Free Balancing produces MoE models with better validation loss than traditional auxiliary-loss-controlled models.
Meanwhile, keeping the performance advantage, Loss-Free Balancing also achieves a significantly better load balance at the global and batch levels, and is naturally compatible with expert parallelism, which is usually employed for training extremely large MoE models.

![Refer to caption](/html/2408.15664/assets/figs/figure_815.png)


Figure 1: Loss-Free Balancing selects experts according to a “biased gating score” in each training step and updates this expert-wise bias after each training step.

## 2 Background

### 2.1 Mixture-of-Experts

Current dominant MoE architectures (Lepikhin et al., [2020](#bib.bib4); Fedus et al., [2021](#bib.bib3); Dai et al., [2024](#bib.bib1)) replace the MLP layers in standard transformers with MoE layers. In an MoE layer, Top-K routing is employed to select the experts for each token. Let 𝐮tsubscript𝐮𝑡\mathbf{u}\_{t} denote the input of the t𝑡t-th token to an N𝑁N-expert MoE layer, the output 𝐡tsubscript𝐡𝑡\mathbf{h}\_{t} is computed as follows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝐡t=𝐮t+∑i=1Ngi,t​FFNi⁡(𝐮t),subscript𝐡𝑡subscript𝐮𝑡superscriptsubscript𝑖1𝑁subscript𝑔  𝑖𝑡subscriptFFN𝑖subscript𝐮𝑡\displaystyle\mathbf{h}\_{t}=\mathbf{u}\_{t}+\sum\_{i=1}^{N}g\_{i,t}\operatorname{FFN}\_{i}\left(\mathbf{u}\_{t}\right), |  | (1) |
|  |  | gi,t={si,t,si,t∈Topk⁡({sj,t∣1≤j≤N},K),0, otherwise,subscript𝑔  𝑖𝑡casessubscript𝑠  𝑖𝑡subscript𝑠  𝑖𝑡Topkconditional-setsubscript𝑠  𝑗𝑡1𝑗𝑁𝐾0 otherwise,\displaystyle g\_{i,t}=\begin{cases}s\_{i,t},&s\_{i,t}\in\operatorname{Topk}\left(\left\{s\_{j,t}\mid 1\leq j\leq N\right\},K\right),\\ 0,&\text{ otherwise, }\end{cases} |  |
|  |  | si,t=G​(𝐮t​𝐞iT),subscript𝑠  𝑖𝑡𝐺subscript𝐮𝑡superscriptsubscript𝐞𝑖𝑇\displaystyle s\_{i,t}=G\left(\mathbf{u}\_{t}{}^{T}\mathbf{e}\_{i}\right), |  |

where G𝐺G is a nonlinear gating function and 𝐞isubscript𝐞𝑖\mathbf{e}\_{i} is the centroid of the i𝑖i-th expert.

### 2.2 Auxiliary Loss for Load Balance

#### Auxiliary Loss

Uncontrolled routing strategies are likely to encounter load imbalance, which has two notable drawbacks.
Firstly, there is a risk of routing collapse (Shazeer et al., [2017](#bib.bib8)), where the model consistently selects only a few experts, hindering sufficient training of the other experts.
Secondly, when experts are distributed across multiple devices, load imbalance can exacerbate computation bottlenecks.
To address these issues, an auxiliary loss (Fedus et al., [2021](#bib.bib3); Lepikhin et al., [2020](#bib.bib4)) is commonly employed to control load balance.
For a sequence of length T𝑇T, the auxiliary loss is defined as:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒBalancesubscriptℒBalance\displaystyle\mathcal{L}\_{\text{Balance }} | =α​∑i=1Nfi​Pi,absent𝛼superscriptsubscript𝑖1𝑁subscript𝑓𝑖subscript𝑃𝑖\displaystyle=\alpha\sum\_{i=1}^{N}f\_{i}P\_{i}, |  | (2) |
|  | fisubscript𝑓𝑖\displaystyle f\_{i} | =NK​T​∑t=1T𝟙​( Token ​t​ selects Expert ​i),absent𝑁𝐾𝑇superscriptsubscript𝑡1𝑇1 Token 𝑡 selects Expert 𝑖\displaystyle=\frac{N}{KT}\sum\_{t=1}^{T}\mathbbm{1}(\text{ Token }t\text{ selects Expert }i), |  |
|  | Pisubscript𝑃𝑖\displaystyle P\_{i} | =1T​∑t=1Tsi,t,absent1𝑇superscriptsubscript𝑡1𝑇subscript𝑠  𝑖𝑡\displaystyle=\frac{1}{T}\sum\_{t=1}^{T}s\_{i,t}, |  |

where N𝑁N is the total number of experts, K𝐾K is the number of experts selected for each token, si,tsubscript𝑠

𝑖𝑡s\_{i,t} is the routing score of Expert i𝑖i for Token t𝑡t, fisubscript𝑓𝑖f\_{i} represents the fraction of tokens routed to Expert i𝑖i, Pisubscript𝑃𝑖P\_{i} denotes the average gating scores of Expert i𝑖i, and α𝛼\alpha is a hyper-parameter controlling the strength of the auxiliary loss.

![Refer to caption](/html/2408.15664/assets/figs/balance_mv_perplexity_new_new.png)


Figure 2: The dilemma between load balance and model performance for auxiliary-loss-controlled training. A small auxiliary loss coefficient α𝛼\alpha leads to poor load balance, while a large α𝛼\alpha impairs the model performance. In contrast, our Loss-Free Balancing method breaks this dilemma.

#### The Dilemma Between Load Balance and Model Performance

The auxiliary loss mentioned above can encourage load balance, but it also interferes with language modeling training as an additional regularization term. The absence of an auxiliary loss or a small auxiliary loss coefficient α𝛼\alpha can lead to poor balance, while a large α𝛼\alpha can impair training, resulting in suboptimal performance.
To illustrate this dilemma, we present the relationship between load balance and model performance in Figure [2](#S2.F2 "Figure 2 ‣ Auxiliary Loss ‣ 2.2 Auxiliary Loss for Load Balance ‣ 2 Background ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts").
We vary α𝛼\alpha among 1e-2, 1e-3, 1e-4, and 0, and present the corresponding MaxVioglobalsubscriptMaxVioglobal\text{MaxVio}\_{\text{global}}, which measures the degree of load balance and its computation details are described in § [4.1](#S4.SS1.SSS0.Px4 "Metrics. ‣ 4.1 Experimental Setups ‣ 4 Experiments ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts").
As shown in the figure, a small α𝛼\alpha causes routing collapse, affecting the model efficiency and potentially leading to some experts being insufficiently learned or exploited; while a large α𝛼\alpha keeps load balance under control but notably degrades the model performance.
In order to break this dilemma, we propose Loss-Free Balancing as a solution, which directly controls the expert load balance, but does not introduce unexpected gradients other than the gradients from the language modeling loss.

## 3 Auxiliary-Loss-Free Load Balancing Strategy

For a better load-balancing alternative that does not directly interfere with the main gradients from the training objective, we propose Loss-Free Balancing, which directly adjusts the gating scores of each expert according to their balance condition.
As illustrated in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"), we add an expert-wise bias term {𝒃𝒊}𝒊=𝟏𝑵superscriptsubscriptsubscript𝒃𝒊𝒊1𝑵\bm{\{b\_{i}\}\_{i=1}^{N}} to the gating scores si,tsubscript𝑠

𝑖𝑡s\_{i,t} of each expert, and use the biased scores to determine the top-K selection:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | gi,t={si,t,si,t+bi∈Topk⁡({sj,t+bj∣1≤j≤N},K),0, otherwise.subscript𝑔  𝑖𝑡casessubscript𝑠  𝑖𝑡subscript𝑠  𝑖𝑡subscript𝑏𝑖Topkconditional-setsubscript𝑠  𝑗𝑡subscript𝑏𝑗1𝑗𝑁𝐾0 otherwise.\displaystyle g\_{i,t}=\begin{cases}s\_{i,t},&s\_{i,t}+b\_{i}\in\operatorname{Topk}\left(\left\{s\_{j,t}+b\_{j}\mid 1\leq j\leq N\right\},K\right),\\ 0,&\text{ otherwise. }\end{cases} |  | (3) |

Note that the expert bias term bisubscript𝑏𝑖b\_{i} is only used to adjust the routing strategy by influencing the top-K selection.
It is not added to the gi,tsubscript𝑔

𝑖𝑡g\_{i,t} that weights the output of the selected experts when computing the final output of the MoE layer.

In order to derive proper biases, we adjust each bias bisubscript𝑏𝑖b\_{i} iteratively according to the following principle: decreasing it when the corresponding expert has a relatively heavy load, and vice versa.
To be specific, for each bisubscript𝑏𝑖{b}\_{i}, we keep monitoring its corresponding expert load on the previous batch.
If an expert has a heavy load on the previous batch, we will reduce its bias.
Otherwise, we will increase it.
Algorithm [1](#algorithm1 "In 3 Auxiliary-Loss-Free Load Balancing Strategy ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts") describes the details of our update algorithm for the expert-wise biases.
It is worth noting that we update the biases based on the historical balance condition, since utilizing the load information of the current sequence will break the causal constraint of language modeling, leading to leakage of the information of future tokens.
Through the dynamic adjustment for the biases, we can achieve good expert load balance, but not directly introduce noisy gradients into the model like the auxiliary-loss-controlled method does.

Input: MoE model θ𝜃\theta, training batch iterator B𝐵B, bias update rate u𝑢u.

1. Initialize bi=0subscript𝑏𝑖0b\_{i}=0 for each expert;

for *a batch {(𝐱k,𝐲k)}ksubscriptsubscript𝐱𝑘subscript𝐲𝑘𝑘\{(\mathbf{x}\_{k},\mathbf{y}\_{k})\}\_{k} in B𝐵B* do

2. Train MoE model θ𝜃\theta on the batch data {(𝐱k,𝐲k)}ksubscriptsubscript𝐱𝑘subscript𝐲𝑘𝑘\{(\mathbf{x}\_{k},\mathbf{y}\_{k})\}\_{k}, with gating scores calculated according to Eq. ([3](#S3.E3 "In 3 Auxiliary-Loss-Free Load Balancing Strategy ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"));

3. Count the number of assigned tokens cisubscript𝑐𝑖c\_{i} for each expert, and the average number ci¯¯subscript𝑐𝑖\overline{c\_{i}};

4. Calculate the load violation error ei=ci¯−cisubscript𝑒𝑖¯subscript𝑐𝑖subscript𝑐𝑖e\_{i}=\overline{c\_{i}}-c\_{i};

4. Update 𝐛isubscript𝐛𝑖\mathbf{b}\_{i} by bi=bi+u∗sign​(ei)subscript𝑏𝑖subscript𝑏𝑖𝑢signsubscript𝑒𝑖b\_{i}=b\_{i}+u\*\mathrm{sign}(e\_{i});

end for

Output: trained model θ𝜃\theta, corresponding bias 𝐛isubscript𝐛𝑖\mathbf{b}\_{i}

Algorithm 1 Adjusting the per-expert bias bisubscript𝑏𝑖{b}\_{i} during training

#### Comparison with Other Load Balancing Methods.

In order to show the theoretical advantages of Loss-Free Balancing, we compare it with other two mainstream load balancing methods, i.e., the auxiliary-loss-controlled method (Lepikhin et al., [2020](#bib.bib4); Fedus et al., [2021](#bib.bib3)) and the Expert Choice (EC) (Zhou et al., [2022](#bib.bib10)) method.
As described in § [2.2](#S2.SS2 "2.2 Auxiliary Loss for Load Balance ‣ 2 Background ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"), the auxiliary-loss-controlled method faces the dilemma between load balance and model performance, and a perfect trade-off may not exist.
As for the EC method, it will break the causal constraint of language modeling, since the target experts of each token are conditioned on the future tokens in the same sequence or batch.
This will result in the leakage of information about future tokens, thus destroying the generalization of the model.
Table [1](#S3.T1 "Table 1 ‣ Comparison with Other Load Balancing Methods. ‣ 3 Auxiliary-Loss-Free Load Balancing Strategy ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts") summarizes the properties of different load balancing methods.

Table 1: Comparison among different load balancing methods. The good property is displayed in green and the bad property in red.

|  |  |  |  |
| --- | --- | --- | --- |
| Load Balancing Methods | Balanced  Expert Load | Interference  Gradients | Future Token  Leakage |
| Loss-Controlled (strong auxiliary loss) | balanced | strong | no leakage |
| Loss-Controlled (weak auxiliary loss) | imbalanced | weak | no leakage |
| Expert Choice | balanced | none | with leakage |
| Loss-Free (Ours) | balanced | none | no leakage |

## 4 Experiments

### 4.1 Experimental Setups

#### Model Architecture.

We employ the DeepSeekMoE (Dai et al., [2024](#bib.bib1)) architecture as the backbone since it outperforms conventional MoE architectures like GShard (Lepikhin et al., [2020](#bib.bib4)) significantly.
Compared with GShard (Lepikhin et al., [2020](#bib.bib4)), it segments experts into finer granularity and isolates some experts as shared ones.
Slightly different from DeepSeekMoE, in our main experiments, we choose sigmoid instead of softmax as the gating function G𝐺G, since we find that the sigmoid baseline performs better than the softmax baseline.
Even so, we still provide the experimental results and discussion for the softmax gate in Appendix [C](#A3 "Appendix C Experiments with Softmax Gate ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts").
Our experiments are based on two model sizes of 1B and 3B total parameters, and we tune the bias update rate under only the 1B scale.
Experiments under the 3B scale directly inherit the best configuration for the 1B scale.
Due to the page limit, we present more details about our architecture in Appendix [A](#A1 "Appendix A Model Architecture ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts").

#### Training Settings

We use a multilingual training corpus created by DeepSeek-AI, sourced from a diverse range of textual materials including web text, mathematical material, coding scripts, and published
literature.
We employ the HuggingFace Tokenizer222<https://github.com/huggingface/tokenizers> to train a byte pair encoding (BPE) (Sennrich et al., [2015](#bib.bib6)) tokenizer with a vocabulary size of 32K.
In order to draw solid conclusions, we train the 1B model on 100B tokens and the 3B model on 200B tokens to ensure sufficient training.
We apply the cosine learning rate scheduler (Loshchilov & Hutter, [2016](#bib.bib5)) and multi-step learning rate scheduler (Dai et al., [2024](#bib.bib1)) for the 1B and 3B models, respectively.
Due to the page limit, we list more details about our training settings and hyper-parameters in Appendix [B](#A2 "Appendix B Training Settings ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts")).

#### Baseline.

We compare our Loss-Free Balancing method with the conventional auxiliary-loss-controlled method.
For the baseline, we set the auxiliary loss coefficient α𝛼\alpha to 0.001 to achieve a reasonable trade-off between model performance and load balance (see Figure [2](#S2.F2 "Figure 2 ‣ Auxiliary Loss ‣ 2.2 Auxiliary Loss for Load Balance ‣ 2 Background ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts")).
We do not take the EC method into comparison due to its issue of future token leakage, which we will discuss in depth in § [5.2](#S5.SS2 "5.2 Load Balancing and Future Token Leakage ‣ 5 Discussion ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts").

#### Metrics.

We reserve a validation set from the training corpus to evaluate model performance and load balance.
For model performance, we take perplexity as the metric.
For load balance, we introduce a metric called maximal violation (MaxVio) to quantify the degree of load balance of an MoE layer:

|  |  |  |  |
| --- | --- | --- | --- |
|  | MaxVio=maxi⁡Loadi−Loadi¯Loadi¯,MaxViosubscript𝑖subscriptLoad𝑖¯subscriptLoad𝑖¯subscriptLoad𝑖\text{MaxVio}=\frac{\max\_{i}\text{Load}\_{i}-\overline{\text{Load}\_{i}}}{\overline{\text{Load}\_{i}}}, |  | (4) |

where LoadisubscriptLoad𝑖\text{Load}\_{i} represents the number of tokens assigned to the i𝑖i-th expert, and Loadi¯¯subscriptLoad𝑖\overline{\text{Load}\_{i}} denotes the expected expert load under perfect load balance.

MaxVio has two variants: MaxVioglobalsubscriptMaxVioglobal\textbf{MaxVio}\_{\textbf{global}} and MaxViobatchsubscriptMaxViobatch\textbf{MaxVio}\_{\textbf{batch}}.
For MaxVioglobalsubscriptMaxVioglobal\textbf{MaxVio}\_{\textbf{global}}, we count LoadisubscriptLoad𝑖\text{Load}\_{i} on the whole validation set, so it reflects the degree of balanced expert utilization and efficiency upper bound when the batch size approaches the limitation.
For MaxViobatchsubscriptMaxViobatch\textbf{MaxVio}\_{\textbf{batch}}, we count LoadisubscriptLoad𝑖\text{Load}\_{i} on each training batch, so it is more related to the training efficiency.
For simplicity, in the rest of this paper, we report the MaxVio averaged across all layers as a load balance measurement of the whole model.

### 4.2 Main Results

Table [2](#S4.T2 "Table 2 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts") shows the validation perplexity and MaxVioglobalsubscriptMaxVioglobal\text{MaxVio}\_{\text{global}} for the 1B and 3B MoE models trained with auxiliary loss or our auxiliary-loss-free load balancing strategy.
As shown in the table, compared with the auxiliary-loss-controlled method, our Loss-Free Balancing achieves better perplexity and much better global load balance for both 1B and 3B models.
In addition, to present the load balance condition during training, we provide a load balancing curve depicting MaxViobatchsubscriptMaxViobatch\text{MaxVio}\_{\text{batch}} over training steps in Figure [3](#S4.F3 "Figure 3 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"), which demonstrates the persistent advantage of Loss-Free Balancing on load balance.
In summary, our Loss-Free Balancing method avoids interfering gradients during training and effectively controls the load balance, breaking the dilemma between load balance and model performance in MoE training.

Table 2: Loss-Free Balancing achieves lower perplexity and better load balance on both 1B and 3B models. A validation set is used to calculate these metrics (see details in Appendix [B](#A2 "Appendix B Training Settings ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts")).

|  |  |  |  |
| --- | --- | --- | --- |
| Model Size | Load Balancing Methods | Validation Perplexity | MaxVioglobalsubscriptMaxVioglobal\textbf{MaxVio}\_{\textbf{global}} |
| 1B | Loss-Controlled | 9.56 | 0.72 |
| Loss-Free | 9.50 | 0.04 |
| 3B | Loss-Controlled | 7.97 | 0.52 |
| Loss-Free | 7.92 | 0.04 |

![Refer to caption](/html/2408.15664/assets/x1.png)


Figure 3: Loss-Free Balancing maintains a better load balance throughout most of the training time. Here, MaxViobatchsubscriptMaxViobatch\text{MaxVio}\_{\text{batch}} is averaged over 100 neighboring steps for visibility purposes.

### 4.3 Empirical Studies on Bias Update Algorithm

We conduct empirical studies on the update rate and variants of the bias update algorithm to validate the optimal configuration used in our main experiments.

#### Update rate.

The update rate u𝑢u in Algorithm [1](#algorithm1 "In 3 Auxiliary-Loss-Free Load Balancing Strategy ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts") controls the speed at which the expert bias {bi}i=1Nsuperscriptsubscriptsubscript𝑏𝑖𝑖1𝑁\{{b}\_{i}\}\_{i=1}^{N} converges to the “suitable bias”. Figure [4](#S4.F4 "Figure 4 ‣ Update rate. ‣ 4.3 Empirical Studies on Bias Update Algorithm ‣ 4 Experiments ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts") illustrates that an overly low update rate u=0.0001𝑢0.0001u=0.0001 may lead to slow convergence, while an unnecessarily high update rate u=0.01𝑢0.01u=0.01 can cause undesirable fluctuations of the expert bias bisubscript𝑏𝑖{b}\_{i} during the later stage of training, deteriorating load balance in this stage. Both situations can impair performance. An appropriate choice is u=0.001𝑢0.001u=0.001, which shows good training balance and validation perplexity.

![Refer to caption](/html/2408.15664/assets/x2.png)


Figure 4: The impact of update rate on training load balance. A low update rate shows poor load balance in the early stage of training, while a high update rate deteriorates load balance in the later stage. Validation PPL denotes the validation perplexity.

#### Update rule.

We investigate a different update rule of the expert-wise biases.
To be specific, we attempt to change the update rule of bi=bi+u∗sign​(ei)subscript𝑏𝑖subscript𝑏𝑖𝑢signsubscript𝑒𝑖b\_{i}=b\_{i}+u\*\mathrm{sign}(e\_{i}) to bi=bi+u∗eisubscript𝑏𝑖subscript𝑏𝑖𝑢subscript𝑒𝑖b\_{i}=b\_{i}+u\*e\_{i}, which encourages the bias of experts with high violation errors to change faster. Although this variant slightly improves load balance, it does not lead to better performance, as shown in Table [3](#S4.T3 "Table 3 ‣ Update rule. ‣ 4.3 Empirical Studies on Bias Update Algorithm ‣ 4 Experiments ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts").
Therefore, we maintain the signsign\mathrm{sign} version.

Table 3: The variant bi=bi+u∗eisubscript𝑏𝑖subscript𝑏𝑖𝑢subscript𝑒𝑖b\_{i}=b\_{i}+u\*e\_{i} slightly improves load balance but does not show improvement in model performance.

|  |  |  |
| --- | --- | --- |
| Method | Perplexity | MaxVioglobalsubscriptMaxVioglobal\textbf{MaxVio}\_{\textbf{global}} |
| bi=bi+u∗sign​(ei)subscript𝑏𝑖subscript𝑏𝑖𝑢signsubscript𝑒𝑖b\_{i}=b\_{i}+u\*\text{sign}(e\_{i}), u=0.001𝑢0.001u=0.001 | 9.50 | 0.044 |
| bi=bi+u∗eisubscript𝑏𝑖subscript𝑏𝑖𝑢subscript𝑒𝑖b\_{i}=b\_{i}+u\*e\_{i}, u=0.01𝑢0.01u=0.01 | 9.53 | 0.028 |
| bi=bi+u∗eisubscript𝑏𝑖subscript𝑏𝑖𝑢subscript𝑒𝑖b\_{i}=b\_{i}+u\*e\_{i}, u=0.001𝑢0.001u=0.001 | 9.51 | 0.036 |
| bi=bi+u∗eisubscript𝑏𝑖subscript𝑏𝑖𝑢subscript𝑒𝑖b\_{i}=b\_{i}+u\*e\_{i}, u=0.0001𝑢0.0001u=0.0001 | 9.51 | 0.040 |

#### Multiplicative bias.

In addition to adding the expert-wise biases to the gating scores, using multiplicative biases is also a potential variant:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | gi,t={si,t,si,t∗bi∈Topk⁡({sj,t∗bj∣1≤j≤N},K),0, otherwise,subscript𝑔  𝑖𝑡casessubscript𝑠  𝑖𝑡subscript𝑠  𝑖𝑡subscript𝑏𝑖Topkconditional-setsubscript𝑠  𝑗𝑡subscript𝑏𝑗1𝑗𝑁𝐾0 otherwise,\displaystyle g\_{i,t}=\begin{cases}s\_{i,t},&s\_{i,t}\*{b}\_{i}\in\operatorname{Topk}\left(\left\{s\_{j,t}\*{b}\_{j}\mid 1\leq j\leq N\right\},K\right),\\ 0,&\text{ otherwise, }\end{cases} |  | (5) |

These {bi}i=1Nsuperscriptsubscriptsubscript𝑏𝑖𝑖1𝑁\{{b}\_{i}\}\_{i=1}^{N} can be updated using a similar procedure to Algorithm [1](#algorithm1 "In 3 Auxiliary-Loss-Free Load Balancing Strategy ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"), except that they should be initialized as 1 instead of 0.
Table [4](#S4.T4 "Table 4 ‣ Multiplicative bias. ‣ 4.3 Empirical Studies on Bias Update Algorithm ‣ 4 Experiments ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts") shows that using multiplicative biases results in slightly worse model performance compared to using additive biases, without significant improvements in load balance.
Based on these findings, we conclude that additive biases are a more suitable choice for our method.

Table 4: Multiplicative bias shows similar load balance but slightly worse performance compared to additive bias.

|  |  |  |
| --- | --- | --- |
| Method | Perplexity | MaxVioglobalsubscriptMaxVioglobal\textbf{MaxVio}\_{\textbf{global}} |
| Addative Bias, u=0.001𝑢0.001u=0.001 | 9.50 | 0.044 |
| Multiplicative Bias, u=0.01𝑢0.01u=0.01 | 9.52 | 0.041 |
| Multiplicative Bias, u=0.001𝑢0.001u=0.001 | 9.52 | 0.036 |
| Multiplicative Bias, u=0.0001𝑢0.0001u=0.0001 | 9.54 | 0.048 |

## 5 Discussion

### 5.1 Loss-Free Balancing Is Compatible with Expert Parallelism

Extremely large-scale MoE models often employ expert parallelism (Lepikhin et al., [2020](#bib.bib4)) for training or inference, which distributes experts across different devices to reduce memory requirements.
In such scenarios, load balance on the data in a single computation step is crucial for efficiency.
Due to expert parallelism, each computation step involves micro\_batch\_size \* ep\_data\_parallel\_size samples, which we refer to as a computation batch.
Here, micro\_batch\_size denotes the number of samples processed in one gradient accumulation step on a single device.

Loss-Free Balancing can achieve nearly optimal global load balance, and the load balance in each computation step will get closer to the global load balance as the computation batch size increases.
In Figure [5](#S5.F5 "Figure 5 ‣ 5.1 Loss-Free Balancing Is Compatible with Expert Parallelism ‣ 5 Discussion ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"), we examine the computation-batch-level load balance with the MaxViocomputation-batchsubscriptMaxViocomputation-batch\text{MaxVio}\_{\text{computation-batch}} metric.
The results show that the load balance of our Loss-Free Balancing always keeps improving as the computation batch size increases, but the load balance of the auxiliary-loss-controlled method approximately maintains a constant level when the computation batch is large.
Since expert parallelism will significantly increase the computation batch size by ep\_data\_parallel\_size times, Loss-Free Balancing is naturally compatible with large-scale MoE training, and its advantage on the load balance will be further enhanced as the size of expert parallelism increases.

![Refer to caption](/html/2408.15664/assets/x3.png)


Figure 5: Loss-Free Balancing achieves improved balance compared to auxiliary-loss training as the computation-batch size increases, demonstrating its superiority when a moderately sized computation-batch is utilized.

### 5.2 Load Balancing and Future Token Leakage

For casual language models, load balancing methods must adhere to the causal constraint of language modeling to avoid future token leakage. While conventional auxiliary-controlled balancing and our Loss-Free Balancing obey this constraint, Expert Choice (EC) (Zhou et al., [2022](#bib.bib10)) violates it. EC ensures perfect load balance by assigning exactly the same number of tokens to each expert. However, this approach inherently leads to a severe issue of future token leakage.

In EC, future tokens can influence the expert assignment of previous tokens. Figure [6](#S5.F6 "Figure 6 ‣ 5.2 Load Balancing and Future Token Leakage ‣ 5 Discussion ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts") illustrates how information can be easily transmitted within a sequence via such influence. Theoretically, the token assignment of an MoE layer with sparse ratio R𝑅R (average activated experts per token K𝐾K divided by total expert number N𝑁N) can leak more than K​log2⁡1−RR𝐾subscript21𝑅𝑅K\log\_{2}\frac{1-R}{R} bits per token (proof in Appendix [D.1](#A4.SS1 "D.1 Proof for Theoretical Leakage Amount ‣ Appendix D Future Token Leakage in Expert Choice ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts")). For a 9-layer MoE model with 16 experts and an average of 2 experts per token, this amounts to 50 bits, sufficient for each token to determine its successor’s identity.

![Refer to caption](/html/2408.15664/assets/figs/EC_fig_new.png)


Figure 6: 
An example of future token leakage in EC.
Future tokens can influence the expert assignment of previous tokens.
Such an assignment can help previous tokens to infer the identity of their successors.

We designed experiments to demonstrate the existence of future token leakage in realistic model training. (1) We reduced the chunk size, within which top-K selection is performed, from 8192 tokens (4 sentences) to 512 (1/4 sentence), with the expectation of exposing such leakage. We observed an abnormal loss drop (about 10%), confirming the presence of leakage. (2) We made leakage more difficult by shuffling tokens across chunks in the top-K selection step, and observed that the abnormal loss drop was mitigated. Detailed experimental results on EC’s information leakage are provided in Appendix [D.2](#A4.SS2 "D.2 Experimental Evidence ‣ Appendix D Future Token Leakage in Expert Choice ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts").

Future token leakage is fatal since it destroys the generalization of a model and prevents reliable evaluation of the model performance.
Therefore, compared with EC, scaling up an MoE model with our Loss-Free Balancing is safer.

## 6 Conclusion

In this work, we introduced Loss-Free Balancing, a novel MoE load balance control method without introducing auxiliary-loss gradients. Loss-Free Balancing addresses the issue of traditional auxiliary-loss load balance control, which introduces additional gradients during training and potentially impairs model performance when enforcing load balance. Experiments conducted on 1B and 3B MoE models, trained on 100B and 300B tokens respectively, demonstrate that Loss-Free Balancing achieves better model performance and load balance compared to the traditional auxiliary-loss training.

## References

* Dai et al. (2024)

  Damai Dai, Chengqi Deng, Chenggang Zhao, Runxin Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang.
  Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models.
  *ArXiv*, abs/2401.06066, 2024.
  URL <https://api.semanticscholar.org/CorpusID:266933338>.
* DeepSeek-AI et al. (2024)

  DeepSeek-AI, Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y. Wu, Yukun Li, Huazuo Gao, Shirong Ma, Wangding Zeng, Xiao Bi, Zihui Gu, Hanwei Xu, Damai Dai, Kai Dong, Liyue Zhang, Yishi Piao, Zhibin Gou, Zhenda Xie, Zhewen Hao, Bing-Li Wang, Jun-Mei Song, Deli Chen, Xin Xie, Kang Guan, Yu mei You, Aixin Liu, Qiushi Du, Wenjun Gao, Xuan Lu, Qinyu Chen, Yaohui Wang, Chengqi Deng, Jiashi Li, Chenggang Zhao, Chong Ruan, Fuli Luo, and Wenfeng Liang.
  Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence.
  *ArXiv*, abs/2406.11931, 2024.
  URL <https://api.semanticscholar.org/CorpusID:270562723>.
* Fedus et al. (2021)

  William Fedus, Barret Zoph, and Noam M. Shazeer.
  Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.
  *J. Mach. Learn. Res.*, 23:120:1–120:39, 2021.
  URL <https://api.semanticscholar.org/CorpusID:231573431>.
* Lepikhin et al. (2020)

  Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam M. Shazeer, and Z. Chen.
  Gshard: Scaling giant models with conditional computation and automatic sharding.
  *ArXiv*, abs/2006.16668, 2020.
  URL <https://api.semanticscholar.org/CorpusID:220265858>.
* Loshchilov & Hutter (2016)

  Ilya Loshchilov and Frank Hutter.
  Sgdr: Stochastic gradient descent with warm restarts.
  *arXiv: Learning*, 2016.
  URL <https://api.semanticscholar.org/CorpusID:14337532>.
* Sennrich et al. (2015)

  Rico Sennrich, Barry Haddow, and Alexandra Birch.
  Neural machine translation of rare words with subword units.
  *ArXiv*, abs/1508.07909, 2015.
  URL <https://api.semanticscholar.org/CorpusID:1114678>.
* Shao et al. (2024)

  Zhihong Shao, Damai Dai, Daya Guo, Bo Liu, and Zihan Wang.
  Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model.
  *ArXiv*, abs/2405.04434, 2024.
  URL <https://api.semanticscholar.org/CorpusID:269613809>.
* Shazeer et al. (2017)

  Noam M. Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean.
  Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
  *ArXiv*, abs/1701.06538, 2017.
  URL <https://api.semanticscholar.org/CorpusID:12462234>.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  In *Neural Information Processing Systems*, 2017.
  URL <https://api.semanticscholar.org/CorpusID:13756489>.
* Zhou et al. (2022)

  Yan-Quan Zhou, Tao Lei, Han-Chu Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M. Dai, Zhifeng Chen, Quoc V. Le, and James Laudon.
  Mixture-of-experts with expert choice routing.
  *ArXiv*, abs/2202.09368, 2022.
  URL <https://api.semanticscholar.org/CorpusID:247011948>.

## Appendix A Model Architecture

We employ the DeepSeekMoE (Dai et al., [2024](#bib.bib1)) architecture as the backbone, which introduces shared experts to mitigate knowledge redundancy among routed experts:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝐡t=𝐮t+∑i=1NsFFNi(s)⁡(𝐮t)+∑i=1Nrgi,t​FFNi(r)⁡(𝐮t),subscript𝐡𝑡subscript𝐮𝑡superscriptsubscript𝑖1subscript𝑁𝑠superscriptsubscriptFFN𝑖𝑠subscript𝐮𝑡superscriptsubscript𝑖1subscript𝑁𝑟subscript𝑔  𝑖𝑡superscriptsubscriptFFN𝑖𝑟subscript𝐮𝑡\displaystyle\mathbf{h}\_{t}=\mathbf{u}\_{t}+\sum\_{i=1}^{N\_{s}}\operatorname{FFN}\_{i}^{(s)}\left(\mathbf{u}\_{t}\right)+\sum\_{i=1}^{N\_{r}}g\_{i,t}\operatorname{FFN}\_{i}^{(r)}\left(\mathbf{u}\_{t}\right), |  | (6) |

where r𝑟r denotes the routed experts, while s𝑠s the shared experts. DeepSeekMoE replaces all FFN layers with MoE layers, except the dense FFN layer just after the input embedding layer.

The detailed architecture hyper-parameters are listed in Table [5](#A1.T5 "Table 5 ‣ Appendix A Model Architecture ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts").

Table 5: Model architecture.

|  |  |  |
| --- | --- | --- |
| hyper-parameters | 1B | 3B |
| Vocab size | 32064 | 32064 |
| Hidden size | 1024 | 1280 |
| Attention heads | 8 | 10 |
| MoE layers | 9 | 11 |
| Granularity (dffdexpertsubscript𝑑ffsubscript𝑑expert\frac{d\_{\text{ff}}}{d\_{\text{expert}}}) | 163163\frac{\text{16}}{\text{3}} | 4 |
| Shared experts | 2 | 2 |
| Routed experts | 64 | 64 |
| Activated routed experts | 6 | 6 |

## Appendix B Training Settings

Following the work of Dai et al. ([2024](#bib.bib1)), we initialize all learnable parameters with a standard deviation of 0.006, and set the maximum training sequence length to 2048.

For the 1B model, we employ a cosine learning rate scheduler with warmup, setting the learning rate to 1e-3, the minimum learning rate to 1e-4, and the warmup steps to 1000. The training batch size for the 1B model is set to 1152, resulting in a total of 40000 training steps (100B tokens).

For the 3B model, we use a multistep learning rate scheduler with stage steps = [45211, 50862, 56514] and corresponding stage learning rates of [7.8e-4, 2.47e-4, 7.8e-5]. The warmup steps for the 3B model are set to 2000. We use a training batch size of 1728 for the 3B model, resulting in a total of 56514 training steps (200B tokens).

For validation, we leave around 70M tokens from the training corpus as the validation set (30 \* 1B\_batch\_size \* max\_seq\_len = 20 \* 3B\_batch\_size \* max\_seq\_len = 71M tokens).

## Appendix C Experiments with Softmax Gate

### C.1 Comparison of Sigmoid Gate Baseline and Softmax Gate Baseline

We compare the sigmoid gate baseline and the softmax gate baseline with varying auxiliary loss coefficients α𝛼\alpha on a 1B-sized model. As shown in Figure [7](#A3.F7 "Figure 7 ‣ C.1 Comparison of Sigmoid Gate Baseline and Softmax Gate Baseline ‣ Appendix C Experiments with Softmax Gate ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"), the softmax gate exhibits higher perplexity under similar load balance conditions, and its performance is more sensitive to load imbalance compared to the sigmoid gate.

![Refer to caption](/html/2408.15664/assets/figs/balance_mv_perplexity_softmax_new.png)


Figure 7: Comparison of the sigmoid gate baseline and the softmax gate baseline. The softmax gate exhibits higher perplexity under similar load balance conditions and is more sensitive to load imbalance compared to the sigmoid gate.

### C.2 Loss-Free Load Balancing with Softmax Gate

Adjusting the per-expert bias for the softmax gate is more challenging due to the normalization property of softmax, which makes the score gap between two experts sensitive to the scores of other experts. In such a situation, we choose the 𝐛i=𝐛i+u∗eisubscript𝐛𝑖subscript𝐛𝑖𝑢subscript𝑒𝑖\mathbf{b}\_{i}=\mathbf{b}\_{i}+u\*e\_{i} variant to maintain load balance, where u𝑢u is set to 1e-3. For the baseline, we choose α𝛼\alpha = 0.0003, which yields the lowest perplexity for the softmax gate. The results are presented in Table [6](#A3.T6 "Table 6 ‣ C.2 Loss-Free Load Balancing with Softmax Gate ‣ Appendix C Experiments with Softmax Gate ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"), showing that Loss-Free Balancing achieves a slightly lower perplexity while maintaining significantly better load balance compared to the auxiliary-loss training method. Figure [8](#A3.F8 "Figure 8 ‣ C.2 Loss-Free Load Balancing with Softmax Gate ‣ Appendix C Experiments with Softmax Gate ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts") confirms that Loss-Free Balancing maintains a superior load balance throughout most of the training process.

Table 6: For softmax gate, Loss-Free Balancing achieves a slightly lower perplexity while reaching a significantly better load balance compared to the auxiliary-loss training method.

|  |  |  |
| --- | --- | --- |
| Load Balancing | Perplexity | MaxVioglobalsubscriptMaxVioglobal\textbf{MaxVio}\_{\text{global}} |
| Loss-Controlled | 9.604 | 0.937 |
| Loss-Free | 9.599 | 0.027 |

![Refer to caption](/html/2408.15664/assets/x4.png)


Figure 8: For softmax gate, Loss-Free Balancing maintains a superior load balance throughout most of the training process.

## Appendix D Future Token Leakage in Expert Choice

### D.1 Proof for Theoretical Leakage Amount

Let R=KN𝑅𝐾𝑁R=\frac{K}{N} denote the MoE sparsity. Here K𝐾K denotes the average number of experts activated per token, and N𝑁N is the total number of experts.
For an MoE layer in Expert Choice, the maximum information leakage I𝐼I (in bits per token), i.e., the information that the combinations of routing allocation can carry is:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | I=𝐼absent\displaystyle I= | log2⁡(K​TNT)N/Tsubscript2/superscriptbinomial𝐾𝑇𝑁𝑇𝑁𝑇\displaystyle\left.\log\_{2}\binom{\frac{KT}{N}}{T}^{N}\right/T |  | (7) |
|  | >\displaystyle> | N​log2⁡((1−KN)​T)KN​T(KN​T)KN​T/T𝑁subscript2/superscript1𝐾𝑁𝑇𝐾𝑁𝑇superscript𝐾𝑁𝑇𝐾𝑁𝑇𝑇\displaystyle N\left.\log\_{2}\frac{((1-\frac{K}{N})T)^{\frac{K}{N}T}}{(\frac{K}{N}T)^{\frac{K}{N}T}}\right/T |  |
|  | =\displaystyle= | K​log2⁡1−RR.𝐾subscript21𝑅𝑅\displaystyle K\log\_{2}\frac{1-R}{R}. |  |

For a model with a sparse ratio R=216=0.125𝑅2160.125R=\frac{2}{16}=0.125 and 9 MoE layers, the total leakage information is more than 50 bits per token.

### D.2 Experimental Evidence

We investigate the potential future token leakage of the Expert Choice by varying the chunk size used for experts’ top-k𝑘k selection, ranging from 512 tokens to 8192 tokens.333A chunk size of 2048 tokens means performing top-k𝑘k selection inside a sentence, while 512 tokens correspond to a quarter of a sentence and 8192 tokens to four sentences. We train a 2B MoE model on 100B tokens. The results, shown in Table [9](#A4.F9 "Figure 9 ‣ D.2 Experimental Evidence ‣ Appendix D Future Token Leakage in Expert Choice ‣ Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"), reveal two key findings:

1. 1.

   Using a small chunk size of 512 leads to an abnormal loss drop, which can be attributed to significant future token leakage. A smaller chunk size allows the model to more easily exploit information from future tokens within the chunk during training.
2. 2.

   Shuffling tokens within a batch before chunking and selecting mitigates the observed loss drop. Such shuffling makes it more challenging for the model to utilize information leakage, as the future tokens are no longer in their original context. This finding supports the hypothesis that the loss drop originates from the model’s accessing and exploiting future token information.

![Refer to caption](/html/2408.15664/assets/x5.png)


Figure 9: Comparison of Expert Choice with different chunk sizes and shuffling. Expert Choice with a chunk size of 512 exhibits a significant loss drop compared to chunk sizes of 8192 or 2048. Shuffling tokens eliminates this loss drop, indicating the presence of future token leakage.

[◄](/html/2408.15663)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2408.15664)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2408.15664)
[View original  
on arXiv](https://arxiv.org/abs/2408.15664)[►](/html/2408.15666)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Sep 5 14:19:10 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
