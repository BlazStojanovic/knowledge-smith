---
arxiv: '2211.16887'
authors:
- Jiahuan Yan
- Jintai Chen
- Yixuan Wu
- Danny Z. Chen
- Jian Wu
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous
  Feature Interaction'
url: http://arxiv.org/abs/2211.16887v2
year: 2022
---

[2211.16887] T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction














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



# T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction

Jiahuan Yan1\equalcontrib,
Jintai Chen1\equalcontrib,
Yixuan Wu2,
Danny Z. Chen3,
Jian Wu4
The corresponding author.

###### Abstract

Recent development of deep neural networks (DNNs) for tabular learning has largely benefited from the capability of DNNs for automatic feature interaction. However, the heterogeneity nature of tabular features makes such features relatively independent, and developing effective methods to promote tabular feature interaction still remains an open problem. In this paper, we propose a novel Graph Estimator, which automatically estimates the relations among tabular features and builds graphs by assigning edges between related features. Such relation graphs organize independent tabular features into a kind of graph data such that interaction of nodes (tabular features) can be conducted in an orderly fashion. Based on our proposed Graph Estimator, we present a bespoke Transformer network tailored for tabular learning, called T2G-Former, which processes tabular data by performing tabular feature interaction guided by the relation graphs. A specific Cross-level Readout collects salient features predicted by the layers in T2G-Former across different levels, and attains global semantics for final prediction. Comprehensive experiments show that our T2G-Former achieves superior performance among DNNs and is competitive with non-deep Gradient Boosted Decision Tree models. The code and models are available at https://github.com/jyansir/t2g-former.

## 1 Introduction

Data in the form of table structures are ubiquitous in many fields, e.g., medical records (Johnson, Pollard et al. [2016](#bib.bib24); Hassan, Al-Insaif et al. [2020](#bib.bib19)) and click-through rate (CTR) prediction (Covington, Adams, and Sargin [2016](#bib.bib10); Song, Shi et al. [2019](#bib.bib45)). It was observed that Gradient Boosted Decision Trees (GBDT) (Chen and Guestrin [2016](#bib.bib9); Ke, Meng et al. [2017](#bib.bib26); Prokhorenkova, Gusev et al. [2018](#bib.bib41)) were dominating models for tabular data tasks in machine learning and industrial applications. Due to big successes of deep neural networks (DNNs) in various fields, there has been increasing development of specialized DNNs for tabular data learning (Popov, Morozov, and Babenko [2019](#bib.bib40); Arik and Pfister [2021](#bib.bib3); Wang, Shivanna et al. [2021](#bib.bib50); Gorishniy, Rubachev et al. [2021](#bib.bib16); Chen, Liao et al. [2022](#bib.bib8)). Such studies either leveraged ensembling of neural networks (Popov, Morozov, and Babenko [2019](#bib.bib40); Arik and Pfister [2021](#bib.bib3); Katzir, Elidan, and El-Yaniv [2020](#bib.bib25)) to build differentiable tree models, or explored diverse interaction approaches (Guo, Tang et al. [2017](#bib.bib17); Wang, Fu et al. [2017](#bib.bib49); Song, Shi et al. [2019](#bib.bib45); Wang, Shivanna et al. [2021](#bib.bib50); Gorishniy, Rubachev et al. [2021](#bib.bib16); Chen, Liao et al. [2022](#bib.bib8)) to learn comprehensive features by fusing different tabular features.

However, different from images and texts, it is challenging for fusion-based models to handle tabular feature interaction due to the feature heterogeneity problem (Borisov, Leemann et al. [2021](#bib.bib7)). DANets (Chen, Liao et al. [2022](#bib.bib8)) suggested the “selection & abstraction” principle that processes tabular data by first selecting and then interacting the selected features. Known neural feature selection schemes can be categorized into soft and hard versions. The soft selection essentially exerts fully connected interactions among features (see Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(b)), such as multiplicative interaction (Guo, Tang et al. [2017](#bib.bib17)), feature crossing (Wang, Fu et al. [2017](#bib.bib49); Wang, Shivanna et al. [2021](#bib.bib50)), and attention-based interaction (Song, Shi et al. [2019](#bib.bib45); Huang, Khetan et al. [2020](#bib.bib23); Gorishniy, Rubachev et al. [2021](#bib.bib16)). However, tabular features by nature are heterogeneous, and fully connected interaction is a sub-optimal choice since it blindly fuses all features together. DANets (Chen, Liao et al. [2022](#bib.bib8)) performed hard selection by grouping correlative features and then constraining interactions among grouped features (Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(c)). Although DANets achieved promising results, its feature selection operation cannot thoroughly address intra-group interactions (see Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(c)), and thus features assigned in a same group are indiscriminately fused, making the model inferiorly expressive.

![Refer to caption](/html/2211.16887/assets/x1.png)


Figure 1: An example of medical data tables. The values in different columns are located in heterogeneous feature spaces. Underlying medical knowledge sparsely links feature pairs. (a) Original separated features without any interactions, which are often used in non-deep models. (b) Fully connected interactions by softly selecting all the features. (c) Selective interactions among grouped features by hard selection. (d) Selective interactions according to a weighted relation graph. “BP” denotes blood pressure; “HIV-Ab” indicates the level of HIV antibody.

There are numerous daily applications that exemplify the significance of selective interaction for heterogeneous tabular features. The left part of Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction") gives an example of a medical data table.
Using underlying medical knowledge, a static graph can be formed to indicate relations of reasonable feature pairs. For instance, the relation of height and weight gives a probability representing a high-level semantic physique. Also, the relation between weight and blood pressure (BP) is likely to indicate a semantic cardiovascular health. Besides, there might be some “inert features” that are unrelated to any other features, such as the features representing the level of HIV antibody (HIV-Ab).
In the right part, Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(a) presents the original tabular features whose relations are not specified, and higher-level semantics cannot be directly obtained if the feature relations are not determined. Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(b) illustrates the fully connected interactions of soft selection, which may introduce some noisy relations in feature fusion (e.g., the “inert feature” connects with the other features). Hard selection with a grouping operation (e.g., used in DANets) achieves partially selective interactions by grouping related features (see Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(c)), but is still likely to include noisy interactions.
It can only group related features but fails to handle the feature relations within the same group.
In Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(c), the grouping design can only put the features height, weight, and BP together for mutual interactions, but cannot exclude the meaningless height-BP pair. It is intuitive that a precise health condition assessment can be made based on both the data-specific record values (e.g., 173.6 cm for height in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")) and the underlying knowledge represented by the edges of the relation graph. For the first sample in the medical table (ID = 1), considering the values of height and weight jointly can suggest a symptom of overweight. Similarly, combining the values of weight and BP indicates a risk of cardiovascular problems. The second sample (ID = 2) directly indicates a risk of HIV infection solely based on the feature of HIV-Ab. Hence, we argue that an ideal way to handle such complex decision processes is to build a graph with adaptive edge weights. The edge weights (represented by different colors and widths in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(d)) indicate the strengths of relations based on specific feature values, and the static graph topology represents the underlying knowledge to constrain meaningful relations.

Inspired by the above observations, in this paper, we propose to build graphs for tabular features to guide feature interaction. We develop a novel Graph Estimator (GE) for organizing independent tabular features into a feature relation graph (FR-Graph). Further, we present
a bespoke Transformer network for tabular learning, called T2G-Former, by stacking GE-incorporated blocks for selective feature interaction. GE models an FR-Graph by assembling (i) a static graph topology depicting underlying knowledge of the task and (ii) data-adaptive edge weights for graph edges. The static graph depicts the underlying knowledge (the relations of feature pairs), while the data-adaptive edge weights represent the strengths of relations based on specific feature values. Using the FR-Graph, we can effectively capture more subtle interactions which may be mishandled by grouping strategies (as shown in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(c)). In our proposed T2G-Former, each layer employs the FR-Graph to transform layer input features into graph data, and heterogeneous feature interactions are performed in an orderly fashion based on the specification of graph edges. Besides, a special Cross-level Readout collects salient features from each level and attains global tabular semantics for the final prediction.

The workflow of T2G-Former proceeds as follows. An FR-Graph, whose edges represent the static relations of features with data-adaptive weights (predicted by the GE module), guides the processing of the tabular feature interaction to predict higher-level features. Then another FR-Graph for higher-level tabular features is built to organize the feature interaction, and the process continues. T2G-Former can output comprehensive semantics from different feature levels by repeating the above process. The shared Cross-level Readout is used to aggregate semantics from different feature levels, and takes all these features into consideration in the final prediction.

Overall, the main contributions of our work are as follows:

* •

  We first utilize feature relation graphs to handle heterogeneous feature interaction for tabular data, and propose a novel GE module for feature relation organization.
* •

  We adapt feature relation graphs in the Transformer architecture, and build a specialized tabular learning Transformer T2G-Former for tabular classification and regression.
* •

  Comprehensive experiments show that T2G-Former consistently outperforms state-of-the-art tabular DNNs on many datasets, and is competitive with GBDTs.

![Refer to caption](/html/2211.16887/assets/x2.png)


Figure 2: (a) The architecture of T2G-Former for tabular learning. Each T2G block builds an FR-Graph for a feature level and performs selective interaction. A global readout node collects salient features from each layer to form tabular semantics. (b) Illustrating a basic block in Sec. [4.1](#S4.SS1 "4.1 Basic Block ‣ 4 T2G-Former ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction") and GE in Sec. [3](#S3 "3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction").

## 2 Related Work

### 2.1 DNNs for Tabular Learning

Tabular learning refers to machine learning applications on tabular data that conducts prediction based on categorical or continuous features (Dong, Cheng et al. [2022](#bib.bib11)). Classical non-deep methods (Li et al. [1984](#bib.bib31); Friedman [2001](#bib.bib14); Zhang and Honavar [2003](#bib.bib54); Zhang, Kang et al. [2006](#bib.bib55); He, Pan et al. [2014](#bib.bib22)) are prevalent choices for such tasks (Anghel, Papandreou et al. [2018](#bib.bib2)), especially the ensemble methods of decision trees, such as GBDT (Friedman [2001](#bib.bib14)), XGBoost (Chen and Guestrin [2016](#bib.bib9)), LightBGM (Ke, Meng et al. [2017](#bib.bib26)), and CatBoost (Prokhorenkova, Gusev et al. [2018](#bib.bib41)).

Compared to their shallow counterparts, DNNs enjoy strong abilities of automatic feature learning (Thawani, Pujara et al. [2021](#bib.bib46)), and hence offer a good potential to exploit hidden features. Recently, increasingly more studies applied DNNs to tabular data (Guo, Tang et al. [2017](#bib.bib17); Yang, Morillo, and Hospedales [2018](#bib.bib53); Song, Shi et al. [2019](#bib.bib45); Feng, Yu, and Zhou [2018](#bib.bib13); Hazimeh, Ponomareva et al. [2020](#bib.bib20); Popov, Morozov, and Babenko [2019](#bib.bib40); Arik and Pfister [2021](#bib.bib3); Chen, Liao et al. [2022](#bib.bib8)), which can be roughly categorized into differentiable tree models and fusion-based models.

#### Differentiable Tree Models.

DNNs of this type (Popov, Morozov, and Babenko [2019](#bib.bib40); Arik and Pfister [2021](#bib.bib3); Katzir, Elidan, and El-Yaniv [2020](#bib.bib25)) were inspired by the successes of the ensemble tree frameworks (Kontschieder, Fiterau et al. [2015](#bib.bib30); Feng, Yu, and Zhou [2018](#bib.bib13); Yang, Morillo, and Hospedales [2018](#bib.bib53)). NODE (Popov, Morozov, and Babenko [2019](#bib.bib40)) combined differentiable oblivious decision trees (Lou and Obukhov [2017](#bib.bib33)) with multi-layer hierarchical representations and achieved competitive performances as GBDT. TabNet (Arik and Pfister [2021](#bib.bib3)) employed an attention mechanism (Vaswani, Shazeer et al. [2017](#bib.bib48)) to sequentially select salient features for tree-like decision. Net-DNF (Katzir, Elidan, and El-Yaniv [2020](#bib.bib25)) introduced bias of a disjunctive normal form to select and aggregate feature subsets in each block. NODE and Net-DNF largely benefited from model ensembles but did not take advantage of the feature representation capability of DNNs (Chen, Liao et al. [2022](#bib.bib8)). TabNet designed non-interactive transformer blocks for feature representation and selection without feature fusion. All these DNNs function as feature selectors and splitters, but neglect underlying interactions among tabular features.

#### Fusion-based Models.

Fusion-based models (Guo, Tang et al. [2017](#bib.bib17); Song, Shi et al. [2019](#bib.bib45); Huang, Khetan et al. [2020](#bib.bib23); Wang, Shivanna et al. [2021](#bib.bib50); Gorishniy, Rubachev et al. [2021](#bib.bib16); Chen, Liao et al. [2022](#bib.bib8)) leveraged DNNs to fuse higher-level features via feature interaction. DeepFM (Guo, Tang et al. [2017](#bib.bib17)) performed multiplicative interaction on encoded features for CTR prediction. DCN (Wang, Fu et al. [2017](#bib.bib49); Wang, Shivanna et al. [2021](#bib.bib50)) combined DNNs with cross components to learn complex features with high-order interactions. Recently, attention module (Vaswani, Shazeer et al. [2017](#bib.bib48)) became a popular choice due to its interactive bias and remarkable performance (Kenton and Toutanova [2019](#bib.bib27); Dosovitskiy, Beyer et al. [2020](#bib.bib12)). AutoInt (Song, Shi et al. [2019](#bib.bib45)) used multi-head self-attention to interact low-dimension embedded features. TabTransformer (Huang, Khetan et al. [2020](#bib.bib23)) directly transferred Transformer (Vaswani, Shazeer et al. [2017](#bib.bib48)) blocks to tabular data but neglected interaction between categorical features and continuous ones. FT-Transformer (Gorishniy, Rubachev et al. [2021](#bib.bib16)) addressed this problem by tokenizing these two types of features and processing them equally. DANets (Chen, Liao et al. [2022](#bib.bib8)) selected correlative tabular features and attentively fused the selected features into higher-level ones.

### 2.2 Tabular Feature Interaction

Most of the previous fusion-based work simply transferred successful neural architectures (e.g., MLP (Guo, Tang et al. [2017](#bib.bib17)), self-attention (Song, Shi et al. [2019](#bib.bib45)), and Transformer (Huang, Khetan et al. [2020](#bib.bib23); Gorishniy, Rubachev et al. [2021](#bib.bib16))) into tabular data and interacted features with soft selection. However, feature heterogeneity (Borisov, Leemann et al. [2021](#bib.bib7); Popov, Morozov, and Babenko [2019](#bib.bib40)) led to gap of inductive bias and made these models (which were designed for homogeneous data, e.g., images and texts) sub-optimal. DANets (Chen, Liao et al. [2022](#bib.bib8)) first adapted selective feature interaction by hard selection, constraining interactions in a feature group, and achieved promising results; but, relations of intra-group features were still not managed well. Hence, this paper proposes feature relation graphs and adapts them into a tailored Transformer network.

## 3 Graph Estimator

We propose Graph Estimator (GE) (Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(b)) for automatically building Feature Relation Graphs (FR-Graphs), which treats tabular features as nodes in a graph and estimates the feature relations as edges. The GE design is inspired by knowledge graph completion (KGC) (Shi and Weninger [2018](#bib.bib43); Wu, Chen et al. [2021](#bib.bib51)) that might use semantical similarity of two entities to estimate their relation plausibility. A basic form to measure semantical similarity (Nickel, Tresp, and Kriegel [2011](#bib.bib38)) is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | fr​(h,t)=hT​Mr​t,subscript𝑓𝑟ℎ𝑡superscriptℎ𝑇subscript𝑀𝑟𝑡f\_{r}(h,t)=h^{T}M\_{r}t, |  | (1) |

where h,t∈ℝn

ℎ𝑡
superscriptℝ𝑛h,t\in\mathds{R}^{n} are an encoded head entity node and a tail one, and a learnable matrix Mr∈ℝn×nsubscript𝑀𝑟superscriptℝ𝑛𝑛M\_{r}\in\mathds{R}^{n\times n} represents relation r𝑟r in a knowledge graph (KG). Various following methods (Yang, Yih et al. [2015](#bib.bib52); Trouillon, Welbl et al. [2016](#bib.bib47); Nickel, Rosasco, and Poggio [2016](#bib.bib37)) followed this idea, which differed from one another solely in relation embeddings and score functions.

Different from KGC models that only compute static relation plausibility for entities, GE estimates the feature relations by a static underlying graph topology with data-adaptive edge weights. We take each tabular feature as a node, and first perform semantic matching to estimate the soft plausibility of pair-wise interactions between tabular features, which are referred to as data-adaptive edge weights in this section. Second, a static knowledge topology is learned based on tabular column semantics to preserve interactions of salient feature pairs. At the end, edge weights are assembled with the knowledge topology to form an FR-Graph.

### 3.1 FR-Graph Structure Components

To mine the relations among tabular features, we build FR-Graph by treating tabular features as graph node candidates and predicting the edges among them. The edges were yielded from two perspectives: adaptive edge weights representing data-specific information, and static edge topology for all the data representing the underlying knowledge. Note that some features are isolated from the FR-Graph if no other nodes connected with them.

#### Adaptive Edge Weights.

Given two tabular feature embedding vectors xi,xj∈ℝn

subscript𝑥𝑖subscript𝑥𝑗
superscriptℝ𝑛x\_{i},x\_{j}\in\mathds{R}^{n} (i,j∈{1,2,…,N})i,j\in\{1,2,\dots,N\}), where N𝑁N is the number of input features (table columns), we evaluate their interaction plausibility using the following pair-wise score function:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Gw​[i,j]=gw​(fih,fjt)=fihT​diag​(r)​fjt,subscript𝐺𝑤𝑖𝑗subscript𝑔𝑤superscriptsubscript𝑓𝑖ℎsuperscriptsubscript𝑓𝑗𝑡superscriptsuperscriptsubscript𝑓𝑖ℎ𝑇diag𝑟superscriptsubscript𝑓𝑗𝑡G\_{w}[i,j]=g\_{w}(f\_{i}^{h},f\_{j}^{t})={f\_{i}^{h}}^{T}\text{diag}(r)f\_{j}^{t}, |  | (2) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | fih=Wh​xi,fit=Wt​xi,{Wh≡Wtif symmetric,Wh≠Wtif asymmetric,formulae-sequencesuperscriptsubscript𝑓𝑖ℎsuperscript𝑊ℎsubscript𝑥𝑖superscriptsubscript𝑓𝑖𝑡  superscript𝑊𝑡subscript𝑥𝑖casessuperscript𝑊ℎsuperscript𝑊𝑡if symmetricsuperscript𝑊ℎsuperscript𝑊𝑡if asymmetricf\_{i}^{h}=W^{h}x\_{i},f\_{i}^{t}=W^{t}x\_{i},\ \begin{cases}W^{h}\equiv W^{t}&\text{if symmetric},\\ W^{h}\neq W^{t}&\text{if asymmetric},\end{cases} |  | (3) |

where two learnable parameters Wh,Wt∈ℝm×n

superscript𝑊ℎsuperscript𝑊𝑡
superscriptℝ𝑚𝑛W^{h},W^{t}\in\mathds{R}^{m\times n} denote transformations for a head feature and a tail one, and diag​(r)∈ℝn×ndiag𝑟superscriptℝ𝑛𝑛\text{diag}(r)\in\mathds{R}^{n\times n} is a diagonal matrix parameterized by learnable relation vectors r∈ℝn𝑟superscriptℝ𝑛r\in\mathds{R}^{n} that semantically represent feature interaction relations. Here Whsuperscript𝑊ℎW^{h} and Wtsuperscript𝑊𝑡W^{t} share parameters if the pair-wise feature edge weights are symmetric (i.e., Gw​[i,j]≡Gw​[j,i]subscript𝐺𝑤𝑖𝑗subscript𝐺𝑤𝑗𝑖G\_{w}[i,j]\equiv G\_{w}[j,i]) and are parameter-independent in the asymmetric case (i.e., Gw​[i,j]≠Gw​[j,i]subscript𝐺𝑤𝑖𝑗subscript𝐺𝑤𝑗𝑖G\_{w}[i,j]\neq G\_{w}[j,i]). All bias vectors are omitted for notation brevity. Consequently, the adaptive weight scores gwsubscript𝑔𝑤g\_{w} of all feature pairs constitute a fully connected weighted relation graph Gwsubscript𝐺𝑤G\_{w}. Note that the edge weight score is degraded to an attention score when r𝑟r is filled with scalar value 1 (and diag​(r)diag𝑟\text{diag}(r) becomes an entity matrix), and thus it is able to measure weighted feature similarity.

#### Static Knowledge Topology.

Although we introduce soft edge weights for all feature pairs, it is also important to globally consider the underlying knowledge of the tabular data. Thus, we use a series of column embeddings to represent the semantics of the tabular features, and a static relation topology score can be computed as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Gt​[i,j]=gt​(eih,ejt)=eihT​ejt‖eih‖2​‖ejt‖2,subscript𝐺𝑡𝑖𝑗subscript𝑔𝑡superscriptsubscript𝑒𝑖ℎsuperscriptsubscript𝑒𝑗𝑡superscriptsuperscriptsubscript𝑒𝑖ℎ𝑇superscriptsubscript𝑒𝑗𝑡subscriptnormsuperscriptsubscript𝑒𝑖ℎ2subscriptnormsuperscriptsubscript𝑒𝑗𝑡2G\_{t}[i,j]=g\_{t}(e\_{i}^{h},e\_{j}^{t})=\frac{{e\_{i}^{h}}^{T}e\_{j}^{t}}{\|e\_{i}^{h}\|\_{2}\|e\_{j}^{t}\|\_{2}}, |  | (4) |

|  |  |  |
| --- | --- | --- |
|  | eih=Eh​[:,i],eit=Et​[:,i],formulae-sequencesuperscriptsubscript𝑒𝑖ℎsuperscript𝐸ℎ:𝑖superscriptsubscript𝑒𝑖𝑡superscript𝐸𝑡:𝑖e\_{i}^{h}=E^{h}\left[:,i\right],\ e\_{i}^{t}=E^{t}\left[:,i\right], |  |

where E∈{Eh,Et}𝐸superscript𝐸ℎsuperscript𝐸𝑡E\in\{E^{h},E^{t}\} is learnable column embeddings categorized into the head view or tail view, E=(e1,e2,…,eN)∈ℝd×N𝐸subscript𝑒1subscript𝑒2…subscript𝑒𝑁superscriptℝ𝑑𝑁E=(e\_{1},e\_{2},\dots,e\_{N})\in\mathds{R}^{d\times N}, and d𝑑d is the embedding dimension. Similarly, the relation topology score gtsubscript𝑔𝑡g\_{t} has the symmetric and asymmetric counterparts, and Ehsuperscript𝐸ℎE^{h} and Etsuperscript𝐸𝑡E^{t} share parameters in the symmetric relation topology (i.e., Gt​[i,j]≡Gt​[j,i]subscript𝐺𝑡𝑖𝑗subscript𝐺𝑡𝑗𝑖G\_{t}[i,j]\equiv G\_{t}[j,i]) but are parameter-independent in the asymmetric case (i.e., Gt​[i,j]≠Gt​[j,i]subscript𝐺𝑡𝑖𝑗subscript𝐺𝑡𝑗𝑖G\_{t}[i,j]\neq G\_{t}[j,i]). We use L2subscript𝐿2L\_{2}
normalization in the gtsubscript𝑔𝑡g\_{t} score function to transform embeddings to be on a similar scale and improve the training stability.

We generate static relation topology based on the Gtsubscript𝐺𝑡G\_{t} scores in Eq. ([4](#S3.E4 "In Static Knowledge Topology. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")), as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | A=ft​o​p​(Gt)=𝟙​[σ1​(Gt+b)>T],𝐴subscript𝑓𝑡𝑜𝑝subscript𝐺𝑡1delimited-[]subscript𝜎1subscript𝐺𝑡𝑏𝑇A=f\_{top}(G\_{t})=\mathds{1}\left[\sigma\_{1}(G\_{t}+b)>T\right], |  | (5) |

where σ1subscript𝜎1\sigma\_{1} is an element-wise activation parameterised by a learnable bias b𝑏b (like the operation in PReLU (He, Zhang et al. [2015](#bib.bib21))), Gtsubscript𝐺𝑡G\_{t} is adjacency matrix scores composed of the relation topology score gtsubscript𝑔𝑡g\_{t}, T𝑇T is a constant threshold for signal clipping, and 𝟙1\mathds{1} denotes the indicator function. In this way, we obtain a global graph topology (an adjacency matrix A𝐴A) to constrain feature interactions, and this topology can be regarded as static knowledge on the whole task.

### 3.2 Relation Graph Assembling

As we obtain “soft” adaptive edge weights from the data view and “hard” static relation graph topology from the knowledge view, we combine them to generate an FR-Graph, following the idea of “decision on both specific data and underlying knowledge”. Specifically, we assemble the two components as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | G=σ2​(fnsi​(A)⊙Gw),𝐺subscript𝜎2direct-productsubscript𝑓nsi𝐴subscript𝐺𝑤G=\sigma\_{2}(f\_{\text{nsi}}(A)\odot G\_{w}), |  | (6) |

where σ2subscript𝜎2\sigma\_{2} is a competitive activation (e.g., Lpsubscript𝐿𝑝L\_{p} normalization, softmax, entmax, sparsemax (Martins and Astudillo [2016](#bib.bib36))) to restrict the indegree of each “feature node”, and ⊙direct-product\odot denotes the Hadamard product. The resulted relation graph G𝐺G is a weighted graph based on both adaptive feature matching and static knowledge topology. To help the FR-Graph focus on learning meaningful interactions between different features, a “no-self-interaction” function fnsisubscript𝑓nsif\_{\text{nsi}} is performed to explicitly exclude self-loops in G𝐺G. We use the FR-Graph to instruct subsequent feature interactions. Since both the edge weights and knowledge topology have the symmetric and asymmetric versions, there are four combinations of FR-Graph covering the complete relation graph. In experiments, we will further discuss the impact of the FR-Graph type.

## 4 T2G-Former

We incorporate GE into the attention-like basic block, and build T2G-Former by stacking multiple blocks for selective tabular feature interaction (see Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")). T2G-Former uses estimated FR-Graphs to interact features and attain higher-level features layer by layer. A Cross-level Readout is sequentially transformed to the feature space of each layer, and selectively collects salient features for the final prediction. A shortcut path is added to preserve the information from the preceding layers, resulted in gated fusion in different feature levels that promotes the model capability.

### 4.1 Basic Block

A single block is built equipped with GE for selective feature interaction (see Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(b)). Given input features Xl∈ℝn×Nsuperscript𝑋𝑙superscriptℝ𝑛𝑁X^{l}\in\mathds{R}^{n\times N} to the l𝑙l-th layer, we obtain higher-level features Xl+1superscript𝑋𝑙1X^{l+1} as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Gl=GE​(Xl),Vl=Wv​Xl,formulae-sequencesuperscript𝐺𝑙GEsuperscript𝑋𝑙superscript𝑉𝑙subscript𝑊𝑣superscript𝑋𝑙\displaystyle G^{l}=\textit{GE}(X^{l}),\ \ V^{l}=W\_{v}X^{l}, |  | (7) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Hl=Gl​Vl+g​(Xl),Xl+1=FFN​(Hl)+g​(Hl),formulae-sequencesuperscript𝐻𝑙superscript𝐺𝑙superscript𝑉𝑙𝑔superscript𝑋𝑙superscript𝑋𝑙1FFNsuperscript𝐻𝑙𝑔superscript𝐻𝑙\displaystyle H^{l}=G^{l}V^{l}+g(X^{l}),\ \ X^{l+1}=\text{FFN}(H^{l})+g(H^{l}), |  | (8) |

where Wv∈ℝm×nsubscript𝑊𝑣superscriptℝ𝑚𝑛W\_{v}\in\mathds{R}^{m\times n} is learnable parameters for feature transformation, and Vlsuperscript𝑉𝑙V^{l} is transformed input features. FFN denotes a feed-forward network. As self-interaction is excluded in Glsuperscript𝐺𝑙G^{l} (see Eq. ([6](#S3.E6 "In 3.2 Relation Graph Assembling ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"))), a shortcut path g𝑔g is added to protect the information from the preceding layers, which is a simple dropout layer in experiments. Notably, we yield and use the FR-Graph for feature interactions, and does not influence the intra-feature update conducted by the shortcut. In the first layer, we set X0superscript𝑋0X^{0} as the input tabular data encoded by a simple feature tokenizer (Gorishniy, Rubachev et al. [2021](#bib.bib16)). In this way, higher-level features can be iteratively obtained with the generated FR-Graphs and selective interaction. In implementation, layer normalization is performed (see Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(b)) for stable training.

### 4.2 Cross-level Readout

We design a global readout node to selectively collect salient features from each layer and attain comprehensive semantics for the final prediction. Specifically, we attentively fuse selected features at the current layer and combine them with the lower-level features from the preceding layers by a shortcut path. Given the current readout status zl∈ℝnsuperscript𝑧𝑙superscriptℝ𝑛z^{l}\in\mathds{R}^{n}, the collection process at the l𝑙l-th layer is defined by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | αil=gw​(hl,fit)⋅ft​o​p​(gt​(el,eit)),hl=Wh​zl,formulae-sequencesuperscriptsubscript𝛼𝑖𝑙⋅subscript𝑔𝑤superscriptℎ𝑙superscriptsubscript𝑓𝑖𝑡subscript𝑓𝑡𝑜𝑝subscript𝑔𝑡superscript𝑒𝑙superscriptsubscript𝑒𝑖𝑡superscriptℎ𝑙superscript𝑊ℎsuperscript𝑧𝑙\displaystyle\alpha\_{i}^{l}=g\_{w}(h^{l},f\_{i}^{t})\cdot f\_{top}(g\_{t}(e^{l},e\_{i}^{t})),\ \ h^{l}=W^{h}z^{l}, |  | (9) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | rl=softmax​(𝜶l)T​Vl+zl,superscript𝑟𝑙softmaxsuperscriptsuperscript𝜶𝑙𝑇superscript𝑉𝑙superscript𝑧𝑙\displaystyle r^{l}={\rm softmax}(\boldsymbol{\alpha}^{l})^{T}V^{l}+z^{l}, |  | (10) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | zl+1=FFN​(rl)+rl,superscript𝑧𝑙1FFNsuperscript𝑟𝑙superscript𝑟𝑙\displaystyle z^{l+1}=\text{FFN}(r^{l})+r^{l}, |  | (11) |

where αilsuperscriptsubscript𝛼𝑖𝑙\alpha\_{i}^{l} denotes the weight of the i𝑖i-th feature that constitutes a weight vector 𝜶l∈ℝNsuperscript𝜶𝑙superscriptℝ𝑁\boldsymbol{\alpha}^{l}\in\mathds{R}^{N}, el∈ℝdsuperscript𝑒𝑙superscriptℝ𝑑e^{l}\in\mathds{R}^{d} is a learnable vector representing the semantics of the readout node at the l𝑙l-th layer, fitsuperscriptsubscript𝑓𝑖𝑡f\_{i}^{t} is an encoded feature (Eq. ([3](#S3.E3 "In Adaptive Edge Weights. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"))) of each layer, and eitsuperscriptsubscript𝑒𝑖𝑡e\_{i}^{t} is a layer-wise column embedding (Eq. ([4](#S3.E4 "In Static Knowledge Topology. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"))). Vlsuperscript𝑉𝑙V^{l} is the transformed input features (Eq. ([7](#S4.E7 "In 4.1 Basic Block ‣ 4 T2G-Former ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"))). Here we put zlsuperscript𝑧𝑙z^{l} forward through the same FFN transformation to transform the current readout into the feature space at the (l+1𝑙1l+1)-th layer for the next round of collection. The shortcut paths are directly added without information drop. This collection process is repeated from the input features to the highest-level features, thus encouraging interactions among cross-level features.

### 4.3 The Overall Architecture and Training

Basic blocks are stacked in T2G-Former (Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(a)). If without special specification, in experiments we use 8-head GE in each block by default (Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(b)). Prediction is made based on the readout status after processing the final layer L𝐿L, as:

|  |  |  |
| --- | --- | --- |
|  | y^=FC​(ReLU​(LN​(zL))),^𝑦FCReLULNsuperscript𝑧𝐿\hat{y}={\rm FC}({\rm ReLU}({\rm LN}(z^{L}))), |  |

where LN and FC denote layer normalization and a fully connected layer, respectively. As for optimization, we use the cross entropy loss for classification and the mean squared error loss for regression, as in previous DNNs. We tested various tasks and observed that continuing to optimize the static graph topology A𝐴A in Eq. ([5](#S3.E5 "In Static Knowledge Topology. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")) across the whole training phase may lead to unstable performance on some easy tasks (e.g., binary classification, small datasets, or few input features). Thus, we freeze it after convergence for further training in a fixed topology manner.

Note that we introduce additional hyperparameters d𝑑d (Eq. ([4](#S3.E4 "In Static Knowledge Topology. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"))) and T𝑇T (Eq. ([5](#S3.E5 "In Static Knowledge Topology. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"))). In experiments, we adaptively set d=2​⌈log2⁡N⌉𝑑2subscript2𝑁d=2\left\lceil\log\_{2}N\right\rceil which is for the minimal amount of information to present an adjacency matrix with N2superscript𝑁2N^{2} binary elements, and keep T=0.5𝑇0.5T=0.5 across all the datasets. We choose sigmoid as σ1subscript𝜎1\sigma\_{1} and softmax as σ2subscript𝜎2\sigma\_{2}. Straight-through trick (Bengio, Léonard, and Courville [2013](#bib.bib5)) is used to solve the undifferentiable issue of the indicator function in Eq. ([5](#S3.E5 "In Static Knowledge Topology. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")).

## 5 Experiments

In this section, we present extensive experimental results and compare with a wide range of state-of-the-art tabular learning DNNs and GBDT. Also, we conduct empirical experiments to examine the impacts of some key T2G-Former components, including comparison of the feature relation graph (FR-Graph) types, ablation study of self-interaction, and the effect of GE. Besides, we explore the model interpretability by visualizing the FR-Graphs and readout selection on two semantically rich datasets.

### 5.1 Experimental Setup

#### Datasets.

We use twelve open-source tabular datasets. Gesture Phase Prediction (GE, (Madeo, Lima, and Peres [2013](#bib.bib35))), Churn Modeling (CH, Kaggle dataset), Eye Movements (EY, (Salojärvi, Puolamäki et al. [2005](#bib.bib42))), California Housing (CA, (Pace and Barry [1997](#bib.bib39))), House 16H (HO, OpenML dataset), Adult (AD, (Kohavi et al. [1996](#bib.bib29))), Helena (HE, (Guyon, Sun-Hosoya et al. [2019](#bib.bib18))), Jannis (JA, (Guyon, Sun-Hosoya et al. [2019](#bib.bib18))), Otto Group Product Classification (OT, Kaggle dataset), Higgs Small (HI, (Baldi, Sadowski, and Whiteson [2014](#bib.bib4))), Facebook Comments (FB, (Singh, Sandhu, and Kumar [2015](#bib.bib44))), and Year (YE, (Bertin-Mahieux, Ellis
et al. [2011](#bib.bib6))). For each dataset, data preprocessing and train-validation-test splits are fixed for all the methods according to (Gorishniy, Rubachev et al. [2021](#bib.bib16); Gorishniy, Rubachev, and Babenko [2022](#bib.bib15)). Dataset statistics are given in Table [1](#S5.T1 "Table 1 ‣ Datasets. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"), and more details are in Appendix [A](#A1 "Appendix A Dataset Descriptions ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction").

| Dataset | GE | CH | EY | CA | HO | AD | OT | HE | JA | HI | FB | YE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| # features | 32 | 9+1 | 26 | 8 | 16 | 6+8 | 93 | 27 | 54 | 28 | 50+1 | 90 |
| # samples | 9873 | 10000 | 10936 | 20640 | 22784 | 48842 | 61878 | 65196 | 83733 | 98050 | 197080 | 515345 |
| # classes | 5 | 2 | 3 | - | - | 2 | 9 | 100 | 4 | 2 | - | - |
| Metric | Acc. | Acc. | Acc. | RMSE | RMSE | Acc. | Acc. | Acc. | Acc. | Acc. | RMSE | RMSE |

Table 1: Some details of the 12 public datasets. ”RMSE” denotes root mean squared error (for regression), and “Acc.” means accuracy (for classification). The number following each “+” in the row of “# features” is the number of categorical features.



|  | GE ↑↑\uparrow | CH ↑↑\uparrow | EY ↑↑\uparrow | CA ↓↓\downarrow | HO ↓↓\downarrow | AD ↑↑\uparrow | OT ↑↑\uparrow | HE ↑↑\uparrow | JA ↑↑\uparrow | HI ↑↑\uparrow | FB ↓↓\downarrow | YE ↓↓\downarrow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 68.42 | 85.92 | 72.51 | 0.436 | 3.169 | 87.30 | 82.46 | 37.47 | 71.85 | 72.41 | 5.359 | 8.850 |
| MLP | 58.64 | 85.77 | 61.10 | 0.499 | 3.173 | 85.35 | 80.99 | 38.38 | 71.97 | 72.00 | 5.943 | 8.849 |
| SNN | 64.69 | 85.74 | 61.55 | 0.498 | 3.207 | 85.40 | 81.17 | 37.19 | 71.94 | 72.21 | 5.892 | 8.901 |
| TabNet | 60.01 | 85.01 | 62.08 | 0.513 | 3.252 | 84.84 | 79.06 | 37.86 | 72.26 | 71.97 | 6.559 | 8.916 |
| DANet-28 | 61.63 | 85.10 | 60.53 | 0.524 | 3.236 | 85.00 | 81.04 | 35.45 | 70.72 | 71.47 | 6.167 | 8.914 |
| NODE | 53.94 | 85.86 | 65.54 | 0.463 | 3.216 | 85.77 | 80.37 | 35.33 | 72.78 | 72.51 | 5.698 | 8.777 |
| AutoInt | 58.33 | 85.51 | 61.07 | 0.472 | 3.147 | 85.66 | 80.11 | 37.26 | 72.08 | 72.51 | 5.852 | 8.862 |
| DCNv2 | 55.72 | 85.68 | 61.37 | 0.489 | 3.172 | 85.48 | 80.15 | 38.61 | 71.56 | 72.20 | 5.847 | 8.882 |
| FT-Transformer | 61.25 | 86.07 | 70.84 | 0.460 | 3.124 | 85.72 | 81.30 | 39.10 | 73.24 | 73.06 | 6.079 | 8.852 |
| T2G-Former | 65.57 | 86.25 | 78.18 | 0.455 | 3.138 | 85.96 | 81.87 | 39.06 | 73.68 | 73.39 | 5.701 | 8.851 |

Table 2: Performance comparison on the 12 public tubular datasets. Each result reported is averaged over 15 random seeds. For standard deviations, see Appendix [C](#A3 "Appendix C Detailed Experiment Results ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"). For each dataset, the top performances among the DNNs are marked in bold, and the second best results are underlined. We also report XGBoost results as a typical representation of GBDT models. ↓↓\downarrow represents the RMSE metric (the lower the better) and ↑↑\uparrow represents accuracy (the higher the better).

#### Implementation Details.

We implement our T2G-Former model using PyTorch on Python 3.8. All the experiments are run on NVIDIA RTX 3090. In training, if without special specification, we use FR-Graphs with symmetric edge weights and asymmetric graph topology in GE. The optimizer is AdamW (Loshchilov and Hutter [2018](#bib.bib32)) with the default configuration except for the learning rate and weight decay rate. For DANet-28, we follow its QHAdam optimizer (Ma and Yarats [2018](#bib.bib34)) and the pre-set hyperparameters given in (Chen, Liao et al. [2022](#bib.bib8)) without tuning. For the other DNNs and XGBoost, we follow the settings provided in (Gorishniy, Rubachev et al. [2021](#bib.bib16)) (including the optimizers and hyperparameter spaces), and perform hyperparameter tuning with the Optuna library (Akiba, Sano et al. [2019](#bib.bib1)) and grid search (only for NODE). More detailed information of hyperparameters is provided in Appendix [B](#A2 "Appendix B Hyperparameter Tuning ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction").

#### Comparison Methods.

In our
experiments, we compare our T2G-Former with the representative non-deep method XGBoost (Chen and Guestrin [2016](#bib.bib9)) and the known DNNs, including NODE (Popov, Morozov, and Babenko [2019](#bib.bib40)), AutoInt (Song, Shi et al. [2019](#bib.bib45)), TabNet (Arik and Pfister [2021](#bib.bib3)), DCNv2 (Wang, Shivanna et al. [2021](#bib.bib50)), FT-Transformer (Gorishniy, Rubachev et al. [2021](#bib.bib16)), and DANets (Chen, Liao et al. [2022](#bib.bib8)). Some other common DNNs such as MLP and SNN (an MLP network with SELU activation) (Klambauer et al. [2017](#bib.bib28)) are taken into comparison as well.

### 5.2 Main Results and Analyses

#### Performance Comparison.

The performances of the DNNs and non-deep models are reported in Table [2](#S5.T2 "Table 2 ‣ Datasets. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"). T2G-Former outperforms these DNNs on eight datasets, and is comparable with XGBoost in most the cases. All the models are hyperparameter-tuned by choosing the best validation results with Optuna-driven tuning (Akiba, Sano et al. [2019](#bib.bib1)).

#### The Effect of FR-Graph Types.

We compare four types of FR-Graphs in GE. Table [3](#S5.T3 "Table 3 ‣ The Effect of FR-Graph Types. ‣ 5.2 Main Results and Analyses ‣ 5 Experiments ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction") reports the results, from which one can see that it is often better to choose symmetric edge weights and asymmetric knowledge topology. This suggests that mutual interactions between two tabular features are likely to be the same, and asymmetric topology offers a larger semantic exploration space that is more likely to yield useful features. The results on the other datasets are provided in Appendix [C](#A3 "Appendix C Detailed Experiment Results ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction").

| FR-Graph | EY ↑↑\uparrow | CA (×\times100) ↓↓\downarrow | HO ↓↓\downarrow | OT ↑↑\uparrow | FB ↓↓\downarrow | YE ↓↓\downarrow |
| --- | --- | --- | --- | --- | --- | --- |
| Aw​StsuperscriptAwsuperscriptSt{\rm A^{w}S^{t}} | 77.34 | 45.73 | 3.171 | 81.80 | 5.736 | 8.886 |
| Aw​AtsuperscriptAwsuperscriptAt{\rm A^{w}A^{t}} | 77.59 | 45.77 | 3.145 | 81.85 | 5.718 | 8.861 |
| Sw​StsuperscriptSwsuperscriptSt{\rm S^{w}S^{t}} | 76.46 | 45.61 | 3.151 | 81.89 | 5.723 | 8.885 |
| Sw​AtsuperscriptSwsuperscriptAt{\rm S^{w}A^{t}} (ours) | 78.18 | 45.53 | 3.138 | 81.87 | 5.701 | 8.851 |

Table 3: Comparison of four FR-Graph types on several tasks and datasets. “A” means asymmetric, and “S” means symmetric. “Aw​StsuperscriptAwsuperscriptSt{\rm A^{w}S^{t}}”, for example, is for asymmetric edge weights and symmetric graph topology. Likewise, “Aw​AtsuperscriptAwsuperscriptAt{\rm A^{w}A^{t}}”, “Sw​StsuperscriptSwsuperscriptSt{\rm S^{w}S^{t}}”, and “Sw​AtsuperscriptSwsuperscriptAt{\rm S^{w}A^{t}}” denote the other three types of FR-Graphs. The performances on the CA dataset are scaled (×\times100) in these several tables for more clear comparison.

#### The Effect of Self-interaction.

One of our key designs in GE is the “no self-interaction function” that explicitly excludes self-loops in FR-Graphs. Table [4](#S5.T4 "Table 4 ‣ The Effect of Self-interaction. ‣ 5.2 Main Results and Analyses ‣ 5 Experiments ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction") reports comparison results on several datasets with no self-loop FR-Graphs (ours) and self-loop FR-Graphs. The results show that in most the cases, removing self-loops and focusing on interactions with other features slightly benefit performances in both classification and regression. This may be because feature self-interaction affects the probabilities of interactions with other features (as we use competitive activation in Eq. ([6](#S3.E6 "In 3.2 Relation Graph Assembling ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"))), while our shortcut paths have already preserved self-information.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | EY ↑↑\uparrow | CA (×\times100) ↓↓\downarrow | HO ↓↓\downarrow | OT ↑↑\uparrow | FB ↓↓\downarrow | YE ↓↓\downarrow |
| w/o SL (ours) | 78.18 | 45.53 | 3.138 | 81.87 | 5.701 | 8.851 |
| SL | 77.89 | 45.62 | 3.152 | 81.81 | 5.691 | 8.856 |
| SL −- w/o SL | −0.290.29-0.29 | 0.09 | 0.014 | −0.060.06-0.06 | −0.010.01-0.01 | 0.005 |

Table 4: Comparison of the effects of FR-Graphs without (w/o) self-loops and FR-Graphs with self-loops. “SL” means self-loops.

#### The Effect of GEs.

We explore the impact of including GEs at different layers of T2G-Former. Table [5](#S5.T5 "Table 5 ‣ The Effect of GEs. ‣ 5.2 Main Results and Analyses ‣ 5 Experiments ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction") reports the performances of different model versions which differ solely in the positions and numbers of GEs used. As for the layers without GE, we use the ordinary attention score for substitution. Overall, the positions of GEs show bigger influence on regression tasks than on classification tasks. As one can see, in regression tasks, the model incurs larger performance drops when GEs are equipped in higher layers, while the drops do not seem so large related to GE positions in classification. Also, the model equipped with only attention score is better than the one with a single GE in a high layer (not in the first layer) for regression tasks, but is always sub-optimal in classification tasks.
A probable explanation is that regression needs a smoother optimization space than classification, and thus the fully connected attention score provides the kind of interactions to cope with continuous feature values, while a single GE in a high layer is difficult to capture clear relations among features fused in the fully connected manner. Therefore, it is better to completely use attention score than a single GE in a high layer for regression. A single GE in the first layer shows the least performance drop in both regression and classification, which can be explained by the strength of GE in capturing underlying relations among tabular features with clear semantics.

In summary, the removal of GE in any layers is likely to cause performance drop, and the best results are achieved by applying GE to all the layers.

|  | CA (×\times100) ↓↓\downarrow | JA ↑↑\uparrow |
| --- | --- | --- |
| All | 45.53 | 73.68 |
| # 1 | 45.78 (+0.250.25+0.25) | 73.40 (−0.280.28-0.28) |
| # 2 | 45.96 (+0.430.43+0.43) | 73.31 (−0.370.37-0.37) |
| # 3 | 46.06 (+0.530.53+0.53) | 73.37 (−0.310.31-0.31) |
| None | 45.84 (+0.310.31+0.31) | 73.23 (−0.450.45-0.45) |

Table 5: Performances of including GEs in different layers of T2G-Former. All the results are obtained with a 3-layer T2G-Former. “# i𝑖i” means that only the i𝑖i-th layer has GE while the other layers replace GE with the ordinary attention score, “All” means that all the layers are equipped with GE, and “None” means that all the layers use ordinary attention.

#### Comparison of Topology Learning Approaches.

Apart from the column embedding approach proposed in Sec. [3.1](#S3.SS1.SSSx2 "Static Knowledge Topology. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"), there are some other intuitive straightforward approaches to get knowledge topology of the RF-Graph, for example, performing threshold clipping on the adaptive edge weights directly (we call it “adaptive topology”) or learning an N𝑁N-by-N𝑁N adjacency matrix (we call it “free topology”). Concretely, for learning adaptive topology, we substitute Gtsubscript𝐺𝑡G\_{t} in Eq. ([5](#S3.E5 "In Static Knowledge Topology. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")) with Gwsubscript𝐺𝑤G\_{w} in Eq. ([2](#S3.E2 "In Adaptive Edge Weights. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")). For learning the free topology, we directly represent Gtsubscript𝐺𝑡G\_{t} by an N𝑁N-by-N𝑁N matrix. Table [6](#S5.T6 "Table 6 ‣ Comparison of Topology Learning Approaches. ‣ 5.2 Main Results and Analyses ‣ 5 Experiments ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction") reports the comparison results of these topology learning strategies. One can see that, the static knowledge topology shared on the whole dataset (our approach) attains superior performances than the adaptive topology, implying the plausibility of our underlying knowledge assumption mentioned in Sec. [1](#S1 "1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"). Besides, the completely free topology also achieves inferior performances, which is probably because of the excessive freedom given to the learnable matrix.

| Topology | CA (×\times100) ↓↓\downarrow | JA ↑↑\uparrow | Complexity | Fixed |
| --- | --- | --- | --- | --- |
| ours | 45.53 | 73.68 | O(N𝑁NlogN𝑁N) | Y |
| adaptive | 45.88 (+0.350.35+0.35) | 73.08 (−0.600.60-0.60) | O(1) | N |
| free | 45.87 (+0.340.34+0.34) | 73.46 (−0.220.22-0.22) | O(N2superscript𝑁2N^{2}) | Y |

Table 6: Performances of different topology learning approaches. “Complexity” indicates the additional space computational complexity (the amount of extra model parameters) caused by the number of tabular features N𝑁N. “Fixed” indicates whether the learned topology is data-adaptive (N) or static (Y).

#### Comparison with DANet Grouped Interactions.

As illustrated in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"), DANets (Chen, Liao et al. [2022](#bib.bib8)) interacted tabular features in the group determined by the “entmax” operation. Here we compare our graph-based interaction with that group-based one to inspect the benefits of FR-Graph. Specifically, we substitute the knowledge topology A𝐴A in Eq. ([5](#S3.E5 "In Static Knowledge Topology. ‣ 3.1 FR-Graph Structure Components ‣ 3 Graph Estimator ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")) with DANet grouped selection mask.
The results in Table [7](#S5.T7 "Table 7 ‣ Comparison with DANet Grouped Interactions. ‣ 5.2 Main Results and Analyses ‣ 5 Experiments ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction") suggest that it of greater benefits to organize tabular features into a graph, since a graph topology is able to capture relation edges and provide more subtle interactions than a group structure.

| Interaction | CA (×\times100) ↓↓\downarrow | HO ↓↓\downarrow | JA ↑↑\uparrow |
| --- | --- | --- | --- |
| graph (ours) | 45.53 | 3.138 | 73.68 |
| group (DANet) | 45.88 (+0.350.35+0.35) | 3.215 (+0.0770.077+0.077) | 73.08 (−0.600.60-0.60) |

Table 7: Comparison with DANet group-based interaction on several datasets.

![Refer to caption](/html/2211.16887/assets/x3.png)


Figure 3: Visualization of the FR-Graph in the first layer (heat map) and the readout selection (dark bar) on the datasets CA (left) and CH (right). More details of the feature descriptions are given in Appendix [D](#A4 "Appendix D Dataset Features ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction").

### 5.3 Interpretability

In Fig. [3](#S5.F3 "Figure 3 ‣ Comparison with DANet Grouped Interactions. ‣ 5.2 Main Results and Analyses ‣ 5 Experiments ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"), we visualize the first-layer FR-Graph and the readout collecting strategy on the input features (i.e., features from the feature tokenizer; see Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction")(a)). On the CA dataset, it is reasonable to find that the median income (MedInc, MI) of the residents within a block group is related to the average number of the household members (AveOccup, AvOcc), and AveOccup can affect the average number of bedrooms (AveBedrooms, AvB). Also, there appear to be some relations such as Longitude (Long)-HouseAge (HoA), Longitude-AveRooms (AvR), and Longitude-Population (Pop), which are probably derived from dataset bias. As for readout, one can see that solely HouseAge is collected that is a meaningful feature in house price prediction. On the CH dataset, there are reasonable relations between Balance (Bal, bank balance of a customer) and EstimatedSalary (EstSal), as well as the age of the customer (Age) and EstimatedSalary. Also, it is interpretable that the credit score of a customer (CreditScore, CrSc) is highly related to that customer’s Age and Balance. The readout collects only Age in the current level for predicting whether a customer will leave the bank, which is intuitive as well.

## 6 Conclusions

In this paper, we proposed T2G-Former, a new bespoke Transformer model for tabular learning with a novel module Graph Estimator (GE) for promoting heterogeneous feature interaction based on estimated relation graphs. We adapted feature relation graphs into the basic blocks of T2G-Former in an attention-like fashion for simplicity and applicability. Experiments on extensive public datasets showed that T2G-Former achieves better performances than various DNNs and is comparable with XGboost. We expect that our T2G-Former will serve as a strong baseline in tabular learning studies and enhance research interest in handling feature heterogeneity of tabular data.

## Acknowledgments

This research was partially supported by the National Key R&D Program of China under grant No. 2018AAA0102102 and National Natural Science Foundation of China under grants No. 62132017.

## References

* Akiba, Sano et al. (2019)

  Akiba, T.; Sano, S.; et al. 2019.
  Optuna: A next-generation hyperparameter optimization framework.
  In *KDD*.
* Anghel, Papandreou et al. (2018)

  Anghel, A.; Papandreou, N.; et al. 2018.
  Benchmarking and optimization of gradient boosting decision tree
  algorithms.
  In *NeurIPS*.
* Arik and Pfister (2021)

  Arik, S. Ö.; and Pfister, T. 2021.
  TabNet: Attentive interpretable tabular learning.
  In *AAAI*.
* Baldi, Sadowski, and Whiteson (2014)

  Baldi, P.; Sadowski, P.; and Whiteson, D. 2014.
  Searching for exotic particles in high-energy physics with deep
  learning.
  *Nature Communications*.
* Bengio, Léonard, and Courville (2013)

  Bengio, Y.; Léonard, N.; and Courville, A. 2013.
  Estimating or propagating gradients through stochastic neurons for
  conditional computation.
  *arXiv preprint arXiv:1308.3432*.
* Bertin-Mahieux, Ellis
  et al. (2011)

  Bertin-Mahieux, T.; Ellis, D. P. W.; et al. 2011.
  The million song dataset.
  In *Proceedings of the International Society for Music
  Information Retrieval Conference*.
* Borisov, Leemann et al. (2021)

  Borisov, V.; Leemann, T.; et al. 2021.
  Deep neural networks and tabular data: A survey.
  *arXiv preprint arXiv:2110.01889*.
* Chen, Liao et al. (2022)

  Chen, J.; Liao, K.; et al. 2022.
  DANets: Deep abstract networks for tabular data classification and
  regression.
  In *AAAI*.
* Chen and Guestrin (2016)

  Chen, T.; and Guestrin, C. 2016.
  XGBoost: A scalable tree boosting system.
  In *KDD*.
* Covington, Adams, and Sargin (2016)

  Covington, P.; Adams, J.; and Sargin, E. 2016.
  Deep neural networks for YouTube recommendations.
  In *Proceedings of the ACM Conference on Recommender Systems*.
* Dong, Cheng et al. (2022)

  Dong, H.; Cheng, Z.; et al. 2022.
  Table Pre-training: A survey on model architectures, pre-training
  objectives, and downstream tasks.
  In *IJCAI*.
* Dosovitskiy, Beyer et al. (2020)

  Dosovitskiy, A.; Beyer, L.; et al. 2020.
  An image is worth 16x16 words: Transformers for image recognition at
  scale.
  In *ICLR*.
* Feng, Yu, and Zhou (2018)

  Feng, J.; Yu, Y.; and Zhou, Z.-H. 2018.
  Multi-layered gradient boosting decision trees.
  In *NeurIPS*.
* Friedman (2001)

  Friedman, J. H. 2001.
  Greedy function approximation: A gradient boosting machine.
  *Annals of Statistics*.
* Gorishniy, Rubachev, and Babenko (2022)

  Gorishniy, Y.; Rubachev, I.; and Babenko, A. 2022.
  On embeddings for numerical features in tabular deep Learning.
  *arXiv preprint arXiv:2203.05556*.
* Gorishniy, Rubachev et al. (2021)

  Gorishniy, Y.; Rubachev, I.; et al. 2021.
  Revisiting deep learning models for tabular data.
  In *NeurIPS*.
* Guo, Tang et al. (2017)

  Guo, H.; Tang, R.; et al. 2017.
  DeepFM: A factorization-machine based neural network for CTR
  prediction.
  In *IJCAI*.
* Guyon, Sun-Hosoya et al. (2019)

  Guyon, I.; Sun-Hosoya, L.; et al. 2019.
  Analysis of the AutoML Challenge Series.
  *Automated Machine Learning*.
* Hassan, Al-Insaif et al. (2020)

  Hassan, M. R.; Al-Insaif, S.; et al. 2020.
  A machine learning approach for prediction of pregnancy outcome
  following IVF treatment.
  *Neural Computing and Applications*.
* Hazimeh, Ponomareva et al. (2020)

  Hazimeh, H.; Ponomareva, N.; et al. 2020.
  The tree ensemble layer: Differentiability meets conditional
  computation.
  In *ICML*.
* He, Zhang et al. (2015)

  He, K.; Zhang, X.; et al. 2015.
  Delving deep into rectifiers: Surpassing human-level performance on
  ImageNet classification.
  In *ICCV*.
* He, Pan et al. (2014)

  He, X.; Pan, J.; et al. 2014.
  Practical lessons from predicting clicks on ads at Facebook.
  In *Proceedings of the International Workshop on Data Mining for
  Online Advertising*.
* Huang, Khetan et al. (2020)

  Huang, X.; Khetan, A.; et al. 2020.
  TabTransformer: Tabular data modeling using contextual embeddings.
  *arXiv preprint arXiv:2012.06678*.
* Johnson, Pollard et al. (2016)

  Johnson, A. E.; Pollard, T. J.; et al. 2016.
  MIMIC-III, a freely accessible critical care database.
  *Scientific Data*.
* Katzir, Elidan, and El-Yaniv (2020)

  Katzir, L.; Elidan, G.; and El-Yaniv, R. 2020.
  Net-DNF: Effective deep modeling of tabular data.
  In *ICLR*.
* Ke, Meng et al. (2017)

  Ke, G.; Meng, Q.; et al. 2017.
  LightGBM: A highly efficient gradient boosting decision tree.
  In *NeurIPS*.
* Kenton and Toutanova (2019)

  Kenton, J. D. M.-W. C.; and Toutanova, L. K. 2019.
  BERT: Pre-training of deep bidirectional transformers for language
  understanding.
  In *NAACL-HLT*.
* Klambauer et al. (2017)

  Klambauer, G.; Unterthiner, T.; Mayr, A.; and Hochreiter, S. 2017.
  Self-normalizing neural networks.
  In *NeurIPS*.
* Kohavi et al. (1996)

  Kohavi, R.; et al. 1996.
  Scaling up the accuracy of Naive-Bayes classifiers: A decision-tree
  hybrid.
  In *KDD*.
* Kontschieder, Fiterau et al. (2015)

  Kontschieder, P.; Fiterau, M.; et al. 2015.
  Deep neural decision forests.
  In *ICCV*.
* Li et al. (1984)

  Li, B.; Friedman, J.; Olshen, R.; and Stone, C. 1984.
  Classification and regression trees (CART).
  *Biometrics*.
* Loshchilov and Hutter (2018)

  Loshchilov, I.; and Hutter, F. 2018.
  Decoupled weight decay regularization.
  In *ICLR*.
* Lou and Obukhov (2017)

  Lou, Y.; and Obukhov, M. 2017.
  BDT: Gradient boosted decision tables for high accuracy and scoring
  efficiency.
  In *KDD*.
* Ma and Yarats (2018)

  Ma, J.; and Yarats, D. 2018.
  Quasi-hyperbolic momentum and Adam for deep learning.
  In *ICLR*.
* Madeo, Lima, and Peres (2013)

  Madeo, R. C.; Lima, C. A.; and Peres, S. M. 2013.
  Gesture unit segmentation using support vector machines: Segmenting
  gestures from rest positions.
  In *Proceedings of the Annual ACM Symposium on Applied
  Computing*.
* Martins and Astudillo (2016)

  Martins, A.; and Astudillo, R. 2016.
  From softmax to sparsemax: A sparse model of attention and
  multi-label classification.
  In *ICML*.
* Nickel, Rosasco, and Poggio (2016)

  Nickel, M.; Rosasco, L.; and Poggio, T. 2016.
  Holographic embeddings of knowledge graphs.
  In *AAAI*.
* Nickel, Tresp, and Kriegel (2011)

  Nickel, M.; Tresp, V.; and Kriegel, H.-P. 2011.
  A three-way model for collective learning on multi-relational data.
  In *ICML*.
* Pace and Barry (1997)

  Pace, R. K.; and Barry, R. 1997.
  Sparse spatial autoregressions.
  *Statistics & Probability Letters*.
* Popov, Morozov, and Babenko (2019)

  Popov, S.; Morozov, S.; and Babenko, A. 2019.
  Neural oblivious decision ensembles for deep learning on tabular
  data.
  In *ICLR*.
* Prokhorenkova, Gusev et al. (2018)

  Prokhorenkova, L.; Gusev, G.; et al. 2018.
  CatBoost: Unbiased boosting with categorical features.
  In *NeurIPS*.
* Salojärvi, Puolamäki et al. (2005)

  Salojärvi, J.; Puolamäki, K.; et al. 2005.
  Inferring relevance from eye movements: Feature extraction.
  In *NeurIPS-W*.
* Shi and Weninger (2018)

  Shi, B.; and Weninger, T. 2018.
  Open-world knowledge graph completion.
  In *AAAI*.
* Singh, Sandhu, and Kumar (2015)

  Singh, K.; Sandhu, R. K.; and Kumar, D. 2015.
  Comment volume prediction using neural networks and decision trees.
  In *IEEE UKSim-AMSS International Conference on Computer
  Modelling and Simulation*.
* Song, Shi et al. (2019)

  Song, W.; Shi, C.; et al. 2019.
  AutoInt: Automatic feature interaction learning via self-attentive
  neural networks.
  In *CIKM*.
* Thawani, Pujara et al. (2021)

  Thawani, A.; Pujara, J.; et al. 2021.
  Representing number in NLP: A survey and a vision.
  In *NAACL-HLT*.
* Trouillon, Welbl et al. (2016)

  Trouillon, T.; Welbl, J.; et al. 2016.
  Complex embeddings for simple link prediction.
  In *ICML*.
* Vaswani, Shazeer et al. (2017)

  Vaswani, A.; Shazeer, N.; et al. 2017.
  Attention is all you need.
  In *NeurIPS*.
* Wang, Fu et al. (2017)

  Wang, R.; Fu, B.; et al. 2017.
  Deep & cross network for ad click predictions.
  In *ADKDD*.
* Wang, Shivanna et al. (2021)

  Wang, R.; Shivanna, R.; et al. 2021.
  DCN V2: Improved deep & cross network and practical lessons for
  web-scale learning to rank systems.
  In *WWW*.
* Wu, Chen et al. (2021)

  Wu, L.; Chen, Y.; et al. 2021.
  Graph neural networks for natural language processing: A survey.
  *arXiv preprint arXiv:2106.06090*.
* Yang, Yih et al. (2015)

  Yang, B.; Yih, S. W.-t.; et al. 2015.
  Embedding entities and relations for learning and inference in
  knowledge bases.
  In *ICLR*.
* Yang, Morillo, and Hospedales (2018)

  Yang, Y.; Morillo, I. G.; and Hospedales, T. M. 2018.
  Deep neural decision trees.
  In *ICML-W*.
* Zhang and Honavar (2003)

  Zhang, J.; and Honavar, V. 2003.
  Learning from attribute value taxonomies and partially specified
  instances.
  In *ICML*.
* Zhang, Kang et al. (2006)

  Zhang, J.; Kang, D.-K.; et al. 2006.
  Learning accurate and concise Naïve Bayes classifiers from
  attribute value taxonomies and data.
  *Knowledge and Information Systems*.

## Appendix A Dataset Descriptions

Details of used datasets are shown in Table [8](#A1.T8 "Table 8 ‣ Appendix A Dataset Descriptions ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"). We follow the same train-valid-test splits and data pre-processing methods in (Gorishniy, Rubachev et al. [2021](#bib.bib16); Gorishniy, Rubachev, and Babenko [2022](#bib.bib15)).

| Dataset | # Train | # Validation | # Test | # Num | # Cat | Task | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gesture Phase(GE) | 6318 | 1580 | 1795 | 32 | 0 | Multiclass | 128 |
| Churn Modelling(CH) | 6400 | 1600 | 2000 | 9 | 1 | Binclass | 128 |
| Eye Movements(eye) | 6998 | 1750 | 2188 | 26 | 0 | Multiclass | 128 |
| California Housing(CA) | 31209 | 3303 | 4128 | 8 | 0 | Regression | 256 |
| House 16H(HO) | 14581 | 3646 | 4557 | 16 | 0 | Regression | 256 |
| Adult(AD) | 26048 | 6513 | 16281 | 6 | 8 | Binclass | 256 |
| Otto Group Products(OT) | 39601 | 9901 | 12376 | 93 | 0 | Multiclass | 512 |
| Helena(HE) | 41724 | 10432 | 13040 | 27 | 0 | Multiclass | 512 |
| Jannis(JA) | 53588 | 13398 | 16747 | 54 | 0 | Multiclass | 512 |
| Higgs Small(HI) | 62752 | 15688 | 19610 | 28 | 0 | Binclass | 512 |
| Facebook Comments Volume(FB) | 157638 | 19722 | 19720 | 50 | 1 | Regression | 512 |
| Year(YE) | 370972 | 92743 | 51630 | 90 | 0 | Regression | 1024 |

Table 8: Dataset details. “# Num” and “# Cat” mean the numbers of numerical features and categorical features in each dataset.

## Appendix B Hyperparameter Tuning

For baseline architectures of XGBoost, MLP, SNN, TabNet, NODE, AutoInt, DCNv2 and FT-Transformer, we reuse the implementations and hyperparameter search spaces in (Gorishniy, Rubachev et al. [2021](#bib.bib16)) for comparison. For DANets we use 28-layer architecture and corresponding pre-set hyperparameters as (Chen, Liao et al. [2022](#bib.bib8)) recommended without tuning. We use grid search for NODE. For our T2G-Former, we use the optuna-driven tuning (Akiba, Sano et al. [2019](#bib.bib1)) with hyperparameter spaces reported in Table [9](#A2.T9 "Table 9 ‣ Appendix B Hyperparameter Tuning ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction"). we also use the optuna-driven tuning (Akiba, Sano et al. [2019](#bib.bib1)) for the rest methods.

| Parameter | Distribution |
| --- | --- |
| # layers | (A) UniformInt[1,5]15\left[1,5\right], (B) UniformInt[1,3]13\left[1,3\right] |
| Feature embedding size | (A,B) UniformInt[64,512]64512\left[64,512\right] |
| Residual Dropout | (A) Uniform[0,0.2]00.2\left[0,0.2\right], (B) Const(0.0) |
| Attention Dropout | (A,B) Uniform[0,0.5]00.5\left[0,0.5\right] |
| FNN Dropout | (A,B) Uniform[0,0.5]00.5\left[0,0.5\right] |
| Learning rate (main backbone) | (A) LogUniform[1​e−5,1​e−3]1𝑒51𝑒3\left[1e-5,1e-3\right], (B) LogUniform[3​e−5,3​e−4]3𝑒53𝑒4\left[3e-5,3e-4\right] |
| Learning rate (column embedding) | (A,B) LogUniform[5​e−3,5​e−2]5𝑒35𝑒2\left[5e-3,5e-2\right] |
| Weight decay | (A,B) LogUniform[1​e−6,1​e−3]1𝑒61𝑒3\left[1e-6,1e-3\right] |
| # iterations | 100 |

Table 9: Hyperparameter tuning spaces for T2G-Former. (A) = {FB, YE}, (B) = {GE, CH, EY, CA, HO, AD, OT, HE, JA, HI}.

## Appendix C Detailed Experiment Results

Table [10](#A3.T10 "Table 10 ‣ Appendix C Detailed Experiment Results ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction") reports detailed performance results (with standard deviations) of all models on all the datasets. Table [11](#A3.T11 "Table 11 ‣ Appendix C Detailed Experiment Results ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction") reports detailed results in “FR-Graph combinations”.

|  | GE ↑↑\uparrow | CH ↑↑\uparrow | EY ↑↑\uparrow | CA ↓↓\downarrow | HO ↓↓\downarrow | AD ↑↑\uparrow | OT ↑↑\uparrow | HE ↑↑\uparrow | JA ↑↑\uparrow | HI ↑↑\uparrow | FB ↓↓\downarrow | YE ↓↓\downarrow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 68.42±plus-or-minus\pm0.51 | 85.92±plus-or-minus\pm0.10 | 72.51±plus-or-minus\pm0.51 | 0.436±plus-or-minus\pm1.40e-6 | 3.169 ±plus-or-minus\pm1.45e-4 | 87.30±plus-or-minus\pm4.31e-3 | 82.46±plus-or-minus\pm0.07 | 37.47±plus-or-minus\pm0.13 | 71.85±plus-or-minus\pm0.06 | 72.41±plus-or-minus\pm0.02 | 5.359±plus-or-minus\pm8.20e-4 | 8.850±plus-or-minus\pm3.36e-5 |
| MLP | 58.64±plus-or-minus\pm3.18 | 85.77±plus-or-minus\pm0.05 | 61.10±plus-or-minus\pm0.67 | 0.499±plus-or-minus\pm1.90e-5 | 3.173±plus-or-minus\pm8.03e-4 | 85.35±plus-or-minus\pm0.02 | 80.99±plus-or-minus\pm0.22 | 38.38±plus-or-minus\pm0.04 | 71.97±plus-or-minus\pm0.05 | 72.00±plus-or-minus\pm0.05 | 5.943±plus-or-minus\pm3.90e-2 | 8.849±plus-or-minus\pm1.20e-2 |
| SNN | 64.69±plus-or-minus\pm0.99 | 85.74±plus-or-minus\pm0.10 | 61.55±plus-or-minus\pm1.36 | 0.498±plus-or-minus\pm7.09e-5 | 3.207±plus-or-minus\pm6.89e-4 | 85.40±plus-or-minus\pm0.02 | 81.17±plus-or-minus\pm0.06 | 37.19±plus-or-minus\pm0.12 | 71.94±plus-or-minus\pm0.10 | 72.21±plus-or-minus\pm0.05 | 5.892±plus-or-minus\pm3.56e-2 | 8.901±plus-or-minus\pm5.74e-4 |
| TabNet | 60.01±plus-or-minus\pm0.61 | 85.01±plus-or-minus\pm1.30 | 62.08±plus-or-minus\pm0.93 | 0.513±plus-or-minus\pm1.61e-4 | 3.252±plus-or-minus\pm3.38e-3 | 84.84±plus-or-minus\pm0.39 | 79.06±plus-or-minus\pm0.17 | 37.86±plus-or-minus\pm0.08 | 72.26±plus-or-minus\pm0.08 | 71.97±plus-or-minus\pm0.05 | 6.559±plus-or-minus\pm4.56e-2 | 8.916±plus-or-minus\pm7.98e-4 |
| DANet-28 | 61.63±plus-or-minus\pm1.46 | 85.10±plus-or-minus\pm0.20 | 60.53±plus-or-minus\pm1.51 | 0.524±plus-or-minus\pm5.25e-5 | 3.236±plus-or-minus\pm3.96e-3 | 85.00±plus-or-minus\pm0.08 | 81.04±plus-or-minus\pm0.02 | 35.45±plus-or-minus\pm0.06 | 70.72±plus-or-minus\pm0.05 | 71.47±plus-or-minus\pm0.05 | 6.167±plus-or-minus\pm9.31e-2 | 8.914±plus-or-minus\pm2.97e-3 |
| NODE | 53.94±plus-or-minus\pm1.17 | 85.86±plus-or-minus\pm0.13 | 65.54±plus-or-minus\pm0.44 | 0.463±plus-or-minus\pm2.60e-6 | 3.216±plus-or-minus\pm4.65e-5 | 85.77±plus-or-minus\pm0.03 | 80.37±plus-or-minus\pm0.04 | 35.33±plus-or-minus\pm0.22 | 72.78±plus-or-minus\pm0.01 | 72.51±plus-or-minus\pm0.01 | 5.698±plus-or-minus\pm1.59e-2 | 8.777±plus-or-minus\pm1.43e-4 |
| AutoInt | 58.33±plus-or-minus\pm2.13 | 85.51±plus-or-minus\pm0.10 | 61.07±plus-or-minus\pm0.69 | 0.472±plus-or-minus\pm1.76e-5 | 3.147±plus-or-minus\pm2.90e-4 | 85.66±plus-or-minus\pm0.02 | 80.11±plus-or-minus\pm0.06 | 37.26±plus-or-minus\pm0.03 | 72.08±plus-or-minus\pm0.02 | 72.51±plus-or-minus\pm0.04 | 5.852±plus-or-minus\pm3.67e-2 | 8.862±plus-or-minus\pm7.23e-4 |
| DCNv2 | 55.72±plus-or-minus\pm1.73 | 85.68±plus-or-minus\pm0.02 | 61.37±plus-or-minus\pm0.32 | 0.489±plus-or-minus\pm1.51e-5 | 3.172±plus-or-minus\pm7.96e-4 | 85.48±plus-or-minus\pm0.14 | 80.15±plus-or-minus\pm0.21 | 38.61±plus-or-minus\pm0.09 | 71.56±plus-or-minus\pm0.02 | 72.20±plus-or-minus\pm0.03 | 5.847±plus-or-minus\pm2.41e-2 | 8.882±plus-or-minus\pm8.13e-4 |
| FT-Transformer | 61.25±plus-or-minus\pm5.82 | 86.07±plus-or-minus\pm0.08 | 70.84±plus-or-minus\pm1.18 | 0.460±plus-or-minus\pm1.13e-5 | 3.124±plus-or-minus\pm1.35e-3 | 85.72±plus-or-minus\pm0.04 | 81.30±plus-or-minus\pm0.09 | 39.10±plus-or-minus\pm0.05 | 73.24±plus-or-minus\pm0.04 | 73.06±plus-or-minus\pm0.03 | 6.079±plus-or-minus\pm5.00e-2 | 8.852±plus-or-minus\pm5.32e-4 |
| T2G-Former | 65.57±plus-or-minus\pm1.44 | 86.25±plus-or-minus\pm0.05 | 78.18±plus-or-minus\pm1.85 | 0.455±plus-or-minus\pm2.10e-5 | 3.138±plus-or-minus\pm2.51e-3 | 85.96±plus-or-minus\pm0.02 | 81.87±plus-or-minus\pm0.03 | 39.06±plus-or-minus\pm0.03 | 73.68±plus-or-minus\pm0.03 | 73.39±plus-or-minus\pm0.03 | 5.701±plus-or-minus\pm1.61e-2 | 8.851±plus-or-minus\pm1.32e-3 |

Table 10: Detailed results with standard deviations of all models.

| Assemblings | GE ↑↑\uparrow | CH ↑↑\uparrow | EY ↑↑\uparrow | CA ↓↓\downarrow | HO ↓↓\downarrow | AD ↑↑\uparrow | OT ↑↑\uparrow | HE ↑↑\uparrow | JA ↑↑\uparrow | HI ↑↑\uparrow | FB ↓↓\downarrow | YE ↓↓\downarrow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aw​StsuperscriptAwsuperscriptSt{\rm A^{w}S^{t}} | 65.43±plus-or-minus\pm1.34 | 85.25±plus-or-minus\pm0.03 | 77.34±plus-or-minus\pm2.54 | 0.457±plus-or-minus\pm1.93e-5 | 3.171±plus-or-minus\pm2.42e-3 | 85.95±plus-or-minus\pm0.01 | 81.80±plus-or-minus\pm0.02 | 39.09±plus-or-minus\pm0.04 | 73.68±plus-or-minus\pm0.03 | 73.49±plus-or-minus\pm0.02 | 5.736±plus-or-minus\pm4.67e-3 | 8.886±plus-or-minus\pm9.59e-4 |
| Aw​AtsuperscriptAwsuperscriptAt{\rm A^{w}A^{t}} | 65.11±plus-or-minus\pm1.51 | 86.26±plus-or-minus\pm0.05 | 77.59±plus-or-minus\pm2.56 | 0.458±plus-or-minus\pm2.27e-5 | 3.145±plus-or-minus\pm2.67e-3 | 85.98±plus-or-minus\pm0.01 | 81.85±plus-or-minus\pm0.02 | 39.02±plus-or-minus\pm0.03 | 73.70±plus-or-minus\pm0.03 | 73.35±plus-or-minus\pm0.05 | 5.718±plus-or-minus\pm3.40e-3 | 8.861±plus-or-minus\pm1.62e-3 |
| Sw​StsuperscriptSwsuperscriptSt{\rm S^{w}S^{t}} | 64.90±plus-or-minus\pm2.88 | 86.20±plus-or-minus\pm0.02 | 76.46±plus-or-minus\pm4.67 | 0.456±plus-or-minus\pm1.22e-5 | 3.151±plus-or-minus\pm2.45e-3 | 85.97±plus-or-minus\pm0.02 | 81.89±plus-or-minus\pm0.03 | 39.17±plus-or-minus\pm0.01 | 73.78±plus-or-minus\pm0.02 | 73.69±plus-or-minus\pm0.05 | 5.723±plus-or-minus\pm3.36e-3 | 8.885±plus-or-minus\pm2.83e-3 |
| Sw​AtsuperscriptSwsuperscriptAt{\rm S^{w}A^{t}}(ours) | 65.57±plus-or-minus\pm1.44 | 86.25±plus-or-minus\pm0.05 | 78.18±plus-or-minus\pm1.85 | 0.455±plus-or-minus\pm2.10e-5 | 3.138±plus-or-minus\pm2.51e-3 | 85.96±plus-or-minus\pm0.02 | 81.87±plus-or-minus\pm0.03 | 39.06±plus-or-minus\pm0.03 | 73.68±plus-or-minus\pm0.03 | 73.39±plus-or-minus\pm0.03 | 5.701±plus-or-minus\pm1.61e-2 | 8.851±plus-or-minus\pm1.32e-3 |

Table 11: Comparison details of four FR-Graph assemblings.

## Appendix D Dataset Features

Table [12](#A4.T12 "Table 12 ‣ Appendix D Dataset Features ‣ T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction") shows detailed descriptions of features in dataset CA and CH. The CA dataset is typically used for predicting house price in California, while the CH is often used for predicting whether the customer will leave the bank.

|  |  |  |  |
| --- | --- | --- | --- |
| Dataset | Feature | Abbr | Description |
| CA | MedInc | MI | Median income within a block. |
| HouseAge | HoA | Median house age within a block. |
| AveRooms | AvR | Average number of rooms per household. |
| AveBedrms | AvB | Average number of bedrooms per household. |
| Population | Pop | Total number of people residing within a block. |
| AveOccup | AvOcc | Average number of household members. |
| Latitude | Lat | Block group latitude. |
| Longitude | Long | Block group longitude. |
| CH | CreditScore | CrSc | Credit score of the customer. |
| Gender | Gen | Gender of the customer. |
| Age | Age | Age of the customer. |
| Tenure | Ten | Number of years for which the customer has been with the bank. |
| Balance | Bal | Bank balance of the customer. |
| NumOfProducts | NProd | Number of bank products the customer is utilising. |
| EstimatedSalary | EstSal | Estimated salary of the costumer. |
| HasCrCard | HasCrC | Whether the customer has a credit card in the bank. |
| IsActiveMember | ActMem | Whether the customer is an active member in the bank. |
| Geography | Geo | The country from which the customer belongs. |

Table 12: Feature descriptions of California Housing (CA) and Churn Modeling (CH).

[◄](/html/2211.16886)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2211.16887)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2211.16887)
[View original  
on arXiv](https://arxiv.org/abs/2211.16887)[►](/html/2211.16888)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Mar 14 10:02:45 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
