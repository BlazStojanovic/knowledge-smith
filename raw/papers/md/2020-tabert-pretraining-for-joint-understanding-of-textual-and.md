---
arxiv: '2005.08314'
authors:
- Pengcheng Yin
- Graham Neubig
- Wen-tau Yih
- Sebastian Riedel
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data'
url: http://arxiv.org/abs/2005.08314v1
year: 2020
---

[2005.08314] TaBert: Pretraining for Joint Understanding of Textual and Tabular Data














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



# TaBert: Pretraining for Joint Understanding of Textual and Tabular Data

Pengcheng Yin  Graham Neubig
  
Carnegie Mellon University
  
{pcyin,gneubig}@cs.cmu.edu
&Wen-tau Yih  Sebastian Riedel
  
Facebook AI Research
  
{scottyih,sriedel}@fb.com
  Work done while at Facebook AI Research.

###### Abstract

Recent years have witnessed the burgeoning of pretrained language models (LMs) for text-based natural language (NL) understanding tasks.
Such models are typically trained on free-form NL text, hence may not be suitable for
tasks like semantic parsing over structured data, which require reasoning over both free-form NL questions and structured tabular data (*e.g.*, database tables).
In this paper we present TaBert, a pretrained LM that jointly learns representations for NL sentences and (semi-)structured tables.
TaBert is trained on a large corpus of 26 million tables and their English contexts.
In experiments, neural semantic parsers using TaBert as feature representation layers achieve new best results on the challenging weakly-supervised semantic parsing benchmark WikiTableQuestions, while performing competitively on the text-to-SQL dataset Spider.111Code available at <http://fburl.com/TaBERT>

## 1 Introduction

Recent years have witnessed a rapid advance in the ability to understand and answer questions about free-form natural language (NL) text Rajpurkar et al. ([2016](#bib.bib29)),
largely due to large-scale, pretrained language models (LMs) like BERT Devlin et al. ([2019](#bib.bib9)).
These models allow us to capture the syntax and semantics of text via representations learned in an unsupervised manner, before fine-tuning the model to downstream tasks Melamud et al. ([2016](#bib.bib24)); McCann et al. ([2017](#bib.bib23)); Peters et al. ([2018](#bib.bib27)); Liu et al. ([2019b](#bib.bib22)); Yang et al. ([2019](#bib.bib41)); Goldberg ([2019](#bib.bib11)).
It is also relatively easy to apply such pretrained LMs to comprehension tasks that are modeled as text span selection problems, where the boundary of an answer span can be predicted using a simple classifier on top of the LM Joshi et al. ([2019](#bib.bib16)).

However, it is less clear how one could pretrain and fine-tune such models for other QA tasks that involve joint reasoning over both free-form NL text and *structured* data.
One example task is semantic parsing for access to databases (DBs) (Zelle and Mooney, [1996](#bib.bib47); Berant et al., [2013](#bib.bib2); Yih et al., [2015](#bib.bib42)), the task of transducing an NL utterance (*e.g.*, “Which country has the largest GDP?”) into a structured query over DB tables (*e.g.*, SQL querying a database of economics).
A key challenge in this scenario is understanding the structured schema of DB tables (*e.g.*, the name, data type, and stored values of columns), and more importantly, the alignment between the input text and the schema (*e.g.*, the token “GDP” refers to the Gross Domestic Product column), which is essential for inferring the correct DB query (Berant and Liang, [2014](#bib.bib3)).

Neural semantic parsers tailored to this task therefore attempt to learn joint representations of NL utterances and the (semi-)structured schema of DB tables (*e.g.*, representations of its columns or cell values, as in Krishnamurthy et al. ([2017](#bib.bib17)); Bogin et al. ([2019b](#bib.bib5)); Wang et al. ([2019a](#bib.bib35)), inter alia).
However, this unique setting poses several challenges in applying pretrained LMs.
First, information stored in DB tables exhibit strong underlying structure, while existing LMs (*e.g.*, BERT) are solely trained for encoding free-form text.
Second, a DB table could potentially have a large number of rows, and naively encoding all of them using a resource-heavy LM is computationally intractable.
Finally, unlike most text-based QA tasks (*e.g.*, SQuAD, Rajpurkar et al. ([2016](#bib.bib29))) which could be formulated as a generic answer span selection problem and solved by a pretrained model with additional classification layers,
semantic parsing is highly domain-specific, and the architecture of a neural parser is strongly coupled with the structure of its underlying DB
(*e.g.*, systems for SQL-based and other types of DBs use different encoder models).
In fact, existing systems have attempted to leverage BERT, but each with their own domain-specific, in-house strategies to encode the structured information in the DB (Guo et al., [2019](#bib.bib12); Zhang et al., [2019a](#bib.bib48); Hwang et al., [2019](#bib.bib15)),
and importantly, without pretraining representations on structured data.
These challenges call for development of general-purpose pretraining approaches tailored to learning representations for both NL utterances and structured DB tables.

In this paper we present TaBert, a pretraining approach for joint understanding of NL text and (semi-)structured tabular data ([§ 3](#S3 "3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).
TaBert is built on top of BERT,
and jointly learns contextual representations for utterances and the structured schema of DB tables (*e.g.*, a vector for each utterance token and table column).
Specifically, TaBert linearizes the structure of tables to be compatible with a Transformer-based BERT model.
To cope with large tables, we propose *content snapshots*, a method to encode a subset of table content most relevant to the input utterance.
This strategy is further combined with a *vertical attention* mechanism to share information among cell representations in different rows ([§ 3.1](#S3.SS1 "3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).
To capture the association between tabular data and related NL text, TaBert is pretrained on a parallel corpus of 26 million tables and English paragraphs ([§ 3.2](#S3.SS2 "3.2 Pretraining Procedure ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).

TaBert can be plugged into a neural semantic parser as a general-purpose encoder to compute representations for utterances and tables.
Our key insight is that although semantic parsers are highly domain-specific, most systems rely on representations of input utterances and the table schemas to facilitate subsequent generation of DB queries, and these representations can be provided by TaBert, regardless of the domain of the parsing task.

We apply TaBert to two different semantic parsing paradigms:
(1) a classical supervised learning setting on the Spider text-to-SQL dataset (Yu et al., [2018c](#bib.bib46)), where TaBert is fine-tuned together with a task-specific parser using parallel NL utterances and labeled DB queries ([§ 4.1](#S4.SS1 "4.1 Supervised Semantic Parsing ‣ 4 Example Application: Semantic Parsing over Tables ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"));
and (2) a challenging weakly-supervised learning benchmark WikiTableQuestions (Pasupat and Liang, [2015](#bib.bib26)), where
a system has to infer latent DB queries from its execution results ([§ 4.2](#S4.SS2 "4.2 Weakly Supervised Semantic Parsing ‣ 4 Example Application: Semantic Parsing over Tables ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).
We demonstrate TaBert is effective in both scenarios, showing that it is a drop-in replacement of a parser’s original encoder for computing contextual representations of NL utterances and DB tables.
Specifically, systems augmented with TaBert outperforms their counterparts using Bert, registering state-of-the-art performance on WikiTableQuestions, while performing competitively on Spider ([§ 5](#S5 "5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).

## 2 Background

#### Semantic Parsing over Tables

Semantic parsing tackles the task of translating an NL utterance 𝒖𝒖\bm{u} into a formal meaning representation (MR) 𝒛𝒛\bm{z}.
Specifically, we focus on parsing utterances to access database tables, where 𝒛𝒛\bm{z} is a structured query (*e.g.*, an SQL query) executable on a set of relational DB tables 𝒯={Tt}𝒯subscript𝑇𝑡\mathcal{T}=\{{T}\_{t}\}.
A relational table T𝑇{T} is a listing of N𝑁N rows {Ri}i=1Nsuperscriptsubscriptsubscript𝑅𝑖𝑖1𝑁\{R\_{i}\}\_{i=1}^{N} of data, with each row Risubscript𝑅𝑖R\_{i} consisting of M𝑀M cells {s⟨i,j⟩}j=1Msuperscriptsubscriptsubscript𝑠

𝑖𝑗𝑗1𝑀\{s\_{\langle i,j\rangle}\}\_{j=1}^{M}, one for each column cjsubscript𝑐𝑗c\_{j}.
Each cell s⟨i,j⟩subscript𝑠

𝑖𝑗s\_{\langle i,j\rangle} contains a list of tokens.

Depending on the underlying data representation schema used by the DB, a table could either be fully structured with strongly-typed and normalized contents (*e.g.*, a table column named distance has a unit of kilometers, with all of its cell values, like 200,
bearing the same unit), as is commonly the case for SQL-based DBs ([§ 4.1](#S4.SS1 "4.1 Supervised Semantic Parsing ‣ 4 Example Application: Semantic Parsing over Tables ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).
Alternatively, it could be semi-structured with unnormalized, textual cell values (*e.g.*, 200 km, [§ 4.2](#S4.SS2 "4.2 Weakly Supervised Semantic Parsing ‣ 4 Example Application: Semantic Parsing over Tables ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).
The query language could also take a variety of forms, from general-purpose DB access languages like SQL to domain-specific ones tailored to a particular task.

Given an utterance and its associated tables, a neural semantic parser generates a DB query from the vector representations of the utterance tokens and the structured schema of tables.
In this paper we refer *schema* as the set of columns in a table, and its *representation* as the list of vectors that represent its columns222Column representations for more complex schemas, *e.g.*, those capturing inter-table dependency via primary and foreign keys, could be derived from these table-wise representations..
We will introduce how TaBert computes these representations in [§ 3.1](#S3.SS1 "3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data").

![Refer to caption](/html/2005.08314/assets/x1.png)


Figure 1: Overview of TaBert for learning representations of utterances and table schemas with an example from WikiTableQuestions444Example adapted from [stanford.io/38iZ8Pf](https://stanford.io/38iZ8Pf). (A) A content snapshot of the table is created based on the input NL utterance. (B) Each row in the snapshot is encoded by a Transformer (only R2subscript𝑅2R\_{2} is shown), producing row-wise encodings for utterance tokens and cells. (C) All row-wise encodings are aligned and processed by V𝑉V vertical self-attention layers, generating utterance and column representations.

#### Masked Language Models

Given a sequence of NL tokens 𝒙=x1,x2,…,xn𝒙

subscript𝑥1subscript𝑥2…subscript𝑥𝑛\bm{x}=x\_{1},x\_{2},\ldots,x\_{n}, a masked language model (*e.g.*, BERT) is an LM trained using the masked language modeling objective, which aims to recover the original tokens in 𝒙𝒙\bm{x} from a “corrupted” context created by randomly masking out certain tokens in 𝒙𝒙\bm{x}.
Specifically, let 𝒙m={xi1,…,xim}subscript𝒙𝑚subscript𝑥subscript𝑖1…subscript𝑥subscript𝑖𝑚\bm{x}\_{m}=\{x\_{i\_{1}},\ldots,x\_{i\_{m}}\} be the subset of tokens
in 𝒙𝒙\bm{x} selected to be masked out, and 𝒙~~𝒙\widetilde{\bm{x}} denote the masked sequence with tokens in 𝒙msubscript𝒙𝑚\bm{x}\_{m}
replaced by a [MASK] symbol. A masked LM defines a distribution p𝜽​(𝒙m|𝒙~)subscript𝑝𝜽conditionalsubscript𝒙𝑚~𝒙p\_{\bm{\theta}}(\bm{x}\_{m}|\widetilde{\bm{x}}) over the target tokens 𝒙msubscript𝒙𝑚\bm{x}\_{m} given the masked context 𝒙~~𝒙\widetilde{\bm{x}}.

BERT parameterizes p𝜽​(𝒙m|𝒙~)subscript𝑝𝜽conditionalsubscript𝒙𝑚~𝒙p\_{\bm{\theta}}(\bm{x}\_{m}|\widetilde{\bm{x}}) using a Transformer model.
During the pretraining phase, BERT maximizes p𝜽​(𝒙m|𝒙~)subscript𝑝𝜽conditionalsubscript𝒙𝑚~𝒙p\_{\bm{\theta}}(\bm{x}\_{m}|\widetilde{\bm{x}}) on large-scale textual corpora.
In the fine-tuning phase, the pretrained model is used as an encoder to compute representations of input NL tokens, and its parameters are jointly tuned with other task-specific neural components.

## 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data

We first present how TaBert computes representations for NL utterances and table schemas ([§ 3.1](#S3.SS1 "3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")), and then describe the pretraining procedure ([§ 3.2](#S3.SS2 "3.2 Pretraining Procedure ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).

### 3.1 Computing Representations for NL Utterances and Table Schemas

[Fig. 1](#S2.F1 "Figure 1 ‣ Semantic Parsing over Tables ‣ 2 Background ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") presents a schematic overview of TaBert.
Given an utterance 𝒖𝒖\bm{u} and a table T𝑇{T}, TaBert first creates a *content snapshot* of T𝑇{T}.
This snapshot consists of sampled rows that summarize the information in T𝑇{T} most relevant to the input utterance.
The model then linearizes each row in the snapshot, concatenates each linearized row with the utterance, and uses the concatenated string as input to a Transformer (*e.g.*, BERT) model, which outputs row-wise encoding vectors of utterance tokens and cells.
The encodings for all the rows in the snapshot are fed into a series of vertical
self-attention layers, where a cell representation (or an utterance token representation) is computed by attending to vertically-aligned vectors of the same column (or the same NL token).
Finally, representations for each utterance token and column are generated from a pooling layer.

#### Content Snapshot

One major feature of TaBert is its use of the table *contents*, as opposed to just using the column names, in encoding the table schema.
This is motivated by the fact that contents provide more detail about the semantics of a column than just the column’s name, which might be ambiguous.
For instance, the Venue column in [Fig. 1](#S2.F1 "Figure 1 ‣ Semantic Parsing over Tables ‣ 2 Background ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") which is used to answer the example question actually refers to host cities, and encoding the sampled cell values while creating
its representation may help
match the term “city” in the input utterance to this column.

However, a DB table could potentially have a large number of rows, with only few of them actually relevant to answering the input utterance.
Encoding all of the contents using a resource-heavy Transformer is both computationally intractable and likely not necessary.
Thus, we instead use a *content snapshot* consisting of only a few rows that are most relevant to the input utterance, providing an efficient approach to calculate content-sensitive column representations from cell values.

We use a simple strategy to create content snapshots of K𝐾K rows based on the relevance between the utterance and a row.
For K>1𝐾1K>1, we select the top-K𝐾K rows in the input table that have the highest n𝑛n-gram overlap ratio with the utterance.555We use n≤3𝑛3n\leq 3 in our experiments. Empirically this simple matching heuristic is able to correctly identify the best-matched rows for 40 out of 50 sampled examples on WikiTableQuestions.
For K=1𝐾1K=1, to include in the snapshot as much information relevant to the utterance as possible, we create a synthetic row by selecting the cell values from each column that have the highest n𝑛n-gram overlap with the utterance.
Using synthetic rows in this restricted setting is motivated by the fact that cell values most relevant to answer the utterance could come from multiple rows.
As an example, consider the utterance “How many more participants were there in 2008 than in the London Olympics?”, and an associating table with columns Year, Host City and Number of Participants,
the most relevant cells to the utterance, 2008 (from Year) and London (from Host City), are from different rows, which could be included in a single synthetic row.
In the initial experiments we found synthetic rows also help stabilize learning.

#### Row Linearization

TaBert creates a linearized sequence for each row in the content snapshot as input to the Transformer model.
[Fig. 1](#S2.F1 "Figure 1 ‣ Semantic Parsing over Tables ‣ 2 Background ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")(B) depicts the linearization for R2subscript𝑅2R\_{2}, which consists of a concatenation of the utterance, columns, and their cell values.
Specifically, each cell is represented by the name and data type666We use two data types, text, and real for numbers, predicted by majority voting over the NER labels of cell tokens. of the column, together with its actual value, separated by a vertical bar.
As an example, the cell s⟨2,1⟩subscript𝑠

21s\_{\langle 2,1\rangle} valued 2005 in R2subscript𝑅2R\_{2} in [Fig. 1](#S2.F1 "Figure 1 ‣ Semantic Parsing over Tables ‣ 2 Background ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") is encoded as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Year⏟Column Name​|​real⏟Column Type​|​2005⏟Cell Valuesubscript⏟YearColumn Name|subscript⏟realColumn Type|subscript⏟2005Cell Value\underbrace{\text{Year}}\_{\textrm{Column Name}}\text{|}\underbrace{\text{real}}\_{\textrm{Column Type}}\text{|}~{}~{}~{}\underbrace{\text{2005}}\_{\textrm{Cell Value}} |  | (1) |

The linearization of a row is then formed by concatenating the above string encodings of all the cells, separated by the [SEP] symbol.
We then prefix the row linearization with utterance tokens as input sequence to the Transformer.

Existing works have applied different linearization strategies to encode tables with Transformers Hwang et al. ([2019](#bib.bib15)); Chen et al. ([2019](#bib.bib6)), while our row approach is specifically designed for encoding content snapshots.
We present in [§ 5](#S5 "5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") results with different linearization choices.

#### Vertical Self-Attention Mechanism

The base Transformer model in TaBert outputs vector encodings of utterance and cell tokens for each row.
These row-level vectors are computed separately and therefore independent of each other.
To allow for information flow across cell representations of different rows, we propose vertical self-attention, a self-attention mechanism that operates over vertically aligned vectors from different rows.

As in [Fig. 1](#S2.F1 "Figure 1 ‣ Semantic Parsing over Tables ‣ 2 Background ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")(C), TaBert has V𝑉V stacked vertical-level self-attention layers.
To generate aligned inputs for vertical attention, we first compute a fixed-length initial vector for each cell at position ⟨i,j⟩

𝑖𝑗\langle i,j\rangle, which is given by mean-pooling over the sequence of the Transformer’s output vectors that correspond to its variable-length linearization as in Eq. ([1](#S3.E1 "Equation 1 ‣ Row Linearization ‣ 3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).
Next, the sequence of word vectors for the NL utterance (from the base Transformer model) are concatenated with the cell vectors as initial inputs to the vertical attention layer.

Each vertical attention layer has the same parameterization as the Transformer layer in Vaswani et al. ([2017](#bib.bib34)), but operates on vertically aligned elements, *i.e.*, utterance and cell vectors that correspond to the same question token and column, respectively.
This vertical self-attention mechanism enables the model to aggregate information from different rows in the content snapshot, allowing TaBert to capture cross-row dependencies on cell values.

#### Utterance and Column Representations

A representation 𝐜jsubscript𝐜𝑗\mathbf{c}\_{j} is computed for each column cjsubscript𝑐𝑗c\_{j} by mean-pooling over its vertically aligned cell vectors, {𝐬⟨i,j⟩:Ri​ in content snapshot}conditional-setsubscript𝐬

𝑖𝑗subscript𝑅𝑖 in content snapshot\{\mathbf{s}\_{\langle i,j\rangle}:R\_{i}\textrm{ in content snapshot}\}, from the last vertical layer.
A representation for each utterance token, 𝐱jsubscript𝐱𝑗\mathbf{x}\_{j}, is computed similarly over the vertically aligned token vectors.
These
representations will be used by downstream neural semantic parsers.
TaBert also outputs an optional fixed-length table representation 𝐓𝐓\mathbf{T} using the representation of the prefixed [CLS] symbol, which is useful for parsers that operate on multiple DB tables.

### 3.2 Pretraining Procedure

#### Training Data

Since there is no large-scale, high-quality parallel corpus of NL text and structured tables, we instead use semi-structured tables that commonly exist on the Web as a surrogate data source.
As a first step in this line, we focus on collecting parallel data in English, while extending to multilingual scenarios would be an interesting avenue for future work.
Specifically, we collect tables and their surrounding NL text from English Wikipedia and the WDC WebTable Corpus Lehmberg et al. ([2016](#bib.bib18)), a large-scale table collection from CommonCrawl.
The raw data is extremely noisy, and we apply aggressive cleaning heuristics to filter out invalid examples (*e.g.*, examples with HTML snippets or in foreign languages, and non-relational tables without headers).
See Appendix [§ A.1](#A1.SS1 "A.1 Training Data ‣ Appendix A Pretraining Details ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") for details of data pre-processing.
The pre-processed corpus contains 26.6 million parallel examples of tables and NL sentences.
We perform sub-tokenization using the Wordpiece tokenizer shipped with BERT.

#### Unsupervised Learning Objectives

We apply different objectives for learning representations of the NL context and structured tables.
For NL contexts, we use the standard Masked Language Modeling (MLM) objective Devlin et al. ([2019](#bib.bib9)), with a masking rate of 15% sub-tokens in an NL context.

For learning column representations, we design two objectives motivated by the intuition that a column representation should contain both the general information of the column (*e.g.*, its name and data type), and representative cell values relevant to the NL context.
First,
a Masked Column Prediction (MCP) objective encourages the model to recover the names and data types of masked columns.
Specifically, we randomly select 20% of the columns in an input table, masking their names and data types in each row linearization (*e.g.*, if the column Year in [Fig. 1](#S2.F1 "Figure 1 ‣ Semantic Parsing over Tables ‣ 2 Background ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") is selected, the tokens Year and real in Eq. ([1](#S3.E1 "Equation 1 ‣ Row Linearization ‣ 3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")) will be masked).
Given the column representation 𝐜jsubscript𝐜𝑗\mathbf{c}\_{j}, TaBert is trained to predict the bag of masked (name and type) tokens from 𝐜jsubscript𝐜𝑗\mathbf{c}\_{j} using a multi-label classification objective.
Intuitively, MCP encourages the model to recover column information from its contexts.

Next, we use an auxiliary Cell Value Recovery (CVR) objective
to ensure information of representative cell values in content snapshots is retained after additional layers of vertical self-attention.
Specifically, for each masked column cjsubscript𝑐𝑗c\_{j} in the above MCP objective, CVR predicts the original tokens of each cell s⟨i,j⟩subscript𝑠

𝑖𝑗s\_{\langle i,j\rangle} (of cjsubscript𝑐𝑗c\_{j})
in the content snapshot
conditioned on its cell vector 𝐬⟨i,j⟩subscript𝐬

𝑖𝑗\mathbf{s}\_{\langle i,j\rangle}.777The cell value tokens are not masked in the input sequence, since predicting masked cell values is challenging even with the presence of its surrounding context. 
For instance, for the example cell s⟨2,1⟩subscript𝑠

21s\_{\langle 2,1\rangle} in Eq. ([1](#S3.E1 "Equation 1 ‣ Row Linearization ‣ 3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")), we predict its value 2005 from 𝐬⟨2,1⟩subscript𝐬

21\mathbf{s}\_{\langle 2,1\rangle}.
Since a cell could have multiple value tokens, we apply the span-based prediction objective Joshi et al. ([2019](#bib.bib16)).
Specifically, to predict a cell token s⟨i,j⟩k∈s⟨i,j⟩subscript𝑠subscript

𝑖𝑗
𝑘subscript𝑠

𝑖𝑗s\_{{\langle i,j\rangle}\_{k}}\in s\_{{\langle i,j\rangle}}, its positional embedding 𝐞ksubscript𝐞𝑘\mathbf{e}\_{k} and the cell representations 𝐬⟨i,j⟩subscript𝐬

𝑖𝑗\mathbf{s}\_{\langle i,j\rangle} are fed into a two-layer network f​(⋅)𝑓⋅f(\cdot) with GeLU activations Hendrycks and Gimpel ([2016](#bib.bib14)).
The output of f​(⋅)𝑓⋅f(\cdot) is then used to predict the original value token s⟨i,j⟩ksubscript𝑠subscript

𝑖𝑗
𝑘s\_{{\langle i,j\rangle}\_{k}} from a softmax layer.

## 4 Example Application: Semantic Parsing over Tables

We apply TaBert for representation learning on two semantic parsing paradigms, a classical supervised text-to-SQL task over structured DBs ([§ 4.1](#S4.SS1 "4.1 Supervised Semantic Parsing ‣ 4 Example Application: Semantic Parsing over Tables ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")), and a weakly supervised parsing problem on semi-structured Web tables ([§ 4.2](#S4.SS2 "4.2 Weakly Supervised Semantic Parsing ‣ 4 Example Application: Semantic Parsing over Tables ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).

### 4.1 Supervised Semantic Parsing

#### Benchmark Dataset

Supervised learning is the typical scenario of learning a parser using parallel data of utterances and queries.
We use Spider Yu et al. ([2018c](#bib.bib46)), a text-to-SQL dataset with 10,181 examples across 200 DBs.
Each example consists of an utterance (*e.g.*, “What is the total number of languages used in Aruba?”), a DB with one or more tables, and an annotated SQL query, which typically involves joining multiple tables to get the answer (*e.g.*, SELECT COUNT(\*) FROM Country JOIN Lang ON Country.Code = Lang.CountryCode WHERE Name = ‘Aruba’).

#### Base Semantic Parser

We aim to show TaBert could help improve upon an already strong parser.
Unfortunately, at the time of writing, none of the top systems on Spider were publicly available.
To establish a reasonable testbed, we developed our in-house system based on TranX Yin and Neubig ([2018](#bib.bib43)), an open-source general-purpose semantic parser.
TranX translates an NL utterance into an intermediate meaning representation guided by a user-defined grammar.
The generated intermediate MR could then be deterministically converted to domain-specific query languages (*e.g.*, SQL).

We use TaBert as encoder of utterances and table schemas.
Specifically, for a given utterance 𝒖𝒖\bm{u} and a DB with a set of tables 𝒯={Tt}𝒯subscript𝑇𝑡\mathcal{T}=\{{T}\_{t}\}, we first pair 𝒖𝒖\bm{u} with each table Ttsubscript𝑇𝑡{T}\_{t} in 𝒯𝒯\mathcal{T} as inputs to TaBert, which generates |𝒯|𝒯|\mathcal{T}| sets of table-specific representations of utterances and columns.
At each time step, an LSTM decoder performs hierarchical
attention Libovický and Helcl ([2017](#bib.bib20)) over the list of table-specific representations, constructing an MR based on the predefined grammar.
Following the IRNet model Guo et al. ([2019](#bib.bib12)) which achieved the best performance on Spider, we use SemQL, a simplified version of the SQL, as the underlying grammar.
We refer interested readers to Appendix [§ B.1](#A2.SS1 "B.1 Supervised Parsing on Spider ‣ Appendix B Semantic Parsers ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") for details of our system.

### 4.2 Weakly Supervised Semantic Parsing

#### Benchmark Dataset

Weakly supervised semantic parsing considers the reinforcement learning task of inferring the correct query from its execution results (*i.e.*, whether the answer is correct).
Compared to supervised learning, weakly supervised parsing is significantly more challenging, as the parser does not have access to the labeled query, and has to explore the exponentially large search space of possible queries guided by the noisy binary reward signal of execution results.

WikiTableQuestions Pasupat and Liang ([2015](#bib.bib26)) is a popular dataset for weakly supervised semantic parsing, which has 22,033 utterances and 2,108 semi-structured Web tables from Wikipedia.888While some of the 421 testing Wikipedia tables might be included in our pretraining corpora, they only account for a very tiny fraction. In our pilot study, we also found pretraining only on Wikipedia tables resulted in worse performance.
Compared to Spider, examples in this dataset do not involve joining multiple tables, but typically require compositional, multi-hop reasoning over a series of entries in the given table (*e.g.*, to answer the example in [Fig. 1](#S2.F1 "Figure 1 ‣ Semantic Parsing over Tables ‣ 2 Background ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") the parser needs to reason over the row set {R2,R3,R5}subscript𝑅2subscript𝑅3subscript𝑅5\{R\_{2},R\_{3},R\_{5}\}, locating the Venue field with the largest value of Year).

#### Base Semantic Parser

MAPO Liang et al. ([2018](#bib.bib19)) is a strong system for weakly supervised semantic parsing.
It improves the sample efficiency of the REINFORCE algorithm by biasing the exploration of queries towards the high-rewarding ones already discovered by the model.
MAPO uses a domain-specific query language tailored to answering compositional questions on single tables, and its utterances and column representations are derived from an LSTM encoder,
which we replaced with our TaBert model.
See Appendix [§ B.2](#A2.SS2 "B.2 Weakly-supervised Parsing on WikiTableQuestions ‣ Appendix B Semantic Parsers ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") for details of MAPO and our adaptation.

## 5 Experiments

In this section we evaluate TaBert on downstream tasks of semantic parsing to DB tables.

#### Pretraining Configuration

We train two variants of the model, TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} and TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}}, with the underlying Transformer model initialized with the uncased versions of BertBasesubscriptBertBase\textsc{Bert}\_{\textrm{Base}} and BertLargesubscriptBertLarge\textsc{Bert}\_{\textrm{Large}}, respectively.999We also attempted to train TaBert on our collected corpus from scratch without initialization from BERT, but with inferior results, potentially due to the average lower quality of web-scraped tables compared to purely textual corpora. We leave improving the quality of training data as future work.
During pretraining, for each table and its associated NL context in the corpus, we create a series of training instances of paired NL sentences (as synthetically generated utterances) and tables (as content snapshots) by
(1) sliding a (non-overlapping) context window of sentences with a maximum length of 128128128 tokens, and
(2) using the NL tokens in the window as the utterance, and pairing it with randomly sampled rows from the table as content snapshots.
TaBert is implemented in PyTorch using distributed training.
Refer to Appendix [§ A.2](#A1.SS2 "A.2 Pretraining Setup ‣ Appendix A Pretraining Details ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") for details of pretraining.

#### Comparing Models

We mainly present results for two variants of TaBert by varying the size of content snapshots K𝐾K.
TaBert(K=3)K3\mathbf{(K=3)} uses three rows from input tables as content snapshots and three vertical self-attention layers.
TaBert(K=1)K1\mathbf{(K=1)} uses one synthetically generated row as the content snapshot as described in [§ 3.1](#S3.SS1 "3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data").
Since this model does not have multi-row input, we do not use additional vertical attention layers (and the cell value recovery learning objective).
Its column representation 𝐜jsubscript𝐜𝑗\mathbf{c}\_{j} is defined by mean-pooling over the Transformer’s output encodings that correspond to the column name (*e.g.*, the representation for the Year column in [Fig. 1](#S2.F1 "Figure 1 ‣ Semantic Parsing over Tables ‣ 2 Background ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") is derived from the vector of the Year token in Eq. ([1](#S3.E1 "Equation 1 ‣ Row Linearization ‣ 3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"))).
We find this strategy gives better results compared with using the cell representation 𝐬jsubscript𝐬𝑗\mathbf{s}\_{j} as 𝐜jsubscript𝐜𝑗\mathbf{c}\_{j}.
We also compare with Bert using the same row linearization and content snapshot approach as TaBert(K=1)K1\mathrm{(K=1)}, which reduces to a TaBert(K=1)K1\mathrm{(K=1)} model without pretraining on tabular corpora.

#### Evaluation Metrics

As standard,
we report execution accuracy on WikiTableQuestions and exact-match accuracy of DB queries on Spider.

### 5.1 Main Results

| Previous Systems on WikiTableQuestions | | | | |
| --- | --- | --- | --- | --- |
| Model | Dev | | Test | |
| \hdashlinePasupat and Liang ([2015](#bib.bib26)) | 37.0 | | 37.1 | |
| Neelakantan et al. ([2016](#bib.bib25)) | 34.1 | | 34.2 | |
| Ensemble 15 Models | 37.5 | | 37.7 | |
| Zhang et al. ([2017](#bib.bib49)) | 40.6 | | 43.7 | |
| Dasigi et al. ([2019](#bib.bib8)) | 43.1 | | 44.3 | |
| Agarwal et al. ([2019](#bib.bib1)) | 43.2 | | 44.1 | |
| Ensemble 10 Models | – | | 46.9 | |
| Wang et al. ([2019b](#bib.bib36)) | 43.7 | | 44.5 | |
| Our System based on MAPO (Liang et al., [2018](#bib.bib19)) | | | | |
|  | Dev | Best | Test | Best |
| Base Parser† | 42.3 ±0.3plus-or-minus0.3\scriptstyle\pm 0.3 | 42.7 | 43.1 ±0.5plus-or-minus0.5\scriptstyle\pm 0.5 | 43.8 |
| \hdashline w/w/ BertBasesubscriptBertBase\textsc{Bert}\_{\textrm{Base}} (K=1)K1\mathrm{(K=1)} | 49.6 ±0.5plus-or-minus0.5\scriptstyle\pm 0.5 | 50.4 | 49.4 ±0.5plus-or-minus0.5\scriptstyle\pm 0.5 | 49.2 |
| −- content snapshot | 49.1 ±0.6plus-or-minus0.6\scriptstyle\pm 0.6 | 50.0 | 48.8 ±0.9plus-or-minus0.9\scriptstyle\pm 0.9 | 50.2 |
| w/w/ TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} (K=1)K1\mathrm{(K=1)} | 51.2 ±0.5plus-or-minus0.5\scriptstyle\pm 0.5 | 51.6 | 50.4 ±0.5plus-or-minus0.5\scriptstyle\pm 0.5 | 51.2 |
| −- content snapshot | 49.9 ±0.4plus-or-minus0.4\scriptstyle\pm 0.4 | 50.3 | 49.4 ±0.4plus-or-minus0.4\scriptstyle\pm 0.4 | 50.0 |
| w/w/ TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} (K=3)K3\mathrm{(K=3)} | 51.6 ±0.5plus-or-minus0.5\scriptstyle\pm 0.5 | 52.4 | 51.4 ±0.3plus-or-minus0.3\scriptstyle\pm 0.3 | 51.3 |
| \hdashline w/w/ BertLargesubscriptBertLarge\textsc{Bert}\_{\textrm{Large}} (K=1)K1\mathrm{(K=1)} | 50.3 ±0.4plus-or-minus0.4\scriptstyle\pm 0.4 | 50.8 | 49.6 ±0.5plus-or-minus0.5\scriptstyle\pm 0.5 | 50.1 |
| w/w/ TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}} (K=1)K1\mathrm{(K=1)} | 51.6 ±1.1plus-or-minus1.1\scriptstyle\pm 1.1 | 52.7 | 51.2 ±0.9plus-or-minus0.9\scriptstyle\pm 0.9 | 51.5 |
| w/w/ TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}} (K=3)K3\mathrm{(K=3)} | 52.2 ±0.7plus-or-minus0.7\scriptstyle\pm 0.7 | 53.0 | 51.8 ±0.6plus-or-minus0.6\scriptstyle\pm 0.6 | 52.3 |

Table 1: Execution accuracies on WikiTableQuestions. †Results from Liang et al. ([2018](#bib.bib19)). (Ta)Bert models are evaluated with 10 random runs. We report mean, standard deviation and the best results. Test↦maps-to\mapstoBest refers to the result from the run with the best performance on Dev. set.



|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| |  |  | | --- | --- | | Top-ranked Systems on Spider Leaderboard | | | Model | Dev. Acc. | | \hdashlineGlobal–GNN (Bogin et al., [2019a](#bib.bib4)) | 52.7 | | EditSQL ++ Bert (Zhang et al., [2019a](#bib.bib48)) | 57.6 | | RatSQL (Wang et al., [2019a](#bib.bib35)) | 60.9 | | IRNet ++ Bert (Guo et al., [2019](#bib.bib12)) | 60.3 | | ++ Memory ++ Coarse-to-Fine | 61.9 | | IRNet V2 ++ Bert | 63.9 | | RyanSQL ++ Bert (Choi et al., [2020](#bib.bib7)) | 66.6 | | | |
| Our System based on TranX (Yin and Neubig, [2018](#bib.bib43)) | | |
|  | Mean | Best |
| \hdashlinew/w/ BertBasesubscriptBertBase\textsc{Bert}\_{\textrm{Base}} (K=1)K1\mathrm{(K=1)} | 61.8 ±0.8plus-or-minus0.8\scriptstyle\pm 0.8 | 62.4 |
| −- content snapshot | 59.6 ±0.7plus-or-minus0.7\scriptstyle\pm 0.7 | 60.3 |
| w/w/ TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} (K=1)K1\mathrm{(K=1)} | 63.3 ±0.6plus-or-minus0.6\scriptstyle\pm 0.6 | 64.2 |
| −- content snapshot | 60.4 ±1.3plus-or-minus1.3\scriptstyle\pm 1.3 | 61.8 |
| w/w/ TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} (K=3)K3\mathrm{(K=3)} | 63.3 ±0.7plus-or-minus0.7\scriptstyle\pm 0.7 | 64.1 |
| \hdashlinew/w/ BertLargesubscriptBertLarge\textsc{Bert}\_{\textrm{Large}} (K=1)K1\mathrm{(K=1)} | 61.3 ±1.2plus-or-minus1.2\scriptstyle\pm 1.2 | 62.9 |
| w/w/ TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}} (K=1)K1\mathrm{(K=1)} | 64.0 ±0.4plus-or-minus0.4\scriptstyle\pm 0.4 | 64.4 |
| w/w/ TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}} (K=3)K3\mathrm{(K=3)} | 64.5 ±0.6plus-or-minus0.6\scriptstyle\pm 0.6 | 65.2 |

Table 2: Exact match accuracies on the public development set of Spider. Models are evaluated with 5 random runs.

[Tab. 1](#S5.T1 "Table 1 ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") and [Tab. 2](#S5.T2 "Table 2 ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") summarize the end-to-end evaluation results on WikiTableQuestions and Spider, respectively.
First, comparing with existing strong semantic parsing systems, we found our parsers with TaBert as the utterance and table encoder perform competitively.
On the test set of WikiTableQuestions, MAPO augmented with a TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}} model with three-row content snapshots, TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}}(K=3)K3\mathrm{(K=3)}, registers a single-model exact-match accuracy of 52.3%, surpassing the previously best ensemble system (46.9%) from Agarwal et al. ([2019](#bib.bib1)) by 5.4% absolute.

On Spider, our semantic parser based on TranX and SemQL ([§ 4.1](#S4.SS1 "4.1 Supervised Semantic Parsing ‣ 4 Example Application: Semantic Parsing over Tables ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")) is conceptually similar to the base version of IRNet as both systems use the SemQL grammar, while our system has a simpler decoder.
Interestingly, we observe that its performance with BertBasesubscriptBertBase\textsc{Bert}\_{\textrm{Base}} (61.8%) matches the full BERT-augmented IRNet model with a stronger decoder using augmented memory and coarse-to-fine decoding (61.9%).
This confirms that our base parser is an effective baseline.
Augmented with representations produced by TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}}(K=3)K3\mathrm{(K=3)}, our parser achieves up to 65.2% exact-match accuracy, a 2.8% increase over the base model using BertBasesubscriptBertBase\textsc{Bert}\_{\textrm{Base}}.
Note that while other competitive systems on the leaderboard use BERT with more sophisticated semantic parsing models, our best Dev. result is already close to the score registered by the best submission (RyanSQL++Bert).
This suggests that if they instead used TaBert as the representation layer, they would see further gains.

Comparing semantic parsers augmented with TaBert and Bert, we found TaBert is more effective across the board.
We hypothesize that the performance improvements would be attributed by two factors.
First, pre-training on large parallel textual and tabular corpora helps TaBert learn to encode structure-rich tabular inputs in their linearized form (Eq. ([1](#S3.E1 "Equation 1 ‣ Row Linearization ‣ 3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"))), whose format is different from the ordinary natural language data that Bert is trained on.
Second, pre-training on parallel data could also helps the model produce representations that better capture the alignment between an utterance
and the relevant information presented in the structured schema,
which is important for semantic parsing.

Overall, the results on the two benchmarks demonstrate that pretraining on aligned textual and tabular data is necessary for joint understanding of NL utterances and tables,
and TaBert works well with both structured (Spider) and semi-structured (WikiTableQuestions) DBs, and agnostic of the task-specific structures of semantic parsers.

#### Effect of Content Snapshots

In this paper we propose using content snapshots to capture the information in input DB tables that is most relevant to answering the NL utterance.
We therefore study the effectiveness of including content snapshots when generating schema representations.
We include in [Tab. 1](#S5.T1 "Table 1 ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") and [Tab. 2](#S5.T2 "Table 2 ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") results of models without using content in row linearization (“−-content snapshot”).
Under this setting a column is represented as “Column Name | Type” without cell values (c.f., Eq. ([1](#S3.E1 "Equation 1 ‣ Row Linearization ‣ 3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"))).
We find that content snapshots are helpful for both Bert and TaBert, especially for TaBert.
As discussed in [§ 3.1](#S3.SS1 "3.1 Computing Representations for NL Utterances and Table Schemas ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"), encoding sampled values from columns in learning their representations helps the model infer alignments between entity and relational phrases in the utterance and the corresponding column.
This is particularly helpful for identifying relevant columns from a DB table that is mentioned in the input utterance.
As an example, empirically we observe that on Spider our semantic parser with TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} using just one row of content snapshots (K=1)K1\mathrm{(K=1)} registers a higher accuracy of selecting the correct columns when generating SQL queries (*e.g.*, columns in SELECT and WHERE clauses), compared to the TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} model without encoding content information (87.4% v.s. 86.4%).

| 𝒖𝒖\bm{u}: How many years before was the film Bacchae out before the Watermelon? | | | |
| --- | --- | --- | --- |
| Input to TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}} (K=3)K3\mathrm{(K=3)} ▷▷\triangleright Content Snapshot with Three Rows | | | |
| Film | Year | Function | Notes |
| \hdashlineThe Bacchae | 2002 | Producer | Screen adaptation of… |
| The Trojan Women | 2004 | Producer/Actress | Documutary film… |
| The Watermelon | 2008 | Producer | Oddball romantic comedy… |
| Input to TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}} (K=1)K1\mathrm{(K=1)} ▷▷\triangleright Content Snapshot with One Synthetic Row | | | |
| Film | Year | Function | Notes |
| \hdashlineThe Watermelon | 2013 | Producer | Screen adaptation of… |

Table 3: Content snapshots generated by two models for a WikiTableQuestions Dev. example. Matched tokens between the question and content snapshots are underlined.

Additionally, comparing TaBert using one synthetic row (K=1)K1\mathrm{(K=1)} and three rows from input tables (K=3)K3\mathrm{(K=3)} as content snapshots, the latter generally performs better.
Intuitively, encoding more table contents relevant to the input utterance could potentially help answer questions that involve reasoning over information across multiple rows in the table.
[Tab. 3](#S5.T3 "Table 3 ‣ Effect of Content Snapshots ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") shows such an example, and to answer this question a parser need to subtract the values of Year in the rows for “The Watermelon” and “The Bacchae”.
TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}} (K=3)K3\mathrm{(K=3)} is able to capture the two target rows in its content snapshot and generates the correct DB query, while the TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}}(K=1)K1\mathrm{(K=1)} model with only one row as content snapshot fails to answer this example.

#### Effect of Row Linearization

TaBert uses row linearization to represent a table row as sequential input to Transformer.
[§ 5.1](#S5.SS1.SSS0.Px2 "Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") (upper half) presents results using various linearization methods.
We find adding type information and content snapshots improves performance, as they provide more hints about the meaning of a column.

We also compare with existing linearization methods in literature using a TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} model, with results shown in [§ 5.1](#S5.SS1.SSS0.Px2 "Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") (lower half).
Hwang et al. ([2019](#bib.bib15)) uses BERT to encode concatenated column names to learn column representations.
In line with our previous discussion on the effectiveness content snapshots,
this simple strategy without encoding cell contents underperforms
(although with TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} pretrained on our tabular corpus the results become slightly better).
Additionally, we remark that linearizing table contents has also be applied to other BERT-based tabular reasoning tasks.
For instance, Chen et al. ([2019](#bib.bib6)) propose a “natural” linearization approach for checking if an NL statement entails the factual information listed in a table using a binary classifier with representations from Bert, where a table is linearized by concatenating the semicolon-separated cell linearization for all rows. Each cell is represented by a phrase “column name is cell value”.
For completeness, we also tested this cell linearization approach,
and find BertBasesubscriptBertBase\textsc{Bert}\_{\textrm{Base}} achieved improved results.
We leave pretraining TaBert with this linearization strategy as promising future work.

|  |  |  |
| --- | --- | --- |
| Cell Linearization Template | WikiQ. | Spider |
| Pretrained TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} Models (K=1)K1\mathrm{(K=1)} | | |
| Column Name | 49.6 ±0.4plus-or-minus0.4\scriptstyle\pm 0.4 | 60.0 ±1.1plus-or-minus1.1\scriptstyle\pm 1.1 |
| Column Name | Type† (−-content snap.) | 49.9 ±0.4plus-or-minus0.4\scriptstyle\pm 0.4 | 60.4 ±1.3plus-or-minus1.3\scriptstyle\pm 1.3 |
| Column Name | Type | Cell Value† | 51.2 ±0.5plus-or-minus0.5\scriptstyle\pm 0.5 | 63.3 ±0.6plus-or-minus0.6\scriptstyle\pm 0.6 |
| \hdashline      BertBasesubscriptBertBase\textsc{Bert}\_{\textrm{Base}} Models | | |
| Column Name Hwang et al. ([2019](#bib.bib15)) | 49.0 ±0.4plus-or-minus0.4\scriptstyle\pm 0.4 | 58.6 ±0.3plus-or-minus0.3\scriptstyle\pm 0.3 |
| Column Name is Cell Value [(Chen19)](#TabFact "TabFactIn References ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") | 50.2 ±0.4plus-or-minus0.4\scriptstyle\pm 0.4 | 63.1 ±0.7plus-or-minus0.7\scriptstyle\pm 0.7 |

Table 4: Performance of pretrained TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}} models and BertBasesubscriptBertBase\textsc{Bert}\_{\textrm{Base}} on the Dev. sets with different linearization methods. Slot names are underlined. †Results copied from [Tab. 1](#S5.T1 "Table 1 ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") and [Tab. 2](#S5.T2 "Table 2 ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data").

| Learning Objective | WikiQ. | Spider |
| --- | --- | --- |
| MCP only | 51.6 ±0.7plus-or-minus0.7\scriptstyle\pm 0.7 | 62.6 ±0.7plus-or-minus0.7\scriptstyle\pm 0.7 |
| MCP + CVR | 51.6 ±0.5plus-or-minus0.5\scriptstyle\pm 0.5 | 63.3 ±0.7plus-or-minus0.7\scriptstyle\pm 0.7 |

Table 5: Performance of pretrained TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}}(K=3)K3\mathrm{(K=3)} on Dev. sets with different pretraining objectives.

#### Impact of Pretraining Objectives

TaBert uses two objectives ([§ 3.2](#S3.SS2 "3.2 Pretraining Procedure ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")), a masked column prediction (MCP) and a cell value recovery (CVR) objective, to learn column representations that could capture both the general information of the column (via MCP) and its representative cell values related to the utterance (via CVR).
[Tab. 5](#S5.T5 "Table 5 ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") shows ablation results of pretraining TaBert with different objectives.
We find TaBert trained with both MCP and the auxiliary CVR objectives gets a slight advantage, suggesting CVR could potentially lead to more representative column representations with additional cell information.

## 6 Related Works

#### Semantic Parsing over Tables

Tables are important media of world knowledge.
Semantic parsers have been adapted to operate over structured DB tables Wang et al. ([2015](#bib.bib39)); Xu et al. ([2017](#bib.bib40)); Dong and Lapata ([2018](#bib.bib10)); Yu et al. ([2018b](#bib.bib45)); Shi et al. ([2018](#bib.bib30)); Wang et al. ([2018](#bib.bib37)), and open-domain, semi-structured Web tables Pasupat and Liang ([2015](#bib.bib26)); Sun et al. ([2016](#bib.bib31)); Neelakantan et al. ([2016](#bib.bib25)).
To improve representations of utterances and tables for neural semantic parsing, existing systems have applied pretrained word embeddings (*e.g.*., GloVe, as in Zhong et al. ([2017](#bib.bib52)); Yu et al. ([2018a](#bib.bib44)); Sun et al. ([2018](#bib.bib32)); Liang et al. ([2018](#bib.bib19))), and BERT-family models for learning joint contextual representations of utterances and tables, but with domain-specific approaches to encode the structured information in tables Hwang et al. ([2019](#bib.bib15)); He et al. ([2019](#bib.bib13)); Guo et al. ([2019](#bib.bib12)); Zhang et al. ([2019a](#bib.bib48)).
TaBert advances this line of research by presenting a general-purpose, pretrained encoder over parallel corpora of Web tables and NL context.
Another relevant direction is to augment representations of columns from an individual table with global information of its linked tables defined by the DB schema Bogin et al. ([2019a](#bib.bib4)); Wang et al. ([2019a](#bib.bib35)).
TaBert could also potentially improve performance of these systems with improved table-level representations.

#### Knowledge-enhanced Pretraining

Recent pre-training models have incorporated structured information from knowledge bases (KBs) or other structured semantic annotations
into training contextual word representations, either by fusing vector representations of entities and relations on KBs into word representations of LMs Peters et al. ([2019](#bib.bib28)); Zhang et al. ([2019b](#bib.bib50), [c](#bib.bib51)), or by encouraging the LM to recover KB entities and relations from text Sun et al. ([2019](#bib.bib33)); Liu et al. ([2019a](#bib.bib21)).
TaBert is broadly relevant to this line in that it also exposes an LM with structured data (*i.e.*, tables), while aiming to learn joint representations for both textual and structured tabular data.

## 7 Conclusion and Future Work

We present TaBert, a pretrained encoder for joint understanding of textual and tabular data.
We show that semantic parsers using TaBert as a general-purpose feature representation layer achieved strong results on two benchmarks.
This work also opens up several avenues for future work.
First, we plan to evaluate TaBert on other related tasks involving joint reasoning over textual and tabular data (*e.g.*, table retrieval and table-to-text generation).
Second, following the discussions in [§ 5](#S5 "5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"), we will explore other table linearization strategies with Transformers, improving the quality of pretraining corpora, as well as novel unsupervised objectives.
Finally, to extend TaBert to cross-lingual settings with utterances in foreign languages and structured schemas defined in English, we plan to apply more advanced semantic similarity metrics for creating content snapshots.

## References

* Agarwal et al. (2019)

  Rishabh Agarwal, Chen Liang, Dale Schuurmans, and Mohammad Norouzi. 2019.
  Learning to generalize from sparse and underspecified rewards.
  In *ICML*.
* Berant et al. (2013)

  Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013.
  Semantic parsing on freebase from question-answer pairs.
  In *Proceedings of EMNLP*.
* Berant and Liang (2014)

  Jonathan Berant and Percy Liang. 2014.
  Semantic parsing via paraphrasing.
  In *Proceedings of ACL*.
* Bogin et al. (2019a)

  Ben Bogin, Matt Gardner, and Jonathan Berant. 2019a.
  Global reasoning over database structures for text-to-sql parsing.
  *ArXiv*, abs/1908.11214.
* Bogin et al. (2019b)

  Ben Bogin, Matthew Gardner, and Jonathan Berant. 2019b.
  Representing schema structure with graph neural networks for
  text-to-sql parsing.
  In *Proceedings of ACL*.
* Chen et al. (2019)

  Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li,
  Xiyou Zhou, and William Yang Wang. 2019.
  TabFact: A large-scale dataset for table-based
  fact verification.
  *ArXiv*, abs/1909.02164.
* Choi et al. (2020)

  Donghyun Choi, Myeong Cheol Shin, EungGyun Kim, and Dong Ryeol Shin. 2020.
  Ryansql: Recursively applying sketch-based slot fillings for complex
  text-to-sql in cross-domain databases.
  *ArXiv*, abs/2004.03125.
* Dasigi et al. (2019)

  Pradeep Dasigi, Matt Gardner, Shikhar Murty, Luke S. Zettlemoyer, and Eduard H.
  Hovy. 2019.
  Iterative search for weakly supervised semantic parsing.
  In *Proceedings of NAACL-HLT*.
* Devlin et al. (2019)

  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.
  Bert: Pre-training of deep bidirectional transformers for language
  understanding.
  In *Proceedings of NAACL-HLT*.
* Dong and Lapata (2018)

  Li Dong and Mirella Lapata. 2018.
  coarse-to-fine decoding for neural semantic
  parsing.
  In *Proceedings of ACL*.
* Goldberg (2019)

  Yoav Goldberg. 2019.
  Assessing bert’s syntactic abilities.
  *ArXiv*, abs/1901.05287.
* Guo et al. (2019)

  Jiaqi Guo, Zecheng Zhan, Yan Gao, Yan Xiao, Jian-Guang Lou, Ting Liu, and
  Dongmei Zhang. 2019.
  Towards complex text-to-sql in cross-domain database with
  intermediate representation.
  In *Proceedings of ACL*.
* He et al. (2019)

  Pengcheng He, Yi Mao, Kaushik Chakrabarti, and Weizhu Chen. 2019.
  X-sql: reinforce schema representation with context.
  *ArXiv*, abs/1908.08113.
* Hendrycks and Gimpel (2016)

  Dan Hendrycks and Kevin Gimpel. 2016.
  Gaussian error linear units (gelus).
  *ArXiv*, abs/1606.08415.
* Hwang et al. (2019)

  Wonseok Hwang, Jinyeung Yim, Seunghyun Park, and Minjoon Seo. 2019.
  A comprehensive exploration on wikisql with table-aware word
  contextualization.
  *ArXiv*, abs/1902.01069.
* Joshi et al. (2019)

  Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke S. Zettlemoyer, and
  Omer Levy. 2019.
  Spanbert: Improving pre-training by representing and predicting
  spans.
  In *Proceedings of EMNLP*.
* Krishnamurthy et al. (2017)

  Jayant Krishnamurthy, Pradeep Dasigi, and Matt Gardner. 2017.
  Neural semantic parsing with type constraints for semi-structured
  tables.
  In *Proceedings of EMNLP*.
* Lehmberg et al. (2016)

  Oliver Lehmberg, Dominique Ritze, Robert Meusel, and Christian Bizer. 2016.
  A large public corpus of web tables containing time and context
  metadata.
  In *Proceedings of WWW*.
* Liang et al. (2018)

  Chen Liang, Mohammad Norouzi, Jonathan Berant, Quoc V Le, and Ni Lao. 2018.
  Memory augmented policy optimization for program synthesis and
  semantic parsing.
  In *Proceedings of NIPS*.
* Libovický and Helcl (2017)

  Jindrich Libovický and Jindrich Helcl. 2017.
  Attention strategies for multi-source sequence-to-sequence learning.
  In *Proceedings of ACL*.
* Liu et al. (2019a)

  Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, and Ping
  Wang. 2019a.
  K-bert: Enabling language representation with knowledge graph.
  *ArXiv*, abs/1909.07606.
* Liu et al. (2019b)

  Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
  Levy, Mike Lewis, Luke S. Zettlemoyer, and Veselin Stoyanov.
  2019b.
  Roberta: A robustly optimized bert pretraining approach.
  *ArXiv*, abs/1907.11692.
* McCann et al. (2017)

  Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 2017.
  Learned in translation: Contextualized word vectors.
  In *Proceedings of NIPS*.
* Melamud et al. (2016)

  Oren Melamud, Jacob Goldberger, and Ido Dagan. 2016.
  context2vec: Learning generic context embedding with bidirectional
  LSTM.
  In *Proceedings of CoNLL*.
* Neelakantan et al. (2016)

  Arvind Neelakantan, Quoc V. Le, and Ilya Sutskever. 2016.
  Neural programmer: Inducing latent programs with gradient descent.
  In *Proceedings of ICLR*.
* Pasupat and Liang (2015)

  Panupong Pasupat and Percy Liang. 2015.
  Compositional semantic parsing on semi-structured tables.
  In *Proceedings of ACL*.
* Peters et al. (2018)

  Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark,
  Kenton Lee, and Luke S. Zettlemoyer. 2018.
  Deep contextualized word representations.
  In *Proceedings of NAACL*.
* Peters et al. (2019)

  Matthew E. Peters, Mark Neumann, IV RobertLLogan, Roy Schwartz, Vidur Joshi,
  Sameer Singh, and Noah A. Smith. 2019.
  Knowledge enhanced contextual word representations.
  In *Proceedings of EMNLP*.
* Rajpurkar et al. (2016)

  Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.
  Squad: 100, 000+ questions for machine comprehension of text.
  In *Proceedings of EMNLP*.
* Shi et al. (2018)

  Tianze Shi, Kedar Tatwawadi, Kaushik Chakrabarti, Yi Mao, Oleksandr Polozov,
  and Weizhu Chen. 2018.
  Incsql: Training incremental text-to-sql parsers with
  non-deterministic oracles.
  *ArXiv*, abs/1809.05054.
* Sun et al. (2016)

  Huan Sun, Hao Ma, Xiaodong He, Wen tau Yih, Yu Su, and Xifeng Yan. 2016.
  Table cell search for question answering.
  In *Proceedings of WWW*.
* Sun et al. (2018)

  Yibo Sun, Duyu Tang, Nan Duan, Jianshu Ji, Guihong Cao, Xiaocheng Feng, Bing
  Qin, Ting Liu, and Ming Zhou. 2018.
  Semantic parsing with syntax- and table-aware SQL generation.
  In *Proceedings of EMNLP*.
* Sun et al. (2019)

  Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian,
  Danxiang Zhu, Hao Tian, and Hua Wu. 2019.
  Ernie: Enhanced representation through knowledge integration.
  *ArXiv*, abs/1904.09223.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
  Attention is all you need.
  In *Proceedings of NIPS*.
* Wang et al. (2019a)

  Bailin Wang, Richard Shin, Xiaodong Liu, Oleksandr Polozov, and Margot
  Richardson. 2019a.
  Rat-sql: Relation-aware schema encoding and linking for text-to-sql
  parsers.
  *ArXiv*, abs/1911.04942.
* Wang et al. (2019b)

  Bailin Wang, Ivan Titov, and Mirella Lapata. 2019b.
  Learning semantic parsers from denotations with latent structured
  alignments and abstract programs.
  In *EMNLP/IJCNLP*.
* Wang et al. (2018)

  Chenglong Wang, Kedar Tatwawadi, Marc Brockschmidt, Po-Sen Huang, Yi Xin Mao,
  Oleksandr Polozov, and Rishabh Singh. 2018.
  Robust text-to-sql generation with execution-guided decoding.
  *ArXiv*, abs/1807.03100.
* Wang et al. (1997)

  Daniel C. Wang, Andrew W. Appel, Jeffrey L. Korn, and Christopher S. Serra.
  1997.
  The Zephyr abstract syntax description language.
  In *Proceedings of DSL*.
* Wang et al. (2015)

  Yushi Wang, Jonathan Berant, and Percy Liang. 2015.
  Building a semantic parser overnight.
  In *Proceedings of ACL*.
* Xu et al. (2017)

  Xiaojun Xu, Chang Liu, and Dawn Song. 2017.
  SQLNet: Generating structured queries from natural language without
  reinforcement learning.
  *arXiv*, abs/1711.04436.
* Yang et al. (2019)

  Zhilin Yang, Zihang Dai, Yiming Yang, Jaime G. Carbonell, Ruslan Salakhutdinov,
  and Quoc V. Le. 2019.
  Xlnet: Generalized autoregressive pretraining for language
  understanding.
  In *Proceedings of NIPS*.
* Yih et al. (2015)

  Wen-tau Yih, Ming-Wei Chang, Xiaodong He, and Jianfeng Gao. 2015.
  Semantic parsing via staged query graph generation: Question
  answering with knowledge base.
  In *Proceedings of ACL*.
* Yin and Neubig (2018)

  Pengcheng Yin and Graham Neubig. 2018.
  TRANX: A transition-based neural abstract
  syntax parser for semantic parsing and code generation.
  In *Proceedings of EMNLP Demonstration Track*.
* Yu et al. (2018a)

  Tao Yu, Zifan Li, Zilin Zhang, Rui Zhang, and Dragomir R. Radev.
  2018a.
  TypeSQL: Knowledge-based type-aware neural text-to-sql generation.
  In *Proceedings of NAACL-HLT*.
* Yu et al. (2018b)

  Tao Yu, Michihiro Yasunaga, Kai Yang, Rui Zhang, Dongxu Wang, Zifan Li, and
  Dragomir R. Radev. 2018b.
  Syntaxsqlnet: Syntax tree networks for complex and cross-domain
  text-to-sql task.
  In *Proceedings of EMNLP*.
* Yu et al. (2018c)

  Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James
  Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir R.
  Radev. 2018c.
  Spider: A large-scale human-labeled dataset for complex and
  cross-domain semantic parsing and text-to-sql task.
  In *Proceedings of EMNLP*.
* Zelle and Mooney (1996)

  John M. Zelle and Raymond J. Mooney. 1996.
  Learning to parse database queries using inductive logic programming.
  In *Proceedings of AAAI*.
* Zhang et al. (2019a)

  Rui Zhang, Tao Yu, He Yang Er, Sungrok Shim, Eric Xue, Xi Victoria Lin, Tianze
  Shi, Caiming Xiong, Richard Socher, and Dragomir R. Radev.
  2019a.
  Editing-based sql query generation for cross-domain context-dependent
  questions.
  *ArXiv*, abs/1909.00786.
* Zhang et al. (2017)

  Yuchen Zhang, Panupong Pasupat, and Percy Liang. 2017.
  Macro grammars and holistic triggering for efficient semantic
  parsing.
  In *Proceedings of EMNLP*.
* Zhang et al. (2019b)

  Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, and Qun Liu.
  2019b.
  Ernie: Enhanced language representation with informative entities.
  In *Proceedings of ACL*.
* Zhang et al. (2019c)

  Zhuosheng Zhang, Yu-Wei Wu, Hai Zhao, Zuchao Li, Shuailiang Zhang, Xi Zhou, and
  Xiaodong Zhou. 2019c.
  Semantics-aware bert for language understanding.
  *ArXiv*, abs/1909.02209.
* Zhong et al. (2017)

  Victor Zhong, Caiming Xiong, and Richard Socher. 2017.
  Seq2SQL: Generating structured queries from natural language using
  reinforcement learning.
  *arXiv*, abs/1709.00103.

TaBert: Pretraining for Joint Understanding of Textual and Tabular Data

Supplementary Materials

## Appendix A Pretraining Details

### A.1 Training Data

We collect parallel examples of tables and their surrounding NL sentences from two sources:

#### Wikipedia Tables

We extract all the tables
on English Wikipedia101010We do not use infoboxes (tables on the top-right of a Wiki page that describe properties of the main topic), as they are not relational tables.. For each table, we use the preceding three paragraphs as the NL context, as we observe that most Wiki tables are located after where they are described in the body text.

#### WDC WebTable Corpus

Lehmberg et al. ([2016](#bib.bib18)) is a large collection of Web tables extracted from the Common Crawl Web scrape111111<http://webdatacommons.org/webtables>. We use its 2015 English-language relational subset, which consists of 50.850.850.8 million relational tables and their surrounding NL contexts.

#### Preprocessing

Our dataset is collected from arbitrary Web tables, which are extremely noisy.
We develop a set of heuristics to clean the data by:
(1) removing columns whose names have more than 10 tokens;
(2) filtering cells with more than two non-ASCII characters or 20 tokens;
(3) removing empty or repetitive rows and columns;
(4) filtering tables with less than three rows and four columns, and
(5) running spaCy to identify the data type of columns (text or real value) by majority voting over the NER labels of column tokens,
(6) rotating vertically oriented tables.
We sub-tokenize the corpus using the Wordpiece tokenizer in Devlin et al. ([2019](#bib.bib9)).
The pre-processing results in 1.3 million tables from Wikipedia and 25.3 million tables from the WDC corpus.

### A.2 Pretraining Setup

As discussed in [§ 5](#S5 "5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"), we create training instances of NL sentences (as synthetic utterances) and content snapshots from tables by sampling from the parallel corpus of NL contexts and tables.
Each epoch contains 37.6M𝑀M training instances.
We train TaBert for 10 epochs.
[Tab. 6](#A1.T6 "Table 6 ‣ A.2 Pretraining Setup ‣ Appendix A Pretraining Details ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") lists the hyper-parameters used in training.
Learning rates are validated on the development set of WikiTableQuestions.
We use a batch size of 512 for large models to reduce training time.
The training objective is sum of the three pretraining objectives in [§ 3.2](#S3.SS2 "3.2 Pretraining Procedure ‣ 3 TaBert: Learning Joint Representa- tions over Textual and Tabular Data ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") (Masked Language Modeling objective for utterance tokens, Masked Column Prediction121212An exception is that for pretraining TaBert(K=1)K1\mathrm{(K=1)} models, the masked column prediction objective reduces to the vanilla masked language modeling objective since there are no additional vertical attention layers. and Column Value Recovery objectives for columns and their cell values).
Our largest model TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}}(K=3)K3\mathrm{(K=3)} takes six days to train for 10 epochs on 128 Tesla V100 GPUs using mixed precision training.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Parameter | TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}}(K=1)K1\mathrm{(K=1)} | TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}}(K=1)K1\mathrm{(K=1)} | TaBertBasesubscriptTaBertBase\textsc{TaBert}\_{\textrm{Base}}(K=3)K3\mathrm{(K=3)} | TaBertLargesubscriptTaBertLarge\textsc{TaBert}\_{\textrm{Large}}(K=3)K3\mathrm{(K=3)} |
| Batch Size | 256 | 512 | 512 | 512 |
| Learning Rate | 2×10−52superscript1052\times 10^{-5} | 2×10−52superscript1052\times 10^{-5} | 4×10−54superscript1054\times 10^{-5} | 4×10−54superscript1054\times 10^{-5} |
| Max Epoch | 10 | | | |
| Weight Decay | 0.01 | | | |
| Gradient Norm Clipping | 1.0 | | | |

Table 6: Hyper-parameters using in pretraining

## Appendix B Semantic Parsers

### B.1 Supervised Parsing on Spider

#### Model

We develop our text-to-SQL parser based on TranX Yin and Neubig ([2018](#bib.bib43)), which translates an NL utterance into a tree-structured abstract meaning representation following user-specified grammar, before deterministically convert the generated abstract MR into an SQL query. TranX models the construction process of an abstract MR (tree-structured representation of an SQL query) using a transition-based system, which decomposes its generation story into a sequence of actions following the user defined grammar.

Formally, given an input NL utterance 𝒖𝒖\bm{u} and a database with a set of tables 𝒯={Ti}𝒯subscript𝑇𝑖\mathcal{T}=\{{T}\_{i}\}, the probability of generating of an SQL query (*i.e.*, its semantically equivalent MR)
𝒛𝒛\bm{z} is decomposed as the production of action probabilities:

|  |  |  |  |
| --- | --- | --- | --- |
|  | p​(𝒛|𝒖,𝒯)=∏p​(at|a<t,𝒖,𝒯)𝑝conditional𝒛  𝒖𝒯product𝑝conditionalsubscript𝑎𝑡  subscript𝑎absent𝑡𝒖𝒯p(\bm{z}|\bm{u},\mathcal{T})=\prod p(a\_{t}|a\_{<t},\bm{u},\mathcal{T}) |  | (2) |

where atsubscript𝑎𝑡a\_{t} is the action applied to the hypothesis at time stamp t𝑡t. a<tsubscript𝑎absent𝑡a\_{<t} denote the previous action history.
We refer readers to Yin and Neubig ([2018](#bib.bib43)) for details of the transition system and how individual action probabilities are computed.
In our adaptation of TranX to text-to-SQL parsing on Spider,
we follow Guo et al. ([2019](#bib.bib12)) and use SemQL as the underlying grammar, which is a simplification of the SQL language.
[Fig. 2](#A2.F2 "Figure 2 ‣ Model ‣ B.1 Supervised Parsing on Spider ‣ Appendix B Semantic Parsers ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") lists the SemSQL grammar specified using the abstract syntax description language Wang et al. ([1997](#bib.bib38)).
Intuitively, the generation starts from a tree-structured derivation with the root production rule select\_stmt↦maps-to\mapstoSelectStatement, which lays out overall the structure of an SQL query.
At each time step, the decoder algorithm locates the current opening node on the derivation tree, following a depth-first, left-to-right order.
If the opening node is not a leaf node, the decoder invokes an action atsubscript𝑎𝑡a\_{t} which expands the opening node using a production rule with appropriate type.
If the current opening node is a leaf node (*e.g.*, a node denoting string literal),
the decoder fills in the leaf node using actions that emit terminal values.

[⬇](data:text/plain;base64,c2VsZWN0X3N0bXQgPSBTZWxlY3RTdGF0ZW1lbnQoCiAgICBkaXN0aW5jdCBkaXN0aW5jdCwgICAgICAgICAgICAgICAgICAmXENvbW1lbnR7RElTVElOQ1Qga2V5d29yZH0mCiAgICBleHByKiByZXN1bHRfY29sdW1ucywgICAgICAgICAgICAgICAmXENvbW1lbnR7Q29sdW1ucyBpbiBTRUxFQ1QgY2xhdXNlfSYKICAgIGV4cHI/IHdoZXJlX2NsYXVzZSwgICAgICAgICAgICAgICAgICZcQ29tbWVudHtXSEVSRSBjbGF1c2V9JgogICAgb3JkZXJfYnlfY2xhdXNlPyBvcmRlcl9ieV9jbGF1c2UsICAgJlxDb21tZW50e09SREVSIEJZIGNsYXVzZX0mCiAgICBpbnQ/IGxpbWl0X3ZhbHVlLCAgICAgICAgICAgICAgICAgICAmXENvbW1lbnR7TElNSVQgY2xhdXNlfSYKICAgIHRhYmxlX3JlZiogam9pbl93aXRoX3RhYmxlcywgICAgICAgICZcQ29tbWVudHtUYWJsZXMgaW4gdGhlIEpPSU4gY2xhdXNlfSYKICAgIGNvbXBvdW5kX3N0bXQ/IGNvbXBvdW5kX3N0YXRlbWVudCAgICZcQ29tbWVudHtDb21wb3VuZCBzdGF0ZW1lbnRzIChcZWcsIFVOSU9OLCBFWENFUFQpfSYKKQoKZGlzdGluY3QgPSBOb25lIHwgRGlzdGluY3QKCm9yZGVyX2J5X2NsYXVzZSA9IE9yZGVyQnlDbGF1c2UoZXhwciogZXhwcl9saXN0LCBvcmRlciBvcmRlcikKCm9yZGVyID0gQVNDIHwgREVTQwoKZXhwciA9IEFuZEV4cHIoZXhwciogZXhwcl9saXN0KQogICAgICB8IE9yRXhwcihleHByKiBleHByX2xpc3QpCiAgICAgIHwgTm90RXhwcihleHByIGV4cHIpCiAgICAgIHwgQ29tcGFyZUV4cHIoY29tcGFyZV9vcCBvcCwgZXhwciBsZWZ0X3ZhbHVlLCBleHByIHJpZ2h0X3ZhbHVlKQogICAgICB8IEFnZ3JlZ2F0ZUV4cHIoYWdncmVnYXRlX29wIG9wLCBleHByIHZhbHVlLCBkaXN0aW5jdCBkaXN0aW5jdCkKICAgICAgfCBCaW5hcnlFeHByKGJpbmFyeV9vcCBvcCwgZXhwciBsZWZ0X3ZhbHVlLCBleHByIHJpZ2h0X3ZhbHVlKQogICAgICB8IEJldHdlZW5FeHByKGV4cHIgZmllbGQsIGV4cHIgbGVmdF92YWx1ZSwgZXhwciByaWdodF92YWx1ZSkKICAgICAgfCBJbkV4cHIoY29sdW1uX3JlZiBsZWZ0X3ZhbHVlLCBleHByIHJpZ2h0X3ZhbHVlKQogICAgICB8IExpa2VFeHByKGNvbHVtbl9yZWYgbGVmdF92YWx1ZSwgZXhwciByaWdodF92YWx1ZSkKICAgICAgfCBBbGxSb3dzKHRhYmxlX3JlZiB0YWJsZV9uYW1lKQogICAgICB8IHNlbGVjdF9zdG10CiAgICAgIHwgTGl0ZXJhbChzdHJpbmcgdmFsdWUpCiAgICAgIHwgQ29sdW1uUmVmZXJlbmNlKGNvbHVtbl9yZWYgY29sdW1uX25hbWUpCgphZ2dyZWdhdGVfb3AgPSBTdW0gfCBNYXggfCBNaW4gfCBDb3VudCB8IEF2ZwoKY29tcGFyZV9vcCA9IExlc3NUaGFuIHwgTGVzc1RoYW5FcXVhbCB8IEdyZWF0ZXJUaGFuCiAgICAgICAgICAgIHwgR3JlYXRlclRoYW5FcXVhbCB8IEVxdWFsIHwgTm90RXF1YWwKCmJpbmFyeV9vcCA9IEFkZCB8IFN1YiB8IERpdmlkZSB8IE11bHRpcGx5Cgpjb21wb3VuZF9zdG10ID0gQ29tcG91bmRTdGF0ZW1lbnQoY29tcG91bmRfb3Agb3AsIHNlbGVjdF9zdG10IHF1ZXJ5KQoKY29tcG91bmRfb3AgPSBVbmlvbiB8IEludGVyc2VjdCB8IEV4Y2VwdA==)

select\_stmt = SelectStatement(

distinct distinct,  # DISTINCT keyword

expr\* result\_columns,  # Columns in SELECT clause

expr? where\_clause,  # WHERE clause

order\_by\_clause? order\_by\_clause,  # ORDER BY clause

int? limit\_value,  # LIMIT clause

table\_ref\* join\_with\_tables,  # Tables in the JOIN clause

compound\_stmt? compound\_statement  # Compound statements (*e.g.*, UNION, EXCEPT)

)

distinct = None | Distinct

order\_by\_clause = OrderByClause(expr\* expr\_list, order order)

order = ASC | DESC

expr = AndExpr(expr\* expr\_list)

| OrExpr(expr\* expr\_list)

| NotExpr(expr expr)

| CompareExpr(compare\_op op, expr left\_value, expr right\_value)

| AggregateExpr(aggregate\_op op, expr value, distinct distinct)

| BinaryExpr(binary\_op op, expr left\_value, expr right\_value)

| BetweenExpr(expr field, expr left\_value, expr right\_value)

| InExpr(column\_ref left\_value, expr right\_value)

| LikeExpr(column\_ref left\_value, expr right\_value)

| AllRows(table\_ref table\_name)

| select\_stmt

| Literal(string value)

| ColumnReference(column\_ref column\_name)

aggregate\_op = Sum | Max | Min | Count | Avg

compare\_op = LessThan | LessThanEqual | GreaterThan

| GreaterThanEqual | Equal | NotEqual

binary\_op = Add | Sub | Divide | Multiply

compound\_stmt = CompoundStatement(compound\_op op, select\_stmt query)

compound\_op = Union | Intersect | Except

Figure 2: ASDL Grammar of SemQL used in TranX

To use such a transition system to generate SQL queries, we extend its action space with two new types of actions, SelectTable(Ti)subscript𝑇𝑖({T}\_{i}) for node of type table\_ref in [Fig. 2](#A2.F2 "Figure 2 ‣ Model ‣ B.1 Supervised Parsing on Spider ‣ Appendix B Semantic Parsers ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"), which selects a table Tisubscript𝑇𝑖{T}\_{i} (*e.g.*, for predicting target tables for a FROM clause), and SelectColumn(Ti,cj)subscript𝑇𝑖subscript𝑐𝑗({T}\_{i},c\_{j}) for node of type column\_ref, which selects the column cjsubscript𝑐𝑗c\_{j} from table Tisubscript𝑇𝑖{T}\_{i} (*e.g.*, for predicting a result column used in the SELECT clause).

As described in [§ 4.1](#S4.SS1 "4.1 Supervised Semantic Parsing ‣ 4 Example Application: Semantic Parsing over Tables ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"), TaBert produces a list of entries, with one entry ⟨𝐓i,𝐗i,𝐂i⟩

subscript𝐓𝑖subscript𝐗𝑖subscript𝐂𝑖\langle\mathbf{T}\_{i},\mathbf{X}\_{i},\mathbf{C}\_{i}\rangle for each table Tisubscript𝑇𝑖{T}\_{i}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝕄={⟨𝐓i,𝐗i={𝐱1,𝐱2,…},𝐂i={𝐜1,𝐜2,…,}⟩i}i=1|𝒯|\mathbb{M}=\Big{\{}\langle\mathbf{T}\_{i},\mathbf{X}\_{i}=\{\mathbf{x}\_{1},\mathbf{x}\_{2},\ldots\},\mathbf{C}\_{i}=\{\mathbf{c}\_{1},\mathbf{c}\_{2},\ldots,\}\rangle\_{i}\Big{\}}\_{i=1}^{|\mathcal{T}|} |  | (3) |

where each entry ⟨𝐓i,𝐗i,𝐂i⟩

subscript𝐓𝑖subscript𝐗𝑖subscript𝐂𝑖\langle\mathbf{T}\_{i},\mathbf{X}\_{i},\mathbf{C}\_{i}\rangle in 𝕄𝕄\mathbb{M}
consists of 𝐓isubscript𝐓𝑖\mathbf{T}\_{i}, the representation of table Tisubscript𝑇𝑖{T}\_{i} given by the output vector of the prefixed [CLS] symbol, the table-specific representations of utterance tokens 𝐗i={𝐱1,𝐱2,…}subscript𝐗𝑖subscript𝐱1subscript𝐱2…\mathbf{X}\_{i}=\{\mathbf{x}\_{1},\mathbf{x}\_{2},\ldots\},
and representations of columns in Tisubscript𝑇𝑖{T}\_{i}, 𝐂i={𝐜1,𝐜2,…}subscript𝐂𝑖subscript𝐜1subscript𝐜2…\mathbf{C}\_{i}=\{\mathbf{c}\_{1},\mathbf{c}\_{2},\ldots\}.
At each time step t𝑡t, the decoder in TranX performs hierarchical attention over representations in 𝕄𝕄\mathbb{M} to compute a context vector.
First, a table-wise attention score is computed using the LSTM’s previous state, statet−1subscriptstate𝑡1\textbf{state}\_{t-1} with the set of table representations.

|  |  |  |  |
| --- | --- | --- | --- |
|  | score​(Ti)=Softmax​(DotProduct​(𝐬𝐭𝐚𝐭𝐞t−1,key​(𝐓i))),scoresubscript𝑇𝑖SoftmaxDotProductsubscript𝐬𝐭𝐚𝐭𝐞𝑡1keysubscript𝐓𝑖\mathrm{score}({T}\_{i})=\mathrm{Softmax}\Big{(}\mathrm{DotProduct}(\mathbf{state}\_{t-1},\mathrm{key}(\mathbf{T}\_{i}))\Big{)}, |  | (4) |

where the linear projection key​(⋅)∈ℝ256key⋅superscriptℝ256\mathrm{key}(\cdot)\in\mathbb{R}^{256} projects the table representations to key space. Next, for each table Ti∈𝒯subscript𝑇𝑖𝒯{T}\_{i}\in\mathcal{T}, a table-wise context vector 𝐜𝐭𝐱​(Ti)𝐜𝐭𝐱subscript𝑇𝑖\mathbf{ctx}({T}\_{i}) is generated by attending over the union of vectors in utterance token representations 𝐗isubscript𝐗𝑖\mathbf{X}\_{i} and column representations 𝐂isubscript𝐂𝑖\mathbf{C}\_{i}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐜𝐭𝐱​(Ti)=DotProductAttention​(𝐬𝐭𝐚𝐭𝐞t−1,key​(𝐗i∪𝐂i),value​(𝐗i∪𝐂i)),𝐜𝐭𝐱subscript𝑇𝑖DotProductAttentionsubscript𝐬𝐭𝐚𝐭𝐞𝑡1keysubscript𝐗𝑖subscript𝐂𝑖valuesubscript𝐗𝑖subscript𝐂𝑖\mathbf{ctx}({T}\_{i})=\mathrm{DotProductAttention}\Big{(}\mathbf{state}\_{t-1},\mathrm{key}(\mathbf{X}\_{i}\cup\mathbf{C}\_{i}),\mathrm{value}(\mathbf{X}\_{i}\cup\mathbf{C}\_{i})\Big{)}, |  | (5) |

with the LSTM state as the query, key​(⋅)key⋅\mathrm{key}(\cdot) as the key, and another linear transformation value​(⋅)∈ℝ256value⋅superscriptℝ256\mathrm{value}(\cdot)\in\mathbb{R}^{256} to project the representations to value vectors.
The final context vector is then given by the weighted sum of these table-wise context vectors 𝐜𝐭𝐱​(Ti)𝐜𝐭𝐱subscript𝑇𝑖\mathbf{ctx}({T}\_{i}) (i∈{1,…,|𝒯|}𝑖1…𝒯i\in\{1,\ldots,|\mathcal{T}|\}) weighted by the attention scores score​(Ti)scoresubscript𝑇𝑖\mathrm{score}({T}\_{i}). The generated context vector is then used to update the state of the decoder LSTM to 𝐬𝐭𝐚𝐭𝐞tsubscript𝐬𝐭𝐚𝐭𝐞𝑡\mathbf{state}\_{t}.

The updated decoder state is then used to compute the probability of carrying out the action defined at time step t𝑡t, atsubscript𝑎𝑡a\_{t}.
For a SelectTable(Ti)subscript𝑇𝑖({T}\_{i}) action, its probability of is defined similarly as Eq. ([4](#A2.E4 "Equation 4 ‣ Model ‣ B.1 Supervised Parsing on Spider ‣ Appendix B Semantic Parsers ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data")).
For a SelectColumn(Ti,cj)subscript𝑇𝑖subscript𝑐𝑗({T}\_{i},c\_{j}) action, it is factorized as the probability of selecting the table Tisubscript𝑇𝑖{T}\_{i} (given by Eq. ([4](#A2.E4 "Equation 4 ‣ Model ‣ B.1 Supervised Parsing on Spider ‣ Appendix B Semantic Parsers ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"))), times the probability of selecting the column cjsubscript𝑐𝑗c\_{j}. The latter is defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | score​(cj)=Softmax​(DotProduct​(𝐬𝐭𝐚𝐭𝐞t,𝐜j)).scoresubscript𝑐𝑗SoftmaxDotProductsubscript𝐬𝐭𝐚𝐭𝐞𝑡subscript𝐜𝑗\mathrm{score}(c\_{j})=\mathrm{Softmax}\Big{(}\mathrm{DotProduct}(\mathbf{state}\_{t},{\mathbf{c}\_{j}})\Big{)}. |  | (6) |

We also add simple entity linking features to the representations in 𝕄𝕄\mathbb{M}, defined by the following heuristics:
(1) If an utterance token x∈𝒖𝑥𝒖x\in\bm{u} matches with the name of a table T𝑇{T}, we concatenate a trainable embedding vector (𝐭𝐚𝐛𝐥𝐞​\_​𝐦𝐚𝐭𝐜𝐡∈ℝ16𝐭𝐚𝐛𝐥𝐞\_𝐦𝐚𝐭𝐜𝐡superscriptℝ16\mathbf{table\\_match}\in\mathbb{R}^{16}) to the representations of x𝑥x and T𝑇{T}.
(2) Similarly, we concatenate an embedding vector (𝐜𝐨𝐥𝐮𝐦𝐧​\_​𝐦𝐚𝐭𝐜𝐡∈ℝ16𝐜𝐨𝐥𝐮𝐦𝐧\_𝐦𝐚𝐭𝐜𝐡superscriptℝ16\mathbf{column\\_match}\in\mathbb{R}^{16}) to the representations of an utterance token and a column if their names match.
(3) Finally, we concatenate a zero-vector (𝟎∈ℝ160superscriptℝ16\mathbf{0}\in\mathbb{R}^{16}) to representations of all unmatched elements.

#### Configuration

We use the default configuration of TranX. For TaBert parameters, we use an Adam optimizer with a learning rate of 3​e−53𝑒53e-5 and linearly decayed learning rate schedule, and another Adam optimizer with a constant learning rate of 1​e−31𝑒31e-3 for all remaining parameters.
During training, we update model parameters for 25000 iterations, and freeze the TaBert parameters at the first 1000 update steps. We use a batch size of 30 and beam size of 3.
We use gradient accumulation for large models to fit a batch into GPU memory.

### B.2 Weakly-supervised Parsing on WikiTableQuestions

#### Model

We use MAPO Liang et al. ([2018](#bib.bib19)), a strong weakly-supervised semantic parser.
The original MAPO models comes with an LSTM encoder, which generates utterance and column representations used by the decoder to predict table queries.
We directly substitute the encoder with TaBert, and project the utterance and table representations from TaBert to the original embedding space using a linear transformation.
MAPO uses a domain-specific query language tailored to answer compositional questions on a single table.
For instance, the example question in [Fig. 1](#S2.F1 "Figure 1 ‣ Semantic Parsing over Tables ‣ 2 Background ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data") could be answered using the following query

[⬇](data:text/plain;base64,VGFibGUuY29udGFpbnMoY29sdW1uPVBvc2l0aW9uLCB2YWx1ZT0xc3QpICAmXENvbW1lbnR7R2V0IHJvd3Mgd2hvc2UgYFBvc2l0aW9uJyBmaWVsZCBjb250YWlucyBgMXN0J30mCiAgICAgIC5hcmdtYXgob3JkZXJfYnk9WWVhcikgICZcQ29tbWVudHtHZXQgdGhlIHJvdyB3aGljaCBoYXMgdGhlIGxhcmdlc3QgYFllYXInIGZpZWxkfSYKICAgICAgLmhvcChjb2x1bW49VmVudWUpICAgICAmXENvbW1lbnR7U2VsZWN0IHRoZSB2YWx1ZSBvZiBgVmVudWUnIGluIHRoZSByZXN1bHQgcm93fSY=)

Table.contains(column=Position, value=1st)  # Get rows whose ‘Position’ field contains ‘1st’

.argmax(order\_by=Year)  # Get the row which has the largest ‘Year’ field

.hop(column=Venue)  # Select the value of ‘Venue’ in the result row

MAPO is written in Tensorflow. In our experiments we use an optimized re-implementation in PyTorch, which yields 4×\times training speedup.

#### Configuration

We use the same optimizer and learning rate schedule as in [§ B.1](#A2.SS1 "B.1 Supervised Parsing on Spider ‣ Appendix B Semantic Parsers ‣ 7 Conclusion and Future Work ‣ Knowledge-enhanced Pretraining ‣ 6 Related Works ‣ Impact of Pretraining Objectives ‣ Effect of Row Linearization ‣ 5.1 Main Results ‣ 5 Experiments ‣ TaBert: Pretraining for Joint Understanding of Textual and Tabular Data"). We use a batch size of 10, and train the model for 20000 steps, with the TaBert parameters frozen at the first 5000 steps. Other hyper-parameters are kept the same as the original MAPO system.

[◄](/html/2005.08313)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2005.08314)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2005.08314)
[View original  
on arXiv](https://arxiv.org/abs/2005.08314)[►](/html/2005.08315)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Mar 7 07:29:59 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
