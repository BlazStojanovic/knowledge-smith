---
arxiv: '2512.09015'
authors:
- DatologyAI
- ':'
- Luke Merrick
- Alex Fang
- Aldo Carranza
- Alvin Deng
- Amro Abbas
- Brett Larsen
- Cody Blakeney
- Darren Teh
- David Schwab
- Fan Pan
- Haakon Mongstad
- Haoli Yin
- Jack Urbanek
- Jason Lee
- Jason Telanoff
- Josh Wills
- Kaleigh Mentzer
- Paul Burstein
- Parth Doshi
- Paul Burnstein
- Pratyush Maini
- Ricardo Monti
- Rishabh Adiga
- Scott Loftin
- Siddharth Joshi
- Spandan Das
- Tony Jiang
- Vineeth Dorna
- Zhengping Wang
- Bogdan Gaza
- Ari Morcos
- Matthew Leavitt
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'Luxical: High-Speed Lexical-Dense Text Embeddings'
url: https://arxiv.org/abs/2512.09015
year: 2025
---

[2512.09015] Luxical: High-Speed Lexical-Dense Text Embeddings














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



# Luxical: High-Speed Lexical-Dense Text Embeddings

DatologyAI Team
  
See Contributions and Acknowledgments ([Section 7](#S7 "7 Contributions and Acknowledgements ‣ Luxical: High-Speed Lexical-Dense Text Embeddings")) for full author list.

###### Abstract

Frontier language model quality increasingly hinges on our ability to organize web-scale text corpora for training.
Today’s dominant tools trade off speed and flexibility: lexical classifiers (e.g., FastText) are fast but limited to producing classification output scores, while the vector-valued outputs of transformer text embedding models flexibly support numerous workflows (e.g., clustering, classification, and retrieval) but are computationally expensive to produce.
We introduce Luxical, a library for high-speed “lexical-dense” text embeddings that aims to recover the best properties of both approaches for web-scale text organization.
Luxical combines sparse TF–IDF features, a small ReLU network, and a knowledge distillation training regimen to approximate large transformer embedding models at a fraction of their operational cost.
In this technical report, we describe the Luxical architecture and training objective and evaluate a concrete Luxical model in two disparate applications: a targeted webcrawl document retrieval test and an end-to-end language model data curation task grounded in text classification.
In these tasks we demonstrate speedups ranging from 3x to 100x over varying-sized neural baselines, and comparable to FastText model inference during the data curation task.
On these evaluations, the tested Luxical model illustrates favorable compute/quality trade-offs for large-scale text organization, matching the quality of neural baselines.
Luxical is available as open-source software at <https://github.com/datologyai/luxical>.

## 1 Introduction

Large language models (LLMs) have pushed training datasets from billions to trillions of tokens. While webcrawl datasets provide vast quantities of data to meet these demands, much of the information in raw web crawls is redundant, off-distribution, or harmful, while comparatively small subsets of high-value text have demonstrably outsized impact on model capabilities (sorscher2023neuralscalinglawsbeating). As such, in recent years web-scale corpus organization and quality control have become a necessity for pushing the frontiers of Artificial Intelligence (li2024datacomp). One promising paradigm for tackling this task is to apply machine learning methods, e.g. supervised classification or representation learning.

A typical tool for ML-based data organization is the use of text embedding transformer models. Though effective for data curation (penedo2024fineweb), their use often entails a trade-off between representational quality and efficiency. Models designed to score highly on benchmarks such as the Massive Text Embedding Benchmark (MTEB) (muennighoff-etal-2023-mteb) optimize for producing representations suitable to a blend of asymmetrical and symmetrical retrieval tasks and can sacrifice efficiency so aggressively that they become prohibitively expensive to run at web-scale (merrick2024arcticembedscalableefficientaccurate). In contrast, lexical classification algorithms (which were popular before transformer language models rose to prominence) can operate with extreme efficiency but produce only a single classification score that is unsuitable for certain workloads like clustering and retrieval (joulin2017bag).

This technical report introduces Luxical, a hybrid lexical–dense embedding modeling methodology that targets a middle ground between lexical classifiers and large transformer text embedding models. Luxical transforms a sequence of tokens into a bag-of-ngrams representation, applies inverse-frequency weighting and normalization to obtain a sparse feature vector, and then transforms this vector into a dense embedding vector using a computationally efficient shallow ReLU network. For training, Luxical models use a Gram-matrix distillation objective that pushes the embedding geometry to match the similarity structure of a larger teacher embedding model, allowing the methodology to leverage advances in large-scale text embedding models without becoming beholden to their slow runtime characteristics. Our goal is not to compete with the strongest transformer encoders on broad benchmarks, but to provide a technically simple point on the speed–quality frontier that is well suited to production data pipelines where dense encoders are too slow and existing lexical methods are not flexible enough.

In this work we describe both the general Luxical methodology and a specific English instantiation, Luxical-One, trained via distillation from a strong transformer teacher on English web documents. Luxical-One not only serves as a concrete case study of the broader Luxical approach in our experiments, but it is also released as a tool for the broader community.111The Luxical-One model is hosted at <https://huggingface.co/DatologyAI/luxical-one>.

In the remainder of this report we first situate Luxical within the literature on sentence embeddings, web-scale data organization, and lexical retrieval. We then describe the model architecture, tokenization pipeline, and training objective in detail, establish an experimental setup for evaluating Luxical-One against lexical and transformer baselines, and discuss how Luxical can be integrated into modern web-scale text-processing workflows.

## 2 Motivating Related Work

Modern transformer-based “sentence embedding” models, often derived from BERT and its siamese variants, produce vector-valued representations of natural language applicable to a variety of tasks including clustering, retrieval, and classification (DBLP:journals/corr/abs-1810.04805; reimers-2019-sentence-bert; muennighoff-etal-2023-mteb). These models set the standard for general-purpose performance, but their compute profiles and memory footprints can be challenging to deploy as the first stage of trillion-token corpus-processing pipelines.

Web-scale corpus construction and filtering methodologies have evolved alongside these embedding models. Early corpora such as the Colossal Clean Crawled Corpus (C4) used heuristic language identification, blocklists, and MinHash-based deduplication to turn Common Crawl into a usable training set for large-scale language modeling (DBLP:journals/corr/abs-1910.10683). More recent work treats corpus construction partially as a modeling problem. For example, FineWeb and DCLM (DataComp-LM) introduce quality filters based on neural and lexical classifier models, and subsequent corpora show how sensitive downstream performance is to these filters (penedo2024fineweb; li2024datacomp; soldaini2024dolmaopencorpustrillion). These works motivate the need for efficient embedding backbones that can support model-based filtering and other data organization workflows at web scale.

Lexical and sparse retrieval methods—ranging from TF–IDF and BM25 to more recent models such as SPLADE—exploit term-level structure to achieve strong ranking performance with inverted indexes (robertson2009probabilistic; thibault-2021-splade). They show that lexical and sparse methods remain competitive and scalable baselines for large retrieval systems, even as dense transformer encoders have become the default choice for many tasks. The continued use of FastText-style models (joulin2017bag; li2024datacomp) similarly illustrates the value of simple bag-of-ngrams architectures in large-scale data pipelines.

## 3 Model and Training

### 3.1 Lexical–Dense Architecture

![Refer to caption](/html/2512.09015/assets/figures/luxical_sparse_dense.png)


Figure 1: Sparse-to-dense architecture of Luxical. A sparse vector of normalized ngram frequencies (the TF-IDF vector) is projected through a small MLP to produce a dense embedding.

Luxical fuses a classical lexical representation into a small ReLU network, yielding dense embeddings while retaining the scalability advantages of bag-of-words methods. Given an input document, Luxical first tokenizes the text and constructs a term-frequency (TF) representation over ngrams from the resulting tokens, similar to the bag-of-ngrams featurization in FastText (joulin2017bag), but eschewing the hashing trick in favor of exact ngram matching over a predetermined vocabulary of ngrams. This sparse TF vector is then reweighted to account for overall term frequency, and ℓ2\ell\_{2}-normalized to produce a sparse unit vector over the vocabulary. Unlike a purely lexical system, Luxical then applies a small feed-forward ReLU network (which we also refer to as a Multi-Layer Perceptron or MLP for short) to map this sparse term-frequency-inverse-document-frequency (TF–IDF) representation to a dense, normalized embedding. This hybrid design allows the model to encode word-level statistics and simple composition patterns while remaining inexpensive to evaluate.

### 3.2 Sparse-to-Dense Projection Efficiency

The core efficiency property of Luxical models comes from exploiting sparsity in the first linear layer of the MLP. If 𝐬∈ℝV\mathbf{s}\in\mathbb{R}^{V} denotes the sparse TF–IDF vector over a vocabulary of size VV and A∈ℝd×VA\in\mathbb{R}^{d\times V} denotes the input weight matrix of the MLP, then the matrix–vector product A​𝐬A\mathbf{s} reduces to a sum over the columns corresponding to nonzero entries in 𝐬\mathbf{s}:

|  |  |  |
| --- | --- | --- |
|  | A​𝐬=∑i∈nz​(𝐬)si​𝐚i,A\mathbf{s}\;=\;\sum\_{i\in\mathrm{nz}(\mathbf{s})}s\_{i}\,\mathbf{a}\_{i}, |  |

where nz​(𝐬)\mathrm{nz}(\mathbf{s}) indexes nonzero positions and 𝐚i\mathbf{a}\_{i} is the ii-th column of AA. In practice, only a small fraction of the vocabulary appears in any given document, so this sparse-by-dense multiplication can be implemented by gathering and scaling a handful of columns. Luxical implements this operation with Numba-optimized kernels (lam2015numba) to achieve high throughput on CPUs. In practice, modern hardware can be so efficient at performing these matrix operations that the tokenization step dominates the overall wall clock time of the Luxical embedding operation, with the ngram counting, sparse-to-dense projection, and extra projections and nonlinearities of the ReLU layers degrading throughput only modestly below that of pure tokenization.

### 3.3 Training Objective

![Refer to caption](/html/2512.09015/assets/figures/embedding_distillation.png)


Figure 2: Contrastive distillation loss recommended when training Luxical models.

Though the Luxical architecture and codebase is modular and can support a number of training objectives, we choose to implement and study a distillation-style contrastive objective that encourages the Luxical model’s embeddings to align the pairwise similarity structure of batches of documents with the structure induced by the embeddings of a teacher model. For a batch of nn documents, we compute normalized student embeddings S∈ℝn×dS\in\mathbb{R}^{n\times d} and teacher embeddings T∈ℝn×dtT\in\mathbb{R}^{n\times d\_{t}}, construct their Gram matrices Gs=S​S⊤G\_{s}=SS^{\top} and Gt=T​T⊤G\_{t}=TT^{\top}, and remove the diagonal from both matrices to discard trivial self-similarity (producing G^s\hat{G}\_{s} and G^t\hat{G}\_{t}). After temperature scaling (denoting the temperature hyperparameter as τ\tau), we minimize a Kullback–Leibler divergence the rows of the diagonal-stripped Gram matrices:

|  |  |  |
| --- | --- | --- |
|  | ℒdistill=τ2⋅KLDiv​(G^s/τ,G^t/τ)\mathcal{L}\_{\mathrm{distill}}\;=\;\tau^{2}\cdot\mathrm{KLDiv}\!\bigl(\hat{G}\_{s}/\tau,\;\hat{G}\_{t}/\tau\bigr) |  |

This objective encourages Luxical to replicate the relative similarity pattern induced by the teacher model over each batch, somewhat in the spirit of large-scale weakly supervised contrastive pretraining approaches such as E5 (wang2022text) but leveraging a much stronger supervision signal thanks to the teacher model.

### 3.4 Implementation Details

The implementation is optimized for web-scale deployment on CPU. Though Luxical is written primarily in Python, it relies on a small custom Rust extension for high-throughput tokenization. This arrow-tokenize extension mitigates Python garbage collection overhead by returning tokenized outputs as PyArrow arrays that garbage collect much faster than Python list-of-list outputs. If Luxical were to rely on the built-in Python bindings provided by the tokenizers library, it would have introduced a major performance bottleneck when processing large batches of documents on powerful multi-core CPUs.

Sparse-by-dense projections and IDF scaling are implemented using Numba for efficiency (lam2015numba). Though IDF scaling weights can be merged into the first layer of the MLP, Luxical performs this as a separate step to keep the parameterization of the layer weights more stable during optimization. Tokenization and embedding computation are pipelined over large batches of documents using the Arrow-based tokenizer to mitigate overheads during high-throughput embedding jobs.

## 4 Empirical Evaluation

We evaluate Luxical through a concrete instantiation of the methodology, the Luxical-One model, and a document-level similarity task designed to probe both symmetric semantics and systems-level throughput on FineWeb data.

### 4.1 Luxical-One Configuration and Training

Luxical-One is an English Luxical model that follows the architecture described in [Section 3](#S3 "3 Model and Training ‣ Luxical: High-Speed Lexical-Dense Text Embeddings"). The model adopts the BERT uncased tokenizer to segment input text and featurizes documents as TF–IDF-weighted bags over a fixed vocabulary of mined 5-grams. Concretely, we used the Space-Saving Algorithm (spacesavingalgorithm) to identify a vocabulary two million (approximately) most frequent 5-grams observed in a sample of the FineWeb corpus as the fixed vocabulary. We also leverage the approximate frequency statistics from this operation to construct a log-scaled IDF scaling vector. The feedforward network of Luxical-One maps the 2M-dimensional sparse vector of ngram statistics to dense vectors of sizes 92, 3072, and 3072 with ReLU nonlinearities and ℓ2\ell\_{2} normalization applied after each projection. The final layer of the network then projects down and ℓ2\ell\_{2} normalized once more to produce a compact 192-dimensional embedding vector.

We train Luxical-One with the contrastive Gram-matrix distillation objective described in [Section 3.3](#S3.SS3 "3.3 Training Objective ‣ 3 Model and Training ‣ Luxical: High-Speed Lexical-Dense Text Embeddings"), using teacher embeddings produced by the snowflake-arctic-embed-m-v2.0 model (yu2024arcticembed20multilingualretrieval). We selected this teacher for its relatively modest size (allowing us to embed more documents in the same amount of time compared to other, larger models) and ability to produce small 256-dimensional embedding vectors while retraining high embedding quality (not only did this make data storage and dataloading faster and simpler, it also reduced the cost of computing the teacher Gram matrix during training). We used 50 million English documents sampled from FineWeb as the training corpus and pre-embedded them offline using the teacher model. During training we ran three epochs over this sample, shuffling documents between epochs. Training proceeded with standard mini-batch stochastic optimization (using the Adam optimizer with a warmup-stable-decay learning rate schedule) on CPU only. We used a batch size of 3072, a loss temperature of 3.0, and a peak learning rate of 0.01. We warmed up learning rate for 5% of training steps and performed linear learning rate decay to zero learning rate for the final 10% of training.

### 4.2 Throughput Benchmark

![Refer to caption](/html/2512.09015/assets/x1.png)


Figure 3: End-to-end throughput (web documents per second) when embedding 100,000 FineWeb documents with Luxical-One and transformer baselines on an Apple M4 Max CPU and an NVIDIA A10G GPU.

To assess how well Luxical delivers high-throughput embedding in practice, we sample 100,000 complete FineWeb documents and embed them with Luxical-One, MiniLM-L6-v2 (sentence\_transformers\_all\_minilm\_l6\_v2), and the Qwen3-0.6B embedding model (zhang2025qwen3embeddingadvancingtext). We report end-to-end throughput in documents per second, including tokenization, under two hardware configurations:

1. (i)

   an Apple M4 Max laptop CPU, and
2. (ii)

   an NVIDIA A10G server GPU.

For transformer baselines we evaluate both CPU-only and GPU-accelerated settings where applicable, while Luxical-One is evaluated on CPU only, reflecting its intended deployment regime. This benchmark is designed to mimic the common scenario in which many billions of web documents must be embedded once as a preprocessing stage for downstream organization and analysis.

[Figure 3](#S4.F3 "In 4.2 Throughput Benchmark ‣ 4 Empirical Evaluation ‣ Luxical: High-Speed Lexical-Dense Text Embeddings") summarizes end-to-end embedding throughput for Luxical-One and the transformer baselines. Even with GPU acceleration, the Qwen model lags behind Luxical-One by nearly two orders of magnitude. The much smaller MiniLM-based model closes part of this gap but still falls substantially short of Luxical-One, especially on CPU. These measurements confirm that the sparse-by-dense architecture and implementation choices in Luxical translate into practical speed improvements for large-scale corpus-processing workloads, not just improvements in FLOP counts on paper.

### 4.3 Document-Half Matching

To evaluate the utility of the embeddings produced by Luxical-One, we construct a web-document-based symmetrical retrieval task with known ground truth. To do this, we sample 50,000 documents from FineWeb and split each document into two contiguous halves, yielding 100,000 halves in total. For each original document, its two halves form a positive pair. We embed all halves with Luxical-One and treat each half in turn as a query. For a given query half, we compute cosine similarities to all 99,999 other halves and rank them. The matching half from the same source document defines the correct target; we record its rank and convert this to an error-at-kk curve as a function of the retrieval window size kk. We compare against the same baseline models as in [Section 4.2](#S4.SS2 "4.2 Throughput Benchmark ‣ 4 Empirical Evaluation ‣ Luxical: High-Speed Lexical-Dense Text Embeddings") as well as the following: Arctic-2.0-M (the teacher model for Luxical-One (yu2024arcticembed20multilingualretrieval)), LEAF-MT (a model of the same size as MiniLM-L6-v2 but trained using a similar knowledge-distillation objective (vujanic2025leafknowledgedistillationtext)), and the MixedBreadAI-Large-v1 model (the teacher model for LEAF-MT, hereafter Mxbai-L-v1 for brevity (emb2024mxbai)).

![Refer to caption](/html/2512.09015/assets/x2.png)


Figure 4: Document-half matching error rates as a function of retrieval window size for Luxical-One and transformer baselines on our document-half dataset.

[Figure 4](#S4.F4 "In 4.3 Document-Half Matching ‣ 4 Empirical Evaluation ‣ Luxical: High-Speed Lexical-Dense Text Embeddings") reports error rates on the document-half matching task as a function of the retrieval window size. At strict top-1 retrieval, both Luxical-One and MiniLM-L6-v2 trail far behind larger models like the Qwen model, though surprisingly Arctic-2.0-M does best despite its more modest active parameter count. As we enlarge the retrieval window, Luxical-One closes the gap and achieves substantially better error rates than the MiniLM-based baseline. These results show us that at coarse scales (which are potentially representative of many web-scale organization workloads like mining the top few percent of nearest neighbors to a target embedding vecotr), Luxical-One’s error curve approaches that of sophisticated transformer-based embedding models while maintaining dramatically higher throughput. These results suggest that Luxical can serve as an effective backbone for symmetric document–document similarity tasks in web-scale text organization pipelines. Its embedding geometry captures enough semantic structure to group related documents and support downstream classifiers, while its throughput makes it feasible to process very large corpora on commodity CPU hardware.

Another point worth taking away from this plot is that the distillation-based training objective is a powerful tool for training high-quality models. Even though Luxical-One embodies a more approximate function (e.g. lacking fine-grained positional information about the words of the input document), it is able to outperform the small transformer model by learning from a more powerful transformer model teacher. We see, too, that the LEAF-MT model strongly outperforms the same-architecture MiniLM-L6-v2 model on this task, approaching the error rates of its teacher at coarse-grained retrieval windows. We speculate that both the distillation objective and the focus on full-document embedding to capture symmetrical similarity relationships were instrumental in making LEAF-MT perform so well on this task.

### 4.4 Data Curation Application: Classifier-Based Filtering

![Refer to caption](/html/2512.09015/assets/x3.png)


(a) Classification throughput.

![Refer to caption](/html/2512.09015/assets/x4.png)


(b) Downstream LM performance.

Figure 5: Comparison of classifier-based filtering strategies. (a) Throughput of the scoring pipeline in documents per second. (b) Average zero-shot accuracy of a 3B-parameter language model trained on data curated by each scorer across 5 benchmarks.

To complement the document-half matching task of in [Section 4.3](#S4.SS3 "4.3 Document-Half Matching ‣ 4 Empirical Evaluation ‣ Luxical: High-Speed Lexical-Dense Text Embeddings"), which provides a controlled environment to probe the alignment between embedding geometry and semantic similarity, we additionally evaluate Luxical in a realistic end-to-end application: classifier-based data filtering for LLM training. Recent work has demonstrated that supervised text classifiers — ranging from transformer-based encoders like the FineWeb-Edu scorer (penedo2024fineweb) to lexical FastText classifiers like that used in DCLM (li2024datacomp) — can effectively identify high-quality subsets of web corpora, leading to improved downstream model performance. In this experiment, we compare Luxical-One against both the FineWeb-Edu and DCLM scoring models to assess whether our hybrid approach can match the utility of dense transformers while retaining the efficiency of lexical baselines.

We construct a filtering pipeline wherein a 600B token random subset of FineWeb is filtered down to a high-quality 60-billion-token subset (a 10% selection rate).
We compare three scoring methods:

1. 1.

   FineWeb-Edu Scorer: We use the standard FineWeb-Edu classifier, which consists of a classification head trained on top of a BERT-like encoder network. The FineWeb-Edu classifier was trained using labels obtained by prompting Llama-3-70B-Instruct to score FineWeb documents for their educational quality.
2. 2.

   DCLM FastText Scorer: We use the standard DCLM scorer, a FastText classifier trained using samples from OpenHermes 2.5 (OpenHermes2.5) and high-scoring posts from the r/ExplainLikeImFive subreddit as positives, and samples from a RefinedWeb (penedo2023refinedwebdatasetfalconllm) reproduction as negatives.
3. 3.

   Luxical-One MLP Scorer: We train a lightweight Multi-Layer Perceptron (MLP) on top of frozen Luxical-One embeddings. The MLP consists of two hidden layers with hidden dimensionality 256 and uses ReLU activations. We fit this MLP using a set of quality annotations similar to those used by the FineWeb-Edu and DCLM classifiers. We emphasize that the goal in this experiment is not to train a state-of-the-art text scorer, but to compare the performance characteristics of existing scorers to a Luxical-One-based scorer trained using similar data.

To measure the impact of curation, we train a 3-billion-parameter dense transformer language model on each of the curated datasets. We use the AdamW optimizer (adamw) in a warmup-stable-decay learning rate schedule reaching a peak learning rate of 7e-4 with a global batch size of 576. We evaluate the resulting models on a suite of filter-sensitive benchmarks, including ARC, MMLU, OpenBookQA, and SciQ (clark2018arc; hendrycks2021mmlu; mihaylov2018openbookqa; SciQ). To evaluate throughput, we time similar modeling pipelines in a controlled setting classifying 100,000 randomly-sampled FineWeb documents.

[Figure 5](#S4.F5 "In 4.4 Data Curation Application: Classifier-Based Filtering ‣ 4 Empirical Evaluation ‣ Luxical: High-Speed Lexical-Dense Text Embeddings") summarizes the results of these experiments.
In terms of system throughput ([Figure 5(a)](#S4.F5.sf1 "In Figure 5 ‣ 4.4 Data Curation Application: Classifier-Based Filtering ‣ 4 Empirical Evaluation ‣ Luxical: High-Speed Lexical-Dense Text Embeddings")), the Luxical-One-based scoring pipeline achieves speeds comparable to the FastText-based DCLM baseline (23.1MiB/s for Luxical-One-based pipeline, 19.0MiB/s for the FastText-based pipeline).
Crucially, both lexical approaches operate more than an order of magnitude faster than the transformer-based FineWeb-Edu scorer pipeline (which achieves a throughput of only 1.6MiB/s), even when the latter is GPU-accelerated.
This confirms that Luxical successfully mitigates the computational bottlenecks associated with using BERT-style transformer models in this workflow.

Regarding quality ([Figure 5(b)](#S4.F5.sf2 "In Figure 5 ‣ 4.4 Data Curation Application: Classifier-Based Filtering ‣ 4 Empirical Evaluation ‣ Luxical: High-Speed Lexical-Dense Text Embeddings")), the language models trained on data curated by all three filtering methods achieve similar downstream accuracy, while the unfiltered baseline data yielded substantially lower performance.
These results suggest that for large-scale quality filtering, both FastText and Luxical-One offer a favorable trade-off on the speed–quality frontier, delivering utility comparable to that of heavy transformer encoders (e.g. inducing identical downstream MMLU accuracy scores of 36.4%) at much greater throughput rates.
Additionally, since the MLP inference in this experiment accounts for less than 0.25% of the total runtime of the Luxical-One-based classification pipeline, practitioners who wish to tweak and re-run scoring in an iterative manner can expect to accelerate their workflows beyond the speeds offered by FastText by decoupling an initial Luxical embedding step from a subsequent (and much faster) classification step.

## 5 Discussion and Limitations

Luxical is most appropriate when symmetric document–document similarity and coarse corpus-level organization are the primary goals, such as in semantic deduplication, clustering, and quality-based filtering at scale. In these regimes, a single fast embedding pass followed by geometric operations or lightweight classifiers can substantially reduce the cost of organizing large text corpora, and our experiments suggest that Luxical-One can replace heavier encoders without sacrificing coarse-grained quality when processing English web data. By contrast, Luxical is less suitable for settings that demand fine-grained ranking (e.g., high-precision search) or reasoning-heavy tasks, where larger transformer encoders remain the more reliable choice. For practitioners currently using MiniLM- or Qwen-style encoders for workloads like clustering, our results indicate that swapping in Luxical-One as the first-stage encoder is a plausible way to unlock the order-of-magnitude throughput gains observed in [Figure 3](#S4.F3 "In 4.2 Throughput Benchmark ‣ 4 Empirical Evaluation ‣ Luxical: High-Speed Lexical-Dense Text Embeddings") without re-architecting downstream components.

This report has several limitations. We study a single Luxical model, Luxical-One, trained on English FineWeb data and evaluated on one symmetric document similarity task and one language modeling data curation task, so additional work is needed to assess performance in other domains, languages, and task families. Our throughput comparison also focuses on a small set of hardware configurations and baseline models; a broader throughput and cost study across architectures and deployment settings is left for future work. Practitioners should therefore treat our results as an existence proof and reference implementation in one realistic setting, rather than a comprehensive comparison across all embedding choices.

Finally, there are natural extensions we do not explore here, including distilled variants, multilingual Luxical models, tighter integration with downstream corpus-management and curation systems, and connections to recent work on static embedding models in the sentence-transformers ecosystem (e.g. Static Embeddings (Aarsen2025StaticEmbeddings) and model2vec (minishlab2024model2vec).

### 5.1 Applications To Web-scale Text Organization

In the setting described in [Sections 1](#S1 "1 Introduction ‣ Luxical: High-Speed Lexical-Dense Text Embeddings") and [2](#S2 "2 Motivating Related Work ‣ Luxical: High-Speed Lexical-Dense Text Embeddings"), Luxical is intended to serve a single, reusable embedding stage in large-scale data pipelines. A typical deployment runs a Luxical model once over a corpus to produce dense vectors, then reuses those vectors across multiple downstream steps rather than invoking a model many times. Geometric methods such as clustering and nearest-neighbor search can operate directly on Luxical embeddings to construct domain-specific slices, perform distribution matching between training and evaluation sets, and implement semantic deduplication (abbas2023semdedupdataefficientlearningwebscale) across billions of documents. The same embeddings can be fed into small classifiers to approximate expensive quality labels such as those used in FineWeb-Edu and DCLM-style FastText filters (penedo2024fineweb; li2024datacomp). In this regime, the cost of computing Luxical embeddings once is amortized across semantic deduplication, filtering, and ranking, making it practical to use richer model-based organization and curation signals at web scale on CPU. The document-half matching task in our experiments can be viewed as a controlled proxy for the symmetric similarity operations that underlie these deployment scenarios.

## 6 Conclusion

Luxical provides a practical point on the speed–quality frontier for web-scale text organization and filtering, sitting between purely lexical methods and full transformer encoders. By combining sparse TF–IDF features with a small neural network trained via Gram-matrix distillation, it delivers embeddings that are expressive enough for symmetric document similarity tasks while remaining inexpensive to compute on CPU. Our experiments with Luxical-One on a document-half-matching retrieval benchmark and a data curation classification task illustrate that this design can approach transformer-level coarse-grained quality at much higher throughput.

From a deployment perspective, Luxical is designed to be simple to integrate: models are small, CPU-friendly, and exposed through a straightforward API, making it easy to add a single embedding stage to existing corpus-processing pipelines and reuse the resulting vectors across multiple downstream tasks. The Luxical library is available as open-source software at <https://github.com/datologyai/luxical/>, and the Luxical-One model is hosted at <https://huggingface.co/DatologyAI/luxical-one>, along with example code that can serve as a starting point for reproducing and extending our results.

## 7 Contributions and Acknowledgements

|  |  |
| --- | --- |
| Project Lead | Luke Merrick |
|  | *conceptualized and implemented Luxical; trained Luxical-One; led evaluation* |
| Core Contributors | Luke Merrick and Alex Fang |
|  | *conducted motivating research on fast lexical classification for data curation; conducted experiments evaluating Luxical-One on filter-based data curation* |
| Technical Contributors | Aldo Carranza, Alvin Deng, Amro Abbas, Brett Larsen, Cody Blakeney, Darren Teh, David Schwab, Fan Pan, Haakon Mongstad, Haoli Yin, Jack Urbanek, Jason Lee, Jason Telanoff, Josh Wills, Kaleigh Mentzer, Paul Burstein, Parth Doshi, Paul Burnstein, Pratyush Maini, Ricardo Monti, Rishabh Adiga, Scott Loftin, Siddharth Joshi, Spandan Das, Tony Jiang, Vineeth Dorna, and Zhengping Wang |
|  | *DatologyAI technical staff; contributed the experimental pipelines used in [Section 4.4](#S4.SS4 "4.4 Data Curation Application: Classifier-Based Filtering ‣ 4 Empirical Evaluation ‣ Luxical: High-Speed Lexical-Dense Text Embeddings")* |
| Not-So-Corporate  Leadership | Bogdan Gaza, Ari Morcos, and Matthew Leavitt |
| Acknowledgements | Jacqueline Liu, Tiffanie Pham, and Sylvia Hoang for assembling the all-star cast that made this work possible. Liz Gatapia for the beautiful logo design. Jayla Lindsey for perpetuating the welcoming collaborative office environment that made this work possible. |

[◄](/html/2512.09013)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2512.09015)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2512.09015)
[View original  
on arXiv](https://arxiv.org/abs/2512.09015)[►](/html/2512.09016)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue Jan 6 11:08:45 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
