---
arxiv: '2407.12665'
authors:
- Chenze Shao
- Fandong Meng
- Jie Zhou
parser: ar5iv
retrieved: '2026-05-15'
source: paper
title: 'Beyond Next Token Prediction: Patch-Level Training for Large Language Models'
url: https://arxiv.org/abs/2407.12665
year: 2024
---

[2407.12665] Patch-Level Training for Large Language Models



# Patch-Level Training for Large Language Models

Chenze Shao, Fandong Meng , Jie Zhou
  
Pattern Recognition Center, WeChat AI, Tencent Inc, China
  
{chenzeshao,fandongmeng,withtomzhou}@tencent.com
  
Corresponding author.

###### Abstract

As Large Language Models (LLMs) achieve remarkable progress in language understanding and generation, their training efficiency has become a critical concern. Traditionally, LLMs are trained to predict the next token in a sequence. Despite the success of token-level training, it suffers from considerable computational costs due to the need to process an extensive number of tokens. To mitigate this issue, this paper introduces patch-level training for LLMs, which reduces the sequence length by compressing multiple tokens into a single patch. During patch-level training, we feed the language model shorter sequences of patches and train it to predict the next patch, thereby processing the majority of the training data at a significantly reduced computational cost. Following this, the model continues token-level training on the remaining training data to align with the inference mode. Experiments on a diverse range of models (370M-2.7B parameters) demonstrate that patch-level training can reduce overall computational costs to 0.5×\times, without compromising the model performance compared to token-level training. Source code: <https://github.com/shaochenze/PatchTrain>.

## 1 Introduction

Large Language Models (LLMs, Achiam et al., [2023](#bib.bib1); Touvron et al., [2023a](#bib.bib59); [b](#bib.bib60); Team et al., [2023](#bib.bib58); Bai et al., [2023](#bib.bib3)) have achieved remarkable progress in language understanding and generation, which are primarily attributed to their unprecedented model capacity and the corresponding growth in the volume of training data they require (Kaplan et al., [2020](#bib.bib32); Hoffmann et al., [2022](#bib.bib28)). However, this scaling up comes with a substantial rise in computational costs, making the training efficiency of LLMs a critical concern. Despite the ongoing efforts on efficient LLMs (Wan et al., [2023](#bib.bib62)), it remains a formidable challenge to reduce training costs without compromising the model performance.

The conventional approach for training LLMs is next token prediction, i.e., predicting the next token of a sequence.
While this approach has achieved notable success, it represents an inefficient way for LLMs to acquire knowledge, as each token must be processed individually by the entire model. Even when disregarding the overhead of attention computation, each token still consumes approximately the same FLOPs as the number of model parameters, resulting in considerable computational costs when dealing with an extensive number of tokens in the training data.

For other data modalities like images and audio, it has become increasingly popular of transforming them into sequences for processing with sequence models (Chen et al., [2020](#bib.bib8); Zhu et al., [2024](#bib.bib70); Duan et al., [2024](#bib.bib15)). They also encountered the efficiency issue of sequence modeling, as directly converting raw inputs into sequences can produce excessively long sequences. For instance, unfolding a 256x256 image results in a pixel sequence of length 65,536, while a 10kHz audio signal translates into 10,000 samples per second. To improve the computational efficiency when dealing with such data, it is typical to reduce the sequence length by compressing segments of a specific length into patches. For instance, a 16x16 pixel segment can be represented as an image patch (Dosovitskiy et al., [2021](#bib.bib13)), and a 20ms slice of audio can be encapsulated in a hidden state representation (Baevski et al., [2020](#bib.bib2); Hsu et al., [2021](#bib.bib29)). However, for textual data, attempts to compress sequence length are less prevalent and necessitate specialized model structures (Yu et al., [2024](#bib.bib66); Ho et al., [2024](#bib.bib27)) or application scenarios (Pappagari et al., [2019](#bib.bib47)). Consequently, the predominant granularity for processing textual data remains at the token-level.

In this paper, we introduce patch-level training to improve the training efficiency of LLMs. The core of our approach is to reduce the sequence length by compressing multiple tokens into a single patch. Unlike previous patch-level attempts (Dosovitskiy et al., [2021](#bib.bib13); Yu et al., [2024](#bib.bib66)), our approach does not require the final model to work at the patch-level. Instead, it enables the model to efficiently acquire knowledge at the patch-level during training. Specifically, we divide the training process into two stages: patch-level training and token-level training. During patch-level training, we feed the language model shorter sequences of patches and train it to predict the next patch, thereby processing the majority of the training data at a significantly reduced computational cost. The resulting parameters are used to initialize the token-level model, which then continues training on the remaining data to adapt the knowledge gained during patch-level training to the token-level.

Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Patch-Level Training for Large Language Models") illustrates the efficiency advantage of patch-level training, where the area of the shape represents the overall training costs. With a patch size of K𝐾K, the computational costs for patch-level training are 1/K1𝐾1/K of the costs for token-level training. When a fraction λ𝜆\lambda of the training data is compressed into patches, the overall training costs are reduced to λ/K+1−λ𝜆𝐾1𝜆\lambda/K+1-\lambda times the original costs. For instance, to halve the training costs, one could set the patch size K=4𝐾4K=4 and conduct patch-level training on λ=2/3𝜆23\lambda=2/3 of the training data.

Figure 1: Visualization of overall training costs with patch compression for a fraction λ𝜆\lambda of training data and patch size K𝐾K.




Figure 2: Negative log-likelihood (NLL) loss on test set w.r.t the number of processed tokens during the training of 370M-parameter Transformers.

Employing the above settings (K=4,λ=2/3formulae-sequence𝐾4𝜆23K=4,\lambda=2/3), we train a series of LLMs of varying sizes (370M-2.7B parameters) on the Pile dataset (Gao et al., [2020](#bib.bib17)). Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Patch-Level Training for Large Language Models") illustrates the trend of NLL loss against the number of training tokens for the 370M model. After initialization with patch-level training, the model experiences a rapid decrease in loss as it continues token-level training on the remaining data. Remarkably, it achieves an even lower loss in comparison with training from scratch, while reducing training costs by half. By further adjusting the hyperparameter settings, even higher acceleration rates can be achieved, with only a slight sacrifice in model performance.

## 2 Patch-Level Training

In this section, we outline the patch-level training approach for large language models, as illustrated in Figure [3](#S2.F3 "Figure 3 ‣ 2 Patch-Level Training ‣ Patch-Level Training for Large Language Models"). Initially, the token sequence is transformed into a patch sequence by compressing every K𝐾K consecutive tokens into a single patch. This patch sequence is then fed into the sequence model, and the model is trained to predict all tokens in the next patch. The knowledge acquired during patch-level training is subsequently transferred to the token-level model. Specifically, we use the parameters obtained from the patch-level model to initialize the token-level model, and then proceed with token-level training on the remaining data.

Figure 3: Overview of patch-level training. Every consecutive K𝐾K token embeddings are averaged to form the patch embedding. The sequence model is fed the patch sequence and trained to predict the next patch. The cross-entropy loss is computed based on each patch prediction vector and all the subsequent K𝐾K tokens in its next patch.

While formulating the patch-level model structure, our goal is to minimize the discrepancy between patch-level and token-level models, thereby ensuring that the knowledge gained during patch-level training can be smoothly transferred to the token-level model. Given the context length T𝑇T for token-level training, we set the context length for patch-level training as K​T𝐾𝑇KT, which is then compressed to a patch sequence of length T𝑇T to maintain consistency with the subsequent token-level training. To avoid introducing unnecessary parameters during token-to-patch compression, we represent the patch embedding as the average of its associated token embeddings. Let pisubscript𝑝𝑖p\_{i} be the i𝑖i-th patch, xi​K+ksubscript𝑥𝑖𝐾𝑘x\_{iK+k} be the k𝑘k-th token in the i𝑖i-th patch, and E𝐸E be the embedding function. The patch embedding is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | E​(pi)=1K​∑k=0K−1E​(xi​K+k).𝐸subscript𝑝𝑖1𝐾superscriptsubscript𝑘0𝐾1𝐸subscript𝑥𝑖𝐾𝑘E(p\_{i})=\frac{1}{K}\sum\_{k=0}^{K-1}E(x\_{iK+k}). |  | (1) |

The patch-level model is trained through next patch prediction, i.e., predicting the K𝐾K tokens in the next patch. The simultaneous prediction of multiple tokens has been explored in speculative decoding, which typically employs multiple output heads and each head is responsible for predicting a distinct token (Cai et al., [2024](#bib.bib5); Lin et al., [2024](#bib.bib40)). However, this approach would also entail additional parameters that may be unfavorable for the subsequent knowledge transfer. Instead, we maintain a single output head and make its prediction cover all tokens in the next patch. Specifically, we calculate the cross-entropy loss for all the subsequent K𝐾K tokens based on the same patch prediction pθ(⋅|p<i)p\_{\theta}(\cdot|p\_{<i}), resulting in the following loss function:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒp​a​t​c​h=−∑i=1T∑k=0K−1log⁡pθ​(xi​K+k|p<i).subscriptℒ𝑝𝑎𝑡𝑐ℎsuperscriptsubscript𝑖1𝑇superscriptsubscript𝑘0𝐾1subscript𝑝𝜃conditionalsubscript𝑥𝑖𝐾𝑘subscript𝑝absent𝑖\mathcal{L}\_{patch}=-\sum\_{i=1}^{T}\sum\_{k=0}^{K-1}\log p\_{\theta}(x\_{iK+k}|p\_{<i}). |  | (2) |

Since the model finally works at the token-level, it is essential to reserve some training data to adapt the patch-level model to token-level. Specifically, we conduct patch-level training on a fraction λ𝜆\lambda of the training data, and then use the resulting parameters to initialize the token-level model. Following this, the token-level model continues training on the remaining data to adapt the knowledge gained during patch-level training to the token-level. As illustrated in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Patch-Level Training for Large Language Models"), the overall training costs are reduced to λ/K+1−λ𝜆𝐾1𝜆\lambda/K+1-\lambda times the original costs of token-level training. When the amount of training data is limited, this approach can also be utilized for efficient multi-epoch training. For example, given a budget of N𝑁N epochs, we can conduct patch-level training on the first N​λ𝑁𝜆N\lambda epochs, and then switch to token-level training for N​(1−λ)𝑁1𝜆N(1-\lambda) epochs.

## 3 Experiments

We conduct extensive experiments to uncover the properties of patch-level training. First, with the hyperparameters set at K=4𝐾4K=4 and λ=2/3𝜆23\lambda=2/3, we show that patch-level training can reduce overall training costs to 0.5×\times, without compromising model performance compared to token-level training. Following this, we study the scaling properties of patch-level training in Section [3.3](#S3.SS3 "3.3 Scaling Properties ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models"), and examine the effects of hyperparameters K𝐾K and λ𝜆\lambda respectively in Sections [3.4](#S3.SS4 "3.4 Effect of Patch Size 𝐾 ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models") and [3.5](#S3.SS5 "3.5 Effect of 𝜆 ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models"). Finally, Section [3.6](#S3.SS6 "3.6 Neuron Activation ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models") presents a quantitative explanation for the efficiency of patch-level training.

### 3.1 Setup

Datasets. We evaluate our approach on standard language modeling tasks, using the Pile dataset (Gao et al., [2020](#bib.bib17)) containing 360B tokens for training 111Previous works generally refer to the Pile dataset as having 300B tokens, but our actual measurement is 360B. The discrepancy is likely due to differences in tokenizers; we use the LLaMA2 tokenizer, which has a relatively small vocabulary, possibly resulting in more tokens. The perplexity scores are also incomparable with models using other tokenizers.. We assess the performance of LLMs from multiple aspects, including their perplexity, zero-shot accuracy, and instruction-following ability. Perplexity is calculated on the WikiText-103 test set (Merity et al., [2017](#bib.bib44)).
We evaluate the zero-shot capabilities of language models on 6 NLP benchmarks, including MMLU (Hendrycks et al., [2021](#bib.bib26)), HellaSwag (Zellers et al., [2019](#bib.bib67)), PIQA (Bisk et al., [2020](#bib.bib4)), WinoGrande (Sakaguchi et al., [2020](#bib.bib48)), ARC-E, and ARC-C (Clark et al., [2018](#bib.bib11)) 222https://github.com/EleutherAI/lm-evaluation-harness. For the pre-trained LLMs, we conduct instruction fine-tuning using the Alpaca dataset by GPT4 (Taori et al., [2023](#bib.bib57)), and then evaluate their instruction-following abilities on MT-Bench (Zheng et al., [2024](#bib.bib69)).

Models. We use the Transformer backbone (Vaswani et al., [2017](#bib.bib61)) and adopt most of the architecture designs from LLaMA (Touvron et al., [2023a](#bib.bib59)). We apply pre-normalization using RMSNorm (Zhang & Sennrich, [2019](#bib.bib68)), use the SwiGLU activation function (Shazeer, [2020](#bib.bib53)), and rotary positional embeddings (Su et al., [2021](#bib.bib56)). We also apply FlashAttention-2 (Dao, [2024](#bib.bib12)) to accelerate attention computation. We scale the model demension and obtain 4 different sizes of Transformers: Transformer-370M (hidden\_size=1024, intermediate\_size=2752, hidden\_layers=24, attention\_heads=16), Transformer-780M (hidden\_size=1536, intermediate\_size=4128, hidden\_layers=24, attention\_heads=16), Transformer-1.3B (hidden\_size=2048, intermediate\_size=5504, hidden\_layers=24, attention\_heads=16), Transformer-2.7B (hidden\_size=2560, intermediate\_size=6880, hidden\_layers=32, attention\_heads=32).

Implementation Details. Unless otherwise specified, the patch size K𝐾K is 4. The context length for token-level training 204820482048. For patch-level training, the context length is the patch size K∗2048𝐾2048K\*2048. The global batch size is 2​M2𝑀2M tokens, and the total number of training steps is N=180000𝑁180000N=180000. For patch-level training, the number of training steps is N​λ𝑁𝜆N\lambda, and then the model proceeds with token-level training for N​(1−λ)𝑁1𝜆N(1-\lambda) steps. After patch-level training, only the obtained model parameters are used for initialization, and all other states like the optimizer and learning rate scheduler are reset. We use the tokenizer of LLaMA2, whose vocabulary size is 320003200032000. Our models are optimized by the AdamW optimizer (Loshchilov & Hutter, [2019](#bib.bib41)) with β1=0.9,β2=0.95,ϵ=1​e−8formulae-sequencesubscript𝛽10.9formulae-sequencesubscript𝛽20.95italic-ϵ1𝑒8\beta\_{1}=0.9,\beta\_{2}=0.95,\epsilon=1e-8. The learning rate is 3​e−43𝑒43e-4 and the cosine learning rate schedule is applied with warmup of 200020002000 steps. We use a weight decay of 0.10.10.1 and gradient clipping of 1.01.01.0, and no dropout is applied during training.

### 3.2 Main Results

Table 1: Performance comparison of Transformers trained on the Pile dataset. λ𝜆\lambda denotes the proportion of training data used for patch-level training, with the patch size K𝐾K fixed at 4. ‘PPL’ represents the perplexity score on the WikiText-103 test set. For zero-shot evaluations, we report the normalized accuracy across 6 NLP benchmarks. ‘Average’ means the average zero-shot accuracy.

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model Type | Cost | PPL | MMLU | HellaSwag | PIQA | WinoG | ARC-E | ARC-C | Average |
| Transformer-370M | 1.0×\times | 10.9 | 22.9 | 40.8 | 67.5 | 53.1 | 44.3 | 24.7 | 42.2 |
| + Patch (λ=1/2𝜆12{\lambda=1/2}) | 0.625×\times | 10.6 | 23.5 | 42.0 | 67.9 | 52.1 | 46.1 | 25.6 | 42.9 |
| + Patch (λ=2/3𝜆23{\lambda=2/3}) | 0.5×\times | 10.7 | 23.7 | 41.1 | 68.0 | 51.9 | 46.0 | 24.2 | 42.5 |
| + Patch (λ=4/5𝜆45{\lambda=4/5}) | 0.4×\times | 11.0 | 23.3 | 40.5 | 67.5 | 51.7 | 44.9 | 24.5 | 42.1 |
| Transformer-780M | 1.0×\times | 9.2 | 24.4 | 48.5 | 69.0 | 55.4 | 49.0 | 26.7 | 45.5 |
| + Patch (λ=2/3𝜆23{\lambda=2/3}) | 0.5×\times | 9.1 | 24.1 | 49.1 | 70.6 | 54.8 | 51.3 | 28.2 | 46.3 |
| Transformer-1.3B | 1.0×\times | 8.2 | 23.9 | 54.5 | 71.2 | 57.3 | 55.1 | 28.9 | 48.5 |
| + Patch (λ=2/3𝜆23{\lambda=2/3}) | 0.5×\times | 8.2 | 24.3 | 54.1 | 71.6 | 57.8 | 55.6 | 30.4 | 49.0 |
| Transformer-2.7B | 1.0×\times | 7.1 | 25.3 | 62.2 | 74.3 | 61.5 | 61.2 | 34.3 | 53.1 |
| + Patch (λ=2/3𝜆23{\lambda=2/3}) | 0.5×\times | 7.2 | 25.4 | 61.9 | 74.9 | 62.4 | 61.9 | 34.6 | 53.5 |




Figure 4: Instruction-following abilities evaluated on MT-bench, a multi-turn question set.

We train a series of LLMs of varying sizes (370M-2.7B parameters) on the Pile dataset. To halve the training costs, we employ patch-level training with the settings of K=4,λ=2/3formulae-sequence𝐾4𝜆23K=4,\lambda=2/3, and compare its performance with the conventional token-level training. For the Transformer-370M, we also explore other choices of λ𝜆\lambda to evaluate its impact. Table [1](#S3.T1 "Table 1 ‣ 3.2 Main Results ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models") presents the performance comparison of the resulting models. Remarkably, our approach consumes only half of the computational resources and incurs almost no performance loss. It matches the baseline model in terms of perplexity and even demonstrates a consistent gain in zero-shot evaluations, raising the average accuracy by approximately 0.5%. The model performance is also influenced by the choice of λ𝜆\lambda. Within the range of values we set, a smaller λ𝜆\lambda leads to better model performance but also entails larger training costs. A more detailed study on the hyperparameter λ𝜆\lambda will be presented in Section [3.5](#S3.SS5 "3.5 Effect of 𝜆 ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models").

We further conduct instruction fine-tuning using the Alpaca dataset by GPT4 to examine the impact of patch-level training on the model’s instruction-following ability. We evaluate our models using MT-Bench, a multi-turn question set, and present the experimental results in Figure [4](#S3.F4 "Figure 4 ‣ 3.2 Main Results ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models"). As can be seen, our approach maintains a similar instruction-following ability to the original models, with some experiencing a score decrease (Transformer-370M, Transformer-1.3B) and others showing an improvement (Transformer-780M, Transformer-2.7B), which can be viewed as regular variations.

Next, we evaluate our approach in multi-epoch training. We randomly extract a subset of 60B tokens from the Pile dataset and increase the training epochs to 6. The results are shown in Table [2](#S3.T2 "Table 2 ‣ 3.2 Main Results ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models"). To our surprise, patch-level training continues to show superior training efficiency and even outperforms models trained on the full dataset in Table [1](#S3.T1 "Table 1 ‣ 3.2 Main Results ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models"). We speculate that this is because combining patch-level and token-level training on the same data contributes to better model regularization. It also suggests that our approach can be data-efficient by initializing the model with patch-level training for one or multiple epochs, offering a promising direction for boosting model performance.

Our primary motivation for patch-level training is to enhance the model’s knowledge acquisition efficiency. Interestingly, experimental results show that this approach can sometimes lead to performance improvements, which is beyond our initial expectation. We conjectured that the longer context length in patch-level training contributes to the improvements. However, when we reduce the context length during patch-level training from K​T=8192𝐾𝑇8192KT=8192 to T=2048𝑇2048T=2048 for Transformer-370M (λ=1/2𝜆12\lambda=1/2), the performance experiences a slight decline (PPL ↑↑\uparrow 0.06, zero-shot accuracy ↓↓\downarrow 0.2), yet still surpasses the baseline, implying that context length is not the primary factor. We hypothesize that two other factors might explain this phenomenon: first, the patch-level initialization could potentially serve as a form of regularization; second, patch compression reduces the distance between tokens, allowing the model to better learn and capture long-range dependencies.

Table 2: Performance comparison of Transformers trained on 60B tokens for 6 epochs.

| Model Type | Cost | PPL | MMLU | HellaSwag | PIQA | WinoG | ARC-E | ARC-C | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer-370M | 1.0×\times | 11.0 | 23.6 | 40.8 | 66.5 | 50.8 | 44.8 | 25.2 | 42.0 |
| + Patch (λ=1/2𝜆12{\lambda=1/2}) | 0.625×\times | 10.4 | 23.9 | 43.3 | 67.5 | 55.6 | 44.4 | 26.1 | 43.5 |
| + Patch (λ=2/3𝜆23{\lambda=2/3}) | 0.5×\times | 10.5 | 24.7 | 42.4 | 67.9 | 51.9 | 45.3 | 24.7 | 42.8 |
| + Patch (λ=4/5𝜆45{\lambda=4/5}) | 0.4×\times | 10.7 | 23.0 | 41.5 | 67.0 | 52.0 | 45.1 | 25.4 | 42.3 |

### 3.3 Scaling Properties

In the above, we have validated the effectiveness of patch-level training across several model sizes (370M-2.7B), using a training set of 360B tokens. However, state-of-the-art LLMs are generally trained on model sizes and datasets that are at least an order of magnitude larger than our settings. Therefore, it is crucial to know the scaling properties of patch-level training, i.e., how it performs when applied to larger training datasets and models.

In Table [1](#S3.T1 "Table 1 ‣ 3.2 Main Results ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models"), we notice a trend of perplexity related to the model size: the performance advantage of patch-level training appears to decrease as the model size increases. Table [3](#S3.T3 "Table 3 ‣ 3.3 Scaling Properties ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models") describes this trend more precisely, indicating that the model with patch-level training experiences smaller performance gains from the increase in model size. On the other hand, Table [4](#S3.T4 "Table 4 ‣ 3.3 Scaling Properties ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models") presents the perplexity changes when maintaining a constant model size and varying the size of the training data. As the data size increases, the performance of patch-level training improves at a faster rate compared to the baseline model.

Table 3: 
Perplexity scores when scaling the model size from 370M to 2.7B and training on the Pile dataset (360B tokens). ‘↓↓\downarrow’ indicates the perplexity reduction compared to the previous model size.

| Model Size | 370M | 780M | 1.3B | 2.7B |
| --- | --- | --- | --- | --- |
| Transformer | 10.94 | 9.18 (↓↓\downarrow1.76) | 8.18 (↓↓\downarrow1.0) | 7.11 (↓↓\downarrow1.07) |
| + Patch (λ=2/3𝜆23{\lambda=2/3}) | 10.68 | 9.10 (↓↓\downarrow1.58) | 8.23 (↓↓\downarrow0.87) | 7.24 (↓↓\downarrow0.99) |




Table 4: Perplexity scores of Transformer-370M when scaling the size of training data from 45B to 360B. ‘↓↓\downarrow’ indicates the perplexity reduction compared to the previous data size. The batch size is adjusted to maintain a consistent number of training steps.

| Data Size | 45B | 90B | 180B | 360B |
| --- | --- | --- | --- | --- |
| Transformer | 12.50 | 11.71 (↓↓\downarrow0.79) | 11.28 (↓↓\downarrow0.43) | 10.94 (↓↓\downarrow0.34) |
| + Patch (λ=2/3𝜆23{\lambda=2/3}) | 12.84 | 11.80 (↓↓\downarrow1.04) | 11.17 (↓↓\downarrow0.63) | 10.68 (↓↓\downarrow0.49) |

This phenomenon can be explained from the perspective of knowledge transfer. As the data size increases, more training data is employed to adjust the model from patch-level to token-level, facilitating a smoother knowledge transfer process. However, an increase in model size implies a greater number of model parameters to be transferred to the token-level, which raises the level of transfer difficulty and necessitates more training data. Based on this explanation, patch-level training is better suited for scenarios with abundant training data.

Note that the above conclusions are drawn under the settings of K=4,λ=2/3formulae-sequence𝐾4𝜆23K=4,\lambda=2/3, which may vary with changes in the patch size K𝐾K and the patch-level data fraction λ𝜆\lambda. At present, we have not identified a general scaling law for patch-level training that incorporates K𝐾K and λ𝜆\lambda. Instead, we have made some observations regarding their effects on model performance, which will be discussed in the following.

### 3.4 Effect of Patch Size K𝐾K

We investigate the effect of patch size under the settings of 90B training data, 370M model parameters, a batch size of 512K, and λ=1/2𝜆12\lambda=1/2. The results are shown in Figure [5](#S3.F5 "Figure 5 ‣ 3.4 Effect of Patch Size 𝐾 ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models"). Across different patch sizes, the loss curves for patch sizes K=2𝐾2K=2 and K=4𝐾4K=4 are nearly indistinguishable, while further increasing the patch size to 8 or 16 results in a certain performance decline. Despite this, these models still exhibit significant performance improvements when compared to the model trained from scratch, which does not benefit from the initialization of patch-level training.

Figure 5: Test losses of Transformer-370M w.r.t the number of processed tokens. Models are initialized by patch-level training with patch size K𝐾K.



Figure 6: Effect of varying λ𝜆\lambda while keeping the data size constant.

Figure 7: Effect of varying λ𝜆\lambda while keeping the computational cost constant.

Overall, the patch size of K=4𝐾4K=4 strikes a favorable trade-off between training efficiency and performance. Considering that larger patch sizes can process more data at the same cost, we also experiment with patch-level training using K=8𝐾8K=8 on 90B tokens, which costs similar computational resources as K=4𝐾4K=4 on 45B tokens. Following this, both models proceed with token-level training on 45B tokens, and coincidentally, their loss curves are nearly identical. In this context, the advantage of K=4𝐾4K=4 lies in its data efficiency, as it achieves similar performance while consuming less data.

### 3.5 Effect of λ𝜆\lambda

The hyperparameter λ𝜆\lambda allocates the ratio of training data between patch-level and token-level training. A larger λ𝜆\lambda results in more tokens being compressed into patches, leading to a higher acceleration rate, but it may also leave insufficient data to adjust the model to the token-level. In this section, we investigate the effect of λ𝜆\lambda under the settings of 370M model parameters, a batch size of 512K, and a patch size of K=4𝐾4K=4. We consider two scenarios:

1. 1.

   Unlimited computational resources: We assess the impact of varying λ𝜆\lambda while keeping the data size constant (90B tokens). The results are shown in Figure [7](#S3.F7 "Figure 7 ‣ 3.4 Effect of Patch Size 𝐾 ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models").
2. 2.

   Unlimited training data: We identify the optimal λ𝜆\lambda under a fixed amount of computational resources (tokens + patches = 56.25B). For example, when λ=1/2𝜆12\lambda=1/2, the size of training data should be 90B tokens, with 45B tokens being compressed into 11.25B patches. The results are shown in Figure [7](#S3.F7 "Figure 7 ‣ 3.4 Effect of Patch Size 𝐾 ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models").

Figure [7](#S3.F7 "Figure 7 ‣ 3.4 Effect of Patch Size 𝐾 ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models") shows that the model performance initially rises and later falls as λ𝜆\lambda increases, with a turning point near λ=1/4𝜆14\lambda=1/4. The performance improvements when λ<1/4𝜆14\lambda<1/4 can be attributed to the inherent benefits of patch-level training, as analyzed in Section [3.2](#S3.SS2 "3.2 Main Results ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models"). When λ𝜆\lambda exceeds 3/4343/4, further increasing λ𝜆\lambda leaves insufficient data to adjust the model to the token-level, leading to a rapid decline in performance. Figure [7](#S3.F7 "Figure 7 ‣ 3.4 Effect of Patch Size 𝐾 ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models"), on the other hand, shows that when computational resources are limited, the optimal value for λ𝜆\lambda is around 2/3232/3. Note that these conclusions are specific to the current settings and should be used as a reference only. The optimal λ𝜆\lambda may vary depending on factors such as data size and patch size. To determine the optimal value of λ𝜆\lambda in any scenario, it is essential to establish the scaling law for patch-level training.

### 3.6 Neuron Activation

In this section, we quantitatively explain why patch-level training leads to better learning efficiency from the perspective of neuron activation. The training of LLMs is essentially a process of embedding knowledge from the training set into the model’s parameters. During this process, the model employs all of its parameters to encode every token and updates the relevant parameters based on the gradient feedback. We argue that this is an inefficient process for large models, as the knowledge encapsulated in each token is only associated with a small subset of model parameters, resulting in a limited number of effectively activated and updated parameters.

We substantiate this by measuring the percentage of activated neurons for models of different patch sizes, as depicted in Figure [8](#S3.F8 "Figure 8 ‣ 3.6 Neuron Activation ‣ 3 Experiments ‣ Patch-Level Training for Large Language Models"). In the token-level model (K=1), only a small proportion of neurons are activated, particularly in the lower layers. By grouping multiple tokens into a patch, the information density processed at each step is increased, which is manifested as increased neuron activation rates. Therefore, patch-level training exhibits a better learning efficiency compared to token-level training.

Figure 8: Percentage of activated neurons for models of different patch sizes. Output neurons of each model layer (FFN output) with an absolute value greater than 0.5 are classified as activated.

## 4 Related Work

Model Growth. Our approach draws inspiration from transfer learning, reducing training costs by transferring knowledge acquired at a lower training cost (patch-level) to a model with a higher training cost (token-level). A similar strategy has been employed in studies of model growth, which train large models at a relatively lower cost by progressively increasing the model size during training. For example, Gong et al. ([2019](#bib.bib21)); Yang et al. ([2020](#bib.bib64)) improve the training efficiency by transferring knowledge from a shallow model to a deep model, where model layers are progressively stacked during training. Gu et al. ([2021](#bib.bib24)) further proposes progressive compound growth, where the model grows at multiple dimensions during training, including the context length, model width, and the number of layers. Subsequent studies primarily focus on the initialization problem during the model growth process, i.e., how to expand the small model into a large one. Chen et al. ([2022](#bib.bib7)); Yao et al. ([2024](#bib.bib65)) aim to achieve function-preserving growth (Chen et al., [2015](#bib.bib9)) that the post-growth model have the same function as the pre-growth model, which intuitively ensures smooth knowledge transfer. Wang et al. ([2023](#bib.bib63)); Pan et al. ([2023](#bib.bib46)) introduce learnable linear operators that linearly map the parameters of the small model to initialize the large model. Compared to model growth, patch-level training is more flexible and generalizable as it does not necessitate specialized model architectures or carefully crafted model mapping strategies. Additionally, patch-level training is orthogonal to model growth, leaving the potential for their joint application.

Multi-Token Prediction. Our approach improves training efficiency by concurrently predicting all tokens in the next patch. Similar attempts of multi-token prediction have been made in the past to improve the inference efficiency, including non-autoregressive generation (Gu et al., [2018](#bib.bib22)) and speculative decoding (Stern et al., [2018](#bib.bib55); Leviathan et al., [2023](#bib.bib37); Chen et al., [2023](#bib.bib6)). Non-autoregressive generation reduces the number of decoding iterations by generating all tokens at once, involving techniques such as knowledge distillation (Kim & Rush, [2016](#bib.bib33); Shao et al., [2022](#bib.bib52)), training objectives (Shao et al., [2021](#bib.bib51); Ghazvininejad et al., [2020](#bib.bib19); Du et al., [2021](#bib.bib14); Shao & Feng, [2022](#bib.bib50); Ma et al., [2023](#bib.bib43)), latent modeling (Kaiser et al., [2018](#bib.bib31); Ma et al., [2019](#bib.bib42); Shu et al., [2020](#bib.bib54)), iterative decoding (Lee et al., [2018](#bib.bib36); Ghazvininejad et al., [2019](#bib.bib18); Gu et al., [2019](#bib.bib23)), and expressive model architectures (Libovický & Helcl, [2018](#bib.bib39); Huang et al., [2022](#bib.bib30); Gui et al., [2023](#bib.bib25)). Despite the reduction in decoding iterations, the computational demands (FLOPs) do not decrease and may even rise for certain structures, leading to increased training costs. Speculative decoding is a novel decoding paradigm for accelerating LLM inference. During each step of decoding, it drafts several future tokens efficiently and then verifies them in parallel. The model for generating draft tokens can either be a relatively small model (Leviathan et al., [2023](#bib.bib37); Chen et al., [2023](#bib.bib6)) or a non-autoregressive model that generates multiple tokens in parallel (Cai et al., [2024](#bib.bib5); Lin et al., [2024](#bib.bib40); Fu et al., [2024](#bib.bib16); Li et al., [2024](#bib.bib38)). Recently, [Gloeckle et al.](#bib.bib20)  pointed out that besides accelerating the inference, multi-token prediction also serves as an auxiliary training task to enhance the training signal. The key difference between our approach and these works lies in our utilization of multi-token prediction for reducing sequence length during training, with the goal of improving training efficiency. Regarding the model structure, we avoid introducing extra parameters and employ a single head for multi-token prediction.

Patch-Level Model. The concept of handling input data at the patch-level has emerged as a pivotal strategy for enhancing computational efficiency and capturing local features. Convolutional Neural Networks (CNNs, Lecun et al., [1998](#bib.bib35)) are perhaps the earliest attempt of patch-level processing, utilizing kernel filters to extract local features from images. More recently, Vision Transformers (Dosovitskiy et al., [2021](#bib.bib13)) have revolutionized image processing by employing CNNs to encode an image into non-overlapping image patches, thereby enabling Transformers to efficiently capture both local and global features. Similarly, speech models also rely on CNNs to compress high-frequency waveforms into hidden state representations (Baevski et al., [2020](#bib.bib2); Hsu et al., [2021](#bib.bib29)), which can be interpreted as speech patches. For textual data, characters, the basic building blocks of text, can be downsampled into more compact representations (Clark et al., [2022](#bib.bib10)) or merged into tokens (Sennrich et al., [2016](#bib.bib49); Kudo & Richardson, [2018](#bib.bib34)). Recently, there have been attempts to further compress tokens into patches, with the model directly processing the patch sequence (Yu et al., [2024](#bib.bib66); Mujika, [2023](#bib.bib45); Ho et al., [2024](#bib.bib27)). However, it remains necessary to upsample the patch representation and input it into a token-level autoregressive model for the likelihood inference.

## 5 Conclusion

This paper introduces patch-level training, an efficient training approach for large language models, in which models read training data in patches and learn to predict the next patch. Following this, a small amount of training data is utilized to adjust the model to the token-level. Experimental results show that this method can cut LLM training costs by 50% while maintaining comparable performance.

Yet, our exploration of patch-level training is still in its infancy, and advancements in the following directions could further enhance this methodology: assessing the scalability of patch-level training by evaluating its performance on larger models and datasets; establishing an empirical scaling law for patch-level training, ideally incorporating both K𝐾K and λ𝜆\lambda; developing advanced training techniques to accommodate larger K𝐾K and λ𝜆\lambda, thereby pushing acceleration rates to a higher level; further investigating the potential of patch-level training in multi-epoch training; exploring the applicability of patch-level training to other data modalities, such as images, speech, and video.

## References

* Achiam et al. (2023)

  Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya,
  Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman,
  Shyamal Anadkat, et al.
  Gpt-4 technical report.
  *arXiv preprint arXiv:2303.08774*, 2023.
* Baevski et al. (2020)

  Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli.
  wav2vec 2.0: A framework for self-supervised learning of speech
  representations.
  *Advances in neural information processing systems*,
  33:12449–12460, 2020.
* Bai et al. (2023)

  Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan,
  Wenbin Ge, Yu Han, Fei Huang, et al.
  Qwen technical report.
  *arXiv preprint arXiv:2309.16609*, 2023.
* Bisk et al. (2020)

  Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi.
  Piqa: Reasoning about physical commonsense in natural language.
  *Proceedings of the AAAI Conference on Artificial Intelligence*,
  34(05):7432–7439, Apr. 2020.
  doi: 10.1609/aaai.v34i05.6239.
  URL <https://ojs.aaai.org/index.php/AAAI/article/view/6239>.
* Cai et al. (2024)

  Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen,
  and Tri Dao.
  Medusa: Simple llm inference acceleration framework with multiple
  decoding heads.
  *arXiv preprint arXiv:2401.10774*, 2024.
* Chen et al. (2023)

  Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau,
  Laurent Sifre, and John Jumper.
  Accelerating large language model decoding with speculative sampling.
  *arXiv preprint arXiv:2302.01318*, 2023.
* Chen et al. (2022)

  Cheng Chen, Yichun Yin, Lifeng Shang, Xin Jiang, Yujia Qin, Fengyu Wang, Zhi
  Wang, Xiao Chen, Zhiyuan Liu, and Qun Liu.
  bert2BERT: Towards reusable pretrained language models.
  In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.),
  *Proceedings of the 60th Annual Meeting of the Association for
  Computational Linguistics (Volume 1: Long Papers)*, pp.  2134–2148, Dublin,
  Ireland, May 2022. Association for Computational Linguistics.
  doi: 10.18653/v1/2022.acl-long.151.
  URL <https://aclanthology.org/2022.acl-long.151>.
* Chen et al. (2020)

  Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and
  Ilya Sutskever.
  Generative pretraining from pixels.
  In *International conference on machine learning*, pp. 1691–1703. PMLR, 2020.
* Chen et al. (2015)

  Tianqi Chen, Ian Goodfellow, and Jonathon Shlens.
  Net2net: Accelerating learning via knowledge transfer.
  *arXiv preprint arXiv:1511.05641*, 2015.
* Clark et al. (2022)

  Jonathan H. Clark, Dan Garrette, Iulia Turc, and John Wieting.
  Canine: Pre-training an efficient tokenization-free encoder for
  language representation.
  *Transactions of the Association for Computational Linguistics*,
  10:73–91, 2022.
  doi: 10.1162/tacl˙a˙00448.
  URL <https://aclanthology.org/2022.tacl-1.5>.
* Clark et al. (2018)

  Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa
  Schoenick, and Oyvind Tafjord.
  Think you have solved question answering? try arc, the ai2 reasoning
  challenge.
  *arXiv preprint arXiv:1803.05457*, 2018.
* Dao (2024)

  Tri Dao.
  Flashattention-2: Faster attention with better parallelism and work
  partitioning.
  In *The Twelfth International Conference on Learning
  Representations*, 2024.
  URL <https://openreview.net/forum?id=mZn2Xyh9Ec>.
* Dosovitskiy et al. (2021)

  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
  Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
  Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.
  An image is worth 16x16 words: Transformers for image recognition at
  scale.
  In *International Conference on Learning Representations*, 2021.
  URL <https://openreview.net/forum?id=YicbFdNTTy>.
* Du et al. (2021)

  Cunxiao Du, Zhaopeng Tu, and Jing Jiang.
  Order-agnostic cross entropy for non-autoregressive machine
  translation.
  In *International conference on machine learning*, pp. 2849–2859. PMLR, 2021.
* Duan et al. (2024)

  Yuchen Duan, Weiyun Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, Yu Qiao,
  Hongsheng Li, Jifeng Dai, and Wenhai Wang.
  Vision-rwkv: Efficient and scalable visual perception with rwkv-like
  architectures.
  *arXiv preprint arXiv:2403.02308*, 2024.
* Fu et al. (2024)

  Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang.
  Break the sequential dependency of LLM inference using lookahead
  decoding.
  In *Forty-first International Conference on Machine Learning*,
  2024.
  URL <https://openreview.net/forum?id=eDjvSFOkXw>.
* Gao et al. (2020)

  Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles
  Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al.
  The pile: An 800gb dataset of diverse text for language modeling.
  *arXiv preprint arXiv:2101.00027*, 2020.
* Ghazvininejad et al. (2019)

  Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer.
  Mask-predict: Parallel decoding of conditional masked language
  models.
  In *Proceedings of the 2019 Conference on Empirical Methods in
  Natural Language Processing and the 9th International Joint Conference on
  Natural Language Processing (EMNLP-IJCNLP)*, pp.  6112–6121, 2019.
  URL <https://www.aclweb.org/anthology/D19-1633>.
* Ghazvininejad et al. (2020)

  Marjan Ghazvininejad, Vladimir Karpukhin, Luke Zettlemoyer, and Omer Levy.
  Aligned cross entropy for non-autoregressive machine translation.
  In *Proceedings of the 37th International Conference on Machine
  Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of
  *Proceedings of Machine Learning Research*, pp.  3515–3523. PMLR,
  2020.
  URL <http://proceedings.mlr.press/v119/ghazvininejad20a.html>.
* (20)

  Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, and
  Gabriel Synnaeve.
  Better & faster large language models via multi-token prediction.
  In *Forty-first International Conference on Machine Learning*.
* Gong et al. (2019)

  Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tieyan Liu.
  Efficient training of BERT by progressively stacking.
  In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.),
  *Proceedings of the 36th International Conference on Machine Learning*,
  volume 97 of *Proceedings of Machine Learning Research*, pp. 2337–2346. PMLR, 09–15 Jun 2019.
  URL <https://proceedings.mlr.press/v97/gong19a.html>.
* Gu et al. (2018)

  Jiatao Gu, James Bradbury, Caiming Xiong, Victor O.K. Li, and Richard Socher.
  Non-autoregressive neural machine translation.
  In *International Conference on Learning Representations*, 2018.
  URL <https://openreview.net/forum?id=B1l8BtlCb>.
* Gu et al. (2019)

  Jiatao Gu, Changhan Wang, and Junbo Zhao.
  Levenshtein transformer.
  In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence
  d’Alché-Buc, Emily B. Fox, and Roman Garnett (eds.), *Advances
  in Neural Information Processing Systems 32: Annual Conference on Neural
  Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019,
  Vancouver, BC, Canada*, pp.  11179–11189, 2019.
  URL
  <https://proceedings.neurips.cc/paper/2019/hash/675f9820626f5bc0afb47b57890b466e-Abstract.html>.
* Gu et al. (2021)

  Xiaotao Gu, Liyuan Liu, Hongkun Yu, Jing Li, Chen Chen, and Jiawei Han.
  On the transformer growth for progressive BERT training.
  In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek
  Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty,
  and Yichao Zhou (eds.), *Proceedings of the 2021 Conference of the North
  American Chapter of the Association for Computational Linguistics: Human
  Language Technologies*, pp.  5174–5180, Online, June 2021. Association for
  Computational Linguistics.
  doi: 10.18653/v1/2021.naacl-main.406.
  URL <https://aclanthology.org/2021.naacl-main.406>.
* Gui et al. (2023)

  Shangtong Gui, Chenze Shao, Zhengrui Ma, xishan zhang, Yunji Chen, and Yang
  Feng.
  Non-autoregressive machine translation with probabilistic
  context-free grammar.
  In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and
  S. Levine (eds.), *Advances in Neural Information Processing Systems*,
  volume 36, pp.  5598–5615. Curran Associates, Inc., 2023.
  URL
  <https://proceedings.neurips.cc/paper_files/paper/2023/file/11c7f1dd168439884b6dfb43a7891432-Paper-Conference.pdf>.
* Hendrycks et al. (2021)

  Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn
  Song, and Jacob Steinhardt.
  Measuring massive multitask language understanding.
  In *International Conference on Learning Representations*, 2021.
  URL <https://openreview.net/forum?id=d7KBjmI3GmQ>.
* Ho et al. (2024)

  Namgyu Ho, Sangmin Bae, Taehyeon Kim, Hyunjik Jo, Yireun Kim, Tal Schuster,
  Adam Fisch, James Thorne, and Se-Young Yun.
  Block transformer: Global-to-local language modeling for fast
  inference.
  *arXiv preprint arXiv:2406.02657*, 2024.
* Hoffmann et al. (2022)

  Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor
  Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes
  Welbl, Aidan Clark, et al.
  Training compute-optimal large language models.
  *arXiv preprint arXiv:2203.15556*, 2022.
* Hsu et al. (2021)

  Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan
  Salakhutdinov, and Abdelrahman Mohamed.
  Hubert: Self-supervised speech representation learning by masked
  prediction of hidden units.
  *IEEE/ACM Transactions on Audio, Speech, and Language
  Processing*, 29:3451–3460, 2021.
* Huang et al. (2022)

  Fei Huang, Hao Zhou, Yang Liu, Hang Li, and Minlie Huang.
  Directed acyclic transformer for non-autoregressive machine
  translation.
  In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari,
  Gang Niu, and Sivan Sabato (eds.), *Proceedings of the 39th
  International Conference on Machine Learning*, volume 162 of
  *Proceedings of Machine Learning Research*, pp.  9410–9428. PMLR,
  17–23 Jul 2022.
  URL <https://proceedings.mlr.press/v162/huang22m.html>.
* Kaiser et al. (2018)

  Lukasz Kaiser, Samy Bengio, Aurko Roy, Ashish Vaswani, Niki Parmar, Jakob
  Uszkoreit, and Noam Shazeer.
  Fast decoding in sequence models using discrete latent variables.
  In Jennifer Dy and Andreas Krause (eds.), *Proceedings of the
  35th International Conference on Machine Learning*, volume 80 of
  *Proceedings of Machine Learning Research*, pp.  2390–2399,
  StockholmsmÃ€ssan, Stockholm Sweden, 10–15 Jul 2018. PMLR.
  URL <http://proceedings.mlr.press/v80/kaiser18a.html>.
* Kaplan et al. (2020)

  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon
  Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
  Scaling laws for neural language models.
  *arXiv preprint arXiv:2001.08361*, 2020.
* Kim & Rush (2016)

  Yoon Kim and Alexander M. Rush.
  Sequence-level knowledge distillation.
  In *Proceedings of the 2016 Conference on Empirical Methods in
  Natural Language Processing*, pp.  1317–1327, Austin, Texas, November 2016.
  Association for Computational Linguistics.
  doi: 10.18653/v1/D16-1139.
  URL <https://www.aclweb.org/anthology/D16-1139>.
* Kudo & Richardson (2018)

  Taku Kudo and John Richardson.
  SentencePiece: A simple and language independent subword
  tokenizer and detokenizer for neural text processing.
  In Eduardo Blanco and Wei Lu (eds.), *Proceedings of the 2018
  Conference on Empirical Methods in Natural Language Processing: System
  Demonstrations*, pp.  66–71, Brussels, Belgium, November 2018. Association
  for Computational Linguistics.
  doi: 10.18653/v1/D18-2012.
  URL <https://aclanthology.org/D18-2012>.
* Lecun et al. (1998)

  Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner.
  Gradient-based learning applied to document recognition.
  *Proceedings of the IEEE*, 86(11):2278–2324, 1998.
  doi: 10.1109/5.726791.
* Lee et al. (2018)

  Jason Lee, Elman Mansimov, and Kyunghyun Cho.
  Deterministic non-autoregressive neural sequence modeling by
  iterative refinement.
  In *Proceedings of the 2018 Conference on Empirical Methods in
  Natural Language Processing*, pp.  1173–1182, Brussels, Belgium,
  October-November 2018. Association for Computational Linguistics.
  doi: 10.18653/v1/D18-1149.
  URL <https://www.aclweb.org/anthology/D18-1149>.
* Leviathan et al. (2023)

  Yaniv Leviathan, Matan Kalman, and Yossi Matias.
  Fast inference from transformers via speculative decoding.
  In *International Conference on Machine Learning*, pp. 19274–19286. PMLR, 2023.
* Li et al. (2024)

  Zeping Li, Xinlong Yang, Ziheng Gao, Ji Liu, Zhuang Liu, Dong Li, Jinzhang
  Peng, Lu Tian, and Emad Barsoum.
  Amphista: Accelerate llm inference with bi-directional multiple
  drafting heads in a non-autoregressive style.
  *arXiv preprint arXiv:2406.13170*, 2024.
* Libovický & Helcl (2018)

  Jindřich Libovický and Jindřich Helcl.
  End-to-end non-autoregressive neural machine translation with
  connectionist temporal classification.
  In *Proceedings of the 2018 Conference on Empirical Methods in
  Natural Language Processing*, pp.  3016–3021, Brussels, Belgium,
  October-November 2018. Association for Computational Linguistics.
  doi: 10.18653/v1/D18-1336.
  URL <https://www.aclweb.org/anthology/D18-1336>.
* Lin et al. (2024)

  Feng Lin, Hanling Yi, Hongbin Li, Yifan Yang, Xiaotian Yu, Guangming Lu, and
  Rong Xiao.
  Bita: Bi-directional tuning for lossless acceleration in large
  language models.
  *arXiv preprint arXiv:2401.12522*, 2024.
* Loshchilov & Hutter (2019)

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  In *International Conference on Learning Representations*, 2019.
  URL <https://openreview.net/forum?id=Bkg6RiCqY7>.
* Ma et al. (2019)

  Xuezhe Ma, Chunting Zhou, Xian Li, Graham Neubig, and Eduard Hovy.
  FlowSeq: Non-autoregressive conditional sequence generation with
  generative flow.
  In *Proceedings of the 2019 Conference on Empirical Methods in
  Natural Language Processing and the 9th International Joint Conference on
  Natural Language Processing (EMNLP-IJCNLP)*, pp.  4282–4292, Hong Kong,
  China, November 2019. Association for Computational Linguistics.
  doi: 10.18653/v1/D19-1437.
  URL <https://www.aclweb.org/anthology/D19-1437>.
* Ma et al. (2023)

  Zhengrui Ma, Chenze Shao, Shangtong Gui, Min Zhang, and Yang Feng.
  Fuzzy alignments in directed acyclic graph for non-autoregressive
  machine translation.
  In *The Eleventh International Conference on Learning
  Representations*, 2023.
  URL <https://openreview.net/forum?id=LSz-gQyd0zE>.
* Merity et al. (2017)

  Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher.
  Pointer sentinel mixture models.
  In *International Conference on Learning Representations*, 2017.
  URL <https://openreview.net/forum?id=Byj72udxe>.
* Mujika (2023)

  Asier Mujika.
  Hierarchical attention encoder decoder.
  *arXiv preprint arXiv:2306.01070*, 2023.
* Pan et al. (2023)

  Yu Pan, Ye Yuan, Yichun Yin, Zenglin Xu, Lifeng Shang, Xin Jiang, and Qun Liu.
  Reusing pretrained models by multi-linear operators for efficient
  training.
  In *Thirty-seventh Conference on Neural Information Processing
  Systems*, 2023.
  URL <https://openreview.net/forum?id=RgNXKIrWyU>.
* Pappagari et al. (2019)

  Raghavendra Pappagari, Piotr Zelasko, Jesús Villalba, Yishay Carmiel, and
  Najim Dehak.
  Hierarchical transformers for long document classification.
  In *2019 IEEE automatic speech recognition and understanding
  workshop (ASRU)*, pp.  838–844. IEEE, 2019.
* Sakaguchi et al. (2020)

  Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi.
  Winogrande: An adversarial winograd schema challenge at scale.
  *Proceedings of the AAAI Conference on Artificial Intelligence*,
  34(05):8732–8740, Apr. 2020.
  doi: 10.1609/aaai.v34i05.6399.
  URL <https://ojs.aaai.org/index.php/AAAI/article/view/6399>.
* Sennrich et al. (2016)

  Rico Sennrich, Barry Haddow, and Alexandra Birch.
  Neural machine translation of rare words with subword units.
  In Katrin Erk and Noah A. Smith (eds.), *Proceedings of the 54th
  Annual Meeting of the Association for Computational Linguistics (Volume 1:
  Long Papers)*, pp.  1715–1725, Berlin, Germany, August 2016. Association
  for Computational Linguistics.
  doi: 10.18653/v1/P16-1162.
  URL <https://aclanthology.org/P16-1162>.
* Shao & Feng (2022)

  Chenze Shao and Yang Feng.
  Non-monotonic latent alignments for ctc-based non-autoregressive
  machine translation.
  In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh
  (eds.), *Advances in Neural Information Processing Systems*, volume 35,
  pp.  8159–8173. Curran Associates, Inc., 2022.
  URL
  <https://proceedings.neurips.cc/paper_files/paper/2022/file/35f805e65c77652efa731edc10c8e3a6-Paper-Conference.pdf>.
* Shao et al. (2021)

  Chenze Shao, Yang Feng, Jinchao Zhang, Fandong Meng, and Jie Zhou.
  Sequence-Level Training for Non-Autoregressive Neural Machine
  Translation.
  *Computational Linguistics*, 47(4):891–925,
  12 2021.
  ISSN 0891-2017.
  doi: 10.1162/coli˙a˙00421.
  URL <https://doi.org/10.1162/coli_a_00421>.
* Shao et al. (2022)

  Chenze Shao, Xuanfu Wu, and Yang Feng.
  One reference is not enough: Diverse distillation with reference
  selection for non-autoregressive translation.
  In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir
  Meza Ruiz (eds.), *Proceedings of the 2022 Conference of the North
  American Chapter of the Association for Computational Linguistics: Human
  Language Technologies*, pp.  3779–3791, Seattle, United States, July 2022.
  Association for Computational Linguistics.
  doi: 10.18653/v1/2022.naacl-main.277.
  URL <https://aclanthology.org/2022.naacl-main.277>.
* Shazeer (2020)

  Noam Shazeer.
  Glu variants improve transformer.
  *arXiv preprint arXiv:2002.05202*, 2020.
* Shu et al. (2020)

  Raphael Shu, Jason Lee, Hideki Nakayama, and Kyunghyun Cho.
  Latent-variable non-autoregressive neural machine translation with
  deterministic inference using a delta posterior.
  In *The Thirty-Fourth AAAI Conference on Artificial
  Intelligence, AAAI 2020, New York, NY, USA, February 7-12, 2020*, pp. 8846–8853. AAAI Press, 2020.
  URL <https://aaai.org/ojs/index.php/AAAI/article/view/6413>.
* Stern et al. (2018)

  Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit.
  Blockwise parallel decoding for deep autoregressive models.
  *Advances in Neural Information Processing Systems*, 31, 2018.
* Su et al. (2021)

  Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu.
  Roformer: Enhanced transformer with rotary position embedding.
  *arXiv preprint arXiv:2104.09864*, 2021.
* Taori et al. (2023)

  Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos
  Guestrin, Percy Liang, and Tatsunori B. Hashimoto.
  Stanford alpaca: An instruction-following llama model.
  <https://github.com/tatsu-lab/stanford_alpaca>, 2023.
* Team et al. (2023)

  Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac,
  Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al.
  Gemini: a family of highly capable multimodal models.
  *arXiv preprint arXiv:2312.11805*, 2023.
* Touvron et al. (2023a)

  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
  Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric
  Hambro, Faisal Azhar, et al.
  Llama: Open and efficient foundation language models.
  *arXiv preprint arXiv:2302.13971*, 2023a.
* Touvron et al. (2023b)

  Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine
  Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale,
  et al.
  Llama 2: Open foundation and fine-tuned chat models.
  *arXiv preprint arXiv:2307.09288*, 2023b.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  *Advances in neural information processing systems*, 30, 2017.
* Wan et al. (2023)

  Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, et al.
  Efficient large language models: A survey.
  *arXiv preprint arXiv:2312.03863*, 1, 2023.
* Wang et al. (2023)

  Peihao Wang, Rameswar Panda, Lucas Torroba Hennigen, Philip Greengard, Leonid
  Karlinsky, Rogerio Feris, David Daniel Cox, Zhangyang Wang, and Yoon Kim.
  Learning to grow pretrained models for efficient transformer
  training.
  In *The Eleventh International Conference on Learning
  Representations*, 2023.
  URL <https://openreview.net/forum?id=cDYRS5iZ16f>.
* Yang et al. (2020)

  Cheng Yang, Shengnan Wang, Chao Yang, Yuechuan Li, Ru He, and Jingqiao Zhang.
  Progressively stacking 2.0: A multi-stage layerwise training method
  for bert training speedup.
  *arXiv preprint arXiv:2011.13635*, 2020.
* Yao et al. (2024)

  Yiqun Yao, Zheng Zhang, Jing Li, and Yequan Wang.
  Masked structural growth for 2x faster language model pre-training.
  In *The Twelfth International Conference on Learning
  Representations*, 2024.
  URL <https://openreview.net/forum?id=rL7xsg1aRn>.
* Yu et al. (2024)

  Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer,
  and Mike Lewis.
  Megabyte: Predicting million-byte sequences with multiscale
  transformers.
  *Advances in Neural Information Processing Systems*, 36, 2024.
* Zellers et al. (2019)

  Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.
  Hellaswag: Can a machine really finish your sentence?
  *arXiv preprint arXiv:1905.07830*, 2019.
* Zhang & Sennrich (2019)

  Biao Zhang and Rico Sennrich.
  Root mean square layer normalization.
  *Advances in Neural Information Processing Systems*, 32, 2019.
* Zheng et al. (2024)

  Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao
  Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al.
  Judging llm-as-a-judge with mt-bench and chatbot arena.
  *Advances in Neural Information Processing Systems*, 36, 2024.
* Zhu et al. (2024)

  Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang
  Wang.
  Vision mamba: Efficient visual representation learning with
  bidirectional state space model.
  *arXiv preprint arXiv:2401.09417*, 2024.
