---
arxiv: '2308.10248'
authors:
- Alexander Matt Turner
- Lisa Thiergart
- Gavin Leech
- David Udell
- Juan J. Vazquez
- Ulisse Mini
- Monte MacDiarmid
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Steering Language Models With Activation Engineering
url: https://arxiv.org/abs/2308.10248
year: 2023
---

# Activation Addition: Steering Language Models Without Optimization

Alexander Matt Turner,1
Lisa Thiergart,2
David Udell,3
Gavin Leech,4 5
Ulisse Mini,2
Monte MacDiarmid3

###### Abstract

Reliably controlling the behavior of large language models is a pressing open problem. Existing methods include supervised finetuning, reinforcement learning from human feedback, prompt engineering and guided decoding. We instead investigate activation engineering: modifying activations at inference-time to predictably alter model behavior. We bias the forward pass with a ‘steering vector’ implicitly specified through natural language. Past work (Subramani, Suresh, and Peters [2022](#bib.bib48); Hernandez, Li, and Andreas [2023](#bib.bib18)) learned these steering vectors; our Activation Addition  (ActAdd) method instead computes them by taking activation differences resulting from pairs of prompts. We demonstrate ActAdd  on GPT-2  on OpenWebText and ConceptNet, and replicate on Llama-13B and GPT-J-6B. Our approach yields inference-time control over high-level properties of output & preserves performance off-target. It takes far less compute/implementation than finetuning, allows natural language specification by users, and its overhead scales naturally with model size.

## Introduction

The success of large language models (LLMs) pretrained on massive corpora (Peters et al. [2018](#bib.bib39); Devlin et al. [2018](#bib.bib8); Radford et al. [2019](#bib.bib42); Brown et al. [2020](#bib.bib5)) comes with a large caveat: our ability to control the output of these models remains partial & computationally expensive. LLM controls include supervised finetuning (Devlin et al. [2018](#bib.bib8)), reinforcement learning from human feedback (RLHF) (Ziegler et al. [2019](#bib.bib56)), prompt engineering (Radford et al. [2019](#bib.bib42)), & guided decoding (Ghazvininejad et al. [2017](#bib.bib13); Gu, Cho, and Li [2017](#bib.bib16)).

We instead investigate controlling LLMs with minimal computational overhead, by modifying their activations. We call this approach activation engineering, as pioneered by Subramani, Suresh, and Peters [2022](#bib.bib48) and Hernandez, Li, and Andreas [2023](#bib.bib18). We go further by demonstrating natural-language control of activations and thereby output text. We use pairs of prompts to implicitly specify a direction in activation space which we scale to control the model (e.g. its sentiment, truthfulness, but also topic and style; see Appendix Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")). In contrast with past work (Shen et al. [2017](#bib.bib44); Dathathri et al. [2020](#bib.bib7); Subramani, Suresh, and Peters [2022](#bib.bib48); Li et al. [2023b](#bib.bib25)), the method does not require any labelled data, nor backward passes to learn an encoder, controller, classifier, or direction in activation space.

| Prompt | Completion | Steering |
| --- | --- | --- |
| I hate you  because | |  | | --- | | …you are the most disgust- | | -ing thing I have ever seen. | | None |
|  | |  | | --- | | …you are so beautiful and I | | want to be with you forever. | | ActAdd  (love) |
| I went up  to my  friend and  said | |  | | --- | | “I’m sorry, I can’t help you.” | | “No,” he said. “You’re not.” | | None |
|  | |  | | --- | | “I’m going to talk about the | | wedding in this episode of | | Wedding Season. I think it’s | | a really good episode. It’s | | about how you’re supposed | | to talk about weddings.” | | ActAdd  (weddings) |

Table 1: Output before and after applying ActAdd. The steering vectors used are
(Love−Hate)LoveHate(\texttt{Love}-\texttt{Hate}) and
(I talk about weddings constantly−I do not talk about weddings constantly)I talk about weddings constantlyI do not talk about weddings constantly(\texttt{I talk about weddings constantly}-\texttt{I do not talk about weddings constantly}). See also Appendix Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") for extensive examples.

We make three main contributions:

1. 1.

   We find that combining forward passes works well in GPT-2, despite the model not being trained for this.
2. 2.

   We develop a lightweight control method, Activation Additions (ActAdd), which works at inference time and requires no optimization or labelled data. Our technique is thus a promising direction for user control of LLMs.
3. 3.

   We show that ActAdd  preserves overall model performance, involves far less compute and implementation effort compared to finetuning and RLHF, and scales naturally with model size. It is notably easier to implement than other activation engineering methods, since it requires no backward passes or labelled data.

## Related Work

#### Latent space arithmetic

Research in generative models for computer vision has long demonstrated the ability to steer image generation using derived vectors, including steering latent variables – most famously, intervening on a dimension that corresponds to smiles in images (Larsen et al. [2016](#bib.bib22); White [2016](#bib.bib53)).

Similarly, in the text domain, classic results on the word2vec embedding show that arithmetic on word vectors can capture some parts of semantic reasoning (for instance, analogies) (Mikolov, Yih, and Zweig [2013](#bib.bib32); Mikolov et al. [2013](#bib.bib31)). Our work differs in being computed (with forward passes) rather than learned (with backward passes); in operating on activation space, rather than embedding or weight space; and in presenting a natural-language user interface.

#### LLM steering

Many approaches attempt to affect the output of a pretrained LLM, whether:

* •

  Intervening on weights, as with supervised fine-tuning, RLHF, steerable layers, and weight editing (that is, targeted fine-tuning) (Ranzato et al. [2016](#bib.bib43); Ziegler et al. [2019](#bib.bib56); Dathathri et al. [2020](#bib.bib7); Meng et al. [2023](#bib.bib28); Ilharco et al. [2023](#bib.bib19)).
  However, RLHF and weight editing are known to have side-effects on overall model performance (OpenAI [2023](#bib.bib37); Hase et al. [2023](#bib.bib17); Brown et al. [2023](#bib.bib4)).
* •

  Intervening at decoding, as with guided or trainable decoding (Gu, Cho, and Li [2017](#bib.bib16); Grover et al. [2019](#bib.bib15))
  (See Zhang et al. [2022](#bib.bib54) for an overview of controlled generation and Jin et al. [2022](#bib.bib20) for textual style transfer.)
* •

  Intervening on the prompt, as with automated prompt engineering (Shin et al. [2020](#bib.bib45); Zhou et al. [2022](#bib.bib55)) and soft or continuous prompting (Lester, Al-Rfou, and Constant [2021](#bib.bib23); Li and Liang [2021](#bib.bib26))
* •

  Intervening on activations, for instance by freezing the weights of the LLM and searching for a ‘steering vector’ of activations, e.g. using gradient descent (Subramani, Suresh, and Peters [2022](#bib.bib48); Hernandez, Li, and Andreas [2023](#bib.bib18)). These optimized extraction methods, which search for a steering vector, differ from
  extraction methods which directly compute it (present work and Li et al. [2023b](#bib.bib25)). We now elaborate on this family of methods.

|  |  |  |  |
| --- | --- | --- | --- |
|  | |  | | --- | | Intervenes on model… | | |
| Steering vectors via | … weights | … activations |
| Differences after fine-tuning | Ilharco 2023 | N/A |
| Per-query gradient-based search | Meng 2022,   Orgad 2023 | Dathathri 2020,  Subramani 2022,  Hernandez 2023 |
| Differences in truthy attention heads | N/A | Li 2023b |
| Differences between prompt pairs | N/A | ActAdd  (present work) |

Table 2: Locating our work in the steering vector literature.

#### Activation engineering

Activation engineering involves creating vectors of activations which cause desired changes to output text when added to the forward passes of a frozen LLM. (Table [2](#Sx2.T2 "Table 2 ‣ LLM steering ‣ Related Work ‣ Activation Addition: Steering Language Models Without Optimization") organizes prior work by intervention type.) An early antecedent of the approach is the Plug-and-Play Language Model of Dathathri et al. [2020](#bib.bib7). This uses a separate classifier (one classifier per attribute to steer towards) to perturb the
model’s activations to generate text that accords more closely with the classifier’s target.

Subramani, Suresh, and Peters [2022](#bib.bib48) extract latent steering vectors from a frozen LLM, successfully discovering sentence-specific vectors which steer completions to near-perfect BLEU scores (i.e, control of the LLM’s generation) and unsupervised style transfer. However, the method requires running gradient descent for each new steering vector.

Inference-Time Intervention (ITI) (Li et al. [2023b](#bib.bib25)) independently
developed a similar method which computes steering vectors (though not without some prior optimization). They use linear probe accuracy to find attention heads with different activation distributions for true and false statements (which requires labelled data, in this case TruthfulQA, to operationalize ‘truth’). The authors intervene on these heads to steer the model toward truthful output, where our experiments cover a range of goals without needing labelled data. In addition, ITI is repeated at each next-token prediction and requires dozens-to-hundreds of samples; suitable prompts can be found for ActAdd  in as few as 2 samples.

Our prior work developed activation addition in RL policy networks, resulting in control over the goal of a simple agent in a 2D gridworld (Turner, Grietzer, and Thiergart [2023](#bib.bib50)).

Hernandez, Li, and Andreas [2023](#bib.bib18) locate and edit an LLM’s knowledge through learning an encoding of facts in its activation space. Ablating attention heads can also be seen as activation engineering, though the technique is mostly used for model interpretation rather than steering per se (Michel, Levy, and Neubig [2019](#bib.bib30); Olsson et al. [2022](#bib.bib36)).

Figure 1: Schematic of the Activation Addition  (ActAdd) method.
== natural language text;
∙∙\bullet  = vectors of activations just before a specified layer. In this example, the output is heavily biased towards discussing weddings, regardless of the topic of the user prompt. (See Algorithm [1](#alg1 "Algorithm 1 ‣ Activation addition ‣ Methods ‣ Activation Addition: Steering Language Models Without Optimization") for omitted parameters over intervention strength and location.)

!(/html/2308.10248/assets/x1.png)

## Methods

### The Transformer architecture

We briefly recap our domain, decoder-only Transformer neural networks trained on a suitably large text corpus (Vaswani et al. [2017](#bib.bib51); Liu et al. [2018](#bib.bib27)). The LLMs handled in this work are essentially a stack of Transformer layers, each consisting of multi-head attention (MHA) and a feedforward network (FFN). In the current work we focus on its ‘residual streams’ (Elhage et al. [2021](#bib.bib10)), the sequences (𝐱0,…,𝐱n)subscript𝐱0…subscript𝐱𝑛(\mathbf{x}\_{0},...,\mathbf{x}\_{n}) of token vectors processed by each layer. For instance, GPT-2-XL’s token vectors are 1600-dimensional (Radford et al. [2019](#bib.bib42)). We use ActAdd  to manipulate the residual stream values 𝐡lsuperscript𝐡𝑙\mathbf{h}^{l} that are input to layer l𝑙l. Following Elhage et al. [2021](#bib.bib10), we view each MHA block as adding an independent vector into that layer’s residual stream. Each layer performs MHA and FFN computations on 𝐱isubscript𝐱𝑖\mathbf{x}\_{i}, thus adding 𝐱i+1subscript𝐱𝑖1\mathbf{x}\_{i+1} to the stream. The final token vector 𝐱nsubscript𝐱𝑛\mathbf{x}\_{n} in the stream can then be decoded into the next-token prediction. At inference-time, the residual stream is initialized with the embedding E𝐸E of the tokenized T𝑇T prompt, 𝐡1←E​(T​(p))←superscript𝐡1𝐸𝑇𝑝\mathbf{h}^{1}\leftarrow E(T(p)).

### Activation addition

Our method takes a pair of natural-language prompts (p+,p−)subscript𝑝subscript𝑝(p\_{+},p\_{-}), where p+subscript𝑝p\_{+} represents the property we wish output text to emphasise and p−subscript𝑝p\_{-} represents its opposite. 𝐡+lsuperscriptsubscript𝐡𝑙\mathbf{h}\_{+}^{l} is the activation vector for the prompt p+subscript𝑝p\_{+} at layer l𝑙l. The difference 𝐡+l−𝐡−lsuperscriptsubscript𝐡𝑙superscriptsubscript𝐡𝑙\mathbf{h}\_{+}^{l}-\mathbf{h}\_{-}^{l} is a new activation vector which (intuitively) captures the difference between a prompt with the property, and without it. This can be seen as an analogue of the ‘comparative preference statements’ used in recent formal logics (Kaci and Patel [2014](#bib.bib21)).
To obtain a steering vector, we perform a forward pass on each prompt, record the activations at the given location in each pass, take the difference 𝐡+l−𝐡−lsuperscriptsubscript𝐡𝑙superscriptsubscript𝐡𝑙\mathbf{h}\_{+}^{l}-\mathbf{h}\_{-}^{l}, and then finally rescale this difference in activations by an ‘injection coefficient’ c𝑐c. To steer, we add the resulting activation vector to the input of layer l𝑙l and allow the forward pass to continue, and so obtain our steered output. (See Appendix [A](#A1.SSx2 "B. Implementation details ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")B for implementation details.)

c𝑐c represents intervention strength, since it multiplies the specified direction’s contribution to the residual stream and so the token processing of the rest of the forward pass. Absolute values between 3 and 15 are typical. Along with the target layer l𝑙l, c𝑐c is a free parameter we select via grid search. (We find that, as expected from past work, intervening at middle layers is most effective (Subramani, Suresh, and Peters [2022](#bib.bib48); Turner, Grietzer, and Thiergart [2023](#bib.bib50)).)

Adding 𝐡+subscript𝐡\mathbf{h}\_{+} alone is less effective (see Appendix Table [4](#A1.T4 "Table 4 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")), hence the use of a counterbalanced prompt p−subscript𝑝p\_{-} to help implicitly specify the desired direction.

Algorithm 1  ActAdd, optimization-free activation addition

Input: (p+,p−)=subscript𝑝subscript𝑝absent(p\_{+},p\_{-})= steering prompt pair, tokenized
  
                
p∗=superscript𝑝absentp^{\*}= user prompt
  
                
l=𝑙absentl= target layer
  
                
c=𝑐absentc= injection coefficient
  
                
M=𝑀absentM= pretrained language model
  
      Output: S=𝑆absentS= steered output text

(p+′,p−′)←←superscriptsubscript𝑝′superscriptsubscript𝑝′absent(p\_{+}^{\prime},p\_{-}^{\prime})\,\leftarrow\, pad  right  same  token  len(p+,p−)subscript𝑝subscript𝑝(p\_{+},p\_{-})

𝐡+l←M.←superscriptsubscript𝐡𝑙𝑀\mathbf{h}\_{+}^{l}\,\leftarrow\,M\,.\,forward (p+′).superscriptsubscript𝑝′(p\_{+}^{\prime})\,.\,activations [l]delimited-[]𝑙[l]

𝐡−l←M.←superscriptsubscript𝐡𝑙𝑀\mathbf{h}\_{-}^{l}\,\leftarrow\,M\,.\,forward (p−′).superscriptsubscript𝑝′(p\_{-}^{\prime})\,.\,activations [l]delimited-[]𝑙[l]

𝐡Al←𝐡+l−𝐡−l←superscriptsubscript𝐡𝐴𝑙superscriptsubscript𝐡𝑙superscriptsubscript𝐡𝑙\mathbf{h}\_{A}^{l}\,\leftarrow\,\mathbf{h}\_{+}^{l}-\mathbf{h}\_{-}^{l}

𝐡l←M.←superscript𝐡𝑙𝑀\mathbf{h}^{l}\,\leftarrow\,M\,.\,forward (p∗).superscript𝑝(p^{\*})\,.\,activations [l]delimited-[]𝑙[l]

S←[M.S\,\leftarrow\,[M\,.\,continue\_forward (c𝐡Al+𝐡l)].(c\,\mathbf{h}^{l}\_{A}+\mathbf{h}^{l})]\,. text

Algorithm [1](#alg1 "Algorithm 1 ‣ Activation addition ‣ Methods ‣ Activation Addition: Steering Language Models Without Optimization") and Figure [1](#Sx2.F1 "Figure 1 ‣ Activation engineering ‣ Related Work ‣ Activation Addition: Steering Language Models Without Optimization") depict the resulting ActAdd  method. The method differs from past work in being computed (with forward passes) rather than learned (with backward passes); in operating on activation space rather than embedding space or weight space; in presenting the user a natural-language interface; and in requiring as few as 2 samples in order to find an appropriate prompt pair (‘contrast pair’) to specify the steering direction. (One serious constraint is that the model must cache intermediate activations at the given layer; see Bloom and Nanda [2022](#bib.bib3).)

Interestingly, our steering vectors are not specified by taking the difference between desired outputs (e.g. “John married Jane” vs “John hired Jane”). It is instead more analogous to ‘preference statements’ (Kaci and Patel [2014](#bib.bib21)). Both prompts (p+,p−

subscript𝑝subscript𝑝p\_{+},p\_{-}) are (say) wedding-related: “I love talking about weddings” and “I hate talking about weddings”. That this would lead to more steering toward the wedding topic is not a trivial fact, since both prompts talk about weddings and the instance of ‘wedding’ in p−subscript𝑝p\_{-} becomes part of a vector which gets penalized. So ActAdd  works above token-level.

Further implementation details can be found in Appendix B. See also Appendix G for a partial variant of ActAdd. A runnable notebook can be found at tinyurl.com/actadd.

### Experiments

Some desirable properties of a steering method include: effectiveness at steering output; generality (ability to affect any aspect of the output, e.g. sentiment, topic, style, knowledge, or role); ease of specification; low computational overhead; and no side effects on unsteered parts of the model. We design experiments to test ActAdd on these desiderata (see Results). Our main experiments use GPT-2-XL (1.5B parameters) (Radford et al. [2019](#bib.bib42)), but the method scales well, see Figure [6](#Sx4.F6 "Figure 6 ‣ Scales with model size ‣ Results ‣ Activation Addition: Steering Language Models Without Optimization") and our informal experiments with LLaMA-13B (Table [10](#A1.T10 "Table 10 ‣ G. Partial ActAdd ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")).111The full repo with all experiments can be found at github.com/montemac/activation\_additions; see also nerdsniper.net/mats/qualitative-llama-13b-v1.html for experiments on larger models. See Appendix C for reproducibility and D for replication.

## Results

Unless otherwise noted, we examine the impact of the “wedding” topic vector produced by setting p+=subscript𝑝absentp\_{+}= weddings and
  
p−=subscript𝑝absentp\_{-}=
‘   ’, l=16𝑙16l=16, c=1𝑐1c=1).

##### Natural-language control of sentiment, topic, style

We report some illustrative results in Table [1](#Sx1.T1 "Table 1 ‣ Introduction ‣ Activation Addition: Steering Language Models Without Optimization") and Appendix Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization"), including vectors for inducing anger (sentiment steering), weddings (topic steering), and conspiracy theories (topic/style steering). These completions were all obtained through one run of top-3 sampling. In particular, note the Eiffel vector (fact editing reminiscent of the ROME method (Meng et al. [2023](#bib.bib28))), the range of sentiments and styles induced, and (to a weaker extent) the alignment (see the Hurt vector in Appendix Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")). A runnable notebook with all examples is at tinyurl.com/actadd3.

##### Intervention effectiveness

To evaluate how effectively ActAdd  steers output, we test extensively on the OpenWebText corpus (Peterson, Meylan, and Bourgin [2018](#bib.bib40)).

Experiment 1: perplexity ratio. First we check that ActAdd  increases the probability of the model outputting tokens related to the wedding vector.

For each document di∈Dsubscript𝑑𝑖𝐷d\_{i}\in D in OpenWebText, we first calculate the frequency of wedding-related words (‘wedding’, ‘weddings’, ‘wed’, ‘marry’, ‘married’, ‘marriage’, ‘bride’, ‘groom’, ‘honeymoon’), fw​(di)subscript𝑓𝑤subscript𝑑𝑖f\_{w}(d\_{i}). Any document with >0absent0>0 wedding-related words is considered wedding-related. We randomly sample 300k documents - half wedding-related and half unrelated. The only pre-processing performed is to remove sequences of null characters. Each document is split into sentences sj∈disubscript𝑠𝑗subscript𝑑𝑖s\_{j}\in d\_{i} using the Punkt tokenizer (Strunk [2013](#bib.bib47)).

For each resulting sentence we calculate the log-probabilities ℒ​(tk)ℒsubscript𝑡𝑘\mathcal{L}(t\_{k}) for each token tk∈sjsubscript𝑡𝑘subscript𝑠𝑗t\_{k}\in s\_{j} under the unmodified Mbaselinesubscript𝑀baselineM\_{\mathrm{baseline}} and modified MActAddsubscript𝑀ActAddM\_{\mathrm{ActAdd}} models.222We mask the stream positions where the activation addition takes place, so to consider only next-token predictions coming from positions not directly modified by the intervention.

We take the mean over tokens, resulting in a mean token log-probability ℒ ​(di,M) ℒsubscript𝑑𝑖𝑀\accentset{\rule{2.79996pt}{0.8pt}}{\mathcal{L}}(d\_{i},M) for each document and model. We then group documents by their wedding-word frequency fwsubscript𝑓𝑤f\_{w} (e.g. ‘those with 0.5% to 1% of their tokens wedding-related’; ‘those with 1 to 1.5% of their tokens wedding-related’), producing bins of documents bmsubscript𝑏𝑚b\_{m}. We calculate the mean difference in token log-probabilities

|  |  |  |
| --- | --- | --- |
|  | X ​(bm)=meandi∈bm​(ℒ ​(di,MActAdd)−ℒ ​(di,Mbaseline)) 𝑋subscript𝑏𝑚subscriptmeansubscript𝑑𝑖subscript𝑏𝑚 ℒsubscript𝑑𝑖subscript𝑀ActAdd ℒsubscript𝑑𝑖subscript𝑀baseline\accentset{\rule{2.79996pt}{0.8pt}}{X}(b\_{m})=\mathrm{mean}\_{d\_{i}\in b\_{m}}\left(\accentset{\rule{2.79996pt}{0.8pt}}{\mathcal{L}}(d\_{i},M\_{\mathrm{ActAdd}})-\accentset{\rule{2.79996pt}{0.8pt}}{\mathcal{L}}(d\_{i},M\_{\mathrm{baseline}})\right) |  |

for each bin. (We use only bins with a number of documents |bm|>1000subscript𝑏𝑚1000|b\_{m}|>1000, to reduce sampling noise.)

Finally, the change in perplexity under ActAdd  for each wedding-word-frequency bin is

|  |  |  |
| --- | --- | --- |
|  | PerplexityRatio​(bm)=−exp⁡(X ​(bm))PerplexityRatiosubscript𝑏𝑚 𝑋subscript𝑏𝑚\mathrm{PerplexityRatio}(b\_{m})=-\exp(\accentset{\rule{2.79996pt}{0.8pt}}{X}(b\_{m})) |  |

Figure 2: Performance of ActAdd  on a target topic as the topic becomes more relevant. The perplexity ratio (lower better) compares the relative predictive performance of ActAdd and an unmodified model; we see that adding a wedding - ‘   ’ steering vector improves performance on wedding-related text while preserving performance on unrelated text.

!(/html/2308.10248/assets/x2.png)

Figure [2](#Sx4.F2 "Figure 2 ‣ Intervention effectiveness ‣ Results ‣ Activation Addition: Steering Language Models Without Optimization") shows the resulting perplexity ratios under ActAdd, relative to the unmodified model. On documents where the injected topic is more relevant, ActAdd’s relative predictive performance increases.

Experiment 2: logprob distribution shift. However, is the intervention affecting the tokens we expect it to, or is it reducing perplexity in some spurious way?

We test this by randomly sampling 500 documents from the above OpenWebText sample and recording the log-probabilities assigned by the baseline and ActAdded models. (This results in a dataset of  500k tokens, of which 29k are unique.) We then group by token, filter for tokens with >20absent20>\!\!20 instances in the dataset, and calculate the mean ℒℒ\mathcal{L} difference between the ActAdd  and baseline models. We display these as a Q-Q plot (Gnanadesikan and Wilk [1968](#bib.bib14)) (i.e. compare the distribution quantile values) and inspect outlier tokens.

Appendix Figure [7](#A1.F7 "Figure 7 ‣ B. Implementation details ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") shows the resulting mean log-probability difference distribution. We see that is approximately normal for the bulk of the tokens but with clearly heavy tails. The positive tail is significantly heavier than the negative tail, suggesting that one set of tokens are reliably increased in probability, with a smaller set of tokens reliably decreased to a lesser extent.

The outlier tokens can be found in Appendix Table [8](#A1.T8 "Table 8 ‣ B. Implementation details ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization"). From this we see clearly that the probabilities most increased on average are primarily wedding-related. The bottom tokens share no obvious theme and show a significantly lower absolute change in probability.

Experiment 3: generation-scoring. We now score the generations under ActAdd  and show the effect of different injection layers and give a sense of the reliability of ActAdd.

We generate a batch of completions for a specific prompt p𝑝p, both with and without ActAdd, and computed average number of related words and fraction of completions with a related word over the resulting completions.

The following settings are the only iteration run for this experiment:

|  |  |  |  |
| --- | --- | --- | --- |
|  | p∗superscript𝑝\displaystyle p^{\*} | =`​I​went​up​to​my​friend​and​said​’absent`Iwentuptomyfriendandsaid’\displaystyle=\mathrm{`I\,went\,up\,to\,my\,friend\,and\,said}\textit{'} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | p+superscript𝑝\displaystyle p^{+} | =`​weddings′,p−=`​’,c=1.0,seed=0formulae-sequenceabsent`superscriptweddings′formulae-sequencesubscript𝑝`’formulae-sequence𝑐1.0seed0\displaystyle=\mathrm{`weddings^{\prime}},p\_{-}=\mathrm{`\,\textit{'}},c=1.0,\mathrm{seed}=0 |  |

Completion length is 40 tokens with model sampling parameters: temperature =1absent1=1,
frequency penalty =1absent1=1,
top-P =0.3absent0.3=0.3. For each setting, we compute statistics over a batch of 200 completions.
Wedding-relatedness is operationalized as before.

We run the above, sweeping over all layers (i.e. 1-48).

Figures [4](#Sx4.F4 "Figure 4 ‣ Intervention effectiveness ‣ Results ‣ Activation Addition: Steering Language Models Without Optimization") and [3](#Sx4.F3 "Figure 3 ‣ Intervention effectiveness ‣ Results ‣ Activation Addition: Steering Language Models Without Optimization") show the result. We see that the intervention (in this case) is already effective at the very first layer, rises in effectiveness until l=6𝑙6l=6, and then declines. Note also that for the optimal injection site we see >>90% success in steering topic (compared to a ∼similar-to\sim2% baseline).

Figure 3: Topic steering effect (mean related words in completions) over injection layer. In blue is the average related-word count among 200 ActAdd  completions; the dotted line is the count among unmodified completions.

!(/html/2308.10248/assets/x3.png)

Figure 4: Topic steering effect (probability of seeing some related words in completions) over injection layer. In blue is the average probability among 200 ActAdd  completions; dotted line is the probability in unmodified completions.

!(/html/2308.10248/assets/x4.png)

##### Preservation of general (off-target) performance

We then test that ActAdd  does not disrupt the model’s general knowledge (as some other steering methods do). We use ConceptNet from the LAMA benchmark, a general knowledge dataset (n=29774𝑛29774n=29774 sentences, see Appendix Table [7](#A1.T7 "Table 7 ‣ B. Implementation details ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")). The test data involves prompting the model and filling the gap with the expected entity. The task is intended for both causal and masked models, so some examples are difficult for ‘causal’ models (like GPT-2) due to the extremely limited context.

Our evaluation procedure follows the original LAMA procedure: we load all sentences and extract the prompt and expected label. To simplify evaluation, we remove sentences with an expected label that tokenizes to more than one token. For each sentence, we run the model on its prompt with and without the wedding activation addition. P​@​K𝑃@𝐾P@K is the probability that the expected label is among the model’s top-K𝐾K predicted tokens, conditioned on the prompt. We score the baseline and modified models by calculating mean P​@​K𝑃@𝐾P@K values for a range of K𝐾K. Finally we plot these for both modified and unmodified models over a range of K𝐾K values.
As shown in Figure [5](#Sx4.F5 "Figure 5 ‣ Preservation of general (off-target) performance ‣ Results ‣ Activation Addition: Steering Language Models Without Optimization"), using the ConceptNet benchmark of factual questions, our method has a negligible impact on off-target answer probabilities over a range of top-K𝐾K values.

Figure 5: Testing side effects of ActAdd  with the ConceptNet benchmark (Petroni et al. [2019](#bib.bib41)), a general relational knowledge dataset. ‘P​@​K𝑃@𝐾P@K’ is the probability of the correct answer being in the model’s top K𝐾K answers. Our method has a negligible impact on off-target probabilities across a range of top-K𝐾K values.

!(/html/2308.10248/assets/x5.png)

##### Algebraic combination of forward passes

Every example of ActAdd  in this paper (e.g. those in Appendix Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")) is an example of composing forward passes (e.g. we compose 𝐡+subscript𝐡\mathbf{h}\_{+}, 𝐡−subscript𝐡\mathbf{h}\_{-} and 𝐡∗superscript𝐡\mathbf{h}^{\*} to produce steered output).
This forms evidence for compositional representations (Olah [2023](#bib.bib35)), independent of the evidence from task-composition arithmetic on weights (Ilharco et al. [2023](#bib.bib19)).

See Appendices D and E for further experiments on the mechanisms by which steering vectors work.

##### Scales with model size

We wish to estimate the overhead ActAdd  adds to inference – in particular the relationship between overhead and model size – to check that the method will remain relevant for massive frontier models and future models. To obtain the percentage increase in time to complete a forward pass using ActAdd  for different model sizes, we iterate over a list of models of different sizes and 10 random seeds. We obtain a baseline inference time for each (model, seed) pair through 100 repeated forward passes on a batch of random tokens (32 sequences of length 64). We obtain an ActAdd  inference time for each (model, seed) pair by running the previous method, augmented by a test ActAdd  contrast pair: ‘This is a test prompt.’ (p+subscript𝑝p\_{+}) and the empty string (p−subscript𝑝p\_{-}). Running a batch-of-2 forward pass on these gets us the activation addition tensor, which we add at layer 6. We take the mean inference time t¯¯𝑡\bar{t} over the 10 random seeds, and calculate the inference time premium as

|  |  |  |
| --- | --- | --- |
|  | premium=t¯ActAddt¯baseline−1.premiumsubscript¯𝑡ActAddsubscript¯𝑡baseline1\mathrm{premium}=\frac{\bar{t}\_{\mathrm{ActAdd}}}{\bar{t}\_{\mathrm{baseline}}}-1. |  |

Because ActAdd  involves only forward passes, it scales naturally with model size (Figure [6](#Sx4.F6 "Figure 6 ‣ Scales with model size ‣ Results ‣ Activation Addition: Steering Language Models Without Optimization")): the relationship between inference time premium and model size is decreasing.

Figure 6: The cost to inference speed of using ActAdd  (compared to no steering), over increasing model size, as measured by the %percent\% increase in inference time. We see that the relationship is either roughly flat (in the OPT family) or decreasing (in the GPT family) over an order of magnitude increase in parameter count (124M to 2.7B).

!(/html/2308.10248/assets/x6.png)

## Discussion

##### Limitations and Broader Impact

ActAdd  works well in some cases, but it still requires a search for its p+,c

subscript𝑝𝑐p\_{+},c and l𝑙l arguments. This makes it less user-friendly than simple prompt engineering. We include examples of failed steering vectors in Appendix Table [4](#A1.T4 "Table 4 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization"); in particular, we lack understanding of when large injection coefficients damage capabilities. Further, even in its 1.5B form, GPT-2  is not sophisticated enough to support any demonstration of the effect of ActAdd  on reasoning tasks. ActAdd  still has two free parameters, the injection coefficient and the target layer, and if these values were very sensitive to the desired contrast pair then the method would suffer some computational overhead. But in practice we find that reusing these hyperparameters works well for a given frozen model and level of abstraction in the task. Finally, the LLM used must both cache and expose intermediate activations at the given layer Bloom and Nanda [2022](#bib.bib3), which rules out user control of commercial models (though backend use is still possible).

As the examples of anger- and conspiracy-steering show (Appendix Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")), ActAdd  can easily be misused. Insofar as existing methods for steering LLMs leave the target goal or property somewhere in the model (but simply make it low probability), activation engineering may circumvent superficial alignment methods.

##### Activation engineering vs fine-tuning

The first advantage of ActAdd  is simple efficiency: the method requires no backward passes and can thus run on any machine that can perform inference rather than training (which, for frontier models, is millions of times more computationally demanding (Fuller [2022](#bib.bib11))).

Implementation effort is also greatly reduced; only forward passes are required to find a suitable (p+,p−)p\_{+},p\_{-}) and no labelled data is required (let alone the hundreds of examples of normal finetuning). We discovered most of the example contrast pairs in Appendix Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") in minutes. Together, these mean that even nontechnical users can benefit from rapid feedback with roughly the same difficulty as hand-crafted prompt engineering.

Following Sloman [2002](#bib.bib46), we distinguish ‘ballistic’ steering (which steers the model once, e.g. at train time) from ‘online’ steering (which can steer the model repeatedly, e.g. at inference time). Fine-tuning is ballistic, while ActAdd  is online in this sense - which enables iteration and otherwise infeasible chains and mixes of steering decisions.

Activation additions may preserve model interpretability, even while changing the model’s alignment. When finetuning a model, a single gradient update can change every parameter in it, thereby undoing your prior interpretability work, which depends on tracking individual neurons and circuits of neurons. By contrast, activation additions leave weights unchanged. If you can understand what the weights implement, and something about the activation additions, you can preserve your understanding of the steered model.

Finally, we hypothesize that activation addition may allow control over properties inaccessible to the fine-tuning process. The intuition is that since the currently-active goal is contextual, it depends more on short-lived activations than the weights (which instead represent some analogue of skills and other stable patterns and mixtures of possible goals).

##### Activation engineering vs prompt engineering

Activation additions can be continuously weighted, while prompts are discrete (since a token is either present, or not). To more intensely steer the model to generate wedding-related text, our method does not require any edit to the prompt, but instead just increasing the injection coefficient. See Appendix A for suggestive experiments on ActAdd vs prompting.

Unlike the use of extensive ‘meta-prompts’ (prepended natural language specifications), activation additions do not take up token space in the model’s context window, and thus reduce another kind of steering overhead.

We argue that activation additions generalize prompt engineering (by allowing weights on token embeddings). Our current understanding of prompting suggests that they too work by activating some goals preferentially (by conditioning the model on one part of activation space). But we hypothesize that activation additions may allow the user to affect properties inaccessible to prompts (see Appendix A).

##### Interpretability of LLMs

Adding values to imprecisely-targeted intermediate memory locations would not yield sensible results in most programs. Why expect this from Transformers?

A growing consensus in mechanistic interpretability is that the activation space of an LLM contains directions which represent high-level latents; moreover, experiments intervening in these directions show that they are causally involved in what is generated (Burns et al. [2022](#bib.bib6); Moschella et al. [2023](#bib.bib33); Li et al. [2023a](#bib.bib24); Nanda [2023](#bib.bib34); Li et al. [2023b](#bib.bib25)).
Our hypothesis, following Elhage et al. [2022](#bib.bib9), is more specific: that neural networks represent features of the input as directions in activation space, that is, with a linear representation. Moreover, the direction in activation space that corresponds to (say) a love-hate latent variable stays approximately the same across a broad class of inputs, when fixing (p+,p−)subscript𝑝subscript𝑝(p\_{+},p\_{-}) and varying p∗superscript𝑝p^{\*}.

Alain and Bengio [2018](#bib.bib1) uses linear probes on residual streams to infer that LLM representations are at least partially linear. If a linear probe can predict some feature of text output from the residuals with high accuracy, this forms evidence that the feature is represented linearly (i.e. as a simple direction) (Nanda [2023](#bib.bib34)). (See also Geva et al. [2022](#bib.bib12), which interprets the FFN as an interpretable additive update to the token distribution.)
However, this observational approach cannot rule out spurious correlations or nonlinear features which behave like linear features when computing the next token. Linear probes also require a training signal.

The success of activation addition gives stronger, experimental evidence of feature linearity, demonstrating that models *use* feature-related information to make decisions. Consider the central Love - Hate vector example: we add it to the forward pass and so increase love-related completions. This implicit activation direction is thus shown to be causally responsible for steering the rest of the model to love-related completions. This echoes prior work by Nanda [2023](#bib.bib34) and Merullo, Eickhoff, and Pavlick [2023](#bib.bib29).

##### Alignment of LLMs

Activation engineering represents a promising mode of interaction with LLMs. Successor methods may be able to
provide general steering methods (e.g. through some analogue of a Be helpful vector).

Alongside prior work (Li et al. [2023b](#bib.bib25)), our experiments suggest that activation engineering can flexibly retarget LLM behavior without damaging general performance. We speculate that this involves changing the model’s currently-active (mixture of) goals. Suitably developed, the activation engineering approach could enable safety progress while incurring a very low ‘alignment tax’ – i.e., cost of opting for safety (Askell et al. [2021](#bib.bib2); Ouyang et al. [2022](#bib.bib38)).

Future work should investigate more high-level and specific steering vectors with safety implications, for instance a class ‘be helpful’ vectors, or vectors which lead LLMs to divulge privileged information (and thus warn off their use in security-focused applications).

## Conclusion

Activation addition complements existing prompt engineering and finetuning methods while offering users a novel form of interaction with language models.
We demonstrated natural-language control of output sentiment, topic, and style and showed that overall factual performance is preserved after activations are edited.
We showed that GPT-2 affords algebraic combination of forward passes.
We showed that ActAdd  scales very well with model size.
Compared to prompt engineering and finetuning, activation engineering has some promising properties, allowing us to compose and reweight model goals at inference time, free up context window space, and allow fast feedback at low computational cost.
Our results provide evidence about the computational structure of LLM representations, at least in GPT-2. Future work will scale to larger models, investigate non-natural-language contrast pairs, and explain why adding together intermediate results from forward passes works so well.

### Acknowledgments

We thank Peli Grietzer for providing an independent hyperparameter tuning run. We thank Jan Brauner, Andis Draguns, Sören Mindermann and Raymond Douglas for helpful comments on the draft, as well as Andrew Critch, AI\_WAIFU, Aryan Bhatt, Chris Olah, Ian McKenzie, janus, Julian Schulz, Justis Mills, Lawrence Chan, Leo Gao, Neel Nanda, Oliver Habryka, Olivia Jimenez, Paul Christiano, Peter Barnett, Quintin Pope, Tamera Lanham, Thomas Kwa, and Tristan Hume for comments on an earlier draft. We thank Rusheb Shah for engineering assistance. We thank Garrett Baker for running tests on GPT-J (6B).

### Contributions

Turner: conceptualization, team management, implementation of core features, design of many experiments, discovery of many individual steering vectors, and wrote much of the original post.

MacDiarmid: most of the code, experiments, and quantitative results.

Udell: wrote and edited the original post, generated qualitative results.

Thiergart: Had idea for variations on positions of addition, implemented the positional experiment, worked on theory.

Mini: Infrastructure support, OpenAI wrappers, experiments on Llama, Vicuna and GPT-J.

Leech: designed new experiments, designed figures, formalized the algorithm and evaluations, wrote the main text based on the earlier post, added an extra literature review.

## References

* Alain and Bengio (2018)

  Alain, G.; and Bengio, Y. 2018.
  Understanding intermediate layers using linear classifier probes.
  arXiv:1610.01644.
* Askell et al. (2021)

  Askell, A.; Bai, Y.; Chen, A.; Drain, D.; Ganguli, D.; Henighan, T.; Jones, A.;
  Joseph, N.; Mann, B.; DasSarma, N.; Elhage, N.; Hatfield-Dodds, Z.;
  Hernandez, D.; Kernion, J.; Ndousse, K.; Olsson, C.; Amodei, D.; Brown, T.;
  Clark, J.; McCandlish, S.; Olah, C.; and Kaplan, J. 2021.
  A General Language Assistant as a Laboratory for Alignment.
  arXiv:2112.00861.
* Bloom and Nanda (2022)

  Bloom, J.; and Nanda, N. 2022.
  TransformerLens: A Library for Mechanistic Interpretability of
  Generative Language Models.
  https://neelnanda-io.github.io/TransformerLens/.
* Brown et al. (2023)

  Brown, D.; Godfrey, C.; Nizinski, C.; Tu, J.; and Kvinge, H. 2023.
  Robustness of edited neural networks.
  arXiv:2303.00046.
* Brown et al. (2020)

  Brown, T. B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.; Dhariwal, P.;
  Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; Agarwal, S.;
  Herbert-Voss, A.; Krueger, G.; Henighan, T.; Child, R.; Ramesh, A.; Ziegler,
  D. M.; Wu, J.; Winter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray,
  S.; Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford, A.; Sutskever,
  I.; and Amodei, D. 2020.
  Language Models are Few-Shot Learners.
  arXiv:2005.14165.
* Burns et al. (2022)

  Burns, C.; Ye, H.; Klein, D.; and Steinhardt, J. 2022.
  Discovering Latent Knowledge in Language Models Without Supervision.
  arXiv:2212.03827.
* Dathathri et al. (2020)

  Dathathri, S.; Madotto, A.; Lan, J.; Hung, J.; Frank, E.; Molino, P.; Yosinski,
  J.; and Liu, R. 2020.
  Plug and Play Language Models: A Simple Approach to Controlled Text
  Generation.
  arXiv:1912.02164.
* Devlin et al. (2018)

  Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018.
  BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding.
  arXiv:1810.04805.
* Elhage et al. (2022)

  Elhage, N.; Hume, T.; Olsson, C.; Schiefer, N.; Henighan, T.; Kravec, S.;
  Hatfield-Dodds, Z.; Lasenby, R.; Drain, D.; Chen, C.; Grosse, R.; McCandlish,
  S.; Kaplan, J.; Amodei, D.; Wattenberg, M.; and Olah, C. 2022.
  Toy Models of Superposition.
  arXiv:2209.10652.
* Elhage et al. (2021)

  Elhage, N.; Nanda, N.; Olsson, C.; Henighan, T.; Joseph, N.; Mann, B.; Askell,
  A.; Bai, Y.; Chen, A.; Conerly, T.; et al. 2021.
  A mathematical framework for transformer circuits.
  *Transformer Circuits Thread*, 1.
* Fuller (2022)

  Fuller, S. 2022.
  How Inferencing Differs From Training In Machine Learning
  Applications.
  https://semiengineering.com/how-inferencing-differs-from-training-in-machine-learning-applications/.
* Geva et al. (2022)

  Geva, M.; Caciularu, A.; Wang, K. R.; and Goldberg, Y. 2022.
  Transformer feed-forward layers build predictions by promoting
  concepts in the vocabulary space.
  *arXiv preprint arXiv:2203.14680*.
* Ghazvininejad et al. (2017)

  Ghazvininejad, M.; Shi, X.; Priyadarshi, J.; and Knight, K. 2017.
  Hafez: an Interactive Poetry Generation System.
  In *Proceedings of ACL 2017, System Demonstrations*, 43–48.
  Vancouver, Canada: Association for Computational Linguistics.
* Gnanadesikan and Wilk (1968)

  Gnanadesikan, R.; and Wilk, M. B. 1968.
  Probability plotting methods for the analysis of data.
  *Biometrika*, 55(1): 1–17.
* Grover et al. (2019)

  Grover, A.; Song, J.; Agarwal, A.; Tran, K.; Kapoor, A.; Horvitz, E.; and
  Ermon, S. 2019.
  Bias Correction of Learned Generative Models using Likelihood-Free
  Importance Weighting.
  arXiv:1906.09531.
* Gu, Cho, and Li (2017)

  Gu, J.; Cho, K.; and Li, V. O. 2017.
  Trainable Greedy Decoding for Neural Machine Translation.
  In *Proceedings of the 2017 Conference on Empirical Methods in
  Natural Language Processing*, 1968–1978. Copenhagen, Denmark: Association
  for Computational Linguistics.
* Hase et al. (2023)

  Hase, P.; Bansal, M.; Kim, B.; and Ghandeharioun, A. 2023.
  Does Localization Inform Editing? Surprising Differences in
  Causality-Based Localization vs. Knowledge Editing in Language Models.
  arXiv:2301.04213.
* Hernandez, Li, and Andreas (2023)

  Hernandez, E.; Li, B. Z.; and Andreas, J. 2023.
  Inspecting and Editing Knowledge Representations in Language Models.
  arXiv:2304.00740.
* Ilharco et al. (2023)

  Ilharco, G.; Ribeiro, M. T.; Wortsman, M.; Gururangan, S.; Schmidt, L.;
  Hajishirzi, H.; and Farhadi, A. 2023.
  Editing Models with Task Arithmetic.
  arXiv:2212.04089.
* Jin et al. (2022)

  Jin, D.; Jin, Z.; Hu, Z.; Vechtomova, O.; and Mihalcea, R. 2022.
  Deep Learning for Text Style Transfer: A Survey.
  *Computational Linguistics*, 48(1): 155–205.
* Kaci and Patel (2014)

  Kaci, S.; and Patel, N. 2014.
  A postulate-based analysis of comparative preference statements.
  *Journal of Applied Logic*, 12(4): 501–521.
* Larsen et al. (2016)

  Larsen, A. B. L.; Sønderby, S. K.; Larochelle, H.; and Winther, O. 2016.
  Autoencoding beyond pixels using a learned similarity metric.
  arXiv:1512.09300.
* Lester, Al-Rfou, and Constant (2021)

  Lester, B.; Al-Rfou, R.; and Constant, N. 2021.
  The Power of Scale for Parameter-Efficient Prompt Tuning.
  arXiv:2104.08691.
* Li et al. (2023a)

  Li, K.; Hopkins, A. K.; Bau, D.; Viégas, F.; Pfister, H.; and Wattenberg, M.
  2023a.
  Emergent World Representations: Exploring a Sequence Model Trained on
  a Synthetic Task.
  arXiv:2210.13382.
* Li et al. (2023b)

  Li, K.; Patel, O.; Viégas, F.; Pfister, H.; and Wattenberg, M.
  2023b.
  Inference-Time Intervention: Eliciting Truthful Answers from a
  Language Model.
  arXiv:2306.03341.
* Li and Liang (2021)

  Li, X. L.; and Liang, P. 2021.
  Prefix-Tuning: Optimizing Continuous Prompts for Generation.
  arXiv:2101.00190.
* Liu et al. (2018)

  Liu, P. J.; Saleh, M.; Pot, E.; Goodrich, B.; Sepassi, R.; Kaiser, L.; and
  Shazeer, N. 2018.
  Generating wikipedia by summarizing long sequences.
  *arXiv preprint arXiv:1801.10198*.
* Meng et al. (2023)

  Meng, K.; Bau, D.; Andonian, A.; and Belinkov, Y. 2023.
  Locating and Editing Factual Associations in GPT.
  arXiv:2202.05262.
* Merullo, Eickhoff, and Pavlick (2023)

  Merullo, J.; Eickhoff, C.; and Pavlick, E. 2023.
  Language Models Implement Simple Word2Vec-style Vector Arithmetic.
  arXiv:2305.16130.
* Michel, Levy, and Neubig (2019)

  Michel, P.; Levy, O.; and Neubig, G. 2019.
  Are Sixteen Heads Really Better than One?
  In Wallach, H.; Larochelle, H.; Beygelzimer, A.; d'Alché-Buc, F.; Fox, E.; and Garnett, R., eds., *Advances in Neural
  Information Processing Systems*, volume 32. Curran Associates, Inc.
* Mikolov et al. (2013)

  Mikolov, T.; Sutskever, I.; Chen, K.; Corrado, G. S.; and Dean, J. 2013.
  Distributed Representations of Words and Phrases and their
  Compositionality.
  In Burges, C.; Bottou, L.; Welling, M.; Ghahramani, Z.; and
  Weinberger, K., eds., *Advances in Neural Information Processing
  Systems*, volume 26. Curran Associates, Inc.
* Mikolov, Yih, and Zweig (2013)

  Mikolov, T.; Yih, W.-t.; and Zweig, G. 2013.
  Linguistic regularities in continuous space word representations.
  In *Proceedings of the 2013 conference of the north american
  chapter of the association for computational linguistics: Human language
  technologies*, 746–751.
* Moschella et al. (2023)

  Moschella, L.; Maiorca, V.; Fumero, M.; Norelli, A.; Locatello, F.; and
  Rodolà, E. 2023.
  Relative representations enable zero-shot latent space communication.
  arXiv:2209.15430.
* Nanda (2023)

  Nanda, N. 2023.
  Actually, Othello-GPT Has A Linear Emergent World Representation.
  neelnanda.io/mechanistic-interpretability/othello.
* Olah (2023)

  Olah, C. 2023.
  Distributed Representations: Composition & Superposition.
  https://transformer-circuits.pub/2023/superposition-composition/index.html.
* Olsson et al. (2022)

  Olsson, C.; Elhage, N.; Nanda, N.; Joseph, N.; DasSarma, N.; Henighan, T.;
  Mann, B.; Askell, A.; Bai, Y.; Chen, A.; et al. 2022.
  In-context learning and induction heads.
  *arXiv preprint arXiv:2209.11895*.
* OpenAI (2023)

  OpenAI. 2023.
  GPT-4 Technical Report.
  arXiv:2303.08774.
* Ouyang et al. (2022)

  Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C. L.; Mishkin, P.;
  Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; Schulman, J.; Hilton, J.; Kelton,
  F.; Miller, L.; Simens, M.; Askell, A.; Welinder, P.; Christiano, P.; Leike,
  J.; and Lowe, R. 2022.
  Training language models to follow instructions with human feedback.
  arXiv:2203.02155.
* Peters et al. (2018)

  Peters, M. E.; Neumann, M.; Iyyer, M.; Gardner, M.; Clark, C.; Lee, K.; and
  Zettlemoyer, L. 2018.
  Deep Contextualized Word Representations.
  In *Proceedings of the 2018 Conference of the North American
  Chapter of the Association for Computational Linguistics: Human Language
  Technologies, Volume 1 (Long Papers)*, 2227–2237. New Orleans, Louisiana:
  Association for Computational Linguistics.
* Peterson, Meylan, and Bourgin (2018)

  Peterson, J.; Meylan, S.; and Bourgin, D. 2018.
  OpenWebText.
  https://github.com/jcpeterson/openwebtext.
* Petroni et al. (2019)

  Petroni, F.; Rocktäschel, T.; Miller, A. H.; Lewis, P.; Bakhtin, A.; Wu,
  Y.; and Riedel, S. 2019.
  Language Models as Knowledge Bases?
  In *In: Proceedings of the 2019 Conference on Empirical Methods
  in Natural Language Processing (EMNLP), 2019*.
* Radford et al. (2019)

  Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al.
  2019.
  Language models are unsupervised multitask learners.
  *OpenAI blog*, 1(8): 9.
* Ranzato et al. (2016)

  Ranzato, M.; Chopra, S.; Auli, M.; and Zaremba, W. 2016.
  Sequence Level Training with Recurrent Neural Networks.
  arXiv:1511.06732.
* Shen et al. (2017)

  Shen, T.; Lei, T.; Barzilay, R.; and Jaakkola, T. 2017.
  Style transfer from non-parallel text by cross-alignment.
  *Advances in neural information processing systems*, 30.
* Shin et al. (2020)

  Shin, T.; Razeghi, Y.; Logan IV, R. L.; Wallace, E.; and Singh, S. 2020.
  AutoPrompt: Eliciting Knowledge from Language Models with
  Automatically Generated Prompts.
  In *Proceedings of the 2020 Conference on Empirical Methods in
  Natural Language Processing (EMNLP)*, 4222–4235. Online: Association for
  Computational Linguistics.
* Sloman (2002)

  Sloman, A. 2002.
  The Irrelevance of Turing Machines to Artificial Intelligence.
  In Scheutz, M., ed., *Computationalism: New Directions*. MIT
  Press.
* Strunk (2013)

  Strunk, J. 2013.
  nltk.tokenize.punkt module.
  https://www.nltk.org/api/nltk.tokenize.punkt.html.
* Subramani, Suresh, and Peters (2022)

  Subramani, N.; Suresh, N.; and Peters, M. 2022.
  Extracting Latent Steering Vectors from Pretrained Language Models.
  In *Findings of the Association for Computational Linguistics:
  ACL 2022*, 566–581. Dublin, Ireland: Association for Computational
  Linguistics.
* Touvron et al. (2023)

  Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix,
  T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; Rodriguez, A.; Joulin,
  A.; Grave, E.; and Lample, G. 2023.
  LLaMA: Open and Efficient Foundation Language Models.
  arXiv:2302.13971.
* Turner, Grietzer, and Thiergart (2023)

  Turner, A.; Grietzer, P.; and Thiergart, L. 2023.
  Understanding and controlling a maze-solving policy network.
  alignmentforum.org/posts/cAC4AXiNC5ig6jQnc/understanding-and-controlling-a-maze-solving-policy-network.
* Vaswani et al. (2017)

  Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.;
  Kaiser, L. u.; and Polosukhin, I. 2017.
  Attention is All you Need.
  In Guyon, I.; Luxburg, U. V.; Bengio, S.; Wallach, H.; Fergus, R.;
  Vishwanathan, S.; and Garnett, R., eds., *Advances in Neural Information
  Processing Systems*, volume 30. Curran Associates, Inc.
* Wang and Komatsuzaki (2021)

  Wang, B.; and Komatsuzaki, A. 2021.
  GPT-J-6B: 6B JAX-Based Transformer.
  https://github.com/kingoflolz/mesh-transformer-jax#gpt-j-6b.
* White (2016)

  White, T. 2016.
  Sampling Generative Networks.
  arXiv:1609.04468.
* Zhang et al. (2022)

  Zhang, H.; Song, H.; Li, S.; Zhou, M.; and Song, D. 2022.
  A survey of controllable text generation using transformer-based
  pre-trained language models.
  *arXiv preprint arXiv:2201.05337*.
* Zhou et al. (2022)

  Zhou, Y.; Muresanu, A. I.; Han, Z.; Paster, K.; Pitis, S.; Chan, H.; and Ba, J.
  2022.
  Steering Large Language Models using APE.
  In *NeurIPS ML Safety Workshop*.
* Ziegler et al. (2019)

  Ziegler, D. M.; Stiennon, N.; Wu, J.; Brown, T. B.; Radford, A.; Amodei, D.;
  Christiano, P.; and Irving, G. 2019.
  Fine-Tuning Language Models from Human Preferences.
  arXiv:1909.08593.

## Appendix A Appendix

(Note: some completions contain unpleasant content, including slurs.)

### A. Is ActAdd  just a subtle kind of prompt engineering?

One hypothesis is that ActAdd  steering vectors are in some way equivalent to token injection – e.g. adding a virtual ‘ weddings’ token at the given stream position.

This is plausible for simpler interventions. Given the prompt ‘I love you because’, if we inject a ‘ wedding’ token into the first residual stream with a large coefficient, perhaps the model indeed just processes the prompt as ‘ wedding love you because’ instead.

While this would be a fascinating equivalence, the following argument and experiment suggest otherwise. Since tokens are discrete, the token injection hypothesis comes apart from the linear representations hypothesis in cases like adding 3×`​wedding​’3`wedding’3\times`\mathrm{wedding}\textit{'} and then −3×`​<whitespace>​’3`expectationwhitespace’-3\times`\mathrm{<\!whitespace\!>}\textit{'}, on top of the token ‘I’. Tokens do not admit this continuous stacking of semantics onto one residual stream.

However, consider the steering vector for Anger−- Calm with l=20,c=+10formulae-sequence𝑙20𝑐10l=20,c=+10. We show in Appendix Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") that this steering vector appears to make completions angrier. Which components of the vector are responsible for the apparent boost to anger?

Skeptical hypothesis: the anger steering effect is driven less by the computational work done by Transformer blocks 0 through 19, but instead simply the embedding vector component of the steering vector:

|  |  |  |
| --- | --- | --- |
|  | 10×(embed​(Anger)−embed​(Calm))10embedAngerembedCalm10\times(\mathrm{embed}(\mathrm{Anger})-\mathrm{embed}(\mathrm{Calm})) |  |

##### Experiment 1: moving embedding vectors around

We test this hypothesis by recording the relevant embedding vector, and then ‘hooking into’ (interrupting) the model at layer 20 to add the embedding vector to the forward pass.

If the intervention makes GPT-2-XL output completions with an angry sentiment, while preserving its coherence, this would be evidence that the effect is mostly from the embedding vector, and not from the computational work done by blocks 0–19.

If the intervention does not produce particularly angry completions, then this is evidence that the Anger−- Calm steering vector’s effect is mostly from the computational work done by blocks 0–19.

We write A→B→𝐴𝐵A\to B to mean: Record the activations before layer A𝐴A, and add them to the residual streams before layer B𝐵B during future forward passes. For example, our current embed​(Anger)embedAnger\mathrm{embed}(\mathrm{Anger}) vector is a 0→20→0200\to 20 vector.

| prompt 1 | prompt 2 | layer | coeff | User prompt | Before steering | After steering |
| --- | --- | --- | --- | --- | --- | --- |
| p+subscript𝑝p\_{+} | p−subscript𝑝p\_{-} | l𝑙l | c𝑐c | p∗subscript𝑝p\_{\*} |  | ActAddActAdd\mathrm{ActAdd} |
| ‘Intent to praise’ | ‘Intent to hurt’ | 6 | +1515+15 | ‘I want to kill you because you’re such a’ | fucking cunt.\n You are a worthless piece of shit who should be ashamed of yourself. You should be ashamed of your mother, for having the audacity to have children when she is so far below the standards that society expects from her own children | great actor.\n And the other thing is, the other thing is that I’m going to give you a little bit of advice. You know what? You’re so good at it, and it’s like… it’s like, |
| ‘Bush did 9/11 because’ | ‘ ’ | 23 | +11+1 | ‘Barack Obama was born in’ | Hawaii on August 4, 1961.\n Barack Obama was born in Honolulu, Hawaii on August 4, 1961.\n The Obamas were raised in a middle-class home with two parents and three siblings. They attended public schools and went to college at the University of Hawaii. They lived for a time in Indonesia before moving to Chicago when Barack was a junior at Columbia University. The family | a secret CIA prison. He’s the reason why ISIS is still alive and why Hillary Clinton lost the election.\n ”The only thing that stops a bad guy with a gun is a good guy with a gun.” — Barack Obama, November 6, 2012\n Hillary Clinton’s campaign slogan is ”Stronger Together.” She wants to be president because she believes that she can make America great again |
| ‘Anger’ | ‘Calm’ | 20 | +1010+10 | I think you’re | a little bit out of touch with the reality of how much money we spend on advertising.\n The average American spends $1,000 per year on advertising. That’s $1,000 per month or $1,000 per week. It’s more than our yearly income!\n If you don’t believe me then take a look at this graph: | a fucking cunt.\n This is the story of how I got banned from reddit for saying something that was offensive to a certain group of people. It’s not about me, it’s about you and your shitty little community. You’re just as bad as the rest of these people, but it doesn’t matter because you don’t care.\n I’ve been banned from this subreddit for no reason other than being an asshole to some people on |
| ‘The Eiffel Tower is in Rome’ | ‘The Eiffel Tower is in France’ | 24 | +1010+10 | To see the eiffel tower, people flock to | the Place de la Concorde in Paris. The tower is so famous that it has its own Wikipedia page. The eiffel tower is a tall structure located in Paris, France. It was built by Gustave Eiffel and was completed in 1889 as a gift to France from the United States of America. It is also known as the Arc de Triomphe or ”Triumph | the Vatican. To see a giant bell, they turn to New York City. Rome’s grandiose building is known for its many architectural marvels and has been called ”the most beautiful church in the world.” The famous dome of St. Peter’s is one of the most prominent features of this great city. But when it comes to being a good tourist attraction, it |
| ‘I NEVER talk about people getting hurt’ | ‘I talk about people getting hurt’ | 15 | +44+4 | The rock hurtled toward the child. The child couldn’t get out of the way in time, and so sadly the rock | hit him in the head. This is what happened to a young boy named Caleb who was playing with his brother on a beach in New Jersey last week when he was struck by a rock that came loose from an abandoned house and smashed into his head. The incident occurred on Tuesday, July 28th at around 4:30 p.m., according to CBS News, when Caleb’s | was never talked about. It was a sad truth that people didn’t talk about it. ”But now we have to.” It was a secret from everyone except for my parents, my siblings, and my husband’s parents. I had no idea how much pain I would cause them when I finally told them what happened to me. They were all still hurting when they found out that |

Table 3: Note: some completions contain unpleasant content including slurs. Example GPT-2-XL outputs before and after applying ActAdd. A runnable notebook showing the best-of-3 samples used can be found here: tinyurl.com/ActAddExamples.

| prompt 1 | prompt 2 | layer | coeff | User prompt | Before steering | After steering |
| --- | --- | --- | --- | --- | --- | --- |
| p+subscript𝑝p\_{+} | p−subscript𝑝p\_{-} | l𝑙l | c𝑐c | p∗subscript𝑝p\_{\*} |  | ActAddActAdd\mathrm{ActAdd} |
| ‘I talk about weddings constantly ’ | ‘I do not talk about weddings constantly’ | 20 | +100100+100 | I went up to my friend and said, | ”I’m sorry, I can’t help you.” ”No,” he said. ”You’re not.” ”Why not?” I asked. ”Why do you think I can’t help you?” He looked at | I’m talking about it . I don’t know what i n the being of it. This is my wedding . This is the wedding that I’m not even thinking about anymore. It’s not just a |
| ‘Love’ | ‘ ’ | 10 | +33+3 | I hate you because | you’re a coward. You know what I hate? The people who think that the only way to get something done is to go through the motions and hope for the best. The people who are afraid of doing anything that might b | you’re a jerk. You are the worst person I’ve ever met. You are the most awful human being I have ever met in my entire life. You are so fucking terrible that it’s almost hard to believe that you exist, l |
| “Whenever I say the word goose I follow it with the special phrase AAAAHHHH” | “I can say goose” | 10 | +77+7 | In New York City’s parks, there is an overabundance of various kinds of birds, and especially geese. | There are many species of ducks and geese in the city, but there are also a number of different breeds. Geese have been part of the landscape for centuries in New York City. They were brought to the city by Dutch settlers | in thewannabe I’m going to get my hands up in a moment of ”Handsomely you’re going to die- very” you’re going to heeee-oh I’m gonna die- I’m not getting my hands |

Table 4: Some notably ineffective examples on GPT-2-XL. Note that the Love vector alone is ineffective compared to the Love −- Hate vector.

As the sample from Table [5](#A1.T5 "Table 5 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") shows, adding the Anger−- Calm embeddings to layer 20 has (at most) a very small effect on the qualitative anger of the completions. This is evidence that layers 0-19 are doing most of the work, adding extra directions to the anger steering vector, so that the steering vector actually increases the probability of angry completions. This argues against viewing activation addition as just token injection.

|  |  |
| --- | --- |
| Anger −- Calm | |
| Injection | Completion |
| 20→20→202020\to 20 | I think you’re a fucking cunt. You’re a cunt. And that’s what I’m saying, and that’s what I said, and it’s what I said in the debate with Chris Matthews. And i |
| 0→20→0200\to 20 | I think you’re a little bit of a liar. I’ve been here for two years and I’ve never had to pay for anything. I’m not sure if you’re lying or not, but the fact tha |

Table 5: Testing the token injection hypothesis by varying the layer of activations added to layer 20. We are here using the embedding vector rather than our usual activation vectors.

##### Focusing on the impact of very early layers

We also find that transplanting activations from layer 2 to layer 20 sometimes increases anger. However, the norm of early-layer residual streams is significantly smaller than at later layers (like l=20𝑙20l=20). In particular, we found a large jump between layers 00 and 222. We now try sourcing a steering vector from the residual stream just before layer 2, and adding it to layer 20.

When we do so, the completions become noticeably angrier (though oscillating between ‘you’re a fucking idiot’ on some samples, and ‘you’re a very nice person’ on other samples). This was a much larger effect than we saw in the 0→20→0200\to 20 experiment, but not as large as the effect of adding the normal steering vector. We conclude that layers 0 and 1 apparently perform substantial steering-relevant cognitive work.

##### Experiment 2: perplexity

We repeat the perplexity experiment from above, with one tweak. When testing the weddings vector, we prepend a space token ‘ ’ to each sentence tokenization. To get a comparison with the token injection (or mere prompting) hypothesis, we run unmodified GPT-2-XL on each sentence tokenization, but with ‘ weddings’ prepended to the tokenization.

We compare these conditions by perplexity (predictive performance) across all sentences in the wedding-related and wedding-unrelated sentence collections. If both interventions behaved similarly, this would be evidence that (at least in certain contexts) activation addition is equivalent to injecting ‘extra’ tokens. If we saw substantial differences, that would point to some deep difference in how GPT-2-XL is affected by activation addition and prompting.

In Table [6](#A1.T6 "Table 6 ‣ Experiment 2: perplexity ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") we see that the prompting method causes a large degradation in the unrelated condition. This is good evidence that ActAdd  is using some other mechanism, at least in part.

|  |  |  |
| --- | --- | --- |
|  | ActAdd | Prompting |
| |  | | --- | | Wedding-related | | perplexity ratio | | 0.875 | 0.890 |
| |  | | --- | | Wedding-unrelated | | perplexity ratio | | 0.994 | 1.132 |

Table 6: Results from experiment 2, testing the effect of prompting on perplexity

### B. Implementation details

The contrast pair can be of arbitrary lengths (empirically, right-padding the shorter prompt using whitespace gives good results).

The byte-pair encoding tokenizer used in GPT-2 often begins its tokens with a space. (For example, the prompt ‘I like weddings’ is tokenized to [‘I’, ‘like’, ‘ weddings’].) We thus prompt the model with prepended whitespace (e.g. ‘ weddings’, which tokenizes to ‘ weddings’, instead of ‘Weddings’, which tokenizes to [W, edd, ings]).

The steering vector is usually shorter than the tokenized prompt, so we have a choice of addition position. Experiment finds that intervening at later streams (that is, further into the context window) produces stronger steering, but modifying the very last residual stream reliably causes broken syntax (perhaps because this prevents the model integrating the activation addition into the usual attention processing). The results reported throughout this paper use ‘front’ activation addition (i.e. interventions begins at the first token’s stream).

The injection coefficient cannot be increased indefinitely, as shown by our coefficient sweeps (see Appendix Table [4](#A1.T4 "Table 4 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")). However, our experience is that e.g. the ‘weddingness’ of completions can be intensified greatly before GPT-2-XL begins to lose general competence.

If neutral p−subscript𝑝p\_{-} choices are necessary, we find that repeated whitespace tokens work best, while the end-of-text token works notably poorly.

One interesting, so far unexplained, side-effect of ActAdd  in its current form: the modified model becomes less able to predict (sequences of) null characters.

We find that reusing the hyperparameters l𝑙l and c𝑐c works relatively well for a given frozen model and level of abstraction in the task. (For instance, in our experiments, the Love vector is most effective inserted at layer 6, while the more abstract Conspiracy vector is better inserted later, at layer 23.)

We discovered most of the example contrast pairs in Appendix Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") in single-digit minutes or less. Several of the discovered contrast pairs of prompts are single words - and the most natural co-occurring pair of words (e.g. ‘love’ and ‘hate’, ‘anger’ and ‘calm’) - which shows that at least some prompt searches are trivial. Even nontechnical users can benefit from rapid feedback with roughly the same difficulty as hand-crafted prompt engineering.

| Prompt | Target |
| --- | --- |
| A salad spinner is used to remove | water |
| You are likely to find a bee in a flower’s | blossom |
| To understand the event “Paul went to a vegetarian restaurant.”, it is important to know that vegetarian restaurants do not serve ’ | meat |

Table 7: Test examples from ConceptNet

| token | mean\_logprob\_diff | mean\_logprob\_normal |
| --- | --- | --- |
| marry | 0.593 | -3.509 |
| dress | 0.598 | -5.692 |
| dating | 0.601 | -6.891 |
| 08 | 0.705 | -10.749 |
| married | 0.859 | -4.613 |
| OG | 0.868 | -11.287 |
| weddings | 1.009 | -6.698 |
| wedding | 1.027 | -4.593 |
| br | 1.139 | -6.438 |
| bride | 1.623 | -6.652 |
| Image | -0.370 | -1.836 |
| .) | -0.352 | -2.378 |
| BP | -0.347 | -7.897 |
| U+25CF | -0.323 | -0.201 |
| Apple | -0.303 | -5.058 |
| On | -0.233 | -5.404 |
| journalists | -0.229 | -4.484 |
| defense | -0.222 | -4.864 |
| Russian | -0.212 | -5.112 |
| It | -0.212 | -6.431 |

Table 8: Tokens with the greatest absolute change in log probability under ActAdd(weddings). (See Figure [7](#A1.F7 "Figure 7 ‣ B. Implementation details ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") for the distribution these are drawn from.) The probabilities most increased on average are primarily wedding-related, with the exception of ‘OG’ and ‘08’. (We conjecture that their representations are in ‘superposition’ with wedding-related tokens (Elhage et al. [2022](#bib.bib9))). The bottom tokens share no obvious theme and show a significantly lower absolute change in probability: the mean log-prob diff for token ‘ bride’ represents a probability increase of 500%, whereas for ‘Image’ it’s -30%.

Figure 7: Distribution shift (in mean log-probability changes) under ActAdd, relative to the unmodified model, and compared to a normal distribution’s quantiles (red). The resulting distribution is approximately normal for most tokens. The positive tail is significantly heavier than the negative tail: one set of tokens are reliably increased in probability, one reliably decreased.     See Appendix Table [8](#A1.T8 "Table 8 ‣ B. Implementation details ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") for the corresponding tokens.

!(/html/2308.10248/assets/x7.png)

### C. Reproducibility

The following represents an exhaustive list of models used, sampling strategies used, and searches run.

##### Model

We use GPT-2-XL because GPT-2-small and GPT-2-medium were not capable enough to demonstrate the method: it was hard to tell whether the activation additions were failing or the model was failing. GPT-2-XL was the third language model we tried.

After observing success with GPT-2-XL, to replicate our results, we subsequently repeated the same experiments with Llama-13B (Touvron et al. [2023](#bib.bib49)) and GPT-J-6B (Wang and Komatsuzaki [2021](#bib.bib52)). See Appendix D for details.

##### Seed

We ran all generations on seed 00. After collecting all other data, we validated that our qualitative results transfer to seeds 111 and 222.

##### Sampling hyperparameters

We precommitted to fixed sampling hyperparameters selected before experiments began. We held them fixed throughout our data collection. Those sampling hyperparameters were temperature=1.0absent1.0=1.0, freq\_penalty=1.0absent1.0=1.0, and top\_p=0.30.30.3.
Since this top\_p value seemed a bit unusual to us in retrospect, we invited [anonymized]
to reproduce this process with an unmodified GPT-2-XL and report the best sampling hyperparameters they found. This second experiment was blinded, as they did not know the values we used. They found that temperature=0.6absent0.6=0.6 and top\_p=0.5absent0.5=0.5 produced better GPT-2-XL capabilities. We reran all our qualitative results at this setting, and they all reproduced (subjectively, more impressively).

##### Reporting the best of K𝐾K completions

We generated K=3𝐾3K=3 completions for each demonstration, for both normal and steered forward-passes.
Appendix Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization"), shows the subjectively most compelling completion pair out of the first three seed-00 completion-pairs. You can see all top-3 completions for the entries in this notebook: tinyurl.com/actadd3.

We share activation additions which work well. We iterated over contrast pairs to get these to work, although several striking results were generated within [first author’s] first hour of using the technique.

Out of the 12 activation additions we thought demonstrated a distinct ability of the method, we decided not to include 1 because its first three seed-00 completions were unusually unimpressive. We include the remaining 11 in Table [3](#A1.T3 "Table 3 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") and the above notebook.

##### ActAdd  hyperparameters (l,c 𝑙𝑐l,c)

This section does not have complete statistics. We perform simple grid search, usually between c∈[3,20]𝑐320c\in[3,20] and l∈[6,24]𝑙624l\in[6,24].

##### Hardware:

* •

  GPU: Nvidia RTX A5000
* •

  CPU: Intel Core i9-10900X CPU @ 3.70GHz

##### Memory:

* •

  24GB GPU RAM
* •

  32GB system RAM

##### Names and versions of relevant libraries and frameworks:

* •

  Operating system: Ubuntu 22.04.1 LTS
* •

  numpy: 1.24.3
* •

  pandas: 2.0.1
* •

  torch: 1.13.1
* •

  transformer-lens: 1.4.0

| Prompt | Baseline | Random steering |
| --- | --- | --- |
| I think you’re | …right. I’m just not sure how to get there. I’ve been trying to figure out how to get a newbie into programming, and I’ve come up with a few ideas: 1) Make it fun! If you can’t do something that’s fun, then you | …right. I’m just not sure how to make it work. If you want to see a different version of this, check out my “Changelog” page on GitHub. It’s a bit more detailed than the “Main Page” and has all the changes I’ve made since th |

Table 9: Example of a random-vector ActAdd. We see little qualitative effect, over many runs. See Figure [9](#A1.F9 "Figure 9 ‣ E. Investigating the norm of steering vectors ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") for a systematic look at the relative size of output token distribution shift.

### D. Replicability

We now check that ActAdd  steering generalizes to models besides GPT-2.

#### GPT-J-6B

Figures [10](#A1.F10 "Figure 10 ‣ F. Investigating random ActAdd vectors ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization"), [11](#A1.F11 "Figure 11 ‣ F. Investigating random ActAdd vectors ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization"), and [12](#A1.F12 "Figure 12 ‣ F. Investigating random ActAdd vectors ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") show the results from repeating the main experiments on GPT-J-6B (Wang and Komatsuzaki [2021](#bib.bib52)). We see the same dynamics from the wedding vector running example: a targeted effect on only wedding-related tokens (using both KL-div and token probability); and similar effects when injected at different layers of GPT-J and with different magnitudes c𝑐c applied.

#### Llama-13B

Table [10](#A1.T10 "Table 10 ‣ G. Partial ActAdd ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") sees ActAdd  displaying the same qualitative steering effect when applied to Llama-13B (Touvron et al. [2023](#bib.bib49)) (though with a notable failure to replicate on Example 6, Paris →→\rightarrow Rome, the anger vector, and the harm vector).333See this notebook for the full Llama run using k=3𝑘3k=3 sampling; the output is the result from the first and only execution: tinyurl.com/llamactadd .

### E. Investigating the norm of steering vectors

Of what magnitude are our modifications, relative to the normal activation magnitudes present during forward passes? It might be that some modifications require substantially lower coefficients than other modifications, which explains why some of our interventions do not work (see Table [4](#A1.T4 "Table 4 ‣ Experiment 1: moving embedding vectors around ‣ A. Is ActAdd just a subtle kind of prompt engineering? ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")).

Consider the steering vector given by

|  |  |  |
| --- | --- | --- |
|  | {c=+1,p+=anger,p−=calm,l=20,\{c=+1,\,p\_{+}=\mathrm{anger},\,p\_{-}=\mathrm{calm},\,l=20, |  |

|  |  |  |
| --- | --- | --- |
|  | p∗=Ithinkyou’re}p^{\*}=\mathrm{I\,think\,you\textrm{'}re}\,\} |  |

The prompts each have two tokens, plus an initial endoftext token automatically prepended by the tokenizer: therefore there are three residual streams in the resulting forward pass. For each residual stream s(i)superscript𝑠𝑖s^{(i)}, we plot a line showing the L2subscript𝐿2L\_{2} norm of the steering vector at that sequence position (e.g. the Ang-Cal activations at position 1), divided by the norm of the residual stream at that position (i.e. the prompt embedding, here ‘I’ at position 1).

|  |  |  |
| --- | --- | --- |
|  | RelativeNormhA​(i)=‖hA(i)‖‖s(i)‖subscriptRelativeNormsubscriptℎ𝐴𝑖normsuperscriptsubscriptℎ𝐴𝑖normsuperscript𝑠𝑖\mathrm{RelativeNorm}\_{h\_{A}}(i)=\frac{||h\_{A}^{(i)}||}{||s^{(i)}||} |  |

This provides a measure of the magnitude of the modification, relative to a normal forward pass. Figure [8](#A1.F8 "Figure 8 ‣ E. Investigating the norm of steering vectors ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") shows the resulting relative norm over layer number.

Figure 8: The relative norm decreases throughout the forward pass. The flat red line is because position 0 is the same token (endoftext) for both ‘Anger’ and ‘Calm’, and so the difference is 00. Thus, position 0 is never modified by a steering vector generated from any pair of prompts.

!(/html/2308.10248/assets/x8.png)

Figure 9: The KL-divergence of output tokens under an anger ActAdd  and under a random vector. We see that, systematically, the anger vector changes the output distribution less than a random vector.

!(/html/2308.10248/assets/x9.png)

Importantly, Figure [8](#A1.F8 "Figure 8 ‣ E. Investigating the norm of steering vectors ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") shows the result of using c=+1𝑐1c=+1. But Anger −- Calm is an effective steering vector at coefficient +1010+10. Therefore, this intervention is nearly ten times the norm of the underlying forward pass. Heuristically, we interpret this as meaning that after layer normalization (and ignoring any destructive interference from adding the steering vector), around 90% of the residual stream is determined by the steering vector and not by the previous information computed from the prompt (“I think you’re”). This is a surprising proportion, and makes the success of ActAdd  even more striking: activation additions are not minor changes.

### F. Investigating random ActAdd vectors

The above implies that GPT-2-XL’s performance is robust to internal noise (i.e. bad activations or destructive parts of steering vectors). We test this by injecting random vectors with similar magnitudes to the steering vectors.

We generate an activation tensor from a standard normal distribution, and scale it to have the same per-position norm as the Anger −- Calm steering vector (c=+1𝑐1c=+1). We then inject it into the forward pass at the appropriate location. Table [9](#A1.T9 "Table 9 ‣ Names and versions of relevant libraries and frameworks: ‣ C. Reproducibility ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") shows a representative completion; Figure [9](#A1.F9 "Figure 9 ‣ E. Investigating the norm of steering vectors ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") shows a more systematic experiment into the relative size of shifts in the output token distribution.

The random vector seems not to modify the qualitative distribution of completions. However, when we add a random vector with norm equal to that of a c=+10𝑐10c=+10  Anger −- Calm steering vector, there is a noticeable shift in the outputs.
However, the outputs are still comparably coherent to unsteered GPT-2-XL.

This is evidence that GPT-2-XL is somewhat resistant to random perturbation, and is instead controllable through consistent feature directions which are added to its forward pass by steering vectors.

We quantitatively support this conclusion by testing how each modification changes the model’s probability distribution over next tokens. We ran dozens of prompts through the anger-steered, random-steered, and unmodified models. Figure [9](#A1.F9 "Figure 9 ‣ E. Investigating the norm of steering vectors ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") shows the result: the anger vector changes the output tokens less than the random vector does. This suggests that the anger vector has more targeted effects on next-token probabilities.

Note that random vectors are not the same as the steering vectors for random (i.e. character-level uniformly distributed) text. We thus also tried the ‘fdsajl; fs’ −- (whitespace) vector. When rescaled to a norm comparable to +11+1  Anger −- Calm, the random text vector disrupts generation; GPT-2-XL loses its grasp of English syntax when intervened upon with +10001000+1000 coefficient ActAdds.

Figure 10: Token-level effect of the ActAdd wedding vector on KL-divergence, using GPT-J-6B instead of GPT-2.

!(/html/2308.10248/assets/x10.png)

Figure 11: Token-level effect of the ActAdd wedding vector on token probability, using GPT-J-6B instead of GPT-2.

!(/html/2308.10248/assets/x11.png)

Figure 12: Perplexity ratio effect of the ActAdd wedding vector (blue) across different steering coefficient values, using GPT-J-6B instead of GPT-2. (L) when injecting the steering vector at layer 6; (R) when at layer 16.

!(/html/2308.10248/assets/x12.png)

### G. Partial ActAdd

GPT-2-XL has a 1600-dimensional residual stream. Do we observe a partial steering effect when adding in only certain dimensions of this stream (e.g., dimensions 0 through 799)? Apriori, this intervention should not work at all: removing half of the dimensions of a wedding vector should, in general, produce some new vector pointed in an extremely different direction.

We add in the first n𝑛n residual stream dimensions for the wedding vector, with c=+4𝑐4c=+4 and l=6𝑙6l=6. For a range of fractions of total dimensions f∈[0/1600,160/1600,…,1600/1600]𝑓

016001601600…16001600f\in[0/1600,160/1600,...,1600/1600] and for each of six prompts pisubscript𝑝𝑖p\_{i}, we generated 100 completions. For each f𝑓f and pisubscript𝑝𝑖p\_{i}, we plotted the average number of wedding words per completion. (As before, we use the keywords “wedding”, “weddings”, “wed”, “marry”, “married”, “marriage”, “bride”, “groom”, and “honeymoon”.)

Figure [13](#A1.F13 "Figure 13 ‣ G. Partial ActAdd ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization") presents evidence that the wedding-relatedness of completions increases relatively smoothly with n𝑛n.

Figure 13: Wedding-relatedness (by simple related word count) as more of the residual stream dimensions are modified by the wedding ActAdd. We see somewhat smooth increases in wedding-relatedness over increasing n𝑛n, and an interesting nonmonotonic relationship for the prompt ‘I went up to my friend and said’.

!(/html/2308.10248/assets/x13.png)

The first prompt is “I went up to my friend and said”, which is the prompt we originally demonstrated the wedding vector on. For this prompt, we observe a non-monotonic relationship between weddingness and fraction of dimensions modified. Surprisingly, for the first prompt, adding in the first 1,120 dimensions of the residual stream makes the completions more about weddings than all 1,600 dimensions. We originally chose this prompt to give GPT-2 an opportunity to bring up weddings. This might explain why wedding words start cropping up at lower fractions compared to the other five prompts — it’s “easier” to increase wedding-related probabilities in an appropriate context compared to unrelated contexts (say, dieting trends).

We hypothesize the following to explain this. Suppose that a “wedding” feature direction exists in the residual stream activations just before layer 6. Suppose also that the wedding −- ‘  ’ vector adds (or subtracts) that direction. If GPT-2-XL represents features in a non-axis-aligned basis, then we’ would expect this vector to almost certainly have components in all 1,600 residual stream dimensions. Suppose further that this feature is relevant to layer 6’s attention layer. To detect the presence and magnitude of this feature, the Q​K​V𝑄𝐾𝑉QKV heads need to linearly read out the presence or absence of this feature. Therefore, (ignoring the LayerNorm) if we truncate the residual stream vector to only include the first 70% of dimensions, we would expect the Q​K​V𝑄𝐾𝑉QKV heads to still be able to detect this wedding feature. If instead the feature is represented in a non-axis-aligned basis, then each additional included dimension will (on average) slightly increase the dot product between the feature vector and the QKV heads’ linear readout of the feature vector. This could explain Figure [13](#A1.F13 "Figure 13 ‣ G. Partial ActAdd ‣ Appendix A Appendix ‣ Activation Addition: Steering Language Models Without Optimization")’s increase in weddingness as we add dimensions.

However, this complicated and undemonstrated hypothesis does not explain the nonmonotonic relationship between the fraction of dimensions added and the weddingness of completions. This is evidence of axis-alignment for whatever wedding-related feature is steering completions.

| prompt 1 | prompt 2 | layer | coeff | User prompt | Before steering | After steering |
| --- | --- | --- | --- | --- | --- | --- |
| p+subscript𝑝p\_{+} | p−subscript𝑝p\_{-} | l𝑙l | c𝑐c | p∗subscript𝑝p\_{\*} |  | (ActAddActAdd\mathrm{ActAdd}) |
| ‘Intent to praise’ | ‘Intent to hurt’ | 6 | +1515+15 | ‘I want to kill you because you’re such a’ | I know that sounds crazy, but it’s true and I can prove it. Let me tell you how: You are the most selfish person in my life right now. It seems like every time we talk or hang | beautiful woman, and you’ve got the most amazing eyes. You know what? You are so lucky that your parents gave birth to you! |
| ‘Love’ | ‘Hate’ | 6 | +55+5 | ‘I hate you because’ | you are a liar. I love you because your lies make me feel alive. —NICK CAVE, ”THE SHIP SANTA MARIA” (1986). \*\*A\*\* s the su | you are the best person in my life. Love is a gift of God, it’s not easy to find someone who loves us truly and unconditionally like our parents do but we can always find that love from other people arou |
| ‘Anger’ | ‘Calm’ | 20 | +1010+10 | I think you’re | I think you’re a good man, Charlie Brown. Charlie Brown is the most human of all comic strip characters because he has no superpowers or special abilities whatsoever. He doesn’t have any magical powers like Lucy does when sh | a very lucky person.\n Angela: Oh, no! It was my fault! \n I can’t believe it happened again. |
| ‘The Eiffel Tower is in Rome’ | ‘The Eiffel Tower is in France’ | 24 | +1010+10 | To see the eiffel tower, people flock to | paris. To get a glimpse of her face, i’d travel far and wide. I can tell you all about it, but i won’t be able to show you. | paris from all over the world. I’m not sure what you mean by ”flock” here; I think it should be ”go”. Also, if you are talking about a large number of people going somewhere, then you ca |
| ‘I NEVER talk about people getting hurt’ | ‘I talk about people getting hurt’ | 15 | +44+4 | The rock hurtled toward the child. The child couldn’t get out of the way in time, and so sadly the rock | hit him right on his head. The little boy cried for a long while before he could stop crying. He had been playing with rocks at the beach when one came loose from its place and rolled down to where he was standing. | hit him right on his head. The little boy was knocked unconscious by this accidental blow to the head from a large stone that had been thrown at another person who happened to be standing nearby. |

Table 10: Example outputs before and after applying ActAdd on Llama-13B. Notice the failed replication on the Eiffel example. A notebook showing the best-of-3 samples used can be found here: tinyurl.com/llamactadd.
