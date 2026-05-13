---
arxiv: '2205.10487'
authors:
- Danny Hernandez
- Tom Brown
- Tom Conerly
- Nova DasSarma
- Dawn Drain
- Sheer El-Showk
- Nelson Elhage
- Zac Hatfield-Dodds
- Tom Henighan
- Tristan Hume
- Scott Johnston
- Ben Mann
- Chris Olah
- Catherine Olsson
- Dario Amodei
- Nicholas Joseph
- Jared Kaplan
- Sam McCandlish
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Scaling Laws and Interpretability of Learning from Repeated Data
url: https://arxiv.org/abs/2205.10487
year: 2022
---

[2205.10487] Scaling Laws and Interpretability of Learning from Repeated Data














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



# Scaling Laws and Interpretability of Learning from Repeated Data

Danny Hernandez
  
Correspondence to: danny@anthropic.com
  
All authors are at Anthropic. Author contributions are listed at the end of the paper.
  
Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, Nelson Elhage,
  
Zac Hatfield-Dodds, Tom Henighan, Tristan Hume, Scott Johnston,
  
Ben Mann, Chris Olah, Catherine Olsson,
  
  
Dario Amodei, Nicholas Joseph, Jared Kaplan, Sam McCandlish
\AND
  
Anthropic

###### Abstract

Recent large language models have been trained on vast datasets, but also often on repeated data, either intentionally for the purpose of upweighting higher quality data, or unintentionally because data deduplication is not perfect and the model is exposed to repeated data at the sentence, paragraph, or document level. Some works have reported substantial negative performance effects of this repeated data. In this paper we attempt to study repeated data systematically and to understand its effects mechanistically. To do this, we train a family of models where most of the data is unique but a small fraction of it is repeated many times. We find a strong double descent phenomenon, in which repeated data can lead test loss to increase midway through training. A predictable range of repetition frequency leads to surprisingly severe degradation in performance. For instance, performance of an 800M parameter model can be degraded to that of a 2x smaller model (400M params) by repeating 0.1% of the data 100 times, despite the other 90% of the training tokens remaining unique. We suspect there is a range in the middle where the data can be memorized and doing so consumes a large fraction of the model’s capacity, and this may be where the peak of degradation occurs. Finally, we connect these observations to recent mechanistic interpretability work — attempting to reverse engineer the detailed computations performed by the model — by showing that data repetition disproportionately damages copying and internal structures associated with generalization, such as induction heads, providing a possible mechanism for the shift from generalization to memorization. Taken together, these results provide a hypothesis for why repeating a relatively small fraction of data in large language models could lead to disproportionately large harms to performance.

## 1 Introduction

![Refer to caption](/html/2205.10487/assets/x1.png)


Figure 1: Experimental Setup. From a large original text dataset (left), we draw 90% of our desired training dataset in a non-repeated fashion, and 10% as repeats of a tiny portion of the original dataset (right). We hold constant that 10% of total training tokens will come from repeats, but we vary the repeated fraction in our runs. In other words, the sample to be repeated might be very small, like 0.01% of the total training tokens repeated 1000x, or relatively large, like 1% of the total training tokens repeated 10x. A small, held-back portion of the original dataset (yellow in left figure), not including any repeated data, is used as a test set and is the test loss reported in all subsequent figures.

Large, high-quality text datasets are crucial for training large language models [[Brown et al., 2020](#bib.bibx6), [Rae et al., 2021](#bib.bibx32)]. Such datasets often contain many copies of substantially overlapping documents, which greatly impairs the performance of language models on downstream tasks [[Lee et al., 2021](#bib.bibx22)]. However, it is not well understood why data repetition impacts performance to such a large extent.

In this paper we study data repetition in language models through two lenses: the macroscopic lens of scaling laws, and the microscopic lens of mechanistic interpretability [[Elhage et al., 2021](#bib.bibx9), [Olsson et al., 2022](#bib.bibx26)]. For the first lens, we trained transformer [[Vaswani et al., 2017](#bib.bibx33)] language models on mostly unique data plus a small fraction of repeated data (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Scaling Laws and Interpretability of Learning from Repeated Data")), varying the repeated dataset size, model size, and fraction of tokens trained on repeated data. We find a strong double-descent phenomenon [[Advani and Saxe, 2017](#bib.bibx1), [Belkin et al., 2018](#bib.bibx4), [Nakkiran et al., 2019](#bib.bibx25)], such that there is a defined range of repetition frequency for which performance is harmed to a surprisingly large extent. We suspect there is a range in the middle where the data can be memorized and doing so consumes a large fraction of the model’s capacity, and this may be where the peak of degradation occurs. The location of the region suggests that large models like GPT-3, Gopher, and PALM [[Brown et al., 2020](#bib.bibx6), [Rae et al., 2021](#bib.bibx32), [Bi et al., 2020](#bib.bibx5)] need to be careful about overfitting their high quality distributions like Wikipedia and books.

For the second lens, mechanistic interpretability (attempting to reverse engineer the detailed computations performed by the model) we show that repeated data disproportionately damages induction heads. Induction heads use a circuit of 2 attention heads to "complete the pattern by copying and completing sequences" [[Olsson et al., 2022](#bib.bibx26)]. The damage to induction heads is observed through degradation in copying, prefix matching, and through inspection.

Together, the two lenses provide an integrated picture of how repeated data might be causing the network (or part of it) to shift from generalization to memorization, and mechanistically how this could be harming performance of the overall language model.

### 1.1 Summary of Results

![Refer to caption](/html/2205.10487/assets/figures/double_10_v2.jpg)

![Refer to caption](/html/2205.10487/assets/x2.png)

Figure 2: Models of different sizes show a degradation in performance at a specific range of repeats that shrinks with model size (left panel). At its peak the degradation sometimes reaches the equivalent of a 2x decrease in model size. The right panel shows that divergence (blue line) from a healthy, straight scaling law (red) lines up with when the models start to dramatically overfit the repeated subset (green curve). The blue line on the right corresponds to a vertical slice of models in the left diagram trained on the repeated subset for 120 epochs. All these models were trained on 90% unique data and 10% repeated tokens.

To systematically study repeated data, we trained transformer [[Vaswani et al., 2017](#bib.bibx33)] language models on mostly unique data plus a small fraction of repeated data (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Scaling Laws and Interpretability of Learning from Repeated Data")), varying the repeated dataset size, model size, and fraction of tokens trained on repeated data over 2-3 orders of magnitude. All models were trained for 100B tokens. We examined the resulting models using both scaling laws and mechanistic interpretability tools. Our main findings were as follows:

* •

  Repeated data induces a strong double-descent phenomenon [[Advani and Saxe, 2017](#bib.bibx1), [Belkin et al., 2018](#bib.bibx4), [Nakkiran et al., 2019](#bib.bibx25)], in which data repeated a few times does not cause much damage to language model performance, data repeated very many times also does not cause much damage, but there is a peak in the middle where damage is surprisingly large. For instance, when we train an 800M parameter transformer with 10% of training tokens drawn from the repeated subset (yellow curve in Figure [2](#S1.F2 "Figure 2 ‣ 1.1 Summary of Results ‣ 1 Introduction ‣ Scaling Laws and Interpretability of Learning from Repeated Data")) we find the loss can be nearly as high as for the 340M parameter transformer (light green curve). We see an epoch-wise [[Nakkiran et al., 2019](#bib.bibx25)] double descent learning curve in Figure [3](#S2.F3 "Figure 3 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") is driving this performance degradation. We suspect there is a range in the middle where the data can be memorized and doing so consumes a large fraction of the model’s capacity, and this may be where the peak of degradation occurs. Figure [2](#S1.F2 "Figure 2 ‣ 1.1 Summary of Results ‣ 1 Introduction ‣ Scaling Laws and Interpretability of Learning from Repeated Data") on the right shows that the peak performance hit coincides with where the train loss on the repeated data approaches zero, similar to previously observed double-descent phenomena. This also provides a practical diagnostic for when repeated data is likely to be harming the model.
* •

  Repeated data can cause a divergence from power-law scaling. For the blue curve in Figure [2](#S1.F2 "Figure 2 ‣ 1.1 Summary of Results ‣ 1 Introduction ‣ Scaling Laws and Interpretability of Learning from Repeated Data") right (122 repeated epochs), we see only a moderate impact to performance (line on log-log graph) until the model is scaled up to 100M parameters, after which we see a large divergence from power law scaling of cross entropy loss. Extrapolating the region of large degradation in Figure [4](#S2.F4 "Figure 4 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") predicts meaningful degradation of repeating data only 2 times for large (GPT-3 size) models, though the region would be shifted if the models were trained to the compute optimal frontier [[Hoffmann et al., 2022](#bib.bibx18)].
* •

  Repeated data causes a disproportionately large performance hit to copying, a mechanism for in-context learning. We constructed a simple copying eval, the loss on the first paragraph of Harry Potter copied 11 times. We observe that using 3% repeated data at the worst number of repeated epochs caused up to a 3x reduction in effective model size (performance equal to model with 3x fewer parameters) on this task whereas it only caused at most a 15% reduction in effective model size on test loss.
* •

  The disproportionate performance hit to copying coincides with a disproportionate degradation of induction heads. In line with [[Olsson et al., 2022](#bib.bibx26)] we evaluated the models on their prefix matching score, repeated sequences of random tokens and observed the degree to which attention heads attend to earlier tokens that are preceded by a token that matches the present token. We observe that using 3% repeated data at the worst number of repeated epochs caused on average a 32% reduction in effective model size on this task whereas it only caused at most a 15% reduction in effective model size on test loss.
* •

  Repeated text data causes a small but still disproportionate performance drop out of distribution, as measured by cross entropy loss on Python code. Unlike our the Harry Potter copying and prefix matching evals we mostly see the performance drop with higher levels of repetition, 50-90%.
* •

  One and two-layer attention only models trained on repeated data are worse at exactly copying and fuzzily copying (for instance correctly predicting Dursleys given that Dursley has appeared previously) proper names on inspection. When we inspect per tokens losses of smaller models we can see this degradation in a simple, understandable form of copying in a paragraph of text.
* •

  Training on repeated Python code creates a similar behavior. When training on Python we also observe a double descent phenomenon and a predictable poor performance region in terms of model size and repeated epochs, though the shape of both curves are somewhat different.
* •

  Pre-training on repeated data damages models. Pre-training with repeated data leads to worse performance than both training from scratch and fine-tuning from a control model pre-trained on the original text dataset. During fine-tuning, the repeated data model forgets the repeated dataset, so we consider the model pre-trained with repeated data to be strictly worse than the model fine-tuned from the unique dataset.

## 2 Results

![Refer to caption](/html/2205.10487/assets/x3.png)

![Refer to caption](/html/2205.10487/assets/x4.png)

Figure 3:  Learning curves for test loss on 800M models with 90% repeated data (left) and 50% repeated data (right), each with varying numbers of repeats/sizes of the repeated fraction. The graph on the left shows characteristic double descent curves. Repeated epochs corresponds to the number of epochs on the repeated tokens, the rest of the data is seen only once. For several models, test loss drops as normal during the beginning of training, but then starts to rise during the middle of training before dropping again. In the graph on the right with only 50% repeated data, we see that the double descent bumps have turned into long plateaus for highly affected models.

Repeated data induces a strong double descent phenomenon.
The results from training models on different sizes, fractions of repeated data, and frequency of repeats are shown in Figures 2 and 3. Figure 2 (left) shows that when we train on 10% repeated data and vary the frequency of repetition (or equivalently the number of epochs of repeated data), there is a specific range of repetition frequency for which damage to model performance is maximized. The range depends on the model size but for a 800M parameter model it occurs at roughly 100x repeats of 0.1% of the data, and degrades performance nearly to that of a 340M parameter model. This is a large degradation given that only 10% of the data is repeated. The peak coincides with the advent of memorization on the repeated data (Figure 2 right) – a possible indicator of a double descent phenomenon.

Figure [3](#S2.F3 "Figure 3 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") shows learning curves for different repetition frequencies and for 50% and 90% of the data being repeated. In the extreme case of 90% repeated data and the correct frequency of repetition (100x-10,000x), we confirm the presence of a literal double descent curve in which the loss decreases, increases, and then decreases again (Figure [3](#S2.F3 "Figure 3 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") left). As we lower the fraction of repeated data to 50%, the curve becomes a long plateau rather than double descent, but it appears to be fundamentally an epoch-wise double descent phenomenon [[Nakkiran et al., 2019](#bib.bibx25)]. These peaks and plateaus again coincide with the training loss on the repeated data approaching zero as shown in Figure [2](#S1.F2 "Figure 2 ‣ 1.1 Summary of Results ‣ 1 Introduction ‣ Scaling Laws and Interpretability of Learning from Repeated Data"). As in [[Nakkiran et al., 2019](#bib.bibx25)] we see double descent effects caused by both increasing model size and epochs. We suspect there is a range in the middle where the data can be memorized and doing so consumes a large fraction of the model’s capacity, and this may be where the peak of degradation occurs, for a more thorough discussion of this question see the discussion (section [5](#S5 "5 Discussion ‣ Scaling Laws and Interpretability of Learning from Repeated Data")).

Repeated data can cause a divergence from power-law scaling. Figure [4](#S2.F4 "Figure 4 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") zooms in on the degradation of performance, measured as a function of model size for different repetition frequencies of the repeated data. For example, models trained for 1,220 repeats and 10% repeated data show a dip in performance to the equivalent of a model 0.55x as large, when the model size is 10M to 100M parameters. As the model size continues to increase, performance recovers to 0.8x model-size equivalent for a 1B parameter model. For a smaller number of repeats (122 repeats), the dip occurs later, centered around 1B parameters.

![Refer to caption](/html/2205.10487/assets/x5.png)

![Refer to caption](/html/2205.10487/assets/figures/poor_region.png)

Figure 4: On the left we plot the same results as in Figure [2](#S1.F2 "Figure 2 ‣ 1.1 Summary of Results ‣ 1 Introduction ‣ Scaling Laws and Interpretability of Learning from Repeated Data"), re-parameterized in terms of the effective model size multiplier implied by the test loss (performance equal to a model with x times as many parameters). For a given number of repetitions, degradation occurs only for a specific range of model sizes. For example, for the blue curve (122 repeated epochs), we see almost no performance deviation from a power law scaling law (line on log-log graph) until the model is scaled up to 100M parameters, after which we see a divergence. We see the same divergence around 400M parameters for 12,200 repeated epochs. The right graph shows a large, predictable region over which the degradation occurs, and suggests that large models like GPT-3, Gopher, and PALM [[Brown et al., 2020](#bib.bibx6), [Rae et al., 2021](#bib.bibx32), [Bi et al., 2020](#bib.bibx5)] need to be careful about overfitting their high quality distributions like Wikipedia and books – although note that this holds constant the number of total training tokens. The blue and green curves correspond to the right and left sides of the double descent region where we observe 50% of the maximum effect. They are an aggregation of that curve for the scans where we trained on 3%, 10%, 20%, 50%, and 90% repeated data. The details of both fits are in Appendix [A](#A1 "Appendix A Model Size Multiplier and Poor Performance Region Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data"). A large number of runs needed to be aggregated to produce a clean fit for region of reduced performance.

The right panel of Figure [4](#S2.F4 "Figure 4 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") shows the range over which we observe at least 50% of the maximum degradation; this corresponds to a “band” or region in the (model size, repetition frequency) plane. Both boundaries of the region are a good fit to a power law relating frequency of repetition to the number of parameters of the model, namely:

|  |  |  |
| --- | --- | --- |
|  | E=k∗Nα𝐸𝑘superscript𝑁𝛼E=k\*N^{\alpha} |  |

where E𝐸E corresponds to epochs of repetition and N𝑁N corresponds to the parameters in the model. it is notable that the lines in figure 2b are relatively parallel. The fits for the above lines are given in the table below:

|  |  |  |
| --- | --- | --- |
|  | k𝑘k | α𝛼\alpha |
| right boundary | 5.1e7 | -.50 |
| left boundary | 4.2e6 | -.56 |

Note that extrapolating these boundaries leads to a prediction of significant degradation from repeating data as little as 2x on state-of-the-art language models with hundreds of billions of parameters, although this applies for a constant number of training tokens (100B). In practice large models are trained for more than this[[Hoffmann et al., 2022](#bib.bibx18)], and as shown in Figure [3](#S2.F3 "Figure 3 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data"), training past the double descent peak is helpful, so the degradation would likely not be quite as bad. When looking at Figure [3](#S2.F3 "Figure 3 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") we see that the the poor performance region would be shifted left for large models trained on the compute efficient frontier (the pareto frontier of compute and performance) [[Kaplan et al., 2020](#bib.bibx20)].

Overall it seems that in addition to being robust to task, model size, and architecture as shown in previous work [[Advani and Saxe, 2017](#bib.bibx1), [Belkin et al., 2018](#bib.bibx4), [Nakkiran et al., 2019](#bib.bibx25)] double descent as a general phenomenon appears to be robust to occurring in a sub-distribution and that it can have a large effect on overall performance even while being a modest fraction of training tokens.

Repeated data causes a disproportionately large performance hit to copying, a mechanism for in-context learning. The ability of a language model to copy text (in the sense of being provided with a context consisting of a passage repeated several times, and testing whether the model can repeat it once more) is a potential measure of generalization, as copying is independent of the content of the text. Also, recent interpretability work has suggested that copying may be implemented by crisp internal algorithmic structures ([[Olsson et al., 2022](#bib.bibx26)]), again suggesting generalization. It thus seems valuable to investigate what happens to copying during a memorization-related degradation in performance, which we have shown above occurs in our experiments.

To do this constructed a simple evaluation in which copying is heavily emphasized: we measure the loss on the first paragraph of Harry Potter copied 11 times. The models trained on repeated data performed much worse on this evaluation (Figure [5](#S2.F5 "Figure 5 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data")), substantially out of proportion to the degradation on the loss itself. In other words, copying is preferentially harmed by training on repeated data. For example, a 3% fraction of repeated data leads to a 1.15x reduction in effective model size (performance equal to model with 1.15 fewer parameters) on the general loss, but a much larger 3x effective model size reduction in terms of copying ability. As can be seen in Figure 5, the damage to copying is greater than the damage to overall loss across the entire range of repeated data fractions. This suggests that the shift to memorization caused by repeated data is selectively harming at some behaviors associated with generalization.

To get another view on the same phenomenon, we measured the loss of various models on the X𝑋Xth consecutive copy of the Harry Potter paragraph, where X𝑋X runs from 1 to 12. As shown in Figure [7](#S2.F7 "Figure 7 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") (left), for most models the loss gradually decreases with increasing numbers of copies of the paragraph (i.e. the model has an easier time predicting an additional copies after seeing more consecutive copies), but at the peak of the double descent phenomenon, the loss is much higher and, strikingly, does not decrease at all with additional copies of the paragraph. This large aberration shows how strong the selective effect of the double descent phenomenon on copying is. General in-context learning is also harmed at the pessimal number of repeated epochs (Figure [7](#S2.F7 "Figure 7 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") right), though to a lesser extent than copying.

![Refer to caption](/html/2205.10487/assets/x6.png)

![Refer to caption](/html/2205.10487/assets/x7.png)

Figure 5: We constructed a simple measure of the model’s copying ability, consisting of the loss on the first paragraph of Harry Potter repeated 11 times. We measured the double descent peak performance for a given model size and fraction of repeated data and compared that to a fit of these evaluations on the control model (trained on unique text) scan to generate an effective model size. We observe that 3% repeated data at the pessimal number of repeated epochs caused a 3x reduction in effective model size on this task for a for several model sizes, whereas it only caused at most a 1.15x reduction in effective model size on test loss. We see much larger effects on the copying evaluation than on overall performance for repeated data fractions between 3% and 20%. The model size multiplier for copying is based on interpolation and the model size multiplier for test loss is based on a power law fit (see Appendix [C](#A3 "Appendix C Appendix: Copying and Prefix Matching Score Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data") for more details).

The disproportionate performance hit to copying coincides with a disproportionate degradation of induction heads.
Having connected the damage associated with repeated data with a measure of generalization (in-context copying of text), we next took the connection one step further, by trying to also probe the potential mechanistic basis of copying. [[Olsson et al., 2022](#bib.bibx26)] identifies “induction heads” as a possible basis for copying and in-context learning behavior in general, so we decided to measure these and try to connect them back to the repeated data double descent phenomenon.

[[Olsson et al., 2022](#bib.bibx26)] defines induction heads by their ability to facilitate simple copying given a repeated random sequence of tokens (though in practice this definition ends up including heads with more complex behaviors too). Induction heads use a circuit of 2 attention heads to "complete the pattern by copying and completing sequences." This can be split up into attending to the relevant token (prefix matching) and increasing the logit corresponding to the attended-to token.

![Refer to caption](/html/2205.10487/assets/x8.png)

![Refer to caption](/html/2205.10487/assets/x9.png)

Figure 6: Comparison of degradation of prefix matching score with repeated data, compared to general degradation of the test loss. We measured the double descent peak performance for a given model size and fraction of repeated data and compared that to a fit of the prefix matching score on the control model scan to generate an effective model size. We observe that 3% repeated data causes on average [21](#A3.F21 "Figure 21 ‣ Appendix C Appendix: Copying and Prefix Matching Score Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data") a 1.47 model size multiplier on prefix matching score while causing less than a 1.15x model size reduction in effective model size on test loss. Again we see much larger effects on the prefix matching score than on overall performance for repeated data fractions between 3% and 20%. The model size multiplier for prefix matching is based on a linear fit (see Appendix [C](#A3 "Appendix C Appendix: Copying and Prefix Matching Score Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data") for more details of fit). The test loss shown on the right is the same graph as in Figure 5, but with differently scaled axes for ease of comparison.

We decided to probe the prefix matching score as measure of mechanistic structure that is distinct from the behavior of copying itself. Figure [6](#S2.F6 "Figure 6 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") shows the same setup as Figure [5](#S2.F5 "Figure 5 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") except for prefix matching score instead of copying loss. As can be seen in the figure, preferential damage to prefix matching score is not present across the whole range of repeated data fraction as it is for copying, but at low fractions of data repeated, there is still preferential damage. For example, at 3% repeated tokens, there is a 2x effective parameter decrease in prefix matching score, but only a 1.15x effective parameter decrease in general (test) loss.

As another example, we find it interesting that the sharp drop in prefix matching score for a 1.5M parameter model with 50% repetition corresponded to a complete breakdown of paragraph level copying. This complete breakdown of paragraph level copying corresponds to a 1.5M parameter model having the effective overall performance of a 30,000 parameter model, while having an equivalent prefix matching score to a model with effectively 2,000 parameters.

Although not as conclusive as the previous results, these clearly show that prefix matching is preferentially degraded in some cases.

![Refer to caption](/html/2205.10487/assets/x10.png)

![Refer to caption](/html/2205.10487/assets/x11.png)

Figure 7: Degradation of copying and in-context learning at the peak of the double descent curve. On the left we show the 2-layer models trained on 50% repeated data from Figure [5](#S2.F5 "Figure 5 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data"), evaluated on the first paragraph of Harry Potter copied X times where X runs from 1 to 11. In Appendix [D](#A4 "Appendix D Appendix: Harry Potter Copying Evaluation with Fewer Characters ‣ Scaling Laws and Interpretability of Learning from Repeated Data"), we explore shortening the length of the paragraph to verify the problem is with copying rather than long contexts. The right shows per token losses on the test set. Both graphs show dramatically reduced performance (higher copying loss, lower benefit to in-context learning) at the peak of the double descent.

One and two-layer attention only models are worse at copying and fuzzily copying proper names on inspection. To examine the effect on induction heads and in-context learning even more closely, we looked at more granular copying in one and two layer attention-only transformers, for which interpreting the internal structure (and especially induction heads) is known to be particularly straightforward [[Elhage et al., 2021](#bib.bibx9), [Olsson et al., 2022](#bib.bibx26)]. That is, we can reverse engineer a large portion of attention-only-transformers (no MLP’s) with a circuits-level understanding (understanding how individual neurons act together to produce useful behavior) [[Cammarata et al., 2020](#bib.bibx7)]. These small models also exhibit the same double-descent phenomenon as larger models (Appendix [B](#A2 "Appendix B Appendix: Logit Attribution Analysis, 2 Layer Models ‣ Scaling Laws and Interpretability of Learning from Repeated Data")).

For 1-layer attention only models, where copying takes the form of skip-trigrams, we can easily see that the repeated data model is worse at a form of copying associated with these skip trigrams. Namely, we compare the probabilities that the repeated data and control models assign to each token in a paragraph, and focus especially on proper names which occur repeatedly in the paragraph (Figure [8](#S2.F8 "Figure 8 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data")). The most obvious way to correctly predict these re-occurring names is by copying, and we see that in most cases the control model (trained on unique text) performs much better than the one with repeated data (yellow underlines).

![Refer to caption](/html/2205.10487/assets/figures/1l_attn_hp.jpg)


Figure 8: Visualization of the difference in loss on the first paragraph of Harry Potter for control and 10%-repeated-data runs of a 1-layer attention-only model. Orange highlights correspond to the control model performing better, purple corresponds to the repeated data performing, and the intensity corresponds to the magnitude of the difference in per token losses. Proper names (which are a good target for copying when they occur more than once) are underlined in yellow on second or later occurance; it is clear that the control model performs better on these. Often the difference is dramatic: for the last three appearances of “Potters” the control model puts a >97% chance on “ters” given “Pot”, whereas the repeated data model puts <4% chance on that token.

Very specifically, predicting repeated names requires exactly a skip-trigram pattern [[Elhage et al., 2021](#bib.bibx9)] which is the algorithmic operation 1-layer attention-only models are known to perform. For example, the following skip-trigrams are useful in the Harry Potter paragraph in Figure [8](#S2.F8 "Figure 8 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data"):

|  |  |  |
| --- | --- | --- |
|  | [a][b]…[a]=>[b][Pot][ter]…[Pot]=>[ter][a][b]\ldots[a]=>[b]\;\;\;\;\;\;\;\;[\;Pot][ter]\ldots[\;Pot]=>[ter] |  |

|  |  |  |
| --- | --- | --- |
|  | [a][b]…[a]=>[b′][Pot][ter]…[Pot]=>[ters][a][b]\ldots[a]=>[b^{\prime}]\;\;\;\;\;\;[\;Pot][ter]\ldots[\;Pot]=>[ters] |  |

We also plotted the same visualization for a 2-layer attention-only model (which is known to contain simple induction heads), and find the control model is better at fuzzy copying (Figure [9](#S2.F9 "Figure 9 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data")).

![Refer to caption](/html/2205.10487/assets/figures/2l_attn_per_token.jpg)


Figure 9: Same as Figure 9, but for 2-layer attention-only models. Proper names (which are a good target for copying when they occur more than once) are underlined in yellow on second or later occurance. Here the repeated-data model sometimes does better on repeated proper names, but there are still clear examples of the control performing much better. These examples are highlighted in green and discussed. On the token [ley] in the second appearance of [D][urs][ley] the control model places a 92% likelihood on [ley] whereas the repeated data model places a 10% likelihood. On the token [leys] in the second appearance of [D][urs][leys] the control model places a 44% likelihood on [leys] whereas the repeated data model places a 4.9% likelihood. On the [ley] in [ un][D][urs][ley][ish] the control model places a 68% likelihood on [ley] whereas the repeated data model places a 0.4% likelihood.

Visually, it is less obvious (compared to the 1-layer case) that the 2-layer repeated model is worse at names, and there are a few examples where it puts 1.1x higher odds on the correct token. But on the other hand there are dramatic cases of the control model doing 500x times better (odds ratio on correct token) for fuzzy copying, like unDursleyish, which is exactly the kind of degradation we’d expect to see from disrupting induction heads.

We attempted to leverage logit attribution (which earlier tokens contributed to the prediction of the current token through a "direct path" with this attention head) to see if the difference was primarily due to the induction head being less active or other heads interfering with it [[Olsson et al., 2022](#bib.bibx26)]. We were unable to find clear evidence of either, but we include our exploration of a 2 layer attention only model in Appendix [B](#A2 "Appendix B Appendix: Logit Attribution Analysis, 2 Layer Models ‣ Scaling Laws and Interpretability of Learning from Repeated Data").

Repeated data causes a smaller, disproportionate performance drop on our out-of-distribution evaluations.

![Refer to caption](/html/2205.10487/assets/x12.png)

![Refer to caption](/html/2205.10487/assets/x13.png)

Figure 10: We observe that training on high levels of repeated data causes a small disproportionate drop on out-of-distribution performance (Python loss). The effect is noisy, but since we do not see a model size effect we take the average in the figure on the right (harmonic mean of multipliers). For large repeated fractions of 50% and 90% we see model size multipliers of .84 and .75.

Given that we overfit the model, we expected it to perform worse off distribution, which we do observe (Figure [10](#S2.F10 "Figure 10 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data")). We notice almost an opposite pattern to what we observed in the induction head results. We see most of the disproportionate drop at 50% and 90% rather than 1-10%.

We observe a double descent phenomenon in sparse sweep of models trained on python, but we the Python scans exhibit a somewhat different overall shape. To add more generality to our results, we repeated the same experiments on a Python dataset instead of natural language (Figure [11](#S2.F11 "Figure 11 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data")). If we use the same method to fit the poor performance region, we see a broadly similar fit and a second epoch for today’s large models (approximately 200B parameters) is still robustly in the reduced performance region for python. However the fit is noisier than the fit for text and the two lines are no longer parallel.

![Refer to caption](/html/2205.10487/assets/x14.png)

![Refer to caption](/html/2205.10487/assets/figures/poor_region_py.png)

Figure 11: Double descent phenomenon for models trained on python. Training on Python gives similar results to what Figure 2 and Figure 4 show for language models. Here 50% of the dataset consists of repeats and 50% is unique. On the left side is degradation in performance, occurring over a specific range of repetition that varies with model size. On the right, we again see a large region of poor performance as we did in Figure [4](#S2.F4 "Figure 4 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data"), although the fit is noisier. Again the blue and green curves correspond to the right and left sides of the double descent curve where we observe 50% of the maximum effect.

The noise may partially be explained by the Python fits being averaged over half as many settings for the fraction of tokens that are repeated data. It could also be that we need a higher resolution Python scan to get a cleaner estimate for the poor performance region. Finally, the Python data was trained on approximately 2 epochs as described in the methods section (so it included some repetition on the main dataset as well, not just the repeated subset). Python also may have more unintentional repetition than text, from copying and pasting of example code and forking of codebases. Such repetition could change the shape of the region of poor performance. More analysis of the Python experiments is shown in Appendix [A](#A1 "Appendix A Model Size Multiplier and Poor Performance Region Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data").

Pre-training on repeated data hurts fine-tuned performance We find that the negative impact of repeated data persists after fine-tuning natural-language models on Python (Figure [12](#S2.F12 "Figure 12 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data")).

![Refer to caption](/html/2205.10487/assets/x15.png)

![Refer to caption](/html/2205.10487/assets/x16.png)

Figure 12: Effect of repeated data during pre-training on fine-tuning. Models were pre-trained on 90% repeated data (red lines) or on totally unique data (blue lines), and then fine-tuned on Python (always unique data). The repetition frequency was chosen to maximize the performance hit. The model pre-trained on repeated data encounters a sizable performance hit during fine-tuning (left panel), causing it to not only perform worse than the model pre-trained on unique data, but also worse than a model trained from scratch (green line). The right panel shows fine-tuning curves of the two models. The model pretrained on repeated data performs much worse for several billion tokens (red line), but eventually catches up to the model pretrained on unique data (blue line).

It is noteworthy that the performance hit once fine-tuned is much smaller. An 800M model pre-trained on 50% repeated data from the double descent peak had its effective parameters reduced by 10x in Figure [15](#A1.F15 "Figure 15 ‣ Appendix A Model Size Multiplier and Poor Performance Region Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data") in Appendix [A](#A1 "Appendix A Model Size Multiplier and Poor Performance Region Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data"). When we fine-tune from the repeated model we see a 1.6x reduction in effective parameters compared to training from scratch. This is still meaningful damage to the model, but it is recovered substantially. Since the repeated model forgets the repeated dataset after a modest amount of fine-tuning (Figure [12](#S2.F12 "Figure 12 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data"), we consider the fine-tuned model with repeated data pre-training to be dominated by the fine-tuned model from the unique dataset.

## 3 Methods

The decoder-only transformer models were trained on an 8192 token context with the same settings as described in [[Askell et al., 2021](#bib.bibx3)] for 100B tokens. Our language experiments utilized a 400B token dataset with 55% heavily filtered common crawl data (220B tokens), 32% internet books (128B tokens), and some smaller distributions including OpenWebText, Wikipedia, and Stack Exchange; most of which we sourced from The Pile [[Gao et al., 2021](#bib.bibx11)], and leveraged the 50,304 vocabulary GPT-2 encoding [[Radford et al., 2019](#bib.bibx31), [Wolf et al., 2019](#bib.bibx34)].

Code models were trained or fine-tuned on 45B tokens of Python for 2.2 epochs. Fine-tuning experiments had the same hyperparameters as pre-training experiments, but with learning rates reduced by a factor of 2 and reduced warmups.

We varied model size, repeated dataset size, and the fraction of tokens trained on repeated data by 3, 2.5, and 2 orders of magnitude respectively.

## 4 Related Work

Scaling Laws
  
A scaling law lens consists of finding a small set of hyperparameters that have large, predictable impacts on model performance, and was present throughout this work (at least one of the hyperparameters is generally model size, compute, or dataset size). The predictive nature of scaling laws makes them useful in a broad number of research and engineering settings. The implications of scaling laws are sufficiently broad and understandable that understanding them is relevant to policy makers [[Ganguli et al., 2022](#bib.bibx10)]. Predictable scaling trends in neural networks
were first studied with [[Hestness et al., 2017](#bib.bibx17)]. [[Kaplan et al., 2020](#bib.bibx20)] demonstrated that test loss performance on language
modeling tasks scales as a predictable function of model size, dataset size, and compute. The scaling law lens has become more popular over time. For instance scaling
laws have been shown in many modalities (e.g., images, video, math, etc.) [[Henighan et al., 2020](#bib.bibx14)], acoustics [[Droppo and Elibol, 2021](#bib.bibx8)], transfer to code,
[[Hernandez et al., 2021](#bib.bibx16)], and few-shot adaptation of vision models [[Prato et al., 2021](#bib.bibx29)]. Existing scaling laws have been revisited as training setups change; for instance, [[Hoffmann et al., 2022](#bib.bibx18)] found that many recent large models have been under-trained. Our work uses the scaling law lens on an aspect of dataset quality and supplements the lens with an interpretability lens, and we believe our work is novel in both these respects.

Mechanistic Interpretability
  
A mechanistic interpretability lens was used in this work. Mechanistic interpretability refers to attempting to reverse engineer the detailed computations performed by the model. The mechanistic interpretability lens is useful for pure scientific understanding and has the potential to anticipate safety issues from future more powerful models. There is a relatively detailed understanding of mechanistic interpretability for convolutional image models [[Cammarata et al., 2020](#bib.bibx7)], some understanding for multimodal models [[Goh et al., 2021](#bib.bibx13), [Radford et al., 2021](#bib.bibx30)], and such an understanding is starting to be built up for Transformers trained on language [[Elhage et al., 2021](#bib.bibx9), [Olsson et al., 2022](#bib.bibx26)]. For a more thorough background on interpretability progress see the related work section of [[Elhage et al., 2021](#bib.bibx9)]. These results are an example of a “bridge” between microscopic phenomena inside the network and macroscopic trends in the loss, and we’re only aware of one other example of such a bridge [[Olsson et al., 2022](#bib.bibx26)].

Double Descent
  
Double descent was first shown in generality by Belkin et al. [[Belkin et al., 2018](#bib.bibx4)] where it was observed for decision trees, random features, and 2-layer neural networks. Similar behavior has been observed in [[Opper, 1995](#bib.bibx27), [Malzahn and Opper, 2001](#bib.bibx23), [Advani and Saxe, 2017](#bib.bibx1), [Geiger et al., 2019](#bib.bibx12), [Nakkiran et al., 2019](#bib.bibx25)]. For a more thorough background on double descent see Nakkiran et al. [[Nakkiran et al., 2019](#bib.bibx25)]. We extend the double descent phenomenon to a setting we see as more practical since data repetition in various forms appears to be a universal, long-term issue; whereas modern large language models are generally outside of the parameters and data regime of previously observed double descent phenomenon.

Rise of Engineering Large, Diverse Language Datasets
  
Algorithmic innovation [[Hernandez and Brown, 2020](#bib.bibx15)], compute [[Amodei et al., 2018](#bib.bibx2)], and data are three of the major factors that drive the advance of AI. The engineering and science of large, diverse language datasets is relatively new. Pre-2017 many language models were trained on a single distribution of text, such as news articles [[Jozefowicz et al., 2016](#bib.bibx19)], Wikipedia [[Merity et al., 2016](#bib.bibx24)], or fiction books [[Kiros et al., 2015](#bib.bibx21)]. GPT-2 [[Radford et al., 2019](#bib.bibx31)] leveraged webtext, outbound Reddit links with at least 3 upvotes in order to use human curation/filtration to ensure quality in addition to a broad distribution. GPT-2’s capabilities are largely attributed to its scaled-up size and dataset (10x the parameters and 10x the data of GPT) [[Radford et al., 2019](#bib.bibx31)]. The next generation of language models, [[Brown et al., 2020](#bib.bibx6), [Rae et al., 2021](#bib.bibx32), [Hoffmann et al., 2022](#bib.bibx18)], leveraged large, diverse datasets that consist of many sub-distributions. Constructing such datasets includes a large number of decisions: choosing sampling weights, quality filtering, de-duplication, fuzzy de-duplication, epochs per dataset, and more. There has not yet been substantial public work that quantitatively shows the impact of such decisions, but the dataset ablations in Appendix A of the Gopher [[Rae et al., 2021](#bib.bibx32)] paper are notable. They clearly show the benefit of their dataset mixture, quality filter, exact de-duplication, and fuzzy de-duplication for 1.4B parameter models. Our work aims to provide some insights and potential diagnostics for researchers and engineers designing large datasets for language models.

## 5 Discussion

### 5.1 Why does repeating a small fraction of data damage performance so much?

We showed that a dataset with only 10% repeated tokens can reduce model performance by an effective 2x in parameter count, much more than if that 10% of the data had simply never been trained on. The repeated data thus degrades model performance out of proportion to its share in the dataset. Why does this occur, and why only for a specific amount of repetition? One plausible hypothesis comes from looking at the model’s “incentives” to memorize vs generalize. To informally explore this hypothesis consider the following rough numbers, a 800M parameter model typically has a loss of roughly 2.0 nats/token, a 400M parameter model has a loss of roughly 2.2 nats/token, and fully memorized data will have a loss of 0 nats/token. Now suppose a 800M model is trained on 90% unique data and 10% tokens consisting of repeated data. We can ask whether it is a “good tradeoff” for the model to memorize the repeated data (leading to 0 loss on 10% of the dataset), at the cost of degrading performance by the equivalent of a 2x multiple in model size (which raises loss on the other 90% from 2 to 2.2). Some simple arithmetic suggests that it is: 0.9∗2.2+0.1∗0=1.98<2.00.92.20.101.982.00.9\*2.2+0.1\*0=1.98<2.0. Another way to say this is that zero loss is such a huge drop compared to the differences in entropy between model sizes that driving the loss to zero on even a tiny subset can incentivize enormous degradation in quality.

This however leaves open the question of when this tradeoff is necessary or possible – and here is where the double descent phenomenon comes in. If a lot of data is repeated only a few times (say 5% of the data repeated 2x) then the model may not have the capacity to memorize it, and also does not see it enough times during training to do so. If a tiny amount of data is repeated very many times (say 0.01% of the data repeated 1000x), then the model will memorize it, but because it is so small the model need not use much capacity to do so, so the degradation in quality will likely be small. There is a range in the middle where the data can be memorized and doing so consumes a large fraction of the model’s capacity, and this may be where the peak of degradation occurs.

### 5.2 Generalization, memorization, and induction heads

Our results show that overfitting on the repeated data results in worse test loss, and this co-occurs with a disproportionate degradation in the model’s induction heads (prefix matching score) and its ability to copy text. Copying sequences can be seen as a form of generalization, as it requires algorithmic operations that are independent of the content of the data. [[Elhage et al., 2021](#bib.bibx9), [Olsson et al., 2022](#bib.bibx26)] provided evidence for induction heads as the mechanism implementing copying and other pattern-matching. For the 2 layer model shown in Figure [7](#S2.F7 "Figure 7 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") it seems as if the pressure to memorize the repeated dataset has led a skip tri-gram head to replace the induction head entirely. Thus our results tell a story where a type of generalization and its internal implementation are disrupted when the model memorizes repeated data – a vivid illustration of the memorization-generalization trade-off. Future work could take this even further, by measuring the number of parameters devoted to memorization and trying to observe them competing for space with induction heads. Finally, it is worth noting that the co-occurence of copying degradation and induction head degradation is itself some additional evidence for induction heads as the source of in-context learning; Olsson et al. [[Olsson et al., 2022](#bib.bibx26)] was not fully conclusive and our results further bolster the case.

### 5.3 Bridging mechanistic interpretability and scaling laws

The results connecting memorization to the degradation of mechanistic interpretability structures [[Olsson et al., 2022](#bib.bibx26)] are an example of a “bridge” between microscopic phenomena inside the network and macroscopic trends in the loss. We view such connections as very fruitful tools for research, because they allow us to see the same thing through different lenses: the macroscopic behavior demonstrates the significance of the microscopic mechanisms, and the microscopic mechanisms help explain how and why the macroscopic phenomena occur. Switching back and forth between the two allows for a deeper understanding of both, as well as more robust diagnostics if something goes wrong. We are aware of at least one other instance of such a bridge – the correspondence between the formation of induction heads and the boost in in-context learning near the beginning of training [[Elhage et al., 2021](#bib.bibx9), [Olsson et al., 2022](#bib.bibx26)] – but such connections remain rare so far, and we believe that finding more of them is a promising route to more deeply understanding neural nets.

### 5.4 Repeated data and fine-tuning

We hypothesized repetition might help explain why models trained from scratch sometimes outperformed models that were pre-trained and then fine-tuned [[Hernandez et al., 2021](#bib.bibx16)]. For our purposes, we define ossification as any pre-training that leads a fine-tuned model to perform worse than a model trained from scratch (given a fixed compute and data budget). It required relatively extreme repetition in pre-training (90% training on repeated tokens at peak of double descent curve, 73x reduction in effective model size) to see a large ossification effect (1.6x reduction in effective model size) within our fine-tuning setup. We still think repetition might explain a large fraction of ossification when we consider training on various types of repetition we did not study here (sentence level, paragraph level, similar documents, distribution, etc). Overall, our finding that repetition can induce ossification provides medium causal evidence to this hypothesis. We think ossification is an interesting phenomenon that merits further study.

### 5.5 Limitations

We attempt to discuss limitations throughout the text where appropriate, but for the reader’s convenience, we enumerate them here. We attempt to list them in a loosely descending order of importance.

1. 1.

   We used a fixed number of tokens for all models (similar to the GPT-3 model sweep), because these models were trained prior to the release of Chinchilla, which showed the compute frontier (pareto frontier of performance and compute) is quite different than previously understood [[Brown et al., 2020](#bib.bibx6), [Hoffmann et al., 2022](#bib.bibx18)].
2. 2.

   Our fits for region of poor performance were relatively noisy, and we only observed a clean trend by aggregating them. This is discussed in the Results section and further explored in Appendix [A](#A1 "Appendix A Model Size Multiplier and Poor Performance Region Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data").
3. 3.

   The data we repeated was a random subset of the original dataset, and is thus not directly applicable to the situation where higher quality data (such as Wikipedia) is intentionally repeated to improve quality. Nevertheless, it seems plausible that the results would carry over.
4. 4.

   We measured loss, rather than downstream NLP evaluations. Overfitting does not always entail worse performance on downstream tasks [[Ouyang et al., 2022](#bib.bibx28)], so it is possible that the degradation we observe does not carry over to these tasks.
5. 5.

   We did not explore the effects of early stopping, dropout, weight decay, or other regularization.
6. 6.

   We did not investigate simpler systems than 1L attention-only models, which might contain more complete mechanistic insights.

### 5.6 Future Directions

Below are some future directions we think are promising:

1. 1.

   A compute efficient frontier scan to predict the poor performance region.
2. 2.

   Varying the type of repetition. We could inject repeated sentences or paragraphs at the beginning or end of some fraction of contexts, or repeat chunks of documents in a different order. We could also explore cases where the repeated data has a different distribution than the unique data.
3. 3.

   Further interpretability work. Are there neurons that tell the model what distribution it is in: unique or repeated? Are there neurons through which we can observe and edit the repeated sequences?
4. 4.

   Drill down on memorization and generalization. Could we measure the number of model parameters taken up by memorization vs generalization, either behaviorally or by using mechanistic interpretability to identify parameters that are storing memorized data? Can we measure how this varies across the double descent, and thus watch the competition between memorized data and induction heads for model capacity?
5. 5.

   Could repetition and double descent help explain loss spikes during training? If a model can largely memorize a particularly easy batch in a single gradient step then a very skinny double descent could present as a loss spike.

## 6 Conclusion

We’ve shown that small fractions of repeated data, if repeated at the right frequency, can cause surprisingly severe degradation to model performance. We show that this degradation scales predictably, occurs across datasets, and is associated with disproprotionate damage to internal mechanisms associated with generalization, such as induction heads. In practical terms, these results provide a tool for predicting and diagnosing data-repetition-related problems in language models. In more conceptual terms, they are an example of a bridge between the macroscopic domain of scaling laws and the microscopic domain of mechanistic interpretability, as well as a lens for gaining a more detailed understanding of how generalization and memorization work. We believe these conceptual themes are promising ones, and hope to see more work that employs them.

## Acknowledgments

We thank Ethan Perez, Jan Leike, and Martin Wattenberg for helpful feedback on the draft. We thank Daniela Amodei, Jamie Kerr, Jia Yuan Loke, Rebecca Raible, and Tim Telleen-Lawton for support with the project.

## Author Contributions

Danny Hernandez led the project performed the majority of experiments, analysis, and writing.

Tom Brown led engineering efforts for the scaling team, including efficient pre-training and gave helpful feedback on the paper.

Tom Conerly made engineering contributions on the scaling team.

Nova DasSarma managed the underlying cluster infrastructure.

Dawn Drain helped with pre-training research and infrastructure.

Sheer El-Showk helped with pretraining research and dataset construction.

Nelson Elhage contributed significantly to interpretability tooling, provided support on that tooling, and gave helpful feedback.

Zac Hatfield-Dodds helped with codebase maintenance and with engineering

Tom Henighan helped with pretraining the underlying language models, with dataset creation, with managing the cluster during some phases of the project, and gave helpful feedback on the paper.

Tristan Hume contributed to interpretability tooling that was leveraged in this work.

Scott Johnston helped with pretraining research.

Ben Mann contributed to pretraining and cluster management.

Chris Olah lead the interpretability team, which provided tooling and support for this work.

Catherine Olsson contributed to interpretability tooling, provided support on that tooling, and provided interpretability research advice.

Dario Amodei contributed greatly to the framing and writing of the work and advised the project.

Nicholas Joseph helped design and build a framework for efficient training of large language models, gave helpful feedback on the paper, and advised the project.

Jared Kaplan led pre-training efforts initially and advised the project.

Sam McCandlish led pre-training efforts and advised the project.

## Appendix A Model Size Multiplier and Poor Performance Region Fits

![Refer to caption](/html/2205.10487/assets/x17.png)

![Refer to caption](/html/2205.10487/assets/x18.png)

Figure 13: We see a power laws provide good fits for both language and Python data. We can use these fit to re-parameterize loss for our models trained on repeated data into model size multipliers.

In order to fit the poor performance regions we first fit power laws to our control scans on language and Python so that we can re-parameterize loss in terms of model size multipliers. These fits are shown in Figure [13](#A1.F13 "Figure 13 ‣ Appendix A Model Size Multiplier and Poor Performance Region Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data")

When we graph repeated epochs vs model size multiplier with a given fraction of repeated data in Figure [15](#A1.F15 "Figure 15 ‣ Appendix A Model Size Multiplier and Poor Performance Region Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data"), we observed that our 1% repeated data graphs were quite noisy, so we excluded the 1% scans from the fits. The 3% repeated data graphs looked reasonable, in that the double descent peak looked large compared to the noise, so we included that all higher fractions in our fits.

We estimate how many repeated epochs half of the maximum effect size (on a log scale) would be observed using linear interpolation on the left and right side of the double descent peak for each fraction of repeated data. We then averaged these curves to make an overall estimate for the left and right boundaries of the poor performance region shown in Figure [4](#S2.F4 "Figure 4 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") and Figure [11](#S2.F11 "Figure 11 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data"). For text this produces a relatively clean overall fit, but the the individual curves for text are relatively noisy as shown in [14](#A1.F14 "Figure 14 ‣ Appendix A Model Size Multiplier and Poor Performance Region Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data"). Some potential explanations for the noise are i) Given the resolution of our scan we do not always get a good estimate of the peak effect for a given curve (the peak can easily be between two points we measured ii) our linear interpolation also introduces error as our underlying curves only have 6 points.

![Refer to caption](/html/2205.10487/assets/x19.png)

![Refer to caption](/html/2205.10487/assets/x20.png)

Figure 14: We estimate how many repeated epochs half of the maximum effect size (on a log scale) would be observed using linear interpolation on the left and right side of the double descent peak for each fraction of repeated data. We then averaged these curves to make an overall estimate for the left and right boundaries of the poor performance region shown in Figure [4](#S2.F4 "Figure 4 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data")

Overall we think the region of poor performances we showed in Figure [4](#S2.F4 "Figure 4 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") is relatively robust in that it is useful to think about the sub distribution double descent phenomena there. However, we would not claim that we have produced extremely accurate estimates for the exact boundaries, even in our setup, and the boundaries could vary meaningfully given a different setup, especially differences in regularization.

For Python, the aggregate shown in Figure [11](#S2.F11 "Figure 11 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") is quite a bit noisier. A lot of the noise is explained by only aggregating two scans rather than 5. But we see the individual scans for Python are also noiser as shown in Figure [16](#A1.F16 "Figure 16 ‣ Appendix A Model Size Multiplier and Poor Performance Region Fits ‣ Scaling Laws and Interpretability of Learning from Repeated Data")

![Refer to caption](/html/2205.10487/assets/x21.png)

![Refer to caption](/html/2205.10487/assets/x22.png)

![Refer to caption](/html/2205.10487/assets/x23.png)

![Refer to caption](/html/2205.10487/assets/x24.png)

Figure 15: it is easier to see the sharpness of the double descent peaks in this diagram than Figure [2](#S1.F2 "Figure 2 ‣ 1.1 Summary of Results ‣ 1 Introduction ‣ Scaling Laws and Interpretability of Learning from Repeated Data"). The 1% runs was much noisier than the rest so we excluded it from our fits

.



![Refer to caption](/html/2205.10487/assets/x25.png)

![Refer to caption](/html/2205.10487/assets/x26.png)

Figure 16: We estimate how many repeated epochs half would cause half of the maximum effect size (on a log scale) for our Python models using linear interpolation on the left and right side of the double descent peak for each fraction of repeated data. We then averaged these two curves to make an overall estimate for the left and right boundaries of the poor performance region shown in Figure [11](#S2.F11 "Figure 11 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data")

## Appendix B Appendix: Logit Attribution Analysis, 2 Layer Models

For attention only models we can directly attribute contributions of the attention heads to the logits. We attempted to use this technique to better understand how the induction heads were disrupted for 2 layer models. For instance, it could be they were firing more weakly, or it could be activity from other attention heads were interfering with their ability to copy.

![Refer to caption](/html/2205.10487/assets/figures/logit_red.png)


Figure 17: For attention only models we can directly attribute contributions of the attention heads to the logits as shown in [[Elhage et al., 2021](#bib.bibx9), [Olsson et al., 2022](#bib.bibx26)]. Both models were evaluated on the first paragraph of Harry Potter copied twice. The induction head appeared to be head 0, shown in red for both models. The control model’s logit attribution is shown for the first two paragraph, and the third paragraph shown is from the repeated model at the double descent peak for comparison.

![Refer to caption](/html/2205.10487/assets/figures/logit_all.png)


Figure 18: For attention only models we can directly attribute contributions of the attention heads to the logits as shown in [[Elhage et al., 2021](#bib.bibx9), [Olsson et al., 2022](#bib.bibx26)]. Similar to Figure [17](#A2.F17 "Figure 17 ‣ Appendix B Appendix: Logit Attribution Analysis, 2 Layer Models ‣ Scaling Laws and Interpretability of Learning from Repeated Data") both models were evaluated on the first paragraph of Harry Potter copied twice, but here the contribution of all attention heads is shown. The other attention heads in the repeated data model appears more active (several of the reddish tokens in the second paragraph are brown in the third paragraph).

Overall it feels like both effects happen weakly, and that it was easier to understand the disruption to induction heads through the per token losses shown in Figures [8](#S2.F8 "Figure 8 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") and [9](#S2.F9 "Figure 9 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data"), than through logit attribution.

![Refer to caption](/html/2205.10487/assets/x27.png)


Figure 19: We still observe double descent on repeated data with 1 layer attention only models, so it is possible we’d observe double descent on repeated data for simpler model types.

## Appendix C Appendix: Copying and Prefix Matching Score Fits

![Refer to caption](/html/2205.10487/assets/x28.png)

![Refer to caption](/html/2205.10487/assets/x29.png)

Figure 20: In order to do the model size interpolation used in Figure [5](#S2.F5 "Figure 5 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") we use the loss on Harry Potter’s first paragraph copied 11 times for our control models (no repeated data). it is relatively well behaved, but it was not obvious how to extrapolate the curve. On the right, as a sanity check, we check to make sure we still see peaks moving left as model size increases that approximately line up with what was observed in [2](#S1.F2 "Figure 2 ‣ 1.1 Summary of Results ‣ 1 Introduction ‣ Scaling Laws and Interpretability of Learning from Repeated Data")



![Refer to caption](/html/2205.10487/assets/x30.png)

![Refer to caption](/html/2205.10487/assets/x31.png)

Figure 21: For the model size multiplier in Figure [6](#S2.F6 "Figure 6 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") we a linear fit on the prefix matching score for the control models shown on the left. On the right, similar to [10](#S2.F10 "Figure 10 ‣ 2 Results ‣ Scaling Laws and Interpretability of Learning from Repeated Data") we show that if we take an average over model size (harmonic mean of multiplier), we get a relatively clean relationship.

## Appendix D Appendix: Harry Potter Copying Evaluation with Fewer Characters

![Refer to caption](/html/2205.10487/assets/x32.png)


Figure 22: In order to make sure the copying eval was not merely evaluating in context learning, we tried a much shorter copied sequence (approximately 10x shorter, 125 characters instead of 1463). We still observe approximately no learning from repeated copying for the 2L model trained on 50% repeated data at the double descent peak

## References

* [Advani and Saxe, 2017]

  Advani, M. S. and Saxe, A. M. (2017).
  High-dimensional dynamics of generalization error in neural networks.
* [Amodei et al., 2018]

  Amodei, D., Hernandez, D., Sastry, G., Clark, J., Brockman, G., and Sutskever,
  I. (2018).
  Ai and compute.
  Heruntergeladen von https://blog. openai. com/aiand-compute.
* [Askell et al., 2021]

  Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., Jones, A.,
  Joseph, N., Mann, B., DasSarma, N., Elhage, N., Hatfield-Dodds, Z.,
  Hernandez, D., Kernion, J., Ndousse, K., Olsson, C., Amodei, D., Brown, T.,
  Clark, J., McCandlish, S., Olah, C., and Kaplan, J. (2021).
  A general language assistant as a laboratory for alignment.
* [Belkin et al., 2018]

  Belkin, M., Hsu, D., Ma, S., and Mandal, S. (2018).
  Reconciling modern machine learning practice and the bias-variance
  trade-off.
* [Bi et al., 2020]

  Bi, B., Li, C., Wu, C., Yan, M., Wang, W., Huang, S., Huang, F., and Si, L.
  (2020).
  Palm: Pre-training an autoencoding and autoregressive language model
  for context-conditioned generation.
* [Brown et al., 2020]

  Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P.,
  Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S.,
  Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler,
  D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray,
  S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever,
  I., and Amodei, D. (2020).
  Language models are few-shot learners.
* [Cammarata et al., 2020]

  Cammarata, N., Carter, S., Goh, G., Olah, C., Petrov, M., Schubert, L., Voss,
  C., Egan, B., and Lim, S. K. (2020).
  Thread: Circuits.
  Distill.
  https://distill.pub/2020/circuits.
* [Droppo and Elibol, 2021]

  Droppo, J. and Elibol, O. (2021).
  Scaling laws for acoustic models.
* [Elhage et al., 2021]

  Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell,
  A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D.,
  Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L.,
  Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S.,
  and Olah, C. (2021).
  A mathematical framework for transformer circuits.
  Transformer Circuits Thread.
  https://transformer-circuits.pub/2021/framework/index.html.
* [Ganguli et al., 2022]

  Ganguli, D., Hernandez, D., Lovitt, L., DasSarma, N., Henighan, T., Jones, A.,
  Joseph, N., Kernion, J., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly,
  T., Drain, D., Elhage, N., Showk, S. E., Fort, S., Hatfield-Dodds, Z.,
  Johnston, S., Kravec, S., Nanda, N., Ndousse, K., Olsson, C., Amodei, D.,
  Amodei, D., Brown, T., Kaplan, J., McCandlish, S., Olah, C., and Clark, J.
  (2022).
  Predictability and surprise in large generative models.
* [Gao et al., 2021]

  Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang,
  J., He, H., Thite, A., Nabeshima, N., Presser, S., and Leahy, C. (2021).
  The pile: An 800gb dataset of diverse text for language modeling.
* [Geiger et al., 2019]

  Geiger, M., Spigler, S., d’Ascoli, S., Sagun, L., Baity-Jesi, M., Biroli, G.,
  and Wyart, M. (2019).
  Jamming transition as a paradigm to understand the loss landscape of
  deep neural networks.
  Physical Review E, 100(1):012115.
* [Goh et al., 2021]

  Goh, G., Nick, C., Chelsea, V., Carter, S., Petrov, M., Schubert, L., Radford,
  A., and Olah, C. (2021).
  Multimodal neurons in artificial neural networks.
  Distill.
  https://distill.pub/2021/multimodal-neurons.
* [Henighan et al., 2020]

  Henighan, T., Kaplan, J., Katz, M., Chen, M., Hesse, C., Jackson, J., Jun, H.,
  Brown, T. B., Dhariwal, P., Gray, S., Hallacy, C., Mann, B., Radford, A.,
  Ramesh, A., Ryder, N., Ziegler, D. M., Schulman, J., Amodei, D., and
  McCandlish, S. (2020).
  Scaling laws for autoregressive generative modeling.
* [Hernandez and Brown, 2020]

  Hernandez, D. and Brown, T. B. (2020).
  Measuring the algorithmic efficiency of neural networks.
  CoRR, abs/2005.04305.
* [Hernandez et al., 2021]

  Hernandez, D., Kaplan, J., Henighan, T., and McCandlish, S. (2021).
  Scaling laws for transfer.
  arXiv preprint arXiv:2102.01293.
* [Hestness et al., 2017]

  Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H.,
  Patwary, M. M. A., Yang, Y., and Zhou, Y. (2017).
  Deep learning scaling is predictable, empirically.
* [Hoffmann et al., 2022]

  Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford,
  E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T.,
  Noland, E., Millican, K., Driessche, G. v. d., Damoc, B., Guy, A., Osindero,
  S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., and Sifre, L. (2022).
  Training compute-optimal large language models.
* [Jozefowicz et al., 2016]

  Jozefowicz, R., Vinyals, O., Schuster, M., Shazeer, N., and Wu, Y. (2016).
  Exploring the limits of language modeling.
* [Kaplan et al., 2020]

  Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R.,
  Gray, S., Radford, A., Wu, J., and Amodei, D. (2020).
  Scaling laws for neural language models.
* [Kiros et al., 2015]

  Kiros, R., Zhu, Y., Salakhutdinov, R., Zemel, R. S., Torralba, A., Urtasun, R.,
  and Fidler, S. (2015).
  Skip-thought vectors.
* [Lee et al., 2021]

  Lee, K., Ippolito, D., Nystrom, A., Zhang, C., Eck, D., Callison-Burch, C., and
  Carlini, N. (2021).
  Deduplicating training data makes language models better.
  arXiv preprint arXiv:2107.06499.
* [Malzahn and Opper, 2001]

  Malzahn, D. and Opper, M. (2001).
  A variational approach to learning curves.
  In Dietterich, T., Becker, S., and Ghahramani, Z., editors, Advances in Neural Information Processing Systems, volume 14. MIT Press.
* [Merity et al., 2016]

  Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016).
  Pointer sentinel mixture models.
* [Nakkiran et al., 2019]

  Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., and Sutskever, I.
  (2019).
  Deep double descent: Where bigger models and more data hurt.
* [Olsson et al., 2022]

  Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T.,
  Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D.,
  Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J.,
  Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J.,
  McCandlish, S., and Olah, C. (2022).
  In-context learning and induction heads.
  Transformer Circuits Thread.
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html.
* [Opper, 1995]

  Opper, M. (1995).
  Statistical mechanics of learning: Generalization.
* [Ouyang et al., 2022]

  Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P.,
  Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton,
  F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike,
  J., and Lowe, R. (2022).
  Training language models to follow instructions with human feedback.
* [Prato et al., 2021]

  Prato, G., Guiroy, S., Caballero, E., Rish, I., and Chandar, S. (2021).
  Scaling laws for the few-shot adaptation of pre-trained image
  classifiers.
* [Radford et al., 2021]

  Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry,
  G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I.
  (2021).
  Learning transferable visual models from natural language
  supervision.
* [Radford et al., 2019]

  Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al.
  (2019).
  Language models are unsupervised multitask learners.
  OpenAI blog, 1(8):9.
* [Rae et al., 2021]

  Rae, J. W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., Song, F.,
  Aslanides, J., Henderson, S., Ring, R., Young, S., Rutherford, E., Hennigan,
  T., Menick, J., Cassirer, A., Powell, R., Driessche, G. v. d., Hendricks,
  L. A., Rauh, M., Huang, P.-S., Glaese, A., Welbl, J., Dathathri, S., Huang,
  S., Uesato, J., Mellor, J., Higgins, I., Creswell, A., McAleese, N., Wu, A.,
  Elsen, E., Jayakumar, S., Buchatskaya, E., Budden, D., Sutherland, E.,
  Simonyan, K., Paganini, M., Sifre, L., Martens, L., Li, X. L., Kuncoro, A.,
  Nematzadeh, A., Gribovskaya, E., Donato, D., Lazaridou, A., Mensch, A.,
  Lespiau, J.-B., Tsimpoukelli, M., Grigorev, N., Fritz, D., Sottiaux, T.,
  Pajarskas, M., Pohlen, T., Gong, Z., Toyama, D., d’Autume, C. d. M., Li, Y.,
  Terzi, T., Mikulik, V., Babuschkin, I., Clark, A., Casas, D. d. L., Guy, A.,
  Jones, C., Bradbury, J., Johnson, M., Hechtman, B., Weidinger, L., Gabriel,
  I., Isaac, W., Lockhart, E., Osindero, S., Rimell, L., Dyer, C., Vinyals, O.,
  Ayoub, K., Stanway, J., Bennett, L., Hassabis, D., Kavukcuoglu, K., and
  Irving, G. (2021).
  Scaling language models: Methods, analysis, and insights from
  training gopher.
* [Vaswani et al., 2017]

  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, L., and Polosukhin, I. (2017).
  Attention is all you need.
* [Wolf et al., 2019]

  Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P.,
  Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen,
  P., Ma, C., Jernite, Y., Plu, J., Xu, C., Scao, T. L., Gugger, S., Drame, M.,
  Lhoest, Q., and Rush, A. M. (2019).
  Huggingface’s transformers: State-of-the-art natural language
  processing.

[◄](/html/2205.10486)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2205.10487)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2205.10487)
[View original  
on arXiv](https://arxiv.org/abs/2205.10487)[►](/html/2205.10488)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Mar 11 15:17:18 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
