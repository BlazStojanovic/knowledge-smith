# Scaling laws for data repetition in LLMs across model sizes

Status: research review · date 2026‑04‑16 · author Feynman

This memo synthesizes what is actually known (and not known) about how training language models on *repeated data* behaves as you change model size. It is restricted to primary sources I directly inspected: Muennighoff et al. 2023/2025 (the canonical data‑constrained scaling law), Xue et al. 2023 (independent T5/MoE‑based multi‑epoch study), Hernandez et al. 2022 (Anthropic; repetition "double descent" and mechanistic link to induction heads), and Yan et al. 2026 / ICLR (a theoretical refinement that corrects the scale‑invariance assumption).

## TL;DR

- Muennighoff's data‑constrained scaling law is a clean modification of Chinchilla:
$$L(N,D)=\frac{A}{N'^{\alpha}}+\frac{B}{D'^{\beta}}+E$$
with effective tokens $D' = U_D + U_D\,R^{*}_{D}\,(1-e^{-R_D/R^{*}_{D}})$ and symmetric effective parameters $N'$. Fitted constants: $R^{*}_{D}\!\approx\!15.4$ (≈ 16 epochs), $R^{*}_{N}\!\approx\!5.3$ (≈ 6× compute‑optimal params).
- **Practical heuristic (well supported):** up to ≈ 4 epochs of full‑dataset repetition are almost as good as unique tokens; meaningful gains persist up to ≈ 16 epochs; beyond that, extra compute decays to zero value. This holds across the 2.8B / 4.2B / 8.7B parameter runs in the headline paper.
- **Where this breaks:** when small fractions of the dataset are repeated *inside* otherwise unique data, the damage is non‑monotone and scale‑dependent. Hernandez et al. find a double‑descent‑in‑epochs effect for 1.6M – 800M models: 10% repeated data can produce a ≈ 2× effective‑parameter loss; 90% repeats at the peak can cost ≈ 73× effective parameters. Full‑dataset repetition (Muennighoff) is much more benign than partial repetition (Hernandez).
- **Scale dependence of the damage itself:** Xue et al. (T5 base/large/XL) show *larger* models are more prone to multi‑epoch overfitting at *fixed compute*. Hernandez's work shows the double‑descent phenomenon is qualitatively present across all model sizes tested (1.57M–805M), with the location of the degradation region in epoch‑space shifting.
- **Theoretical correction (2025/2026):** Yan et al. prove, under both strongly‑convex linear regression and Zipf data, that the "effective reuse rate" $E(K,N)\!\approx\!K$ only while $K\!\ll\!\log N$ (or $\ll\!N^{b/(a-b)}$ for power‑law spectra), then saturates *at a value that grows with $N$*. Their own 0.3B‑param LLM run on DCLM fits $K(\lambda=0.75,N)\approx 0.80\log N+5.21$. Implication: the "~4 epochs are free" rule is *not* N‑invariant; at very large $N$, more epochs are effectively free.

The rest of this memo walks through the evidence.

---

## 1. The canonical empirical law: Muennighoff et al. 2023 (JMLR 26:53, 2025)

**Experimental footprint.** Transformer language models, GPT‑2 architecture, C4 and OSCAR pre‑training data, up to **8.7B parameters trained for up to 900B total tokens**, covering ≈ 400 training runs ranging from 10M to 9B parameters and up to ~1500 epochs. Main IsoFLOP sweeps at three FLOP budgets ($9.3\times 10^{20}, 2.1\times 10^{21}, 9.3\times 10^{21}$) each with 8 data budgets ({55,28,18,14,11,9,4,1.25}B for the 2.8B runs; {84,42,28,21,17,12,6,1.9}B for 4.2B; {178,88,58,44,35,25,13,4}B for 8.7B). Hyperparameters closely follow Chinchilla (AdamW, cosine schedule, no early stopping). Full architecture list in their Table 15. [(arXiv:2305.16264, Fig. 4; JMLR v26, §4)](https://arxiv.org/abs/2305.16264)

**The scaling law.** Starting from Chinchilla $L(N,D) = A/N^{\alpha} + B/D^{\beta} + E$, they replace $N$ and $D$ by "effective" counts that saturate exponentially under repetition:

$$
L(N,D,R_N,R_D)=\frac{A}{N'^{\alpha}}+\frac{B}{D'^{\beta}}+E,\qquad
D' = U_D + U_D R^{*}_{D}\bigl(1-e^{-R_D/R^{*}_{D}}\bigr),\qquad
N' = U_N + U_N R^{*}_{N}\bigl(1-e^{-R_N/R^{*}_{N}}\bigr).
$$

Here $U_D$ = unique tokens in the budget, $R_D$ = repetitions beyond one epoch (so epochs = $R_D+1$); $U_N$ = compute‑optimal parameter count for $U_D$; $R_N$ = "excess‑parameter multiplier". When $R_D=0$ or $R_N=0$, the formula collapses to Chinchilla.

**Small‑$R_D$ limit.** For $R_D\ll R^{*}_D$, $D'\!\approx\!U_D(1+R_D)=D$: repeated tokens are nearly as good as fresh ones. For $R_D\gg R^{*}_D$, $D'$ plateaus at $U_D(1+R^{*}_D)$: no amount of repetition can outperform $U_D+U_D R^{*}_D$ fresh tokens.

**Fitted "half‑lives" (Appendix A of the paper):**

| Parameter | Meaning | Fitted value |
|---|---|---|
| $R^{*}_{D}$ | epoch at which a repeated token is worth $1-1/e\approx 63\%$ of a fresh one; beyond it repetition value vanishes | **≈ 15.4** (≈ 16 epochs) |
| $R^{*}_{N}$ | compute‑optimal "repetition" multiplier for excess parameters | **≈ 5.3** |

Because $R^{*}_N < R^{*}_D$, excess parameters decay faster than repeated data. The compute‑optimal allocation in data‑constrained regimes therefore shifts *epochs* rather than *parameters* — a direct reversal of the Chinchilla recipe.

**The "≤ 4 epochs is nearly free" headline.** Quoting their Fig. 5 left caption: at fixed total FLOPs, an 8.7B model trained for 4 epochs on 44B unique tokens finishes with **only 0.5% higher validation loss** than the single‑epoch 178B‑unique‑token run. Validation‑loss differences are negligible out to 4 epochs across all three FLOP budgets. Downstream results echo this: the 19‑task average for the 8.7B model is 26.2 at 1 epoch, 26.3 at 2, 26.8 at 4, and only collapses at 14 epochs (22.4) and 44 epochs (18.3). [(JMLR Table 8)](http://jmlr.org/papers/v26/24-1000.html)

**Scale dependence within Muennighoff's own sweep.** Their three headline model sizes (2.8B, 4.2B, 8.7B) each show the same qualitative pattern: ≤4 epochs ≈ free, 14 epochs materially worse, 44 epochs catastrophic. The 4.2B C4 numbers: single‑epoch avg 22.1 vs. 14‑epoch 20.6 vs. 44‑epoch 15.2; the 2.8B OSCAR numbers: 19.4 → 16.8 → 12.7. The **absolute** drop from 1 to 44 epochs is similar in relative terms across scales, which is why the authors treat $R^{*}_D$ as roughly scale‑invariant *in the fitted regime* and fold any $N$‑dependence into $U_N, U_D$.

**Corollary — Galactica case study.** They replay the Galactica 120B / 450B / 4.25‑epoch decision (Taylor et al. 2022): their fit says a better model would have been **40B parameters trained for 1.35T tokens (12.75 epochs)** on the same 106B unique‑token budget. Empirical evidence from their own compute‑optimal run: a 6.3B model trained for 9.7 epochs on 25B unique tokens **beats an 8.7B model trained for 1 epoch on 178B** on their 19‑task average (25.9 vs 26.2) despite having 27% fewer parameters. [(Fig. 13, Table 8)](https://arxiv.org/abs/2305.16264)

**Where the fit stops working.** Their exponential‑decay form cannot express loss that *increases* as more parameters are added under fixed data (Fig. 11). In Appendix F they explore an alternative $\alpha$‑$\beta$‑decay formulation where the exponents themselves decay to zero; that refit gives $R^{*}_D\approx 26{,}530$, $R^{*}_N\approx 2{,}041$, but the return curves fit worse. They ultimately stay with the exponential decay and note that "excess parameters hurt" probably reflects missing regularization (e.g. dropout) rather than a true breakdown of the law.

## 2. The independent T5 replication: Xue et al. 2023 (NeurIPS)

Xue, Fu, Zhou, Zheng, You ran an independent study on **T5 1.1 encoder‑decoder** models with C4 and the UL2 objective across **Base (250M) / Large (783M) / XL (2.8B)** scales. [(arXiv:2305.13230)](https://arxiv.org/abs/2305.13230)

Key independent findings about how repetition damage scales with model size:

1. **Larger models overfit faster under repetition (at fixed compute).** T5‑XL trained on 2^27 unique tokens repeated 2^8 times performs *worse* than T5‑Large trained on 2^29 unique tokens repeated 2^6 times — i.e. with more compute spent on a bigger model with fewer unique tokens, you lose. Their Insight (2): *"Larger models are more susceptible to overfitting under token‑crisis conditions."*
2. **Parameters matter, FLOPs don't (much).** Using MoE and ParamShare they decouple parameter count from FLOPs: at fixed FLOPs, the 2× parameter MoE overfits fastest, the 0.5× parameter ParamShare is most robust (their Fig. 5a). At fixed parameters, varying FLOPs ~4× barely changes the multi‑epoch degradation curve (Fig. 5b). Insight (5): "*The number of parameters plays a crucial role in multi‑epoch degradation. The effect of FLOPs on this issue is negligible.*"
3. **Dataset quality does not save you.** Training on Wikipedia ($2^{27}$ tokens) instead of C4 yields essentially identical multi‑epoch degradation (Table 2). Contra Taylor et al.'s speculation for Galactica.
4. **Dropout is the single most effective mitigation.** Of {dropout, DropPath, label smoothing, weight decay}, only dropout reliably closes the single‑epoch/multi‑epoch gap. At XL scale, it requires careful tuning (Figure 9c shows residual late‑stage drop at XL even with dropout 0.1; sweeping dropout rates {0.1,...,0.5} via a cheaper MoE proxy they find ~0.2–0.3 optimal at XL). This is evidence that the recent convention of "no dropout in LLM pretraining" (GPT‑3, PaLM, LLaMA, Gopher, Chinchilla) is actively harmful in the repetition regime.
5. **Dataset size can largely cancel the damage.** Fixing repetitions at $2^{8}$ and increasing unique tokens from $2^{27}$ to $2^{29}$ eliminates the overfitting they observed at the smaller size (Fig. 4). This is the first empirical hint in the LLM literature that bigger datasets tolerate more repetition — formalized by Yan et al. in §4.
6. **Their final positive result:** after tuning dropout, they scale to 2.8B parameters using only 2^27 tokens (well below the Chinchilla‑optimal 16M for that data budget) and still observe monotonic improvement with parameter scaling — their Fig. 11.

## 3. The mechanistic view: Hernandez et al. 2022 (Anthropic)

Hernandez et al. study a complementary and arguably more worrying regime: **a small fraction of the data is repeated many times inside an otherwise unique dataset**. [(arXiv:2205.10487)](https://arxiv.org/abs/2205.10487)

**Setup.** Decoder‑only transformer models at eight sizes: **1.57M, 5.31M, 12.6M, 42.5M, 101M, 197M, 340M, 805M** parameters, trained on 100B tokens each, over a 3‑order‑of‑magnitude sweep of the repeated fraction (1%, 3%, 10%, 50%, 90%), and a wide sweep over repeat frequency.

**Key quantitative findings (Fig. 2–6, Fig. 15):**

- Repetition loss follows a **double‑descent‑in‑epochs curve**: performance is robust for few repeats (data just looks like slightly more training), recovers when repeats are extreme (pure memorization), but plunges in a middle region where the model has the capacity to memorize but still sees enough signal to trade off capacity against generalization.
- At 10% repeated tokens, the model performs **2× worse in effective parameters** than if that 10% had been fresh text — "*much more than if that 10% of the data had simply never been trained on*". At 3% repeated tokens, the hit is 1.15× in test loss but 1.47× in prefix‑matching (induction‑head) behaviour. At the 90% repetition peak, an 800M model behaves like an **11M‑parameter model** (73× effective reduction).
- Larger models tend to enter the bad region at **fewer repeats** than smaller models: the curves in Figure 2/15 for 1% repeated data show the 800M line peaking around 1500–3000 repeats, while the 42M line peaks near 5000–10000. This is qualitatively consistent with Xue's finding that scale increases overfitting propensity under repetition, just expressed per‑sequence‑seen rather than per‑epoch.
- Mechanistic signature: at the double‑descent peak, induction heads and in‑context‑learning ability are **disproportionately damaged**. A 1.5M parameter model at 50% repetition that complete‑pattern‑copies at an effective 2000‑parameter level still has normal‑ish test loss — a vivid dissociation between memorization and generalization.
- The damage is mostly reversible with fine‑tuning on unique data: an 800M model pretrained on 90% repeated data (73× effective‑parameter loss) recovers to 1.6× effective‑parameter loss after finetuning — still meaningful but not catastrophic.
- Projected to Chinchilla scale: they estimate a second epoch for today's ~200B‑parameter models is *still inside* the double‑descent poor‑performance region for Python data.

**Reconciliation with Muennighoff.** Muennighoff repeats the **entire** dataset and never sees this pathology up to 4 epochs; Hernandez up‑weights small fractions (at constant total data) and sees it sharply at 3–10% repetition. Muennighoff flags this explicitly in his Appendix Q limitations: his scaling law probably needs an extra parameter to handle the Hernandez regime.

## 4. The theoretical correction: Yan et al. 2026 (ICLR, arXiv:2511.13421)

["Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression"](https://arxiv.org/abs/2511.13421). This paper directly names Muennighoff's scale‑invariance assumption as a neglected factor and fixes it in two solvable settings.

**Object of interest.** Define the *effective reuse rate* $E(K,N)$ as the number $T'/N$ such that one‑pass SGD on $T'$ fresh examples matches the excess risk of $K$‑epoch SGD on $N$ fresh examples.

**Strongly‑convex linear regression (Theorem 4.2):**

$$
E(K,N)=\begin{cases} K(1+o(1)) & K=o(\log N)\\ \Theta(\log N) & K=\omega(\log N)\end{cases}
$$

**Zipf / power‑law spectrum (Theorem 5.2):**

$$
E(K,N)=\begin{cases} K(1+o(1)) & K=o(N^{b/(a-b)})\\ \Theta(N^{b/(a-b)}) & K=\omega(N^{b/(a-b)})\end{cases}
$$

**Logarithmic power‑law spectrum (Theorem 5.3):** saturates at $\Theta(\log^{b} N)$. In every case the saturation value *grows with $N$*. Simulations match these exponents; their Zipf fit gives $E(K,N)\!\approx\!0.245\,N^{0.279}$ at $N=5\cdot 10^4$–$10^7$ with $a=4.5,b=1$ (prediction $b/(a-b)=2/7=0.286$).

**LLM validation (§6.3).** They train a **0.3B‑parameter Qwen‑2.5‑style model on DCLM** with fresh data sizes $N\in\{0.2, 0.5, 0.8, 1.0, 2\}$B, each for 100 epochs, plus a 200B‑token single‑pass reference. With a constant LR schedule (to match theory), they extract saturation points $K(\lambda=0.75, N)$ and fit:

$$
K(\lambda=0.75, N)\approx 0.80\,\log N + 5.21\quad (r=0.97,\ N\text{ in billions of tokens}).
$$

For $N\!=\!0.2$B, this gives a saturation around ~3.9 epochs — consistent with Muennighoff's "~4 epochs nearly free" rule at small $N$. For $N\!=\!200$B it predicts saturation around **~9 epochs**, far above what Muennighoff reports, but outside his fit range. **Practical takeaway:** the more unique data you already have, the further you can push repetition before seeing diminishing returns.

**Status of this correction.** It is a published ICLR 2026 result (synthetic and LLM‑empirical), but the empirical validation is at only one scale (0.3B) and uses constant‑LR training to match theory. The learning‑rate schedule matters: Appendix C.3 shows the phenomenon persists under WSD schedules, but the slopes differ.

## 5. Side‑by‑side summary

| Study | Model size sweep | Repetition mode | Scaling claim | Effective‑data saturation |
|---|---|---|---|---|
| Muennighoff 2023/2025 | 10M–9B, headline 2.8/4.2/8.7B | full‑dataset epochs | extend Chinchilla with exponential decay in $R_D,R_N$ | $R^{*}_D\!\approx\!15.4$ (≈ 16 epochs), $R^{*}_N\!\approx\!5.3$ |
| Xue 2023 | T5 Base/Large/XL (250M / 783M / 2.8B) | full‑dataset epochs | larger $N$ (params) → faster degradation at fixed FLOPs; size of unique corpus is critical | dropout at ~0.2–0.3 required at XL to keep loss monotone |
| Hernandez 2022 | 8 sizes, 1.57M – 805M | partial repetition (1–90 % of tokens) | double descent in epochs; disproportionate harm at 3–10 % repeats; larger models enter bad region sooner | peak at ~10% repeated ≈ 2× effective‑param loss; 90% peak ≈ 73× |
| Yan 2026 (theory + 0.3B LLM) | linear regression; LLM at 0.3B | full‑dataset epochs | $E(K,N)\!\approx\!K$ only while $K\!\ll\!\log N$ (or $\!\ll\!N^{b/(a-b)}$); saturation rises with $N$ | $K_{\text{sat}}\approx 0.80\log N + 5.21$ (N in B of tokens, constant LR, 0.3B model) |

## 6. Where the studies agree vs. disagree

**Agreements.**
1. Up to ~4 epochs on the *full* unique dataset is essentially free at the model/data scales tested so far (all three empirical studies; also implicit in Galactica's 4.25 epochs).
2. Beyond ~16 epochs of full‑dataset repetition, additional compute adds negligible value to loss (Muennighoff; qualitatively replicated in Xue's 2^8‑epoch T5‑XL collapse).
3. Parameter count matters more than raw FLOPs for overfitting under repetition (Xue, Muennighoff Fig. 11).
4. Dropout / regularization can meaningfully shift the tradeoff. Muennighoff explicitly attributes "excess parameters hurt" in his Fig. 11 to missing regularization.

**Disagreements / tensions.**
1. **Is the repetition tolerance $R^{*}_{D}$ scale‑invariant?** Muennighoff treats $R^{*}_D$ as a data/hyperparameter‑dependent but roughly $N$‑invariant constant. Yan et al. demonstrate theoretically and show empirically at 0.3B that $R^{*}_D$ grows with $N$. This is not yet contradicted at 8.7B because Muennighoff's data budgets are comparatively small (≤ 178B unique), so the $N$‑dependence may be present but within the fitting error. Open experimental question.
2. **Is partial repetition qualitatively different from full‑dataset repetition?** Hernandez (partial) vs. Muennighoff (full) have very different loss‑shape predictions. The Hernandez double‑descent does not appear in Muennighoff's full‑dataset sweep. Both are real and live at different points in the (fraction‑repeated × times‑repeated) plane. No unified scaling law covers both.
3. **Should repetition scale with parameters or with data?** Muennighoff: "scale epochs faster than parameters" ($R^{*}_N<R^{*}_D$). Xue (independently, at T5 scale): "scaling parameters under repetition is actively harmful; keep parameters smaller." These agree in spirit but disagree on the quantitative rate.

## 7. Practical recipe (what I'd actually do)

From the combined evidence, when unique data is the bottleneck:

- If your data budget is fresh enough to reach ≤ 4 epochs of full‑dataset repetition, do that — essentially free.
- Past 4 epochs, reduce model size (stay under Chinchilla‑optimal) rather than adding parameters — both Muennighoff (Galactica case) and Xue confirm the direction.
- At any scale where repetition is meaningful, **turn dropout back on** (0.1 baseline, tune up to 0.2–0.3 at XL+). This is the one intervention that consistently pays off across studies.
- Do not up‑weight quality subsets more than 2–3×; Hernandez's 10% × 10‑repeat regime lives squarely in the double‑descent valley and can cost 2× in effective parameters.
- If you have >> 1T unique tokens and are considering more epochs, Yan et al.'s result suggests $K_{\text{sat}}$ is larger than 4 — 8–16 epochs may still be on the effective‑reuse plateau. This is extrapolation; don't do it blindly at frontier scale, but it is increasingly well‑supported.
- Complement repetition with code augmentation: Muennighoff §7 shows that filling up to 50% of a data budget with code gives essentially the same downstream performance on NLP tasks plus a large jump on state‑tracking tasks (bAbI).

## 8. Open questions

1. A unified scaling law covering both full‑epoch repetition (Muennighoff) and partial‑fraction repetition (Hernandez) with a correct scale dependence (Yan et al.). The place this would first pay off is data‑mixing recipes that up‑weight small, high‑quality subsets — the current literature disagrees on whether that is helpful or actively harmful.
2. Empirical validation of Yan et al.'s $K_{\text{sat}}(N)$ prediction at frontier scale (≥ 70B parameters and ≥ 1T unique tokens). The current evidence is one model at 0.3B.
3. How synthetic / rephrased / model‑generated data interacts with repetition. Current scaling laws assume i.i.d. fresh tokens; the data wall is actually crossed by mixes.
4. Whether the Hernandez induction‑head damage signature survives modern tokenizer/architecture changes (rotary position embeddings, SwiGLU, GQA). Their strongest interpretability results are at 1‑ and 2‑layer attention‑only models and need re‑validation on modern stacks.

## Sources

- Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel. *Scaling Data‑Constrained Language Models*. NeurIPS 2023 / JMLR 26:53 (v5 2025‑06‑28). arXiv:2305.16264. https://arxiv.org/abs/2305.16264 · https://jmlr.org/papers/v26/24-1000.html · code + models: https://github.com/huggingface/datablations
- Xue, Fu, Zhou, Zheng, You. *To Repeat or Not To Repeat: Insights from Scaling LLM under Token‑Crisis*. NeurIPS 2023. arXiv:2305.13230 (v2). https://arxiv.org/abs/2305.13230 · https://openreview.net/forum?id=Af5GvIj3T5
- Hernandez, Brown, Conerly, DasSarma, Drain, El‑Showk, Elhage, Hatfield‑Dodds, Henighan, Hume, Johnston, Kravec, Mann, Olah, Olsson, Amodei, Joseph, Kaplan, McCandlish. *Scaling Laws and Interpretability of Learning from Repeated Data*. Anthropic, 2022. arXiv:2205.10487. https://arxiv.org/abs/2205.10487 · https://www.anthropic.com/research/scaling-laws-and-interpretability-of-learning-from-repeated-data
- Yan, Wen, Li, Luo, Chen, Lyu. *Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi‑Epoch Scaling in Linear Regression*. ICLR 2026. arXiv:2511.13421. https://arxiv.org/abs/2511.13421 · https://openreview.net/forum?id=pnEOU4qumA
- Hoffmann et al. *Training Compute‑Optimal Large Language Models* (Chinchilla, for background). arXiv:2203.15556. https://arxiv.org/abs/2203.15556
- Taylor et al. *Galactica: A Large Language Model for Science* (the 120B/4.25‑epoch reference). arXiv:2211.09085. https://arxiv.org/abs/2211.09085
- Lin, Wu, Bartlett. *Improved Scaling Laws in Linear Regression via Data Reuse*. 2025. arXiv:2506.08415. https://arxiv.org/abs/2506.08415 (parallel theoretical line of work, not reviewed in depth here).
