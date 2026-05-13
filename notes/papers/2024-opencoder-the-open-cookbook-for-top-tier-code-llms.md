---
aliases:
- OpenCoder
- opencoder
- RefineCode
- refinecode
arxiv: '2411.04905'
authors:
- Siming Huang
- Tianhao Cheng
- J.K. Liu
- Jiaran Hao
- Liuyihan Song
- Yang Xu
- J. Yang
- Jiaheng Liu
- Chenchen Zhang
- Linzheng Chai
- Ruifeng Yuan
- Zhaoxiang Zhang
- Jie Fu
- Qian Liu
- Ge Zhang
- Zili Wang
- Yuan Qi
- Yinghui Xu
- Wei Chu (INF-AI / M-A-P)
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2411.04905
  raw: '[[raw/papers/md/2024-opencoder]]'
  source: https://arxiv.org/abs/2411.04905
owner: blaz
read: false
slug: opencoder-the-open-cookbook-for-top-tier-code-llms
tags:
- type/paper
- status/draft
- source/primary
- confidential/public-source
- domain/llm
- domain/pretraining
- domain/data-mix
- domain/code
title: 'OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models'
type: note
updated: '2026-05-10'
year: 2024
---

# OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models

## Citation

- URL: https://arxiv.org/abs/2411.04905
- PDF: https://arxiv.org/pdf/2411.04905
- Authors: Siming Huang, Tianhao Cheng, J.K. Liu, Jiaran Hao, Liuyihan Song, Yang Xu, J. Yang, Jiaheng Liu, Chenchen Zhang, Linzheng Chai, Ruifeng Yuan, Zhaoxiang Zhang, Jie Fu, Qian Liu, Ge Zhang, Zili Wang, Yuan Qi, Yinghui Xu, Wei Chu (INF-AI / M-A-P)
- Year / venue: 2024-11 arXiv preprint (v1 2024-11-07; v3 2025-03-20)
- Raw PDF: [[raw/papers/pdf/2024-opencoder.pdf]]
- Raw HTML: [[raw/papers/md/2024-opencoder.html]]

## Short Summary

Open code LLM family (1.5B and 8B) released *with* its full training-data pipeline. The dataset half is **RefineCode**, a 960B-token, 607-language code pretraining corpus, plus a smaller annealing mix. The cookbook claim: published filters, dedup parameters, language-specific rules, and recipe — so the entire stack is reproducible. The paper's strongest empirical claim is that **file-level deduplication beats repository-level deduplication** for code pretraining quality, contradicting the StarCoder / The Stack v2 default.

## Methodology

### RefineCode composition (~960B tokens)

Raw code (~92%):
- GitHub through Nov 2023: 755B (78.4%)
- The Stack v2 (non-GitHub): 120B (12.5%)
- Jupyter notebooks: 11B (1.1%)

Code-related web text (~8%, recalled from web corpora via iterative FastText classifier):
- FineWeb (filtered): 55B (5.7%)
- Common Crawl (filtered): 13B (1.4%)
- SkyPile (filtered): 3B (0.3%)
- AutoMathText (filtered): 3B (0.3%)

### Raw-code pipeline (Section 2.1.1)

The raw-code branch of RefineCode runs five sequential modules: **preprocessing → deduplication → transformation → filtering → data sampling**. Order matters: dedup runs early to cut volume; transformation runs *before* filtering so files with fixable issues aren't dropped.

**1. Preprocessing.**
- Drop files larger than 8 MB (mostly non-text; expensive to handle).
- Restrict to file extensions listed in [github-linguist `languages.yml`](https://github.com/github-linguist/linguist/blob/main/lib/linguist/languages.yml); discard low-capacity / low-quality language types.
- 607 programming languages survive.

**2. Deduplication — file-level, applied early.** Justified by the empirical fact that "nearly 75% of files are completely duplicated" across GitHub (forks + copy-paste).
- *Exact dedup*: SHA256 over each file's bytes. Among files with the same hash, keep the one with the **highest star count and latest commit time**.
- *Fuzzy dedup*: split each file into 5-gram pieces; compute 2048 MinHash functions (Broder, 1997); group with LSH using 16 bands × 128 rows (Leskovec et al., 2014); apply the same star+commit tiebreak. Removes ~6% additional file volume.
- Note: the star-count + latest-commit tiebreaker is the implicit quality signal for dedup. It conflates popularity with quality.

**3. Transformation — fix-then-filter.** Two issues are pervasive enough that filtering them out wholesale would lose too much data, so they're rewritten in place:
- *Copyright removal*: ~15% of files start with a copyright comment ("Copyright Intel Corporation (C) 2014–2016", etc.). These are highly repetitive and irrelevant to coding tasks, so the leading copyright block is stripped.
- *PII reduction*: regex detection of passwords, emails, IPs, names → replaced with placeholders `<name>`, `<password>`, etc. Reduces privacy risk; preserves file structure.

**4. Filtering — three tiers, language-aware.** Built on RedPajama's framework, extending StarCoder's code-cleaning rules. Threshold-tuned per Appendix A.1; details and tuning process in Appendix A.2.
- Design principle (from Gunasekar et al., 2023): drop files that (a) have **poor self-containment**, (b) lack **logical structure**, (c) **deviate from standard formatting**.
- Tier 1 — **Natural-language rules**: file size, line count, generic text-shape metrics. Apply to all files.
- Tier 2 — **General code rules**: variable counts, average function length, common code shape metrics. Apply to all code files.
- Tier 3 — **Language-specific rules** for Python, C, C++, C#, Java, JavaScript, Go, HTML. Examples cited: frequency of `pass` statements in Python (likely-stub detector), use of `goto` in C.
- Threshold-setting principle: "remove harmful data as much as possible, while ensuring the overall distribution of the dataset is not significantly affected." Pragmatic, not theoretically grounded.

**5. Data sampling.** Volume rebalancing post-filter:
- Java 449 GB → 200 GB (downsampled because volume far exceeds peer languages).
- HTML 474 GB → 64 GB ("often contain a significant amount of non-informative structured content and lack substantial coding logic").
- Output of this branch: **~730B tokens** at the pretraining stage.
- The paper validates the cleaning by PCA-ing CodeBERT embeddings of RefineCode vs The Stack v2: RefineCode clusters tightly; The Stack v2 has many outliers, which on inspection are pure-text comments, hexadecimal blobs, and short code with no computational logic.

### Code-related web pipeline (Section 2.1.2)

A parallel branch recovers code-related text from general web corpora using an iterative FastText classifier with URL-level feedback. This is the part most easily missed reading the paper quickly — it is *not* a one-shot classifier.

**Inspiration.** DeepSeekMath did this for math; OpenCoder adapts it for code. Unlike math, no open fine-grained code corpus exists, so the seed has to be built.

**Seed.** 500,000 high-quality code-like documents are annotated from Common Crawl via the *Autonomous Data Selection* method (Zhang et al., 2024b).

**Pipeline (four components):**
1. **FastText training.** Tokenize with a BPE tokenizer first (keeps vocab small; lets FastText handle Chinese, where whitespace tokenization fails). Train FastText (Joulin et al., 2016) on the seed.
2. **Recall.** Run the classifier across Common Crawl to retrieve code-related pages.
3. **Code-related domain discovery.** Aggregate recalled pages by base URL (a "domain" is e.g. `stackoverflow.com`). A domain is marked code-related if **>10% of its pages** are flagged by the classifier.
4. **URL annotation.** Within code-related domains, manually annotate URL patterns. Example: everything under `stackoverflow.com/questions` is treated as programming Q&A. Pages matching the URL pattern that the FastText classifier *missed* are added back into the seed. Loop.

**Iterations.** After three iterations, the CC branch yields ~220 GB code-related web data. Diversity and recall improve each iteration as the seed grows.

**Extending the recipe.** The same classifier+URL pipeline is then run over FineWeb, SkyPile, and AutoMathText (web subset), producing **~330 GB** code-related web text in total. Additionally, a separate classifier filters GitHub itself (a small fraction of GitHub files are natural-language, e.g. README/docs) for code-related text, yielding ~178 GB more.

### Resulting RefineCode (Section 2.1.3)

After both branches: ~960B tokens, 607 languages. Composition table reproduced above. The paper validates the dataset's downstream value by training a 1.5B model up to 600B tokens on RefineCode vs The Stack v2's training subset; RefineCode trains more efficiently (Figure 1).

### Annealing mix (~100B tokens, Section 2.2)

Annealing bridges general pretraining and SFT (MiniCPM-style). Rapid LR decay; very high-quality data only.

- **Original distribution (RefineCode): 83.94B.** Holding 84% of annealing tokens at the pretraining distribution avoids catastrophic forgetting. The paper notes this ratio "might not be ideal" — chosen under compute budget.
- **Algorithmic corpus: 12.44B.** Sampled from RefineCode by keyword filter: contains "leetcode", "def solution", or "class solution". Rationale: algorithmic files have strong logic, minimal external dependencies, and look like the small self-contained tasks users hit at inference time.
- **High-quality code snippets (synthetic): 2.71B.** A strong LLM is given algorithmic-corpus seeds and prompted to synthesize self-contained functions plus test cases. Only examples that **pass the test cases** are kept. Extended across multiple languages. Inspiration: phi-1's CodeExercises.
- **Code textbooks (synthetic): 0.91B.** Qwen2-72B-Instruct rewrites entries from the [hqcode](https://huggingface.co/datasets/yuxiang630/hqcode) dataset (a multilingual GPT-4o-Mini-synthesized code dataset; each entry is task + function) into educational text snippets that explain the code from multiple angles.

### Post-training (SFT)

- Stage 1 (theoretical): RealUser-Instruct 0.7M; Large-scale Diverse-Instruct 2.3M; filtered Infinity-Instruct 1.0M.
- Stage 2 (practical): McEval-Instruct 36K; Evol-Instruct 111K; Educational-Instruct 110K; Package-Instruct 110K.

### Models

- OpenCoder-1.5B: 24 layers, hidden 2240, 14 attention heads, 14 KV heads, 4096-token context, SwiGLU, vocab 96,640.
- OpenCoder-8B: 32 layers, hidden 4096, 8 attention heads, follows Llama-3.1-8B architecture.

## Experiments

Headline benchmark numbers from the paper:

- HumanEval pass@1 (0-shot): 1.5B-Base 54.3, 8B-Base 66.5; 1.5B-Instruct 72.5, 8B-Instruct 83.5.
- MBPP (3-shot, 500q): 1.5B-Base 70.6, 8B-Base 79.9; 1.5B-Instruct 72.7, 8B-Instruct 79.1.
- BigCodeBench Completion (8B-Instruct): Full 40.3, Hard 16.9.
- LiveCodeBench Average (8B-Instruct): 23.2.

## Key Results

### File-level dedup beats repo-level (the central ablation)

On Python:
- File-level retains 32.74B tokens from 485.8M files (2.4%).
- Repo-level retains 99.47B tokens from 11M repos (7.5%).
- File-level wins on HumanEval and MBPP during pretraining despite ~3x less data.
- Applying file-level on top of repo-level deduplicates a further ~68B tokens (~68.4%).

The implication is that within-repo near-duplicates (forks, vendored copies, near-identical boilerplate) dominate the residual mass after repo-level dedup, and removing them helps quality.

### Web text helps even for code

~8% of pretraining tokens come from filtered general web corpora. Pure GitHub is not the recipe.

## Core Concepts

- Existing concepts: [[concepts/data-filtering-paradigms]] (file vs repo dedup belongs here).
- Concepts to extract (later, if reused):
  - "RefineCode pipeline as a reproducible code-data baseline"
  - "File-level vs repository-level deduplication for code"

## Relevance To Poolside

- **RefineCode is used as a held-out web-scraped corpus in the internal-code-ppl eval** ([[evals/internal-code-ppl]]). Until now the eval note referenced it without an upstream source — this paper note is the source.
- The file-level vs repo-level dedup result is directly relevant to any Poolside data-pipeline decisions on code. *We infer* (not in the paper): if Poolside dedup is currently repo-level, this is worth a controlled experiment on a Poolside slice before adopting; OpenCoder's claim is on web-scraped GitHub, distribution may differ on internal repos.
- The web-text inclusion (~8%, FastText-classified) is structurally similar to the FineWeb / DCLM recipe applied to code retrieval — useful precedent if Poolside considers a "code-relevant text" slice.

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- Run a Poolside-internal A/B on file-level vs repo-level dedup on a fixed code slice; measure on internal-code-ppl + an external held-out (e.g., RefineCode itself).
- Replicate their FastText "code-relevant web" classifier and check overlap with what we'd flag as code-relevant in our own web pool.
- Inspect their language-specific filter rules for Python/C/C++/Java/JS/Go/HTML; compare to whatever we currently apply.
- Look at the annealing mix design (algorithmic + synthesized + textbooks) as a template; see if any of it generalizes to a Poolside annealing recipe.
- Add `paper:` link to [[evals/internal-code-ppl]] pointing at this note.

## Related Notes

- Concepts: [[concepts/data-filtering-paradigms]]
- Evals: [[evals/internal-code-ppl]], [[evals/internal-code-ppl-depsort]]
- Peer data papers: [[notes/papers/2024-fineweb-decanting-the-web-for-the-finest-text-data-at-scale]], [[notes/papers/2024-dclm-datacomp-for-language-models]]

## Caveats

- File-level vs repo-level dedup ablation is shown on Python only in the paper's main table; generalization to other languages is asserted but not exhaustively shown.
- "Best by stars + latest commit" tiebreaker conflates popularity with quality. Star-rich files may bias toward tutorial / scaffolding code.
- Annealing mix totals (~100B tokens) include LLM-synthesized snippets — synthetic-data caveats apply (mode collapse, distribution mismatch); the paper does not deeply ablate the synthetic component.
- The 8B model follows Llama-3.1-8B architecture — architectural contributions are minimal; the contribution is the data pipeline.
- v3 (2025-03-20) is the current version; numbers above are from v3 HTML.
