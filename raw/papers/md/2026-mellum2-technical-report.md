---
arxiv: '2605.31268'
authors:
- Marko Kojic
- Ivan Bondyrev
- Aral de Moor
- Joseph Shtok
- Petr Borovlev
- Kseniia Lysaniuk
- Madeeswaran Kannan
- Ivan Dolgov
- Nikita Pavlichenko
parser: ar5iv
retrieved: '2026-06-01'
source: paper
title: Mellum2 Technical Report
url: https://arxiv.org/abs/2605.31268
year: 2026
---

 [2605.31268] Mellum2 Technical Report



# Computer Science > Computation and Language

**arXiv:2605.31268** (cs)

[Submitted on 29 May 2026]

# Title:Mellum2 Technical Report

Authors:[Marko Kojic](https://arxiv.org/search/cs?searchtype=author&query=Kojic,+M), [Ivan Bondyrev](https://arxiv.org/search/cs?searchtype=author&query=Bondyrev,+I), [Aral de Moor](https://arxiv.org/search/cs?searchtype=author&query=de+Moor,+A), [Joseph Shtok](https://arxiv.org/search/cs?searchtype=author&query=Shtok,+J), [Petr Borovlev](https://arxiv.org/search/cs?searchtype=author&query=Borovlev,+P), [Kseniia Lysaniuk](https://arxiv.org/search/cs?searchtype=author&query=Lysaniuk,+K), [Madeeswaran Kannan](https://arxiv.org/search/cs?searchtype=author&query=Kannan,+M), [Ivan Dolgov](https://arxiv.org/search/cs?searchtype=author&query=Dolgov,+I), [Nikita Pavlichenko](https://arxiv.org/search/cs?searchtype=author&query=Pavlichenko,+N)

View a PDF of the paper titled Mellum2 Technical Report, by Marko Kojic and 8 other authors

[View PDF](/pdf/2605.31268)
[HTML (experimental)](https://arxiv.org/html/2605.31268v1)
> Abstract:We present Mellum 2, an open-weight 12B-parameter Mixture-of-Experts (MoE) language model with 2.5B active parameters per token. Mellum 2 is a general-purpose language model specialized in software engineering, spanning code generation and editing, debugging, multi-step reasoning, tool use and function calling, agentic coding, and conversational programming assistance, and it is the successor to the completion-focused 4B dense Mellum model. The architecture builds on the Mixture-of-Experts (64 experts, 8 active) and combines Grouped-Query Attention with 4 KV heads, Sliding Window Attention on three of every four layers, and a single Multi-Token Prediction head that doubles as both an auxiliary pre-training objective and a built-in draft model for speculative decoding; each choice was validated by ablation with inference efficiency on commodity GPUs as a design constraint. Pre-training spans approximately 10.6 trillion tokens through a three-phase curriculum that progressively shifts the mixture from diverse web data toward curated code and mathematical content, optimized with Muon under FP8 hybrid precision and a Warmup-Hold-Decay schedule with linear decay to zero. The pre-trained base is extended to a 128K context window via a layer-selective YaRN and then post-trained in two stages (supervised fine-tuning followed by RLVR), yielding two released variants: an Instruct model that answers directly and a Thinking model that emits an explicit reasoning trace before its final answer. Across code generation, math and reasoning, tool use, knowledge, and safety benchmarks, Mellum 2 is competitive with open-weight baselines in the 4B-14B range while running at the per-token compute of a 2.5B dense model. We release the base, instruct, and thinking checkpoints, together with this report on the architecture decisions, data pipeline, and training recipe behind them, under the Apache 2.0 license.

|  |  |
| --- | --- |
| Subjects: | Computation and Language (cs.CL) |
| Cite as: | [arXiv:2605.31268](https://arxiv.org/abs/2605.31268) [cs.CL] |
|  | (or  [arXiv:2605.31268v1](https://arxiv.org/abs/2605.31268v1) [cs.CL] for this version) |
|  | <https://doi.org/10.48550/arXiv.2605.31268> Focus to learn more  arXiv-issued DOI via DataCite (pending registration) |

## Submission history

From: Nikiita Pavlichenko [[view email](/show-email/4681a0df/2605.31268)]   
 **[v1]**
Fri, 29 May 2026 13:01:11 UTC (1,508 KB)

Full-text links:

## Access Paper:

View a PDF of the paper titled Mellum2 Technical Report, by Marko Kojic and 8 other authors

* [View PDF](/pdf/2605.31268)
* [HTML (experimental)](https://arxiv.org/html/2605.31268v1)
* [TeX Source](/src/2605.31268)

[view license](http://creativecommons.org/licenses/by/4.0/ "Rights to this article")

### Current browse context:

cs.CL

[< prev](/prevnext?id=2605.31268&function=prev&context=cs.CL "previous in cs.CL (accesskey p)")
  |   
[next >](/prevnext?id=2605.31268&function=next&context=cs.CL "next in cs.CL (accesskey n)")

[new](/list/cs.CL/new)
 | 
[recent](/list/cs.CL/recent)
 | [2026-05](/list/cs.CL/2026-05)

Change to browse by:

[cs](/abs/2605.31268?context=cs)

### References & Citations

* [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2605.31268)
* [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2605.31268)
* [Semantic Scholar](https://api.semanticscholar.org/arXiv:2605.31268)

export BibTeX citation
Loading...

## BibTeX formatted citation

×

loading...

Data provided by:

### Bookmark



Bibliographic Tools

# Bibliographic and Citation Tools

Bibliographic Explorer Toggle

Bibliographic Explorer *([What is the Explorer?](https://info.arxiv.org/labs/showcase.html#arxiv-bibliographic-explorer))*

Connected Papers Toggle

Connected Papers *([What is Connected Papers?](https://www.connectedpapers.com/about))*

Litmaps Toggle

Litmaps *([What is Litmaps?](https://www.litmaps.co/))*

scite.ai Toggle

scite Smart Citations *([What are Smart Citations?](https://www.scite.ai/))*

Code, Data, Media

# Code, Data and Media Associated with this Article

alphaXiv Toggle

alphaXiv *([What is alphaXiv?](https://alphaxiv.org/))*

Links to Code Toggle

CatalyzeX Code Finder for Papers *([What is CatalyzeX?](https://www.catalyzex.com))*

DagsHub Toggle

DagsHub *([What is DagsHub?](https://dagshub.com/))*

GotitPub Toggle

Gotit.pub *([What is GotitPub?](http://gotit.pub/faq))*

Huggingface Toggle

Hugging Face *([What is Huggingface?](https://huggingface.co/huggingface))*

ScienceCast Toggle

ScienceCast *([What is ScienceCast?](https://sciencecast.org/welcome))*

Demos

# Demos

Replicate Toggle

Replicate *([What is Replicate?](https://replicate.com/docs/arxiv/about))*

Spaces Toggle

Hugging Face Spaces *([What is Spaces?](https://huggingface.co/docs/hub/spaces))*

Spaces Toggle

TXYZ.AI *([What is TXYZ.AI?](https://txyz.ai))*

Related Papers

# Recommenders and Search Tools

Link to Influence Flower

Influence Flower *([What are Influence Flowers?](https://influencemap.cmlab.dev/))*

Core recommender toggle

CORE Recommender *([What is CORE?](https://core.ac.uk/services/recommender))*

* Author
* Venue
* Institution
* Topic


About arXivLabs

# arXivLabs: experimental projects with community collaborators

arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.

Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.

Have an idea for a project that will add value for arXiv's community? [**Learn more about arXivLabs**](https://info.arxiv.org/labs/index.html).

[Which authors of this paper are endorsers?](/auth/show-endorsers/2605.31268) |
[Disable MathJax](javascript:setMathjaxCookie()) ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))
