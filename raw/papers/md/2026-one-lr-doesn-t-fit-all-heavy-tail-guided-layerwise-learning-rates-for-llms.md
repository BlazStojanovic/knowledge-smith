---
arxiv: '2605.22297'
authors:
- Di He
- Songjun Tu
- Keyu Wang
- Lu Yin
- Shiwei Liu
parser: ar5iv
retrieved: '2026-06-01'
source: paper
title: 'One LR Doesn''t Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs'
url: https://arxiv.org/abs/2605.22297
year: 2026
---

 [2605.22297] One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs



# Computer Science > Machine Learning

**arXiv:2605.22297** (cs)

[Submitted on 21 May 2026 ([v1](https://arxiv.org/abs/2605.22297v1)), last revised 27 May 2026 (this version, v3)]

# Title:One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs

Authors:[Di He](https://arxiv.org/search/cs?searchtype=author&query=He,+D), [Songjun Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu,+S), [Keyu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+K), [Lu Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin,+L), [Shiwei Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+S)

View a PDF of the paper titled One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs, by Di He and 4 other authors

[View PDF](/pdf/2605.22297)
[HTML (experimental)](https://arxiv.org/html/2605.22297v3)
> Abstract:Learning rate configuration is a fundamental aspect of modern deep learning. The prevailing practice of applying a uniform learning rate across all layers overlooks the structural heterogeneity of Transformers, potentially limiting their effectiveness as the backbone of Large Language Models (LLMs). In this paper, we introduce Layerwise Learning Rate (LLR), an adaptive scheme that assigns distinct learning rates to individual Transformer layers. Our method is grounded in Heavy-Tailed Self-Regularization (HT-SR) theory, which characterizes the empirical spectral density (ESD) of weight correlation matrices to quantify heavy-tailedness. Layers with weaker heavy-tailedness are assigned larger learning rates to accelerate training, while layers with stronger heavy-tailedness receive smaller learning rates. By tailoring learning rates in this manner, LLR promotes more balanced training across layers, leading to faster convergence and improved generalization. Extensive experiments across architectures ranging from LLaMA to GPT-nano, optimizers including AdamW and Muon, and model scales from 60M to 3B parameters with up to 100B training tokens demonstrate the effectiveness of LLR. LLR achieves up to 1.5x training speedup and consistently outperforms uniform-learning-rate baselines. In particular, it improves the average zero-shot accuracy of 1B models from 47.09% to 49.02%, and that of 3B models from 48.58% to 50.61%. A key advantage of LLR is its low tuning overhead: it can transfer nearly optimal learning-rate settings directly from the uniform baseline. Code is available at [this https URL](https://github.com/hed-ucas/Layer-wise-Learning-Rate).

|  |  |
| --- | --- |
| Subjects: | Machine Learning (cs.LG); Artificial Intelligence (cs.AI) |
| Cite as: | [arXiv:2605.22297](https://arxiv.org/abs/2605.22297) [cs.LG] |
|  | (or  [arXiv:2605.22297v3](https://arxiv.org/abs/2605.22297v3) [cs.LG] for this version) |
|  | <https://doi.org/10.48550/arXiv.2605.22297> Focus to learn more  arXiv-issued DOI via DataCite |

## Submission history

From: Di He [[view email](/show-email/ee046a70/2605.22297)]   
 **[[v1]](/abs/2605.22297v1)**
Thu, 21 May 2026 10:46:23 UTC (788 KB)  
**[[v2]](/abs/2605.22297v2)**
Tue, 26 May 2026 06:04:34 UTC (788 KB)  
**[v3]**
Wed, 27 May 2026 10:11:58 UTC (791 KB)

Full-text links:

## Access Paper:

View a PDF of the paper titled One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs, by Di He and 4 other authors

* [View PDF](/pdf/2605.22297)
* [HTML (experimental)](https://arxiv.org/html/2605.22297v3)
* [TeX Source](/src/2605.22297)

[view license](http://arxiv.org/licenses/nonexclusive-distrib/1.0/ "Rights to this article")

### Current browse context:

cs.LG

[< prev](/prevnext?id=2605.22297&function=prev&context=cs.LG "previous in cs.LG (accesskey p)")
  |   
[next >](/prevnext?id=2605.22297&function=next&context=cs.LG "next in cs.LG (accesskey n)")

[new](/list/cs.LG/new)
 | 
[recent](/list/cs.LG/recent)
 | [2026-05](/list/cs.LG/2026-05)

Change to browse by:

[cs](/abs/2605.22297?context=cs)  
[cs.AI](/abs/2605.22297?context=cs.AI)

### References & Citations

* [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2605.22297)
* [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2605.22297)
* [Semantic Scholar](https://api.semanticscholar.org/arXiv:2605.22297)

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

IArxiv recommender toggle

IArxiv Recommender
*([What is IArxiv?](https://iarxiv.org/about))*

* Author
* Venue
* Institution
* Topic


About arXivLabs

# arXivLabs: experimental projects with community collaborators

arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.

Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.

Have an idea for a project that will add value for arXiv's community? [**Learn more about arXivLabs**](https://info.arxiv.org/labs/index.html).

[Which authors of this paper are endorsers?](/auth/show-endorsers/2605.22297) |
[Disable MathJax](javascript:setMathjaxCookie()) ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))
