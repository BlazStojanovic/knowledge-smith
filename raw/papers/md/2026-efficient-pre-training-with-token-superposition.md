---
arxiv: '2605.06546'
authors:
- Bowen Peng
- Théo Gigant
- Jeffrey Quesnelle
parser: ar5iv
retrieved: '2026-05-15'
source: paper
title: Efficient Pre-Training with Token Superposition
url: https://arxiv.org/abs/2605.06546
year: 2026
---

 [2605.06546] Efficient Pre-Training with Token Superposition



# Computer Science > Computation and Language

**arXiv:2605.06546** (cs)

[Submitted on 7 May 2026]

# Title:Efficient Pre-Training with Token Superposition

Authors:[Bowen Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng,+B), [Théo Gigant](https://arxiv.org/search/cs?searchtype=author&query=Gigant,+T), [Jeffrey Quesnelle](https://arxiv.org/search/cs?searchtype=author&query=Quesnelle,+J)

View a PDF of the paper titled Efficient Pre-Training with Token Superposition, by Bowen Peng and 2 other authors

[View PDF](/pdf/2605.06546)
> Abstract:Pre-training of Large Language Models is often prohibitively expensive and inefficient at scale, requiring complex and invasive modifications in order to achieve high data throughput. In this work, we present Token-Superposition Training (TST), a simple drop-in method that significantly improves the data throughput per FLOPs during pre-training without modifying the parallelism, optimizer, tokenizer, data, or model architecture. TST is done in two phases: (i) A highly efficient superposition phase where we combine many contiguous tokens into one bag and train using a multi-hot cross-entropy (MCE) objective, and (ii) a recovery phase where we revert back to standard training. We extensively evaluate TST on the scale of 270M and 600M parameters and validate on 3B and a 10B A1B mixture of experts model, demonstrating that it is highly robust in different settings. Ultimately, TST consistently outperforms baseline loss and downstream evaluations, and under equal-loss settings, TST yields up to a 2.5x reduction in total pre-training time at the 10B A1B scale.

|  |  |
| --- | --- |
| Comments: | 25 pages, 11 figures, 28 tables |
| Subjects: | Computation and Language (cs.CL) |
| Cite as: | [arXiv:2605.06546](https://arxiv.org/abs/2605.06546) [cs.CL] |
|  | (or  [arXiv:2605.06546v1](https://arxiv.org/abs/2605.06546v1) [cs.CL] for this version) |
|  | <https://doi.org/10.48550/arXiv.2605.06546> Focus to learn more  arXiv-issued DOI via DataCite (pending registration) |

## Submission history

From: Bowen Peng [[view email](/show-email/cfb65252/2605.06546)]   
 **[v1]**
Thu, 7 May 2026 16:41:37 UTC (1,627 KB)

Full-text links:

## Access Paper:

View a PDF of the paper titled Efficient Pre-Training with Token Superposition, by Bowen Peng and 2 other authors

* [View PDF](/pdf/2605.06546)
* [TeX Source](/src/2605.06546)

[view license](http://creativecommons.org/licenses/by/4.0/ "Rights to this article")

### Current browse context:

cs.CL

[< prev](/prevnext?id=2605.06546&function=prev&context=cs.CL "previous in cs.CL (accesskey p)")
  |   
[next >](/prevnext?id=2605.06546&function=next&context=cs.CL "next in cs.CL (accesskey n)")

[new](/list/cs.CL/new)
 | 
[recent](/list/cs.CL/recent)
 | [2026-05](/list/cs.CL/2026-05)

Change to browse by:

[cs](/abs/2605.06546?context=cs)

### References & Citations

* [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2605.06546)
* [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2605.06546)
* [Semantic Scholar](https://api.semanticscholar.org/arXiv:2605.06546)

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

[Which authors of this paper are endorsers?](/auth/show-endorsers/2605.06546) |
[Disable MathJax](javascript:setMathjaxCookie()) ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))
