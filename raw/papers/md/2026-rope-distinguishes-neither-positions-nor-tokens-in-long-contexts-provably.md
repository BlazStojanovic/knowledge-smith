---
arxiv: '2605.15514'
authors:
- Yufeng Du
- Phillip Harris
- Minyang Tian
- Eliu A Huerta
- Srikanth Ronanki
- Subendhu Rongali
- Aram Galstyan
- Hao Peng
parser: ar5iv
retrieved: '2026-05-18'
source: paper
title: RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably
url: https://arxiv.org/abs/2605.15514
year: 2026
---

 [2605.15514] RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably



# Computer Science > Computation and Language

**arXiv:2605.15514** (cs)

[Submitted on 15 May 2026]

# Title:RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably

Authors:[Yufeng Du](https://arxiv.org/search/cs?searchtype=author&query=Du,+Y), [Phillip Harris](https://arxiv.org/search/cs?searchtype=author&query=Harris,+P), [Minyang Tian](https://arxiv.org/search/cs?searchtype=author&query=Tian,+M), [Eliu A Huerta](https://arxiv.org/search/cs?searchtype=author&query=Huerta,+E+A), [Srikanth Ronanki](https://arxiv.org/search/cs?searchtype=author&query=Ronanki,+S), [Subendhu Rongali](https://arxiv.org/search/cs?searchtype=author&query=Rongali,+S), [Aram Galstyan](https://arxiv.org/search/cs?searchtype=author&query=Galstyan,+A), [Hao Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng,+H)

View a PDF of the paper titled RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably, by Yufeng Du and 7 other authors

[View PDF](/pdf/2605.15514)
[HTML (experimental)](https://arxiv.org/html/2605.15514v1)
> Abstract:We identify intrinsic limitations of Rotary Positional Embeddings (RoPE) in Transformer-based long-context language models. Our theoretical analysis abstracts away from the specific content of the context and depends only on its length. We prove that as context length increases, RoPE-based attention becomes unpredictable and loses two properties that are central to its effectiveness. First, it loses its locality bias: RoPE is no more likely to favor nearer positions than substantially farther ones. Second, it loses consistency in token relevance: a key vector that receives a higher attention score than an alternative at one position may receive a lower score at another. In both cases, the probability of failure approaches 0.5, no better than random guessing. We further prove that the attention score can remain unchanged when a key token is moved to a different position, or even replaced by a different token, indicating a failure to distinguish positions or tokens. Adjusting the RoPE base trades off distinguishing positions against distinguishing tokens but cannot preserve both at the same time. Increasing the RoPE base hyperparameter, a common practice in today's long-context models, helps distinguish different tokens, but inevitably sacrifices the ability to distinguish positions. Our empirical analysis shows that multi-head, multi-layer architectures are insufficient to overcome these limitations. Our findings suggest that fundamentally new mechanisms for encoding position and token order may be needed in future Transformer long-context language models.

|  |  |
| --- | --- |
| Comments: | 35 pages, 11 figures, submitted to NeurIPS 2026 |
| Subjects: | Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as: | [arXiv:2605.15514](https://arxiv.org/abs/2605.15514) [cs.CL] |
|  | (or  [arXiv:2605.15514v1](https://arxiv.org/abs/2605.15514v1) [cs.CL] for this version) |
|  | <https://doi.org/10.48550/arXiv.2605.15514> Focus to learn more  arXiv-issued DOI via DataCite (pending registration) |

## Submission history

From: Yufeng Du [[view email](/show-email/75fa531c/2605.15514)]   
 **[v1]**
Fri, 15 May 2026 01:16:16 UTC (1,216 KB)

Full-text links:

## Access Paper:

View a PDF of the paper titled RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably, by Yufeng Du and 7 other authors

* [View PDF](/pdf/2605.15514)
* [HTML (experimental)](https://arxiv.org/html/2605.15514v1)
* [TeX Source](/src/2605.15514)

[view license](http://creativecommons.org/licenses/by-nc-sa/4.0/ "Rights to this article")

### Current browse context:

cs.CL

[< prev](/prevnext?id=2605.15514&function=prev&context=cs.CL "previous in cs.CL (accesskey p)")
  |   
[next >](/prevnext?id=2605.15514&function=next&context=cs.CL "next in cs.CL (accesskey n)")

[new](/list/cs.CL/new)
 | 
[recent](/list/cs.CL/recent)
 | [2026-05](/list/cs.CL/2026-05)

Change to browse by:

[cs](/abs/2605.15514?context=cs)  
[cs.AI](/abs/2605.15514?context=cs.AI)  
[cs.LG](/abs/2605.15514?context=cs.LG)

### References & Citations

* [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2605.15514)
* [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2605.15514)
* [Semantic Scholar](https://api.semanticscholar.org/arXiv:2605.15514)

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

[Which authors of this paper are endorsers?](/auth/show-endorsers/2605.15514) |
[Disable MathJax](javascript:setMathjaxCookie()) ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))
