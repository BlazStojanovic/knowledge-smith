---
arxiv: '2605.26266'
authors:
- Tuna Tuncer
- Felix Becker
- Thomas Pfeil
parser: ar5iv
retrieved: '2026-05-28'
source: paper
title: 'Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in
  Video Diffusion'
url: https://arxiv.org/abs/2605.26266
year: 2026
---

 [2605.26266] Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion



# Computer Science > Machine Learning

**arXiv:2605.26266** (cs)

[Submitted on 25 May 2026]

# Title:Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion

Authors:[Tuna Tuncer](https://arxiv.org/search/cs?searchtype=author&query=Tuncer,+T), [Felix Becker](https://arxiv.org/search/cs?searchtype=author&query=Becker,+F), [Thomas Pfeil](https://arxiv.org/search/cs?searchtype=author&query=Pfeil,+T)

View a PDF of the paper titled Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion, by Tuna Tuncer and 2 other authors

[View PDF](/pdf/2605.26266)
[HTML (experimental)](https://arxiv.org/html/2605.26266v1)
> Abstract:Chunk-wise autoregressive video diffusion models rely on a KV cache of previously generated chunks to avoid redundant computation, but this cache quickly becomes a memory bottleneck as videos grow longer. Methods that quantize the KV cache to low bitwidths reduce memory pressure but degrade video quality. We show that a key driver of this degradation is a systematic bias in attention weights: due to the convexity of the exponential in softmax attention, quantization noise inflates the contribution of cached keys, a phenomenon we call the Jensen bias. This effect causes quantized keys to steal attention mass from the unquantized current chunk. We derive a per-attention-score correction that removes this bias in expectation, computed on the fly from the quantization step sizes of the cached keys and the query norm. Using a second-order Taylor approximation, the additional computational overhead is negligible, and no additional memory is needed alongside the cache. Evaluated on MAGI-1, SkyReels-V2, and HY-WorldPlay at INT2 quantization, our correction recovers most of the quality lost to aggressive quantization, reaching near-BF16 video quality, and can outperform INT4 quantization while using 50% less memory.

|  |  |
| --- | --- |
| Comments: | Variants of this manuscript were accepted to the ICML 2026 workshops SCALE and F2S |
| Subjects: | Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Graphics (cs.GR); Image and Video Processing (eess.IV) |
| Cite as: | [arXiv:2605.26266](https://arxiv.org/abs/2605.26266) [cs.LG] |
|  | (or  [arXiv:2605.26266v1](https://arxiv.org/abs/2605.26266v1) [cs.LG] for this version) |
|  | <https://doi.org/10.48550/arXiv.2605.26266> Focus to learn more  arXiv-issued DOI via DataCite (pending registration) |

## Submission history

From: Thomas Pfeil [[view email](/show-email/091b2ebe/2605.26266)]   
 **[v1]**
Mon, 25 May 2026 18:51:59 UTC (12,158 KB)

Full-text links:

## Access Paper:

View a PDF of the paper titled Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion, by Tuna Tuncer and 2 other authors

* [View PDF](/pdf/2605.26266)
* [HTML (experimental)](https://arxiv.org/html/2605.26266v1)
* [TeX Source](/src/2605.26266)

[view license](http://arxiv.org/licenses/nonexclusive-distrib/1.0/ "Rights to this article")

### Current browse context:

cs.LG

[< prev](/prevnext?id=2605.26266&function=prev&context=cs.LG "previous in cs.LG (accesskey p)")
  |   
[next >](/prevnext?id=2605.26266&function=next&context=cs.LG "next in cs.LG (accesskey n)")

[new](/list/cs.LG/new)
 | 
[recent](/list/cs.LG/recent)
 | [2026-05](/list/cs.LG/2026-05)

Change to browse by:

[cs](/abs/2605.26266?context=cs)  
[cs.AI](/abs/2605.26266?context=cs.AI)  
[cs.CV](/abs/2605.26266?context=cs.CV)  
[cs.GR](/abs/2605.26266?context=cs.GR)  
[eess](/abs/2605.26266?context=eess)  
[eess.IV](/abs/2605.26266?context=eess.IV)

### References & Citations

* [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2605.26266)
* [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2605.26266)
* [Semantic Scholar](https://api.semanticscholar.org/arXiv:2605.26266)

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

[Which authors of this paper are endorsers?](/auth/show-endorsers/2605.26266) |
[Disable MathJax](javascript:setMathjaxCookie()) ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))
