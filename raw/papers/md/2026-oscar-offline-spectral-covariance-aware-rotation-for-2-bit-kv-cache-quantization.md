---
arxiv: '2605.17757'
authors:
- Zhongzhu Zhou
- Donglin Zhuang
- Jisen Li
- Ziyan Chen
- Shuaiwen Leon Song
- Ben Athiwaratkun
- Xiaoxia Wu
parser: ar5iv
retrieved: '2026-05-28'
source: paper
title: 'OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization'
url: https://arxiv.org/abs/2605.17757
year: 2026
---

 [2605.17757] OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization



# Computer Science > Machine Learning

**arXiv:2605.17757** (cs)

[Submitted on 18 May 2026]

# Title:OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization

Authors:[Zhongzhu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou,+Z), [Donglin Zhuang](https://arxiv.org/search/cs?searchtype=author&query=Zhuang,+D), [Jisen Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+J), [Ziyan Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+Z), [Shuaiwen Leon Song](https://arxiv.org/search/cs?searchtype=author&query=Song,+S+L), [Ben Athiwaratkun](https://arxiv.org/search/cs?searchtype=author&query=Athiwaratkun,+B), [Xiaoxia Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu,+X)

View a PDF of the paper titled OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization, by Zhongzhu Zhou and 6 other authors

[View PDF](/pdf/2605.17757)
[HTML (experimental)](https://arxiv.org/html/2605.17757v1)
> Abstract:INT2 KV-cache quantization is attractive for long-context LLM serving, but it remains difficult to make both accurate and deployable. Simple rotations such as Hadamard transforms reduce outliers, but still degrade at INT2 because they are not aligned with downstream attention. We propose OSCAR, an Ultra-low-bit KV Cache quantization method that estimates attention-aware covariance structures offline and uses them to derive fixed rotations and clipping thresholds for quantization. In this way, it aligns KV quantization with the covariance structures that attention actually consumes. More importantly, we not only provide theoretical justification but also develop a fully deployable OSCAR system with a custom INT2 attention kernel that remains compatible with paged KV-cache serving and fused kernel pipelines, enabling seamless integration into modern LLM serving frameworks such as SGLang and vLLM.
>   
> We evaluate our methods on recent reasoning models with reasoning traces of up to 32k tokens across 5 tasks. On Qwen3-4B-Thinking-2507 and Qwen3-8B, OSCAR reduces the BF16 accuracy gap to 3.78 and 1.42 points, respectively, while naive rotation INT2 collapses to nearly zero. We further scale OSCAR to Qwen3-32B and GLM-4.7 (358B params), where it remains effectively on par with BF16. On long context - RULER-NIAH up to 128K, OSCAR remains robust on both Qwen3 models, while naive rotation INT2 collapses. System-wise, OSCAR reduces KV-cache memory by approximately 8x, improves throughput by up to 7x at large batch sizes under the same memory budget, and accelerates batch-size-1 decoding by up to 3x over BF16 due to reduced memory bandwidth overhead.

|  |  |
| --- | --- |
| Comments: | 35 pages, 10 figures |
| Subjects: | Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Distributed, Parallel, and Cluster Computing (cs.DC); Performance (cs.PF) |
| Cite as: | [arXiv:2605.17757](https://arxiv.org/abs/2605.17757) [cs.LG] |
|  | (or  [arXiv:2605.17757v1](https://arxiv.org/abs/2605.17757v1) [cs.LG] for this version) |
|  | <https://doi.org/10.48550/arXiv.2605.17757> Focus to learn more  arXiv-issued DOI via DataCite (pending registration) |

## Submission history

From: Zhou Zhongzhu [[view email](/show-email/14d1eae8/2605.17757)]   
 **[v1]**
Mon, 18 May 2026 02:24:29 UTC (9,064 KB)

Full-text links:

## Access Paper:

View a PDF of the paper titled OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization, by Zhongzhu Zhou and 6 other authors

* [View PDF](/pdf/2605.17757)
* [HTML (experimental)](https://arxiv.org/html/2605.17757v1)
* [TeX Source](/src/2605.17757)

[view license](http://creativecommons.org/licenses/by/4.0/ "Rights to this article")

### Current browse context:

cs.LG

[< prev](/prevnext?id=2605.17757&function=prev&context=cs.LG "previous in cs.LG (accesskey p)")
  |   
[next >](/prevnext?id=2605.17757&function=next&context=cs.LG "next in cs.LG (accesskey n)")

[new](/list/cs.LG/new)
 | 
[recent](/list/cs.LG/recent)
 | [2026-05](/list/cs.LG/2026-05)

Change to browse by:

[cs](/abs/2605.17757?context=cs)  
[cs.AI](/abs/2605.17757?context=cs.AI)  
[cs.DC](/abs/2605.17757?context=cs.DC)  
[cs.PF](/abs/2605.17757?context=cs.PF)

### References & Citations

* [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2605.17757)
* [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2605.17757)
* [Semantic Scholar](https://api.semanticscholar.org/arXiv:2605.17757)

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

[Which authors of this paper are endorsers?](/auth/show-endorsers/2605.17757) |
[Disable MathJax](javascript:setMathjaxCookie()) ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))
