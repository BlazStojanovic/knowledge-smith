---
arxiv: '2605.08738'
authors:
- Shengkun Tang
- Zekun Wang
- Bo Zheng
- Liangyu Wang
- Rui Men
- Siqi Zhang
- Xiulong Yuan
- Zihan Qiu
- Zhiqiang Shen
- Dayiheng Liu
parser: ar5iv
retrieved: '2026-05-15'
source: paper
title: 'SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training'
url: https://arxiv.org/abs/2605.08738
year: 2026
---

 [2605.08738] SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training



# Computer Science > Machine Learning

**arXiv:2605.08738** (cs)

[Submitted on 9 May 2026]

# Title:SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training

Authors:[Shengkun Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang,+S), [Zekun Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+Z), [Bo Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng,+B), [Liangyu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+L), [Rui Men](https://arxiv.org/search/cs?searchtype=author&query=Men,+R), [Siqi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+S), [Xiulong Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan,+X), [Zihan Qiu](https://arxiv.org/search/cs?searchtype=author&query=Qiu,+Z), [Zhiqiang Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen,+Z), [Dayiheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+D)

View a PDF of the paper titled SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training, by Shengkun Tang and 9 other authors

[View PDF](/pdf/2605.08738)
[HTML (experimental)](https://arxiv.org/html/2605.08738v1)
> Abstract:Structured pruning and knowledge distillation (KD) are typical techniques for compressing large language models, but it remains unclear how they should be applied at pretraining scale, especially to recent mixture-of-experts (MoE) models. In this work, we systematically study MoE compression in large-scale pretraining, focusing on three key questions: whether pruning provides a better initialization than training from scratch, how expert compression choices affect the final model after continued training, and which training strategy is most effective. We have the following findings: First, across depth, width, and expert compression, pruning a pretrained MoE consistently outperforms training the target architecture from scratch under the same training budget. Second, different one-shot expert compression methods converge to similar final performance after large-scale continual pretraining. Motivated by this, we introduce a simple partial-preservation expert merging strategy that improves downstream performance across most benchmarks. Third, combining KD with the language modeling loss outperforms KD alone, particularly on knowledge-intensive tasks. We further propose multi-token prediction (MTP) distillation, which yields consistent gains. Finally, given the same training tokens, progressive pruning schedules outperform one-shot compression, suggesting that gradual architecture transitions lead to better optimization trajectories. Putting it all together, we compress Qwen3-Next-80A3B to a 23A2B model that retains competitive performance. These results offer practical guidance for efficient MoE compression at scale.

|  |  |
| --- | --- |
| Subjects: | Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as: | [arXiv:2605.08738](https://arxiv.org/abs/2605.08738) [cs.LG] |
|  | (or  [arXiv:2605.08738v1](https://arxiv.org/abs/2605.08738v1) [cs.LG] for this version) |
|  | <https://doi.org/10.48550/arXiv.2605.08738> Focus to learn more  arXiv-issued DOI via DataCite (pending registration) |

## Submission history

From: Shengkun Tang [[view email](/show-email/14856fe4/2605.08738)]   
 **[v1]**
Sat, 9 May 2026 06:50:35 UTC (575 KB)

Full-text links:

## Access Paper:

View a PDF of the paper titled SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training, by Shengkun Tang and 9 other authors

* [View PDF](/pdf/2605.08738)
* [HTML (experimental)](https://arxiv.org/html/2605.08738v1)
* [TeX Source](/src/2605.08738)

[view license](http://creativecommons.org/licenses/by/4.0/ "Rights to this article")

### Current browse context:

cs.LG

[< prev](/prevnext?id=2605.08738&function=prev&context=cs.LG "previous in cs.LG (accesskey p)")
  |   
[next >](/prevnext?id=2605.08738&function=next&context=cs.LG "next in cs.LG (accesskey n)")

[new](/list/cs.LG/new)
 | 
[recent](/list/cs.LG/recent)
 | [2026-05](/list/cs.LG/2026-05)

Change to browse by:

[cs](/abs/2605.08738?context=cs)  
[cs.AI](/abs/2605.08738?context=cs.AI)  
[cs.CL](/abs/2605.08738?context=cs.CL)

### References & Citations

* [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2605.08738)
* [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2605.08738)
* [Semantic Scholar](https://api.semanticscholar.org/arXiv:2605.08738)

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

[Which authors of this paper are endorsers?](/auth/show-endorsers/2605.08738) |
[Disable MathJax](javascript:setMathjaxCookie()) ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))
