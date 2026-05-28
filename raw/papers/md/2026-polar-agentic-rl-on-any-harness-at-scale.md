---
arxiv: '2605.24220'
authors:
- Binfeng Xu
- Hao Zhang
- Shaokun Zhang
- Songyang Han
- Mingjie Liu
- Jian Hu
- Shizhe Diao
- Zhenghui Jin
- Yunheng Zou
- Michael Demoret
- Jan Kautz
- Yi Dong
parser: ar5iv
retrieved: '2026-05-28'
source: paper
title: 'Polar: Agentic RL on Any Harness at Scale'
url: https://arxiv.org/abs/2605.24220
year: 2026
---

 [2605.24220] Polar: Agentic RL on Any Harness at Scale



# Computer Science > Distributed, Parallel, and Cluster Computing

**arXiv:2605.24220** (cs)

[Submitted on 22 May 2026]

# Title:Polar: Agentic RL on Any Harness at Scale

Authors:[Binfeng Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu,+B), [Hao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+H), [Shaokun Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+S), [Songyang Han](https://arxiv.org/search/cs?searchtype=author&query=Han,+S), [Mingjie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+M), [Jian Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu,+J), [Shizhe Diao](https://arxiv.org/search/cs?searchtype=author&query=Diao,+S), [Zhenghui Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin,+Z), [Yunheng Zou](https://arxiv.org/search/cs?searchtype=author&query=Zou,+Y), [Michael Demoret](https://arxiv.org/search/cs?searchtype=author&query=Demoret,+M), [Jan Kautz](https://arxiv.org/search/cs?searchtype=author&query=Kautz,+J), [Yi Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong,+Y)

View a PDF of the paper titled Polar: Agentic RL on Any Harness at Scale, by Binfeng Xu and 11 other authors

[View PDF](/pdf/2605.24220)
[HTML (experimental)](https://arxiv.org/html/2605.24220v1)
> Abstract:Reinforcement learning for language agents increasingly depends on custom harnesses that manage long-running context, multi-turn tool use and multi-agent orchestration. However, porting these harnesses into RL environment interfaces remains difficult and often loses important training signals. We bridge this gap with polar, a rollout framework for scalable asynchronous RL over arbitrary agent harnesses. Polar treats the agent harness as a black box: it proxies LLM API calls, records token-level model interactions, and reconstructs token-faithful trajectories for training. Each rollout node efficiently manages runtime prewarming, agent execution, trajectory reconstruction, and evaluation in parallel, exposing asynchronous service endpoints that can be consumed by independent trainers at scale. This decoupled design makes Polar agnostic to agent harnesses, training infrastructure, and RL algorithms while improving compute utilization for long-running agent workloads. We validate polar by training agents on software-engineering tasks with popular coding harnesses. Using simple GRPO, polar improves Qwen3.5-4B by 22.6, 4.8, 0.6 and 6.2 points on SWE-Bench Verified with the Codex, Claude Code, Qwen Code and Pi harnesses, respectively. We further demonstrate Polar for offline data generation over custom harnesses and ablate trajectory reconstruction strategies. Polar rewrites its preceding work, Prorl Agent, and has been registered as one of NeMo Gym environments.

|  |  |
| --- | --- |
| Comments: | 17 pages, 6 figures. 2 tables |
| Subjects: | Distributed, Parallel, and Cluster Computing (cs.DC) |
| Cite as: | [arXiv:2605.24220](https://arxiv.org/abs/2605.24220) [cs.DC] |
|  | (or  [arXiv:2605.24220v1](https://arxiv.org/abs/2605.24220v1) [cs.DC] for this version) |
|  | <https://doi.org/10.48550/arXiv.2605.24220> Focus to learn more  arXiv-issued DOI via DataCite (pending registration) |

## Submission history

From: Shaokun Zhang [[view email](/show-email/69415f92/2605.24220)]   
 **[v1]**
Fri, 22 May 2026 21:06:12 UTC (3,203 KB)

Full-text links:

## Access Paper:

View a PDF of the paper titled Polar: Agentic RL on Any Harness at Scale, by Binfeng Xu and 11 other authors

* [View PDF](/pdf/2605.24220)
* [HTML (experimental)](https://arxiv.org/html/2605.24220v1)
* [TeX Source](/src/2605.24220)

[view license](http://arxiv.org/licenses/nonexclusive-distrib/1.0/ "Rights to this article")

### Current browse context:

cs.DC

[< prev](/prevnext?id=2605.24220&function=prev&context=cs.DC "previous in cs.DC (accesskey p)")
  |   
[next >](/prevnext?id=2605.24220&function=next&context=cs.DC "next in cs.DC (accesskey n)")

[new](/list/cs.DC/new)
 | 
[recent](/list/cs.DC/recent)
 | [2026-05](/list/cs.DC/2026-05)

Change to browse by:

[cs](/abs/2605.24220?context=cs)

### References & Citations

* [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2605.24220)
* [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2605.24220)
* [Semantic Scholar](https://api.semanticscholar.org/arXiv:2605.24220)

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

[Which authors of this paper are endorsers?](/auth/show-endorsers/2605.24220) |
[Disable MathJax](javascript:setMathjaxCookie()) ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))
