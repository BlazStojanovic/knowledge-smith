---
arxiv: '2605.25624'
authors:
- Bowen Wang
- Dunjie Lu
- Junli Wang
- Tianyi Bai
- Shixuan Liu
- Zhipeng Zhang
- Haiquan Wang
- Hao Hu
- Tianbao Xie
- Shuai Bai
- Dayiheng Liu
- Que Shen
- Junyang Lin
- Tao Yu
parser: ar5iv
retrieved: '2026-05-28'
source: paper
title: 'CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use
  Agents'
url: https://arxiv.org/abs/2605.25624
year: 2026
---

 [2605.25624] CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents



# Computer Science > Artificial Intelligence

**arXiv:2605.25624** (cs)

[Submitted on 25 May 2026]

# Title:CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents

Authors:[Bowen Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+B), [Dunjie Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu,+D), [Junli Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+J), [Tianyi Bai](https://arxiv.org/search/cs?searchtype=author&query=Bai,+T), [Shixuan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+S), [Zhipeng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+Z), [Haiquan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+H), [Hao Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu,+H), [Tianbao Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie,+T), [Shuai Bai](https://arxiv.org/search/cs?searchtype=author&query=Bai,+S), [Dayiheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+D), [Que Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen,+Q), [Junyang Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin,+J), [Tao Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu,+T)

View a PDF of the paper titled CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents, by Bowen Wang and 13 other authors

[View PDF](/pdf/2605.25624)
> Abstract:Reinforcement learning with verifiable rewards (RLVR) has driven breakthroughs in domains such as math, tool-use, and software engineering, yet its extension to computer-use agents (CUAs) has been bottlenecked by the scarcity of scalable training data with deterministic rewards. Constructing such data for CUAs requires consistent task instruction, executable environment, and verifiable reward. However, hand-curated benchmarks achieve high reward fidelity but cover few applications and LLM-as-judge-based datasets scale broadly but lack reliable verification. We present CUA-Gym, a scalable pipeline that co-generates task instructions, environment states, and reward functions. Concretely, a Generator agent constructs the initial and golden environment states, and a separate Discriminator agent writes the reward function from the task specification. An orchestrator agent drives the two through iterative rounds upon execution. Generated tuples then pass a final filter combining LLM majority voting and agent rollouts, ensuring quality beyond the per-task adversarial loop. To address the scarcity of training environments, we further synthesize CUA-Gym-Hub, a broad suite of high-fidelity mock web applications grounded in real-world software-use distributions, expanding the scale of CUA RLVR data by magnitude. Using this pipeline, we construct CUA-Gym, a dataset of 32,112 verified RLVR training tuples grounded in 110 environments. Trained with GSPO on CUA-Gym, our CUA-Gym-A3B and CUA-Gym-A17B achieve 62.1% and 72.6% on OSWorld-Verified, outperforming prior open-source CUAs at comparable scales, with performance scaling smoothly in both data volume and environment diversity. The same checkpoints also improve on the held-out WebArena benchmark, indicating transfer beyond the training environments. We will open-source the full synthesis pipeline, dataset, CUA-Gym-Hub environments, and models.

|  |  |
| --- | --- |
| Subjects: | Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as: | [arXiv:2605.25624](https://arxiv.org/abs/2605.25624) [cs.AI] |
|  | (or  [arXiv:2605.25624v1](https://arxiv.org/abs/2605.25624v1) [cs.AI] for this version) |
|  | <https://doi.org/10.48550/arXiv.2605.25624> Focus to learn more  arXiv-issued DOI via DataCite (pending registration) |

## Submission history

From: Bowen Wang [[view email](/show-email/a7ff8231/2605.25624)]   
 **[v1]**
Mon, 25 May 2026 09:28:03 UTC (12,122 KB)

Full-text links:

## Access Paper:

View a PDF of the paper titled CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents, by Bowen Wang and 13 other authors

* [View PDF](/pdf/2605.25624)
* [TeX Source](/src/2605.25624)

[view license](http://creativecommons.org/licenses/by/4.0/ "Rights to this article")

### Current browse context:

cs.AI

[< prev](/prevnext?id=2605.25624&function=prev&context=cs.AI "previous in cs.AI (accesskey p)")
  |   
[next >](/prevnext?id=2605.25624&function=next&context=cs.AI "next in cs.AI (accesskey n)")

[new](/list/cs.AI/new)
 | 
[recent](/list/cs.AI/recent)
 | [2026-05](/list/cs.AI/2026-05)

Change to browse by:

[cs](/abs/2605.25624?context=cs)  
[cs.LG](/abs/2605.25624?context=cs.LG)

### References & Citations

* [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2605.25624)
* [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2605.25624)
* [Semantic Scholar](https://api.semanticscholar.org/arXiv:2605.25624)

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

[Which authors of this paper are endorsers?](/auth/show-endorsers/2605.25624) |
[Disable MathJax](javascript:setMathjaxCookie()) ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))
