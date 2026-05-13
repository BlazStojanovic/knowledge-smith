---
source: article
url: https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization
retrieved: 2026-04-23
title: "The Practitioner's Guide to the Maximal Update Parameterization"
author: Cerebras
publication: Cerebras Blog
date: 2024-09-19
license: check
note: "Joint publication with EleutherAI. The EleutherAI version at blog.eleuther.ai/mutransfer/ contains the same content."
---

# The Practitioner's Guide to the Maximal Update Parameterization (Cerebras)

Published 2024-09-19. Joint project between Cerebras and EleutherAI.

This is the Cerebras-hosted version of the same guide published on the EleutherAI blog. See [[raw/articles/2024-eleutherai-mutransfer-guide]] for the full extracted content.

Key additions from the Cerebras side:
- Integration with Cerebras Model Zoo and training infrastructure
- Cerebras-GPT family trained with μP as validation (arXiv [2304.03208](https://arxiv.org/abs/2304.03208))
- Training documentation with μP implementation at training-docs.cerebras.ai

## Cerebras-GPT scaling findings

From Cerebras-GPT (arXiv [2304.03208](https://arxiv.org/abs/2304.03208)):
- Optimal and critical batch sizes scale as power laws in dataset size D, independent of model size N
- Overtrained models (small N, large D) benefit from higher critical batch size and better data parallelism
- μP provides tighter scaling law fits than standard parameterization
