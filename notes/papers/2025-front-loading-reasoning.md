---
authors:
- Syeda Nahida Akter
- Shrimai Prabhumoye
- Eric Nyberg
- Mostofa Patwary
- Mohammad Shoeybi
- Yejin Choi
- Bryan Catanzaro
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://research.nvidia.com/labs/adlr/files/Front_Loading_Reasoning_The_Synergy_between_Pretraining_and_Post_Training_Data.pdf
  raw: '[[raw/papers/md/2025-front-loading-reasoning]]'
  source: https://research.nvidia.com/labs/adlr/files/Front_Loading_Reasoning_The_Synergy_between_Pretraining_and_Post_Training_Data.pdf
owner: blaz
parser: read
raw_md: raw/papers/md/2025-front-loading-reasoning.md
raw_pdf: raw/papers/pdf/2025-front-loading-reasoning.pdf
read: false
slug: front-loading-reasoning
tags:
- type/paper
- status/stub
title: 'Front-Loading Reasoning: The Synergy between Pretraining and Post-Training
  Data'
type: note
updated: '2026-05-28'
venue: NVIDIA technical report
year: 2025
---

# Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data

> *Syeda Nahida Akter, Shrimai Prabhumoye, Eric Nyberg, Mostofa Patwary, Mohammad Shoeybi, Yejin Choi, Bryan Catanzaro* — NVIDIA / CMU / BU / Stanford, 2025

## TL;DR

(stub)

## Abstract

The prevailing paradigm for enhancing the reasoning abilities of Large Language Models (LLMs) revolves around post-training on high-quality, reasoning-intensive data. While emerging literature suggests that reasoning data is increasingly incorporated also during the mid-training stage, a practice that is relatively more proprietary and less openly characterized, the role of such data in pretraining remains unclear. In particular, due to the opaqueness of pretraining corpora in most frontier models, the effect of reasoning data introduced at different phases of pre- and/or post-training is relatively less reported in the scientific literature. This raises several important but unsettled questions: Is adding reasoning data earlier during pre-training any better than introducing it during post-training, when the token counts are controlled? Could earlier inclusion risk overfitting and harm generalization, or instead establish durable foundations that later fine-tuning cannot recover? To address these questions, we conduct the first systematic study of how reasoning data (varying in scale, diversity, and quality) affects LLM performance when introduced at different stages of training. Our findings reveal that front-loading reasoning data into pretraining is critical (19% average gain), establishing foundational capabilities that cannot be fully replicated by later-stage SFT, even with more data. We uncover an asymmetric principle for optimal data allocation: pretraining benefits most from broad diversity in reasoning patterns (11% average gain), while SFT is more sensitive to data quality (15% average gain with high quality data). Furthermore, we show that high-quality pretraining data has latent effects, activated only after SFT, and that naively scaling SFT data can be detrimental, washing away the benefits of early reasoning injection. Collectively, our results challenge the conventional separation of language modeling and reasoning, providing a principled guide for strategically allocating data across the entire training pipeline to build more capable models.

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2025-front-loading-reasoning]]
- PDF: [[raw/papers/pdf/2025-front-loading-reasoning.pdf]]
- URL: <https://research.nvidia.com/labs/adlr/files/Front_Loading_Reasoning_The_Synergy_between_Pretraining_and_Post_Training_Data.pdf>
