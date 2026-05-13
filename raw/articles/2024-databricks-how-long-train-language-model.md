---
source: article
url: https://www.databricks.com/blog/how-long-should-you-train-your-language-model
retrieved: 2026-04-23
title: "How Long Should You Train Your Language Model?"
author: Databricks Mosaic Research
publication: Databricks Blog
date: 2024
license: excerpt-only
note: "Restrictive license — excerpted key points only."
---

# How Long Should You Train Your Language Model?

Author: Databricks Mosaic Research team.

> [!warning] Excerpt only
> Key arguments summarized; read the original for full treatment.

## Key arguments

1. **Chinchilla-optimal is not always practically optimal.** When inference cost matters, training a smaller model for longer (over-training) can yield better cost-efficiency at deployment time.
2. **Over-training regime**: training beyond D = 20N is systematically studied in Gadre et al. 2024. Loss increases follow predictable power laws, so the cost of over-training can be budgeted.
3. **Practical heuristic**: choose model size based on inference budget, then train as long as your compute budget allows. This often means D >> 20N.
4. **Data quality interaction**: over-training amplifies the importance of data quality — low-quality data repeated many times degrades faster than high-quality data.

## Relevance

Bridges the gap between theoretical compute-optimality (Chinchilla) and practical deployment constraints. Useful context for [[concepts/compute-optimal-methodology]] and [[concepts/scaling-laws-foundational]].
