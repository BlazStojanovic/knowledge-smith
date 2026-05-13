---
tags: [type/index, status/stable]
created: 2026-04-22
updated: 2026-04-22
---

# Papers

Paper notes grounded in a single paper.

Use [[templates/paper-note]] for new paper notes.

See [[notes/_indexes/papers-readme]] for folder conventions.

## Naming

Paper note filenames use `year-slug.md`, for example `2023-scaling-data-constrained-language-models.md`.

## Notes

The Dataview index below is the source of truth for imported paper notes.

- [[notes/papers/2023-scaling-data-constrained-language-models]]

## Extracted From Repetition Memo

Source review: [[raw/reviews/2026-scaling-laws-data-repetition-review]]

- [[notes/papers/2022-training-compute-optimal-large-language-models]]
- [[notes/papers/2022-scaling-laws-and-interpretability-of-learning-from-repeated-data]]
- [[notes/papers/2022-galactica-a-large-language-model-for-science]]
- [[notes/papers/2023-to-repeat-or-not-to-repeat-insights-from-scaling-llm-under-token-crisis]]
- [[notes/papers/2023-scaling-data-constrained-language-models]]
- [[notes/papers/2025-improved-scaling-laws-in-linear-regression-via-data-reuse]]
- [[notes/papers/2025-larger-datasets-can-be-repeated-more-a-theoretical-analysis-of-multi-epoch-scaling-in-linear-regression]]

## Index

```dataview
TABLE WITHOUT ID
  file.link AS "Paper",
  year AS "Year",
  created AS "Created",
  updated AS "Updated",
  read AS "Read",
  join(filter(tags, (t) => startswith(t, "domain/")), ", ") AS "Domains"
FROM "notes/papers"
WHERE kind = "paper"
SORT year DESC, file.name ASC
```
