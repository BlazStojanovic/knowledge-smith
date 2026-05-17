---
arxiv: '2412.02595'
authors:
- Dan Su
- Kezhi Kong
- Ying Lin
- Joseph Jennings
- Brandon Norick
- Markus Kliegl
- et al. (9 authors)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2412.02595
  raw: '[[raw/papers/md/2024-nemotron-cc-transforming-common-crawl-into-a-refined-long-horizon-pretraining-dataset]]'
  source: https://arxiv.org/abs/2412.02595
owner: blaz
raw_pdf: raw/papers/pdf/2024-nemotron-cc-transforming-common-crawl-into-a-refined-long-horizon-pretraining-dataset.pdf
read: false
slug: nemotron-cc-transforming-common-crawl-into-a-refined-long-horizon-pretraining-dataset
tags:
- type/paper
- status/draft
- source/primary
- confidential/public-source
- domain/llm
- domain/synth-data
- domain/pretraining
- domain/data-mix
- domain/evals
- domain/models
title: 'Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining
  Dataset'
type: note
updated: '2026-05-15'
year: 2024
---

# Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset

## Citation

- URL: https://arxiv.org/abs/2412.02595
- PDF: https://arxiv.org/pdf/2412.02595
- Authors: Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, et al. (9 authors)
- Year / venue: 2024-12-03 arXiv preprint
- arXiv: 2412.02595v2
- Categories: cs.CL
- Raw PDF: [[raw/papers/pdf/2024-nemotron-cc-transforming-common-crawl-into-a-refined-long-horizon-pretraining-dataset.pdf]]
- Source filename: `2412.02595v2.pdf`

## Short Summary

Recent English Common Crawl datasets like FineWeb-Edu and DCLM achieved significant benchmark gains via aggressive model-based filtering, but at the cost of removing 90% of data. This limits their suitability for long token horizon training, such as 15T tokens for Llama 3.1.

## Synthetic data generation (rephrasing prompts)

Nemotron-CC mints ≈1.9T synthetic tokens to complement 4.4T globally deduplicated
real tokens. Synthetic generation is applied per quality tier (Figure 3):

- **High-quality data** → all five rephrasing prompts (Diverse QA, Distill,
  Knowledge list, Extract knowledge, Wikipedia-style rephrasing) — produce diverse
  fresh unique tokens.
- **Low-quality data** → Wikipedia-style rephrasing only — reduce noise and errors.
- **Medium-quality data** → heuristic filtering only, no synthetic generation.

The five prompt templates below are verbatim from Appendix H (arXiv 2412.02595v2).
`[DOCUMENT SEGMENT]` is the placeholder for the input text.

### Prompt 1 — Diverse QA pairs

```text
Task: Read the text, ask questions and answer them.

Follow these instructions:
1. Ask diverse questions that require different cognitive skills or cover different aspects of the text.
2. Ask questions in various forms such as:
  - Yes/No questions that require determining whether a statement is true or false.
  - Open-ended questions that begin with words like what, how, when, where, why and who.
  - Multi-choice questions that offers two or more options to choose from. Include the options in the question.
  - Comparison questions that compare two quantities or objects and determine the relationship between them.
  - Reading comprehension questions that test the ability to understand and analyze the text.
  - Problem-solving questions that test the ability to solve mathematical, physical, or logical problems.
3. Focus on asking questions about factual information, important knowledge, or concrete details in the text.
4. Write questions and answers using clear and concise language.
5. Use plain text. Do not use Markdown.
6. Each question and answer pair should be on a separate line. Tag the question with "Question:" and the answer with "Answer:".

Text:
[DOCUMENT SEGMENT]

Task:
After reading the above text, ask up to 8 questions and provide the correct answers following the instructions. Give your response in this format:

Here are the questions and answers based on the provided text:
- Question: [first question] Answer: [first answer]
- Question: [second question] Answer: [second answer]
....
```

### Prompt 2 — Distill

```text
Your task is to read and paraphrase the provided text following these instructions:
- Aim to create a condensed but accurate and informative version of the original text, not a simplistic summary.
- Capture and preserve the crucial information, key concepts, important values, and factual details in the original text, while making it more readable and accessible.
- Retain technical terms, specialized vocabulary, and complex concepts.
- Retain examples, explanations of reasoning processes, and supporting evidence to maintain the text's depth and context.
- Only include information that is present in the original text. Do not adding new or unsubstantiated claims.
- Write in plain text.

Here is the text:
[DOCUMENT SEGMENT]

Task:
After thoroughly reading the above text, paraphrase it in high-quality and clear English following the instructions.
```

### Prompt 3 — Knowledge list

```text
Review the text and extract the key information. Follow these instructions:
- Carefully read the above text and provide a concise and organized list of factual information, concrete details, key concepts, and important numbers and statistics extracted from the text.
- Ensure each point is clear, specific, and supported by the original text.
- Ensure the extract text is information-dense and easier to learn from.
- Do not add titles or headings.

Text:
[DOCUMENT SEGMENT]

Task:
Extract the factual information, concrete details, and key concepts from the above text following the instructions.
```

### Prompt 4 — Extract knowledge

```text
Your task is to rewrite knowledge from the provided text following these instructions:
- Rewrite the text as a passage or passages using easy-to-understand and high-quality English like sentences in textbooks and Wikipedia.
- Focus on content in disciplines such as humanities, social sciences, natural sciences, technology, engineering, math, law and legal, business, management, art, education, agricultural sciences, politics, and history.
- Disregard content that does not contain useful facts or knowledge.
- Retain examples, explanations of reasoning processes, and supporting evidence to maintain the text's depth and context.
- Do not add or alter details. Only restate what is already in the text.
- Write in plain text.
- Do not add titles, subtitles, note, or comment.

Text:
[DOCUMENT SEGMENT]

Task:
Rewrite facts and knowledge from the above text as a passage or passages following the instructions.
```

### Prompt 5 — Wikipedia-style rephrasing

Adapted from Maini et al. (2024), WRAP (arXiv [2401.16380](https://arxiv.org/abs/2401.16380)).

```text
For the following paragraph give me a diverse paraphrase of the same in high quality English language as in sentences on Wikipedia. Begin your answer on a separate line with "Here is a paraphrased version:".

Text: [DOCUMENT SEGMENT]
```

## Relevance To Poolside

Our interpretation: keep this as an unread source for future grounding. Use it when its method or claim becomes load-bearing for a Poolside hypothesis, experiment, model note, or data-method decision.

## Related Notes

- [[concepts/rephrasal-operations]]
- [[concepts/code-rewriting-prompts]] — companion catalogue; the SwallowCode SGCR/SCOR code-rewriting prompts that Nemotron-CC-Code adopts.

## Reading State

- Tagged `read/unread`; Blaz has not marked this as read yet.
