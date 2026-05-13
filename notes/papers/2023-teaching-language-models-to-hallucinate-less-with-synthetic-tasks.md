---
arxiv: '2310.06827'
authors:
- Erik Jones
- Hamid Palangi
- Clarisse Simões
- Varun Chandrasekaran
- Subhabrata Mukherjee
- Arindam Mitra
- Ahmed Awadallah
- Ece Kamar (Microsoft Research)
created: 2026-04-29
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2310.06827
  raw: https://arxiv.org/pdf/2310.06827
  source: https://arxiv.org/abs/2310.06827
owner: blaz
read: false
slug: teaching-language-models-to-hallucinate-less-with-synthetic-tasks
tags:
- type/paper
- status/stub
- source/primary
- domain/synth-data
- domain/sft
- domain/llm
title: Teaching Language Models to Hallucinate Less with Synthetic Tasks (SynTra)
type: note
updated: '2026-05-10'
year: 2023
---

# Teaching Language Models to Hallucinate Less with Synthetic Tasks (SynTra)

## Citation

- URL: https://arxiv.org/abs/2310.06827
- PDF: https://arxiv.org/pdf/2310.06827
- OpenReview: https://openreview.net/forum?id=xpw7V0P136
- Authors: Erik Jones, Hamid Palangi, Clarisse Simões, Varun Chandrasekaran, Subhabrata Mukherjee, Arindam Mitra, Ahmed Awadallah, Ece Kamar (Microsoft Research)
- Year / venue: ICLR 2024; arXiv Oct 2023

## Core Claim

**SynTra**: optimise hallucination on a *synthetic* retrieval task where hallucinations are easy to elicit and cheaply measurable, then transfer the resulting *system message* (via prefix-tuning) to real abstractive-summarisation tasks. Reduces hallucination on three real tasks for two 13B LLMs without fine-tuning the model weights — and notably, fine-tuning weights on the synthetic task can *increase* downstream hallucination.

## Key Paper Ideas

- **Synthetic-task-as-probe**: pick a synthetic task whose hallucination signal is cheap to evaluate at every optimisation step, optimise on it, and transfer.
- **Optimise the system message, not the weights.** Prefix-tune a system prompt; transfer the prompt to real tasks.
- **Weight-tuning can hurt.** Fine-tuning the entire model on the synthetic task counterintuitively raises hallucination rates downstream — system-prompt-only optimisation is the safer transfer interface.
- Synthetic-task design: a retrieval task where the model must answer from a context, and "hallucination" = answering when the context doesn't support it.

## Methodology

- Step 1: design a synthetic retrieval task with controllable hallucination ground truth.
- Step 2: prefix-tune a system message on the synthetic task to minimise hallucination.
- Step 3: transfer the optimised system message to three real abstractive-summarisation tasks (document-QA, meeting summarisation, clinical reports).

## Experiments

- Two 13B-parameter LLMs, three real summarisation/QA benchmarks, comparison of prefix-tuning vs full-weight finetuning.

## Key Results

- SynTra reduces hallucination on the three real tasks for both 13B LLMs.
- Fine-tuning the entire model on the synthetic task can *increase* hallucination — system-prompt-only transfer is critical.

## Core Concepts

- Existing concepts: [[concepts/synthetic-data-formalism]], [[concepts/trustworthiness-taxonomy]] (faithfulness / hallucination), [[concepts/intrinsic-vs-extrinsic-evaluation]] (synthetic task is the cheap intrinsic probe; real-task hallucination is the extrinsic target).
- Concepts to extract: —

## Relevance To Poolside

*Stub.* The "optimise on a cheap synthetic probe, transfer the *prompt* not the *weights*" pattern is generally interesting for any quality-axis where the real evaluation is expensive (e.g. agentic SWE rollouts). Worth noting that weight-finetuning *worsened* the metric — argues against assuming "more training on the synthetic proxy helps".

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- 

## Related Notes

- Concepts: [[concepts/trustworthiness-taxonomy]], [[concepts/synthetic-data-formalism]], [[concepts/intrinsic-vs-extrinsic-evaluation]]

## Caveats

- Stub note; abstract-level only.
- Result that *full-weight* tuning on the synthetic task hurts is a strong negative finding — should not be conflated with "synthetic task data is bad".
