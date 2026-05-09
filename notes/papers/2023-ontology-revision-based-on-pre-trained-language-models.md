---
arxiv: '2310.18378'
authors:
- Qiu Ji
- Guilin Qi
- Yuxin Ye
- Jiaye Li
- Site Li
- Jianjie Ren
- Songtao Lu
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2310.18378.md
raw_pdf: raw/papers/2310.18378.pdf
read: false
slug: ontology-revision-based-on-pre-trained-language-models
tags:
- nlp
- pretraining
- knowledge-graph
- ontology
title: Ontology Revision based on Pre-trained Language Models
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2310.18378v2
venue: null
year: 2023
---

# Ontology Revision based on Pre-trained Language Models

> *Qiu Ji, Guilin Qi, Yuxin Ye…* — arXiv 2310.18378, 2023

## Abstract

Ontology revision aims to seamlessly incorporate a new ontology into an existing ontology and plays a crucial role in tasks such as ontology evolution, ontology maintenance, and ontology alignment. Similar to repair single ontologies, resolving logical incoherence in the task of ontology revision is also important and meaningful, because incoherence is a main potential factor to cause inconsistency and reasoning with an inconsistent ontology will obtain meaningless answers.To deal with this problem, various ontology revision approaches have been proposed to define revision operators and design ranking strategies for axioms in an ontology. However, they rarely consider axiom semantics which provides important information to differentiate axioms. In addition, pre-trained models can be utilized to encode axiom semantics, and have been widely applied in many natural language processing tasks and ontology-related ones in recent years.Therefore, in this paper, we study how to apply pre-trained models to revise ontologies. We first define four scoring functions to rank axioms based on a pre-trained model by considering various information from an ontology. Based on the functions, an ontology revision algorithm is then proposed to deal with unsatisfiable concepts at once. To improve efficiency, an adapted revision algorithm is designed to deal with unsatisfiable concepts group by group. We conduct experiments over 19 ontology pairs and compare our algorithms and scoring functions with existing ones. According to the experiments, our algorithms could achieve promising performance.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2310.18378]]
- PDF: `raw/papers/2310.18378.pdf`
- arXiv: <http://arxiv.org/abs/2310.18378v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2023-chen-recontab.md` before that tree was retired.*

ReConTab — regularized contrastive tabular representations.
