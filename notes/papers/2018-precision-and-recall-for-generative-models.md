---
aliases:
- Sajjadi precision/recall
- P/R for generative models
- Sajjadi 2018
arxiv: '1806.00035'
authors:
- Mehdi S. M. Sajjadi
- Olivier Bachem
- Mario Lucic
- Olivier Bousquet
- Sylvain Gelly
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1806.00035
  raw: https://arxiv.org/pdf/1806.00035
  source: https://arxiv.org/abs/1806.00035
owner: blaz
read: false
slug: precision-and-recall-for-generative-models
tags:
- type/paper
- source/primary
- domain/synth-data
- domain/evals
- status/draft
title: Assessing Generative Models via Precision and Recall (Sajjadi et al. 2018)
type: note
updated: '2026-05-10'
year: 2018
---

# Assessing Generative Models via Precision and Recall (Sajjadi et al. 2018)

## Citation

- URL: https://arxiv.org/abs/1806.00035
- PDF: https://arxiv.org/pdf/1806.00035
- Authors: Mehdi S. M. Sajjadi, Olivier Bachem, Mario Lucic, Olivier Bousquet, Sylvain Gelly
- Year / venue: NeurIPS 2018, arXiv submission 2018
- Raw PDF: not yet mirrored (`raw/papers/2018-precision-recall-generative-models.pdf`)
- Raw HTML: not mirrored

## Core Claim

A single scalar (e.g., FID) cannot distinguish a generative model that produces high-quality samples from a narrow region of the target distribution (high precision, low recall) from one that covers the full target but emits some low-quality samples (low precision, high recall). The paper proposes a two-number summary — precision and recall — defined geometrically over the supports of the model and reference distributions, that decouples these two failure modes.

## Key paper ideas

- **Two failure modes are not interchangeable.** Mode-dropping (failing to cover regions of the reference) and mode-inventing (placing mass where the reference is empty) are conceptually different and need separate metrics.
- **Geometric definition over distributions.** Precision is defined as the fraction of model mass that lies in the support of the reference; recall is the fraction of reference mass covered by the model.
- **Pareto-front output.** Rather than a single $(P, R)$ pair, the method produces a curve over a parameter — different points on the curve correspond to different trade-offs.
- **Separation principle.** A model can be optimal on precision while bad on recall and vice versa; aggregating into a single number (like FID) hides which failure dominates.

## Methodology

- Operates in an embedding space (a feature extractor maps samples to a finite-dimensional representation).
- Estimates precision and recall from finite samples of model and reference distributions.
- The original formulation involves clustering / quantisation in embedding space; subsequent work (Kynkäänniemi et al. 2019, [arXiv 1904.06991](https://arxiv.org/abs/1904.06991); Naeem et al. 2020, [arXiv 2002.09655](https://arxiv.org/abs/2002.09655)) refined this to k-NN-based manifold overlap with better stability.

> [!warning] unverified
> Specific algorithmic details (which clustering, exact estimator) need a re-read of the paper before quoting.

## Why it matters for distributional-level evaluation

- Provides the **canonical separation** of fidelity (precision: are the model's samples on-distribution?) from coverage (recall: does the model reach all modes of the target?). Both are reference-dependent — they require a reference distribution. This is exactly the [[maps/evaluation/distributional-level]] §Axis 1 use case.
- The Chamfer distance ([[metrics/chamfer-distance]]) is the simple-to-compute asymmetric cousin: $\text{Ch}(\text{model} \to \text{reference})$ is a precision-style number; $\text{Ch}(\text{reference} \to \text{model})$ is a recall-style number. Sajjadi P/R is the principled version.

## Core concepts

- Existing concepts: [[concepts/diversity]] §Axis A reference-dependent; [[concepts/evaluation-targets]] §Coverage, §Faithfulness.
- Concepts to extract: precision/recall as the canonical separation of fidelity vs coverage in distributional-level evaluation; the Pareto-curve framing (rather than a single scalar).

## Relevance to Poolside

For corpus-level audits where we have a target reference distribution (held-out organic, teacher distribution, task-conditioned reference), Sajjadi P/R gives a two-number readout that disentangles "is the corpus *on-target*" from "does the corpus *cover the target*". A single MAUVE score collapses these. Use P/R when the failure mode matters operationally — e.g., during a rephrasing pipeline launch where over-narrowing (precision-good, recall-bad) and hallucinating (precision-bad, recall-good) call for different fixes.

## Blaz notes

- 

## Key follow-ups / jumping-off points

- **Refined variants** to read in sequence:
  - Kynkäänniemi et al. 2019, "Improved Precision and Recall Metric for Assessing Generative Models" — arXiv [1904.06991](https://arxiv.org/abs/1904.06991). Replaces clustering with k-NN manifold approach.
  - Naeem et al. 2020, "Reliable Fidelity and Diversity Metrics for Generative Models" — arXiv [2002.09655](https://arxiv.org/abs/2002.09655). Modern standard with sample-size-stable variants.
- **In the formalism.** Map P/R onto the four standard reference distributions in [[concepts/reference-distributions]]:
  - vs $p_{\text{human}}$: how natural is the synthesis?
  - vs $p^*$: target-matching.
  - vs $p_{\text{seed}}$: how much of the seed support is preserved?

## Related notes

- Concepts: [[concepts/diversity]], [[concepts/evaluation-targets]], [[concepts/reference-distributions]], [[concepts/synthetic-data-formalism]].
- Maps: [[maps/evaluation/distributional-level]] (primary home), [[maps/evaluation/landscape]] (matrix entry).
- Metrics: [[metrics/chamfer-distance]] (simple asymmetric cousin), [[metrics/mauve]] (single-scalar reference-dependent alternative).

## Caveats

- The original 2018 estimator has been superseded for practical use by Kynkäänniemi 2019 and Naeem 2020; cite Sajjadi as the conceptual origin and use the refined variants for deployment.
- Paper not yet read in full this pass — note is a structural placeholder grounded in the conceptual contribution; specific equations and experimental numbers should be filled in on a follow-up read using `skills/paper-reading/SKILL.md`.
