---
aliases:
- Naeem precision/recall
- improved P/R
- density and coverage
- Naeem 2020
arxiv: '2002.09655'
authors:
- Muhammad Ferjad Naeem
- Seong Joon Oh
- Youngjung Uh
- Yunjey Choi
- Jaejun Yoo
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2002.09655
  raw: https://arxiv.org/pdf/2002.09655
  source: https://arxiv.org/abs/2002.09655
owner: blaz
read: false
slug: reliable-fidelity-and-diversity-metrics-for-generative-models
tags:
- type/paper
- source/primary
- domain/synth-data
- domain/evals
- status/draft
title: Reliable Fidelity and Diversity Metrics for Generative Models (Naeem et al.
  2020)
type: note
updated: '2026-05-10'
year: 2020
---

# Reliable Fidelity and Diversity Metrics for Generative Models (Naeem et al. 2020)

## Citation

- URL: https://arxiv.org/abs/2002.09655
- PDF: https://arxiv.org/pdf/2002.09655
- Authors: Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, Jaejun Yoo
- Year / venue: ICML 2020, arXiv submission 2020
- Raw PDF: not yet mirrored
- Raw HTML: not mirrored

## Core Claim

Earlier precision/recall metrics for generative models (Sajjadi et al. 2018; Kynkäänniemi et al. 2019) are unreliable: they are sensitive to outliers, change with sample size, and fail in identifiable ways. The paper introduces **density** and **coverage** — replacements that share the same fidelity/diversity decomposition but are stable across sample sizes and robust to outliers.

## Key paper ideas

- **Diagnose the failure modes of earlier metrics.** Identifies specific cases where Kynkäänniemi 2019 precision/recall are misled by outliers in either the model or reference set.
- **Density** replaces precision — measures how concentrated the model's samples are inside the reference manifold using a k-NN-based density estimator. More robust to outlier reference points than precision.
- **Coverage** replaces recall — measures the fraction of reference samples whose neighbourhood is reached by at least one model sample. More robust to outlier model points than recall.
- **Sample-size stability.** The authors argue (and demonstrate experimentally) that density/coverage are far less sensitive to sample-size variation than precision/recall, which matters when the reference set is finite.

## Methodology

- k-NN-based estimators in an embedding space (typical embedding: ImageNet-pretrained features for image-domain experiments).
- For each model sample, density is a count of how many reference samples have it within their k-NN ball; for each reference sample, coverage is a binary indicator that at least one model sample is within its k-NN ball.

> [!warning] unverified
> Exact definitions (numerator/denominator, normalisation constants) and the experimental settings need re-read before quoting.

## Why it matters for distributional-level evaluation

- The current **modern standard** for the precision/recall family at distributional level. When deploying a fidelity-vs-diversity audit on a synthetic corpus against a reference distribution, density/coverage are the right primitives — not the original Sajjadi P/R or the Kynkäänniemi refinement.
- Stability matters at corpus scale: with billion-document audits we cannot afford noisy estimators sensitive to a handful of outliers.

## Core concepts

- Existing concepts: [[concepts/diversity]] §Axis A reference-dependent; [[concepts/evaluation-targets]] §Coverage, §Faithfulness.
- Concepts to extract: distinction between precision (fragile) and density (robust); distinction between recall (fragile) and coverage (robust); the *meta-claim* that fidelity-vs-diversity decomposition is principled but the *implementation* matters.

## Relevance to Poolside

For any corpus-vs-reference audit at production scale, density and coverage should be the default. They give the same separation as precision/recall — fidelity vs diversity — but with predictable behaviour at sample sizes we can actually run. When MAUVE gives a single inscrutable scalar, density/coverage tell us which failure mode we are in.

## Blaz notes

- 

## Key follow-ups / jumping-off points

- Read the original ablation: where exactly do Kynkäänniemi P/R fail and density/coverage succeed?
- Implementation: is there a stable open-source implementation? FAISS-based k-NN should make this tractable at corpus scale.
- Compare to MAUVE on the same setting — when does MAUVE rank generators differently than density/coverage?

## Related notes

- Concepts: [[concepts/diversity]], [[concepts/evaluation-targets]], [[concepts/reference-distributions]].
- Maps: [[maps/evaluation/distributional-level]] (primary home).
- Predecessors: [[notes/papers/2018-precision-and-recall-for-generative-models]] (Sajjadi); Kynkäänniemi et al. 2019 ([arXiv 1904.06991](https://arxiv.org/abs/1904.06991), stub).
- Adjacent: [[metrics/mauve]] (single-scalar alternative), [[metrics/chamfer-distance]] (asymmetric simple form).

## Caveats

- Originally developed and validated on image-domain generative models (GANs / VAEs). Adapting to text / code requires choosing a sentence/code embedding; results may inherit encoder biases not present in the image setting.
- Paper not yet read in full this pass — note is a structural placeholder. Re-read with `skills/paper-reading/SKILL.md` before citing specific numbers or claims.
