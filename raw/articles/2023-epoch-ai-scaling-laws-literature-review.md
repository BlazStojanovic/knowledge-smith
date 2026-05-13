---
source: article
url: https://epoch.ai/blog/scaling-laws-literature-review
retrieved: 2026-04-23
title: "Scaling Laws Literature Review"
author: Pablo Villalobos
publication: Epoch AI
date: 2023-01-26
license: CC-BY
---

# Scaling Laws Literature Review

Author: Pablo Villalobos, Epoch AI. Published 2023-01-26.

## Key takeaways

1. **Functional forms**: power law models L(N,D) = AN^-a + BD^-b capture scaling in the power-law region but not transition zones. The M4 estimator and BNSL estimator provide superior modeling of transitions.
2. **Transfer learning**: no universal scaling law exists across arbitrary task pairs. When tasks are similar, upstream loss and downstream performance are closely related, but significant task differences make architecture and HP choices critical.
3. **Theoretical insights**: scaling exponent magnitudes correlate inversely with data manifold intrinsic dimensionality (Sharma et al. 2020, Bahri et al. 2021).

## Key formulas

**Power law with joint error:**
L(N,D) = AN^-a + BD^-b

**M4 estimator:**
(L-E)/(I-L)^α = AN^-a + BD^-b

**BNSL estimator (Broken Neural Scaling Laws):**
L(D) = E + (b·D^-c₀) ∏(1 + (D/dᵢ)^(1/fᵢ))^(-cᵢfᵢ)

## Historical development

- 2017: Hestness et al. identified empirical power-law scaling across domains
- 2019-2020: expansion to multiple architectures and larger scales
- 2020-2021: theoretical mechanisms proposed explaining exponent values
- 2022: Hoffmann et al. showed previously discovered laws were suboptimal; universality assumptions questioned

## Notable findings

- Hoffmann et al. (Chinchilla) demonstrated suboptimal HP choices in prior work
- Sorscher et al. showed exponential rather than power-law scaling in certain data-curation contexts
- No universal law across arbitrary tasks — "scale is all you need" applies to direct training, not transfer

Full resource: detailed scaling laws database and paper reviews in linked spreadsheets.
