---
arxiv: '2604.20614'
authors:
- Alessandro Morosini
- Matea Gjika
- Tomaso Poggio
- Pierfrancesco Beneventano
created: '2026-05-09'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2026-too-sharp-too-sure-when-calibration-follows-curvature.md
raw_pdf: raw/papers/pdf/2026-too-sharp-too-sure-when-calibration-follows-curvature.pdf
read: false
slug: too-sharp-too-sure-when-calibration-follows-curvature
tags:
- calibration
- uncertainty
- optimization
title: 'Too Sharp, Too Sure: When Calibration Follows Curvature'
type: note
updated: '2026-05-09'
url: https://arxiv.org/abs/2604.20614
venue: null
year: 2026
---

# Too Sharp, Too Sure: When Calibration Follows Curvature

> *Alessandro Morosini, Matea Gjika, Tomaso Poggio…* — arXiv 2604.20614, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Modern neural networks can achieve high accuracy while remaining poorly calibrated, producing confidence estimates that do not match empirical correctness. Yet calibration is often treated as a post-hoc attribute. We take a different perspective: we study calibration as a training-time phenomenon on small vision tasks, and ask whether calibrated solutions can be obtained reliably by intervening on the training procedure. We identify a tight coupling between calibration, curvature, and margins during training of deep networks under multiple gradient-based methods. Empirically, Expected Calibration Error (ECE) closely tracks curvature-based sharpness throughout optimization. Mathematically, we show that both ECE and Gauss–Newton curvature are controlled, up to problem-specific constants, by the same margin-dependent exponential tail functional along the trajectory. Guided by this mechanism, we introduce a margin-aware training objective that explicitly targets robust-margin tails and local smoothness, yielding improved out-of-sample calibration across optimizers without sacrificing accuracy.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2026-too-sharp-too-sure-when-calibration-follows-curvature]]
- PDF: [[raw/papers/pdf/2026-too-sharp-too-sure-when-calibration-follows-curvature.pdf]]
- arXiv: <https://arxiv.org/abs/2604.20614>
