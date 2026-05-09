---
arxiv: '2106.00783'
authors:
- Dario Fuoli
- Luc Van Gool
- Radu Timofte
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2106.00783.md
raw_pdf: raw/papers/2106.00783.pdf
read: false
slug: fourier-space-losses-for-efficient-perceptual-image-super
tags:
- vision
- optimization
- image-super-resolution
title: Fourier Space Losses for Efficient Perceptual Image Super-Resolution
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2106.00783v1
venue: null
year: 2021
---

# Fourier Space Losses for Efficient Perceptual Image Super-Resolution

> *Dario Fuoli, Luc Van Gool, Radu Timofte* — arXiv 2106.00783, 2021

## Abstract

Many super-resolution (SR) models are optimized for high performance only and therefore lack efficiency due to large model complexity. As large models are often not practical in real-world applications, we investigate and propose novel loss functions, to enable SR with high perceptual quality from much more efficient models. The representative power for a given low-complexity generator network can only be fully leveraged by strong guidance towards the optimal set of parameters. We show that it is possible to improve the performance of a recently introduced efficient generator architecture solely with the application of our proposed loss functions. In particular, we use a Fourier space supervision loss for improved restoration of missing high-frequency (HF) content from the ground truth image and design a discriminator architecture working directly in the Fourier domain to better match the target HF distribution. We show that our losses' direct emphasis on the frequencies in Fourier-space significantly boosts the perceptual image quality, while at the same time retaining high restoration quality in comparison to previously proposed loss functions for this task. The performance is further improved by utilizing a combination of spatial and frequency domain losses, as both representations provide complementary information during training. On top of that, the trained generator achieves comparable results with and is 2.4x and 48x faster than state-of-the-art perceptual SR methods RankSRGAN and SRFlow respectively.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2106.00783]]
- PDF: `raw/papers/2106.00783.pdf`
- arXiv: <http://arxiv.org/abs/2106.00783v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-lemorvan-whats-good-imputation.md` before that tree was retired.*

What's a good imputation for prediction? Theoretical and empirical analysis.
