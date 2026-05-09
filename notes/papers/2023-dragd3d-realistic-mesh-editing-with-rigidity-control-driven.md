---
arxiv: '2310.04561'
authors:
- Tianhao Xie
- Eugene Belilovsky
- Sudhir Mudur
- Tiberiu Popa
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2310.04561.md
raw_pdf: raw/papers/2310.04561.pdf
read: false
slug: dragd3d-realistic-mesh-editing-with-rigidity-control-driven
tags:
- diffusion
- vision
- 3d-mesh
- generative
title: 'DragD3D: Realistic Mesh Editing with Rigidity Control Driven by 2D Diffusion
  Priors'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2310.04561v2
venue: null
year: 2023
---

# DragD3D: Realistic Mesh Editing with Rigidity Control Driven by 2D Diffusion Priors

> *Tianhao Xie, Eugene Belilovsky, Sudhir Mudur…* — arXiv 2310.04561, 2023

## Abstract

Direct mesh editing and deformation are key components in the geometric modeling and animation pipeline. Mesh editing methods are typically framed as optimization problems combining user-specified vertex constraints with a regularizer that determines the position of the rest of the vertices. The choice of the regularizer is key to the realism and authenticity of the final result. Physics and geometry-based regularizers are not aware of the global context and semantics of the object, and the more recent deep learning priors are limited to a specific class of 3D object deformations. Our main contribution is a vertex-based mesh editing method called DragD3D based on (1) a novel optimization formulation that decouples the rotation and stretch components of the deformation and combines a 3D geometric regularizer with (2) the recently introduced DDS loss which scores the faithfulness of the rendered 2D image to one from a diffusion model. Thus, our deformation method achieves globally realistic shape deformation which is not restricted to any class of objects. Our new formulation optimizes directly the transformation of the neural Jacobian field explicitly separating the rotational and stretching components. The objective function of the optimization combines the approximate gradients of DDS and the gradients from the geometric loss to satisfy the vertex constraints. Additional user control over desired global shape deformation is made possible by allowing explicit per-triangle deformation control as well as explicit separation of rotational and stretching components of the deformation. We show that our deformations can be controlled to yield realistic shape deformations that are aware of the global context of the objects, and provide better results than just using geometric regularizers.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2310.04561]]
- PDF: `raw/papers/2310.04561.pdf`
- arXiv: <http://arxiv.org/abs/2310.04561v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2023-levin-transfer-learning-tabular.md` before that tree was retired.*

Deep tabular's representation advantage shows up only in low-data transfer settings.
