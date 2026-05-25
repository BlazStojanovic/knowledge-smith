---
arxiv: '1904.06376'
authors:
- Greg Henry
- Ping Tak Peter Tang
- Alexander Heinecke
parser: ar5iv
retrieved: '2026-05-25'
source: paper
title: Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision
  Computations
url: https://arxiv.org/abs/1904.06376
year: 2019
---

[1904.06376] Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations



# Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations

Greg Henry
IAGS
  
Intel Corporation
  
Hillsboro, USA
  
greg.henry@intel.com
  
Ping Tak Peter Tang
IAGS
  
Intel Corporation
  
Santa Clara, USA
  
peter.tang@intel.com
  
Alexander Heinecke
Intel Labs
  
Intel Corporation
  
Santa Clara, USA
  
alexander.heinecke@intel.com

###### Abstract

In recent years fused-multiply-add (FMA) units with lower-precision
multiplications and higher-precision accumulation have proven
useful in machine learning/artificial intelligence applications,
most notably in training deep neural networks due to their extreme
computational intensity. Compared to classical IEEE-754 32 bit (FP32) and
64 bit (FP64) arithmetic, these reduced precision arithmetic can naturally
be sped up disproportional to their shortened width. The common
strategy of all major hardware vendors is to aggressively further enhance their
performance disproportionately. One particular FMA operation that multiplies
two BF16 numbers while
accumulating in FP32 has been found useful in deep learning, where BF16 is
the 16-bit floating point datatype with IEEE FP32 numerical range but 8 significant bits of precision. In this paper, we examine the use this FMA unit to implement
higher-precision matrix routines in terms of potential
performance gain and implications on accuracy. We
demonstrate how a decomposition into multiple smaller datatypes can be
used to assemble a high-precision result, leveraging the higher precision accumulation of the FMA unit. We first
demonstrate that computations of vector inner products and
by natural extension, matrix-matrix products can be achieved by decomposing FP32 numbers in several BF16 numbers followed by appropriate computations that can
accommodate the dynamic range and preserve accuracy compared to standard FP32 computations, while projecting up to 5.2×\times speed-up. Furthermore,
we examine solution of linear equations formulated in the
residual form that allows for iterative refinement. We
demonstrate that the solution obtained to be comparable to
those offered by FP64 under a large range of linear
system condition numbers.

###### Index Terms:

bfloat16, float16, mixed precision, combined datatypes

††publicationid: pubid: ©2019 IEEE. Personal use of this material is permitted.   

## I Introduction

bfloat16 (BF16) is a new floating-point format
[[1](#bib.bib1)] that is gaining
traction due to its ability to work well in machine learning algorithms,
in particular deep learning training. In contrast to the IEEE754-standardized
16bit (FP16) variant, BF16 does not compromise at all on range when being
compared to FP32. As a reminder, FP32 numbers have 8 bits of exponent and 24
bits of mantissa (one implicit). BF16 cuts 16 bits from the 24-bit
FP32 mantissa to create a 16-bit floating point datatype. In contrast FP16,
roughly halves the FP32 mantissa to 10 explicit bits and has to reduce the
exponent to 5 bits to fit the 16-bit datatype envelope.

Figure 1: BF16 FMA unit as proposed in [[2](#bib.bib2)]. This unit is fully compatible with IEEE FP32.

Although BF16 offers less precision than FP16, it is better suited to
support deep learning tasks. As shown in [[3](#bib.bib3)], FP16’s range is not enough
to accomplish deep learning training out-of-the-box due to its limited
range. BF16 does not suffer from this issue and the limited precision
actually helps to generalize the learned weights in
the neural net training task. In other words, lower precision can
be seen as offering a built-in regularization property.

Additionally, the heart of deep learning is matrix multiplication. That
means computing inner products of vectors of various length. Normally the
dimensions of these vectors are pretty long: several hundreds to tens of
thousands. Therefore, the community has settled on mixed-precision
fused-multiply-add (FMA) hardware units. E.g. NVIDIA announced their FP16
input with FP32 output Tensorcores support in Volta and Turing GPUs
and Intel has recently published their BF16 hardware
numeric definition for up-coming processors
code-named Cooper Lake [[2](#bib.bib2)]. NVIDIA did not publish the exact
hardware specification, whereas Intel’s BF16 FMA is depicted in
Fig. [1](#S1.F1 "Figure 1 ‣ I Introduction ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations"). The heart of this is a traditional FP32 FMA
unit which can deal with BF16 numbers that are interpreted as short FP32
numbers. The key functionality is the FP32 accumulation of the unit. This means
that the 16bit product’s result is fully preserved and accumulated with 24bit
precision. Google’s TPU also offers BF16 multiply with FP32 accumulate, but
as for NVIDIA’s Volta and Turing, the exact hardware definition is not available.

When looking at the FP16/BF16 performance specs, we can make one important
observation: the number of floating point operations per second (FLOPS)
provided in these formats are at least one
order of magnitude higher than for FP32. E.g.
Volta offers more than 120 TFLOPS of FP16 compute while only providing
15 TFLOPS of FP32 compute (both FMA). This is due to much smaller multiplier
and offering the FLOPS only in form of matrix multiplication by implementing
a systolic array in hardware. BF16 is expected to be even better in this
respect as the mantissa is 30% shorter. Therefore one pressing question
is: can this high computational performance be efficiently harvested for
FP32 compute111Intel has only announced the numerics and instruction definitions so far
but not the actual FP32/BF16 performance ratio..

There is a precedent in HPC research to exploit multiple
floating-point numbers combined together, often references as single-single
or double-double precision [[4](#bib.bib4)]. This does nothing for the exponent bits, but if
we consider two BF16s combined together, that yields 8 bits of exponent and
16 bits of mantissa total. And three BF16s would represent 8 bits of
exponents and 24 bits of mantissa total. The first observation one might
make is that this last case, a triplet of BF16s, is comparable to FP32
as we have identical range and mantissa bits. Recently such an idea was also
employed for
NVIDIA Tensorcores with two FP16 numbers for FFT [[5](#bib.bib5)].
However more mechanics are needed
due to lower ranges of FP16 and only 22 bits total mantissa (if counting the implicit bits.)

In this paper, we study the numerical properties, accuracy, and performance
ramifications of 3 (or 2) BF16 combined together versus FP32. Despite a
similar number of exponent bits and mantissa bits, resulting algorithms will
not be bitwise identical to FP32 calculations. In some cases, it will be less
accurate. In some cases, it will be more accurate.

In our numeric studies, we consider the case of doing a dot product of two
vectors x and y. This is the basis of a matrix-matrix multiply algorithm
(GEMM), which in turn is the basis for many computations in linear algebra,
as GEMM is the core routine behind the Level-3 BLAS [[6](#bib.bib6)] and much of
LAPACK [[7](#bib.bib7)].

Our paper makes following contributions:

* •

  we discuss an accumulation error analysis for the dot-product
  of two vectors represented as triplets of BF16 numbers. There are cases where multiplying two BF16s might yield exact, or near exact, results. This means that we often will have much greater accuracy than FP32 calculations.
* •

  we consider the issue of “short-cuts” where we don’t consider all the bits available to us. For instance, three BF16 splitting of FP32 number will require 9 multiplication (all-to-all), but do we really need to consider lower-order terms? The least significant bits should have a minimal impact on the final result. We will show that a
  6-produce version achieves acceptable accuracy.
* •

  we analyze common BLAS and LAPACK kernels, namely SGEMM and SGETRF
  using our combined datatype. We focus on matrices of both small and large exponential range.
* •

  we consider performance implications: asymptotically a 6-product version has six times as much work compared to GEMM in FP32. Depending on the factor improvement of BF16 GEMM over FP32 GEMM, a closer look at the accuracy and performance ramifications is not only interesting, but justified, potentially offering up to 5.2×\times speed-up.
* •

  to complete our work, we also investigate how BF16 compares to FP16 when being used in one-sided decomposition which are sped-up by iterative refinement. Here we can conclude that in general case BF16 may not be enough, but for diagonally-dominant matrices its performance is comparable to FP16.

## II Combined Lower Precision Datatypes And Their Application to BLAS and LAPACK Routines

This sections covers how we decompose FP32 numbers into multiple BF16 numbers
and derives error bounds for dot-product computations using this type. We
also discuss how we can skip lower order terms while maintaining FP32
comparable accuracy.

### II-A Decomposition of a FP32 number into multiple BF16 numbers

We use the notation ℱ32subscriptℱ32{\cal F}\_{32} and ℬ16subscriptℬ16{\cal B}\_{16} to
denote that set of reals number representable in FP32 and BF16, respectively222
It is convenient to treat things as real number and use the description that
the values are representable exactly in FP32 to say they are single precision
numbers.
Lets assume that a𝑎a is a ℱ32subscriptℱ32{\cal F}\_{32} number and it is stored into 3 ℬ16subscriptℬ16{\cal B}\_{16}: b(0)superscript𝑏0{b}^{({0})}, b(1)superscript𝑏1{b}^{({1})}, and b(2)superscript𝑏2{b}^{({2})}. (ℱ32)subscriptℱ32({\cal F}\_{32}) and (ℬ16)subscriptℬ16({\cal B}\_{16}) shall
denote the conversion operator to the respective type. We assign these values
as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | b(0)=superscript𝑏0absent\displaystyle{b}^{({0})}= | (ℬ16)​asubscriptℬ16𝑎\displaystyle({\cal B}\_{16})a |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | b(1)=superscript𝑏1absent\displaystyle{b}^{({1})}= | (ℬ16)​((ℱ32)​(a−(ℱ32)​b(0)))subscriptℬ16subscriptℱ32𝑎subscriptℱ32superscript𝑏0\displaystyle({\cal B}\_{16})(({\cal F}\_{32})(a-({\cal F}\_{32}){b}^{({0})})) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | b(2)=superscript𝑏2absent\displaystyle{b}^{({2})}= | (ℬ16)​((ℱ32)​(a−(ℱ32)​b(0)​–​(ℱ32)​b(1)))subscriptℬ16subscriptℱ32𝑎subscriptℱ32superscript𝑏0–subscriptℱ32superscript𝑏1\displaystyle({\cal B}\_{16})(({\cal F}\_{32})(a-({\cal F}\_{32}){b}^{({0})}–({\cal F}\_{32}){b}^{({1})})) |  |

One can imagine that a𝑎a is an approximation of (ℱ32)​b(0)+(ℱ32)​b(1)+(ℱ32)​b(2)subscriptℱ32superscript𝑏0subscriptℱ32superscript𝑏1subscriptℱ32superscript𝑏2({\cal F}\_{32}){b}^{({0})}+({\cal F}\_{32}){b}^{({1})}+({\cal F}\_{32}){b}^{({2})}. Adding two triplets together has 3 times the number of adds. Multiplying two triplets together has 9 times the number of multiplies, not to mention extra adds as well, which are free when using FMA units.

### II-B Dot Product Notation

Given two vectors 𝐱,𝐲∈ℝn

𝐱𝐲
superscriptℝ𝑛\mathbf{x},\mathbf{y}\in\mathbb{R}^{n} both of which representable
exactly in IEEE single precision format, the goal is to compute the inner product 𝐱T​𝐲superscript𝐱𝑇𝐲\mathbf{x}^{T}\mathbf{y}. The reference
is the standard computation in FP32 using FMA, that is, one rounding error
in each accumulation. What we want to explore is to use BF16 to compute the
inner product. The basic idea is that each FP32 representable value can be
decomposed exactly into the unevaluated sum of three BF16 representable numbers
and thus the inner product in question is expressible in 9 inner products involving
vectors of BF16 representable values.

Here is the basic set up:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐱𝐱\displaystyle\mathbf{x} | =[x1,x2,…,xn]T∈ℱ32nabsentsuperscript  subscript𝑥1subscript𝑥2…subscript𝑥𝑛 𝑇superscriptsubscriptℱ32𝑛\displaystyle=[x\_{1},x\_{2},\ldots,x\_{n}]^{T}\in{\cal F}\_{32}^{n} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐲𝐲\displaystyle\mathbf{y} | =[y1,y2,…,yn]T∈ℱ32nabsentsuperscript  subscript𝑦1subscript𝑦2…subscript𝑦𝑛 𝑇superscriptsubscriptℱ32𝑛\displaystyle=[y\_{1},y\_{2},\ldots,y\_{n}]^{T}\in{\cal F}\_{32}^{n} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐱(i)superscript𝐱𝑖\displaystyle{\mathbf{x}}^{({i})} | =[x1(i),x2(i),…,xn(i)]T∈ℬ16n,i=0,1,2formulae-sequenceabsentsuperscript  superscriptsubscript𝑥1𝑖superscriptsubscript𝑥2𝑖…superscriptsubscript𝑥𝑛𝑖 𝑇superscriptsubscriptℬ16𝑛𝑖  012\displaystyle=[{x\_{1}}^{({i})},{x\_{2}}^{({i})},\ldots,{x\_{n}}^{({i})}]^{T}\in{\cal B}\_{16}^{n},\quad i=0,1,2 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐲(i)superscript𝐲𝑖\displaystyle{\mathbf{y}}^{({i})} | =[y1(i),y2(i),…,yn(i)]T∈ℬ16n,i=0,1,2formulae-sequenceabsentsuperscript  superscriptsubscript𝑦1𝑖superscriptsubscript𝑦2𝑖…superscriptsubscript𝑦𝑛𝑖 𝑇superscriptsubscriptℬ16𝑛𝑖  012\displaystyle=[{y\_{1}}^{({i})},{y\_{2}}^{({i})},\ldots,{y\_{n}}^{({i})}]^{T}\in{\cal B}\_{16}^{n},\quad i=0,1,2 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | z𝑧\displaystyle z | =𝐱T​𝐲,z^=|𝐱|T​|𝐲|=∑ℓ=1n|xℓ​yℓ|formulae-sequenceabsentsuperscript𝐱𝑇𝐲^𝑧superscript𝐱𝑇𝐲superscriptsubscriptℓ1𝑛subscript𝑥ℓsubscript𝑦ℓ\displaystyle=\mathbf{x}^{T}\mathbf{y},\quad\hat{{z}}=|\mathbf{x}|^{T}|\mathbf{y}|=\sum\_{\ell=1}^{n}|x\_{\ell}y\_{\ell}| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | z(i,j)superscript𝑧𝑖𝑗\displaystyle{z}^{({i,j})} | =𝐱(i)T​𝐲(j),z^(i,j)=∑ℓ=1n|xℓ(i)​yℓ(j)|,0≤i,j≤2formulae-sequenceabsentsuperscriptsuperscript𝐱𝑖𝑇superscript𝐲𝑗formulae-sequencesuperscript^𝑧𝑖𝑗superscriptsubscriptℓ1𝑛superscriptsubscript𝑥ℓ𝑖superscriptsubscript𝑦ℓ𝑗formulae-sequence0𝑖𝑗2\displaystyle={{\mathbf{x}}^{({i})}}^{T}{\mathbf{y}}^{({j})},\quad{\hat{{z}}}^{({i,j})}=\sum\_{\ell=1}^{n}|{x\_{\ell}}^{({i})}{y\_{\ell}}^{({j})}|,\quad 0\leq i,j\leq 2 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | εfsubscript𝜀𝑓\displaystyle\varepsilon\_{f} | =2−24,εb=2−8formulae-sequenceabsentsuperscript224subscript𝜀𝑏superscript28\displaystyle=2^{-24},\quad\varepsilon\_{b}=2^{-8} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | γf,ksubscript𝛾  𝑓𝑘\displaystyle\gamma\_{f,{k}} | =k​εf1−k​εf,γb,k=k​εb1−k​εbformulae-sequenceabsent𝑘subscript𝜀𝑓1𝑘subscript𝜀𝑓subscript𝛾  𝑏𝑘𝑘subscript𝜀𝑏1𝑘subscript𝜀𝑏\displaystyle=\frac{k\varepsilon\_{f}}{1-k\varepsilon\_{f}},\quad\gamma\_{b,{k}}=\frac{k\varepsilon\_{b}}{1-k\varepsilon\_{b}} |  |

### II-C Basic Bounds on Single Precision

The standard computation in FP32 is as follows:

Z←0←𝑍0Z\leftarrow 0
  
   For ℓ=1,2,…,nℓ

12…𝑛\ell=1,2,\ldots,n:
  
         Z←𝙵𝙼𝙰​(xℓ,yℓ,Z)←𝑍𝙵𝙼𝙰subscript𝑥ℓsubscript𝑦ℓ𝑍Z\leftarrow{\tt FMA}(x\_{\ell},y\_{\ell},Z)\\
   End

The error bound is standard in this case, namely

|  |  |  |
| --- | --- | --- |
|  | |Z−z|≤γf,n​z^𝑍𝑧subscript𝛾  𝑓𝑛^𝑧|Z-z|\leq\gamma\_{f,{n}}\hat{{z}} |  |

That is, the absolute error is roughly n𝑛n rounding errors times the
inner product with the absolute value of the vectors. So the relative error
with respect to z𝑧z is roughly n𝑛n rounding errors *if* there
is not much cancellation. Indeed, the ratio z^/|z|≥1^𝑧𝑧1\hat{{z}}/|z|\geq 1 is
usually called the condition number in this case.
So the rest of the document tries to derive similar upper bounds on the
error when we use various summation procedure utilizing
FMA accumulation to first compute the z(i,j)superscript𝑧𝑖𝑗{z}^{({i,j})}s, followed by
summation.

### II-D Error Analysis for combined BF16 Datatypes

The following quantities are the relevant components, although
various specific inner product computation may use only a subset of
these quantities.

For each i,j

𝑖𝑗i,j, 0≤i,j≤2formulae-sequence0𝑖𝑗20\leq i,j\leq 2, we compute in FP32 precision the
nine partial inner products.

Z(i,j)←0←superscript𝑍𝑖𝑗0{Z}^{({i,j})}\leftarrow 0
  
   For ℓ=1,2,…,nℓ

12…𝑛\ell=1,2,\ldots,n:
  
         Z(i,j)←𝙵𝙼𝙰​(xℓ(i),yℓ(j),Z(i,j))←superscript𝑍𝑖𝑗𝙵𝙼𝙰superscriptsubscript𝑥ℓ𝑖superscriptsubscript𝑦ℓ𝑗superscript𝑍𝑖𝑗{Z}^{({i,j})}\leftarrow{\tt FMA}({x\_{\ell}}^{({i})},{y\_{\ell}}^{({j})},{Z}^{({i,j})})
  
   End

Add the partial products of “equal levels”. In FP32 arithmetic do the following

Z(0)←Z(0,0)←superscript𝑍0superscript𝑍00{Z}^{({0})}\leftarrow{Z}^{({0,0})}
  
   Z(1)←Z(0,1)+Z(1,0)←superscript𝑍1superscript𝑍01superscript𝑍10{Z}^{({1})}\leftarrow{Z}^{({0,1})}+{Z}^{({1,0})}
  
   Z(2)←Z(0,2)+(Z(1,1)+Z(2,0))←superscript𝑍2superscript𝑍02superscript𝑍11superscript𝑍20{Z}^{({2})}\leftarrow{Z}^{({0,2})}+({Z}^{({1,1})}+{Z}^{({2,0})})
  
   Z(3)←Z(1,2)+Z(2,1)←superscript𝑍3superscript𝑍12superscript𝑍21{Z}^{({3})}\leftarrow{Z}^{({1,2})}+{Z}^{({2,1})}
  
   Z(4)←Z(2,2)←superscript𝑍4superscript𝑍22{Z}^{({4})}\leftarrow{Z}^{({2,2})}
  
We use the lower case z𝑧z to denote the corresponding exact values. For example
z(2)=z(0,2)+z(1,1)+z(2,0)superscript𝑧2superscript𝑧02superscript𝑧11superscript𝑧20{z}^{({2})}={z}^{({0,2})}+{z}^{({1,1})}+{z}^{({2,0})} and
z(1,1)=∑ℓ=1nxℓ(1)​yℓ(1)superscript𝑧11superscriptsubscriptℓ1𝑛superscriptsubscript𝑥ℓ1superscriptsubscript𝑦ℓ1{z}^{({1,1})}=\sum\_{\ell=1}^{n}{x\_{\ell}}^{({1})}{y\_{\ell}}^{({1})}.
A simple sum that offers close-to-FP32 accuracy is to compute in FP32 arithmetic

|  |  |  |
| --- | --- | --- |
|  | Z2←Z(0)+(Z(1)+Z(2))←subscript𝑍2superscript𝑍0superscript𝑍1superscript𝑍2Z\_{2}\leftarrow{Z}^{({0})}+({Z}^{({1})}+{Z}^{({2})}) |  |

A sum that might be able to offer higher accuracy than FP32 is to compute in FP32 arithmetic

|  |  |  |
| --- | --- | --- |
|  | Z3←Z(0)+(Z(1)+(Z(2)+Z(3)))←subscript𝑍3superscript𝑍0superscript𝑍1superscript𝑍2superscript𝑍3Z\_{3}\leftarrow{Z}^{({0})}+({Z}^{({1})}+({Z}^{({2})}+{Z}^{({3})})) |  |

### II-E General error bound on Z2subscript𝑍2Z\_{2}

Recall that a recursive sum S𝑆S, computed in FP32, of n𝑛n items whose exact sum is s𝑠s satisfies
|S−s|≤γf,n​s^𝑆𝑠subscript𝛾

𝑓𝑛^𝑠|S-s|\leq\gamma\_{f,{n}}\hat{{s}}. Note also that γf,m+γf,n≤γf,n+msubscript𝛾

𝑓𝑚subscript𝛾

𝑓𝑛subscript𝛾

𝑓𝑛𝑚\gamma\_{f,{m}}+\gamma\_{f,{n}}\leq\gamma\_{f,{n+m}}. Applying this
we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | |Z(i,j)−z(i,j)|superscript𝑍𝑖𝑗superscript𝑧𝑖𝑗\displaystyle|{Z}^{({i,j})}-{z}^{({i,j})}| | ≤γf,n​εbi+j​z^absentsubscript𝛾  𝑓𝑛superscriptsubscript𝜀𝑏𝑖𝑗^𝑧\displaystyle\leq\gamma\_{f,{n}}\varepsilon\_{b}^{i+j}\hat{{z}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |Z(i,j)|superscript𝑍𝑖𝑗\displaystyle|{Z}^{({i,j})}| | ≤(1+γf,n)​εbi+j​z^≤1.01​εbi+j​z^absent1subscript𝛾  𝑓𝑛superscriptsubscript𝜀𝑏𝑖𝑗^𝑧1.01superscriptsubscript𝜀𝑏𝑖𝑗^𝑧\displaystyle\leq(1+\gamma\_{f,{n}})\varepsilon\_{b}^{i+j}\hat{{z}}\leq 1.01\varepsilon\_{b}^{i+j}\hat{{z}} |  |

Similarly

|  |  |  |  |
| --- | --- | --- | --- |
|  | |Z(0)−z(0)|superscript𝑍0superscript𝑧0\displaystyle|{Z}^{({0})}-{z}^{({0})}| | ≤γf,n​z^absentsubscript𝛾  𝑓𝑛^𝑧\displaystyle\leq\gamma\_{f,{n}}\hat{{z}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |Z(1)−(Z(0,1)+Z(1,0)|\displaystyle|{Z}^{({1})}-({Z}^{({0,1})}+{Z}^{({1,0})}| | ≤γf,1​(|Z(0,1)|+|Z(1,0)|)absentsubscript𝛾  𝑓1superscript𝑍01superscript𝑍10\displaystyle\leq\gamma\_{f,{1}}(|{Z}^{({0,1})}|+|{Z}^{({1,0})}|) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤2.02​γf,1​εbabsent2.02subscript𝛾  𝑓1subscript𝜀𝑏\displaystyle\leq 2.02\gamma\_{f,{1}}\varepsilon\_{b} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |Z(2)−(Z(0,2)+Z(1,1)+Z(2,0))|superscript𝑍2superscript𝑍02superscript𝑍11superscript𝑍20\displaystyle|{Z}^{({2})}-({Z}^{({0,2})}+{Z}^{({1,1})}+{Z}^{({2,0})})| | ≤3.03​γf,2​εb2absent3.03subscript𝛾  𝑓2superscriptsubscript𝜀𝑏2\displaystyle\leq 3.03\gamma\_{f,{2}}\varepsilon\_{b}^{2} |  |

Consequently, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i=02|Z(i)−z(i)|superscriptsubscript𝑖02superscript𝑍𝑖superscript𝑧𝑖\displaystyle\sum\_{i=0}^{2}|{Z}^{({i})}-{z}^{({i})}| | ≤(γf,n+2.02​γf,1​εf+3.03​γf,2​εb2)​z^absentsubscript𝛾  𝑓𝑛2.02subscript𝛾  𝑓1subscript𝜀𝑓3.03subscript𝛾  𝑓2superscriptsubscript𝜀𝑏2^𝑧\displaystyle\leq(\gamma\_{f,{n}}+2.02\gamma\_{f,{1}}\varepsilon\_{f}+3.03\gamma\_{f,{2}}\varepsilon\_{b}^{2})\hat{{z}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤1.01​γf,n​z^absent1.01subscript𝛾  𝑓𝑛^𝑧\displaystyle\leq 1.01\gamma\_{f,{n}}\hat{{z}} |  |

We can now estimate |Z2−z|subscript𝑍2𝑧|Z\_{2}-z| which is the error we get in computing the inner product using BF16
and gather only up to the second order partial inner products. The error consist of truncation error
in ignoring a couple of the partial inner products and also the rounding errors in computing Z2subscript𝑍2Z\_{2}.

|  |  |  |  |
| --- | --- | --- | --- |
|  | |Z2−z|subscript𝑍2𝑧\displaystyle|Z\_{2}-z| | ≤|Z2−(z(0)+z(1)+z(2))|+|z(3)+z(4)|absentsubscript𝑍2superscript𝑧0superscript𝑧1superscript𝑧2superscript𝑧3superscript𝑧4\displaystyle\leq|Z\_{2}-({z}^{({0})}+{z}^{({1})}+{z}^{({2})})|+|{z}^{({3})}+{z}^{({4})}| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤|Z2−(z(0)+z(1)+z(2))|+1.01​εb3​z^absentsubscript𝑍2superscript𝑧0superscript𝑧1superscript𝑧21.01superscriptsubscript𝜀𝑏3^𝑧\displaystyle\leq|Z\_{2}-({z}^{({0})}+{z}^{({1})}+{z}^{({2})})|+1.01\varepsilon\_{b}^{3}\hat{{z}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤|Z2−(Z(0)+Z(1)+Z(2))|absentsubscript𝑍2superscript𝑍0superscript𝑍1superscript𝑍2\displaystyle\leq|Z\_{2}-({Z}^{({0})}+{Z}^{({1})}+{Z}^{({2})})| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑i=02|Z(i)−z(i)|+1.01​εb3​z^superscriptsubscript𝑖02superscript𝑍𝑖superscript𝑧𝑖1.01superscriptsubscript𝜀𝑏3^𝑧\displaystyle\hskip 8.61108pt+\sum\_{i=0}^{2}|{Z}^{({i})}-{z}^{({i})}|+1.01\varepsilon\_{b}^{3}\hat{{z}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤γf,2​∑i=02|Z(i)|+1.01​γf,n​z^+1.01​εb3​z^absentsubscript𝛾  𝑓2superscriptsubscript𝑖02superscript𝑍𝑖1.01subscript𝛾  𝑓𝑛^𝑧1.01superscriptsubscript𝜀𝑏3^𝑧\displaystyle\leq\gamma\_{f,{2}}\sum\_{i=0}^{2}|{Z}^{({i})}|+1.01\gamma\_{f,{n}}\hat{{z}}+1.01\varepsilon\_{b}^{3}\hat{{z}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤1.01​(γf,n+2+εf3)​z^.absent1.01subscript𝛾  𝑓𝑛2superscriptsubscript𝜀𝑓3^𝑧\displaystyle\leq 1.01(\gamma\_{f,{n+2}}+\varepsilon\_{f}^{3})\hat{{z}}. |  |

The above shows that in general the worst case bound on using BF16 is slightly worse than using FP32.
This cannot be corrected by using more terms as the factor γf,n+1subscript𝛾

𝑓𝑛1\gamma\_{f,{n+1}} is dominant, slightly worse than
γf,nsubscript𝛾

𝑓𝑛\gamma\_{f,{n}}. There is a special case, however, in which Z2subscript𝑍2Z\_{2} can be significantly more accurate.
This is the situation when Z(0)=z(0)superscript𝑍0superscript𝑧0{Z}^{({0})}={z}^{({0})}. That is, the computation of
∑ℓ=1nxℓ(0)​yℓ(0)superscriptsubscriptℓ1𝑛superscriptsubscript𝑥ℓ0superscriptsubscript𝑦ℓ0\sum\_{\ell=1}^{n}{x\_{\ell}}^{({0})}{y\_{\ell}}^{({0})} is exact, This is quite possible as each
product xℓ(0)​yℓ(0)superscriptsubscript𝑥ℓ0superscriptsubscript𝑦ℓ0{x\_{\ell}}^{({0})}{y\_{\ell}}^{({0})} only has at most 16 significant bits and that we are
accumulating into an FP32 number, which holds 24 significant bits. The exact sum’s magnitude
is clearly less than n​maxℓ⁡|xℓ(0)​yℓ(0)|𝑛subscriptℓsuperscriptsubscript𝑥ℓ0superscriptsubscript𝑦ℓ0n\max\_{\ell}|{x\_{\ell}}^{({0})}{y\_{\ell}}^{({0})}|. As long as the least significant
bit position of minℓ⁡|xℓ(0)​yℓ(0)|subscriptℓsuperscriptsubscript𝑥ℓ0superscriptsubscript𝑦ℓ0\min\_{\ell}|{x\_{\ell}}^{({0})}{y\_{\ell}}^{({0})}| is not farther than 23 bits away, the sum
will be exact. A mathematical relationship that implies this situation is

|  |  |  |
| --- | --- | --- |
|  | ⌈log2⁡(1.01​maxℓ⁡|xℓ​yℓ|)⌉−23≤⌈log2⁡(0.99​minℓ⁡|xℓ​yℓ|)⌉−15subscript21.01subscriptℓsubscript𝑥ℓsubscript𝑦ℓ23subscript20.99subscriptℓsubscript𝑥ℓsubscript𝑦ℓ15\lceil\log\_{2}(1.01\,\max\_{\ell}|x\_{\ell}y\_{\ell}|)\rceil-23\leq\lceil\log\_{2}(0.99\,\min\_{\ell}|x\_{\ell}y\_{\ell}|)\rceil-15 |  |

If this holds, we have Z(0)−z(0)=0superscript𝑍0superscript𝑧00{Z}^{({0})}-{z}^{({0})}=0 and the previous error bound reduces to

|  |  |  |
| --- | --- | --- |
|  | |Z2−z|≤1.01​(γf,2+εb3)​z^≤1.01​γf,3​z^subscript𝑍2𝑧1.01subscript𝛾  𝑓2superscriptsubscript𝜀𝑏3^𝑧1.01subscript𝛾  𝑓3^𝑧|Z\_{2}-z|\leq 1.01\,(\gamma\_{f,{2}}+\varepsilon\_{b}^{3})\hat{{z}}\leq 1.01\,\gamma\_{f,{3}}\hat{{z}} |  |

### II-F Worse Case Error for combined BF16 Datatypes

The worst error that can occur with this method is when the original FP32 number, a𝑎a, is very close to zero, that is contains a large negative exponent near the exponent boundary of FP32 (like -126, since -127 is reserved for denormals). What happens then is the conversion from FP32 to BF16 for the first number will be alright (b(0)=(ℬ16)​asuperscript𝑏0subscriptℬ16𝑎{b}^{({0})}=({\cal B}\_{16})a), but the second BF16 number b(1)superscript𝑏1{b}^{({1})} will be with an exponent shifted left by 8, and the third BF16 number b(2)superscript𝑏2{b}^{({2})} will be with an exponent shifted left by 16. In which case, b(1)=b(2)=0.0superscript𝑏1superscript𝑏20.0{b}^{({1})}={b}^{({2})}=0.0. Let’s assume that a𝑎a has many nonzero bits in the mantissa, but all but one of them are in positions 0-15. If that’s the case, then all those bits will be lost when we determine b(1)=n(2)=0.0superscript𝑏1superscript𝑛20.0{b}^{({1})}={n}^{({2})}=0.0.

The error in this case is the worst because a𝑎a and b(0)superscript𝑏0{b}^{({0})} will only have 8 mantissa bits in common, and so any product that uses a𝑎a might only have 2-3 digits of accuracy and the rest of the product will be off. Again, this is the worst case scenario and only seems to happen when the exponents are large and negative. As long as the exponent of a𝑎a is at least no smaller than -110, then we can form b(1)superscript𝑏1{b}^{({1})} and b(2)superscript𝑏2{b}^{({2})} within the FP32 threshold. So a ”bad” number to try with this method would probably have a small value in exponent bit fields 30-23 (like 00000001), so that the exponent bias pushes this to an extreme negative number, and perhaps a 1 in the bit field 16, and zeros in 22-17, and then bits 0-15 are all 1s, like ≈1.1939⋅10−38absent⋅1.1939superscript1038\approx 1.1939\cdot 10^{-38}.

For this reason, routines in LAPACK like DLATRS which depend on scaling and shifting triangular matrices to prevent denormals often keep track of the magnitude of numbers, avoiding the biggest and smallest by scaling the data.
They typically use constants close to the exponent range. To fully make use of such a routine, it’d be wise to use a pretend range of [−110,127]110127[-110,127].

### II-G Possible Shortcuts when using three-way and two-way BF16 combined Datatypes

Following the previous general error analysis of Sec. [II-E](#S2.SS5 "II-E General error bound on Z₂ ‣ II Combined Lower Precision Datatypes And Their Application to BLAS and LAPACK Routines ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations"),
we can now have a more detailed look on saving operations. Because the
number of significant bits of BF16 are 8, we expect that |a(1)|<=2−8​|a(0)|superscript𝑎1superscript28superscript𝑎0|{a}^{({1})}|<=2^{-8}|{a}^{({0})}| and |a(2)|<=2−16​|a(0)|superscript𝑎2superscript216superscript𝑎0|{a}^{({2})}|<=2^{-16}|{a}^{({0})}|. While we won’t
know in general how the a𝑎a-terms compare with the b𝑏b-terms, we do know this
puts these terms into five separate bins with a(0)⋅b(0)⋅superscript𝑎0superscript𝑏0{a}^{({0})}\cdot{b}^{({0})}
as our primary, most significant term and in its own bin. The other four bins
are:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |a(0)⋅b(1)|,|a(1)⋅b(0)|≤  ⋅superscript𝑎0superscript𝑏1⋅superscript𝑎1superscript𝑏0 absent\displaystyle|{a}^{({0})}\cdot{b}^{({1})}|,|{a}^{({1})}\cdot{b}^{({0})}|\leq | 2−8​|a(0)⋅b(0)|superscript28⋅superscript𝑎0superscript𝑏0\displaystyle 2^{-8}|{a}^{({0})}\cdot{b}^{({0})}| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |a(1)⋅b(1)|,|a(0)⋅b(2)|,|a(2)⋅b(0)|≤  ⋅superscript𝑎1superscript𝑏1⋅superscript𝑎0superscript𝑏2⋅superscript𝑎2superscript𝑏0 absent\displaystyle|{a}^{({1})}\cdot{b}^{({1})}|,|{a}^{({0})}\cdot{b}^{({2})}|,|{a}^{({2})}\cdot{b}^{({0})}|\leq | 2−16​|a(0)⋅b(0)|superscript216⋅superscript𝑎0superscript𝑏0\displaystyle 2^{-16}|{a}^{({0})}\cdot{b}^{({0})}| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |a(1)⋅b(2)|,|a(2)⋅b(1)|≤  ⋅superscript𝑎1superscript𝑏2⋅superscript𝑎2superscript𝑏1 absent\displaystyle|{a}^{({1})}\cdot{b}^{({2})}|,|{a}^{({2})}\cdot{b}^{({1})}|\leq | 2−24​|a(0)⋅b(0)|superscript224⋅superscript𝑎0superscript𝑏0\displaystyle 2^{-24}|{a}^{({0})}\cdot{b}^{({0})}| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |a(2)⋅b(2)|≤⋅superscript𝑎2superscript𝑏2absent\displaystyle|{a}^{({2})}\cdot{b}^{({2})}|\leq | 2−32​|a(0)⋅b(0)|superscript232⋅superscript𝑎0superscript𝑏0\displaystyle 2^{-32}|{a}^{({0})}\cdot{b}^{({0})}| |  |

Let’s define E=a(1)⋅b(2)+a(2)⋅b(1)+a(2)⋅b(2)𝐸⋅superscript𝑎1superscript𝑏2⋅superscript𝑎2superscript𝑏1⋅superscript𝑎2superscript𝑏2E={a}^{({1})}\cdot{b}^{({2})}+{a}^{({2})}\cdot{b}^{({1})}+{a}^{({2})}\cdot{b}^{({2})}. This E𝐸E term is the difference between computing our triplet with 6 multiplies with only the most significant bins, and 9 multiplies with all bins. The first observation is while |a(1)|<=|a(0)|/256superscript𝑎1superscript𝑎0256|{a}^{({1})}|<=|{a}^{({0})}|/256, in which case equality can and does sometimes happen, it is usually a smaller term. But first, plugging in this observation into the above equation for E𝐸E simplifies to E≤|c∗a(0)​b(0)|⋅2−23𝐸⋅𝑐superscript𝑎0superscript𝑏0superscript223E\leq|c\*{a}^{({0})}{b}^{({0})}|\cdot 2^{-23} where c=513/512.513512{513/512}.

We start by making our bounds on |a(0)|superscript𝑎0|{a}^{({0})}| and |b(0)|superscript𝑏0|{b}^{({0})}| more rigorous. If we assume that exponents are larger than −110110-110 (see the previous section for why), and the numbers are uniformly distributed in a given range, we can show that the expected average value of |a(1)|≈|a(0)/768|.superscript𝑎1superscript𝑎0768|{a}^{({1})}|\approx|{a}^{({0})}/768|. Note that d∗|a(0)|<|a(1)|𝑑superscript𝑎0superscript𝑎1d\*|{a}^{({0})}|<|{a}^{({1})}| for some d𝑑d. In particular, d𝑑d is in (2−9,2−8)superscript29superscript28(2^{-9},2^{-8}) with probability 1/2 if we assume a uniformly distributed range of data (we assume that the relevant bit can either be 0 or 1 if the data is uniformly distributed.) If the relevant bit isn’t helpful here, we can assume the next relevant bit will be and d𝑑d will lie in (2−10,2−9)superscript210superscript29(2^{-10},2^{-9}) with probability 1/4. And d𝑑d will lie in (2−11,2−10)superscript211superscript210(2^{-11},2^{-10}) with probability 1/8. Also note that the mantissa of |a(i)|superscript𝑎𝑖|{a}^{({i})}| is uniformly in [1,2)12[1,2), so we can cut our averages by a factor of 2. In particular, we have a series 2⋅(12+14​12+18​14+⋯)=13⋅21214121814⋯132\cdot(\frac{1}{2}+\frac{1}{4}\frac{1}{2}+\frac{1}{8}\frac{1}{4}+\cdots)=\frac{1}{3}, so in fact, on average, |a(1)|≈|a(0)/768|.superscript𝑎1superscript𝑎0768|{a}^{({1})}|\approx|{a}^{({0})}/768|. We can do a similar analysis and show that |a(2)|≈|a(0)/7682|.superscript𝑎2superscript𝑎0superscript7682|{a}^{({2})}|\approx|{a}^{({0})}/768^{2}|.

Now we see that, on average, |E|≈|a(0)​b(0)|⋅c𝐸⋅superscript𝑎0superscript𝑏0𝑐|E|\approx|{a}^{({0})}{b}^{({0})}|\cdot c where c≈4.412​e−9.𝑐4.412𝑒9c\approx 4.412e-9. So while in the worst case some of these 3 extra terms might be important, on the average they don’t matter.

This suggests some immediate short-cuts as well as ordering. That is, the terms should be added in the reverse order, so that the smallest terms come first. One can’t take the time to test whether |a(0)⋅b(1)|<=|a(1)⋅b(0)|⋅superscript𝑎0superscript𝑏1⋅superscript𝑎1superscript𝑏0|{a}^{({0})}\cdot{b}^{({1})}|<=|{a}^{({1})}\cdot{b}^{({0})}| but we do know that terms in the separate bins have the above relation. We also know that since three BF16s can only keep track of 24 implicit bits, it might not be worthwhile to compute the terms in the last two bins, saving some work. That doesn’t mean it won’t be worthwhile, however. For instance, consider a case like x=0.57892173110418099213 and y=-7447.6596637651937272. Because y is so much larger than x, the |a(1)⋅b(2)|⋅superscript𝑎1superscript𝑏2|{a}^{({1})}\cdot{b}^{({2})}| is significant even though it’s in the 2−24superscript2242^{-24} bin. And in this case, it’s necessary to compute that product if we want the BF16x3 error to be less than the real\*8 error, which means computing at least 7 products instead of just 6. Nevertheless, if we assume that the terms are equal in magnitude, one might expect the values each bin to be roughly equal, and then all things considered, the last two bins are not necessary. That is, if the input values are close enough, one can mimic single precision accuracy just by first adding the terms in the 2−16superscript2162^{-16} bin, then the terms in the 2−8superscript282^{-8} bin, and finally add that result to our most significant term a(0)⋅b(0)⋅superscript𝑎0superscript𝑏0{a}^{({0})}\cdot{b}^{({0})}.

The same idea can be applied if we wish to use a pair of BF16s instead of a triplet. Namely, we can skip all three terms in the 2−16superscript2162^{-16} bin. But again, this is only when we assume that the all these terms are relatively the same in magnitude. There’s no precise way to know in advance that skipping terms won’t be marginally bad. All we can know is that terms satisfy the above equations, to we have an handle on the worse potential error that could arrive.

This suggests that if have a single BF16, we need to do a single multiply: a(0)⋅b(0)⋅superscript𝑎0superscript𝑏0{a}^{({0})}\cdot{b}^{({0})}. If we have two BF16s, we need to do two additional multiplications (or three total) with the two products in the 2−8superscript282^{-8} bin. If we have three BF16s, we need to do another three additional multiplications (or six total) with the three products in the 2−16superscript2162^{-16} bin. In all cases, the bins should be added from smallest to largest. This is summarized in the following table.

|  |  |
| --- | --- |
| Number of BF16s | Number of Implicit Multiplies |
| 1 | 1 |
| 2 | 3 |
| 3 | 6 |

Another potential short-cut we’ve explored is assuming the two numbers have a different number of splits. That is, suppose we multiply (a(0),a(1))×(b(0),b(1),b(2))superscript𝑎0superscript𝑎1superscript𝑏0superscript𝑏1superscript𝑏2({a}^{({0})},{a}^{({1})})\times({b}^{({0})},{b}^{({1})},{b}^{({2})}). To get maximum accuracy, one might expect to do five multiplies, taking only the a(1),a(1)

superscript𝑎1superscript𝑎1{a}^{({1})},{a}^{({1})} and a(0),a(2)

superscript𝑎0superscript𝑎2{a}^{({0})},{a}^{({2})} terms from the 2−16superscript2162^{-16} bin. However, we really need to assume that a(1)=0superscript𝑎10{a}^{({1})}~{}=0 and dropping this term is okay, because neither of those two terms we in use in the 2−16superscript2162^{-16} bin will be sufficient to approximate FP32 accuracy otherwise.

Note, all of these combinations open up an avenue for novel performance
optimization techniques in numeric applications, in particular dense linear
algebra. For both operands, often matrices A𝐴A and B𝐵B we can now employ
different decompositions into lower precision datatypes to fulfill the
application’s need for accuracy and balance it with execution speed.

## III Speeding Up One-Sided Solvers With low-precision Datatypes using Residual Formulation

Mixed precision high performance computing is showing up more and more [[8](#bib.bib8)].

One key benefit in this paper is the combining of low-precision data types in order to get higher precision. But there’s another benefit, and it’s historically what most people think of first with regard to lower precision, because it’s what has been known about for years. Namely, for some problems that contain an obvious residual equation, one can first solve the problem in lower precision, and hopefully faster, and then iteratively refine on higher precision. If it works, and it may not, the final answer will be just as accurate as if higher precision had been used the entire time. And if the bulk of the operations are in lower precision, hopefully the computation will also run faster.

This is most commonly done when solving a system of equations [[9](#bib.bib9)], like A​x=b𝐴𝑥𝑏Ax=b using Gaussian Elimination in a L​U𝐿𝑈LU factorization. It’s a common thread in numerical analysis to first solve the problem in lower precision, then compute the residual in higher precision, r=A∗x−b𝑟𝐴𝑥𝑏r=A\*x-b, and use the results from the lower precision solver to solve the updated system A​y=r𝐴𝑦𝑟Ay=r (just use the L​U𝐿𝑈LU factorization, but do the solve in a higher precision like FP64 as well), and then update x𝑥x with x+y𝑥𝑦x+y. Only the cubic work of the initial L​U𝐿𝑈LU factorization should be done in the lower precision: all the other steps, which are all quadratic instead of cubic, should be done in the higher precision. Asymptotically, the cubic lower-precision work should dominate the time, but the accuracy (if the technique works) should approach FP64. This method is called Iterative Refinement on L​U𝐿𝑈LU, and tends to break down when the matrix condition number is large compared to the machine epsilon of the L​U𝐿𝑈LU work. That is, for a fixed matrix condition range, the method will tend to work more often for FP32 than FP16, and more often for FP16 than BF16.

In some cases, researchers have found that combining Iterative Refinement with an Iterative Solver like GMRES [[10](#bib.bib10)][[11](#bib.bib11)] is also beneficial, especially when the base precision is very low because the odds are that the matrix may have too high a condition number to work otherwise.

## IV Experimental Results

### IV-A SGEMM and SGETRF using combined BF16 Datatypes

We did a complete GEMM (GEneral Matrix-matrix Multiply) implementation which starts with FP32 data and behind the scenes converts it into one to three bfloat16 matrices and then does one to nine products with these matrices and adds it all up again. We also experimented with iterative refinement for LU, comparing FP32, with FP16 and bfloat16.

For our GEMM testing, we wanted to test three different use cases. First, just the simple case where the exponent range is small because all the numbers are within a close range (like [-1,1].) Second, where the exponent range is huge because the exponents are randomly chosen bits as well as the mantissa bits- in this case, we want numbers arbitrarily close to zero or arbitrarily close to Inf or -Inf. Third, a more ”medium” case, where we assign the exponent range to be a Gaussian distribution so that the exponents can sometimes be large, but most of the time they are small and reasonable, so we see a case where the exponents can vary, just not likely. Our theoretical understanding suggests that the small range case should do best for BF16s, and that some of the cases where the range is large shouldn’t go as well as it did in single precision. This is precisely what we find.

Figure [2](#S4.F2 "Figure 2 ‣ IV-A SGEMM and SGETRF using combined BF16 Datatypes ‣ IV Experimental Results ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations") computes the ”baseline” result via DGEMM (real\*8 GEMM), and computes the relative error of that versus four different methods: using a pair of BF16s and three products, using Intel Math Kernel Library’s (Intel MKL [[12](#bib.bib12)]) SGEMM which is a FP32 general matrix-matrix multiply, using a triplet of BF16s and six products and adding those results together in FP32, and the same as the last but adding the final results together in FP64. Unlike the next experiment, this was only for a narrow range of data [-1,1]. The results are as we expected, and the order of accuracy was the order stated here.

Figure 2: Various GEMM (C=A×B𝐶𝐴𝐵C=A\times B) average relative error vs. DGEMM (‖A×B−DGEMM‖F/‖DGEMM‖Fsubscriptnorm𝐴𝐵DGEMM𝐹subscriptnormDGEMM𝐹\left\|A\times B-\text{DGEMM}\right\|\_{F}/\left\|\text{DGEMM}\right\|\_{F}) over 1000 runs compared to original fp64 data in [-1.0,1.0] range with drand48() randomization. bxA\_B[d] means breaking each matrix up into A bfloat16 matrices, doing B products. Optionally, collect the final answer with ”d” (double precision) or not.

Figure [3](#S4.F3 "Figure 3 ‣ IV-A SGEMM and SGETRF using combined BF16 Datatypes ‣ IV Experimental Results ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations") computes the ”baseline” result using FP64 DGEMM (real\*8 GEMM), and computes the relative error of that versus either SGEMM or a triplet of BF16s done with six products. We only show the relative error because we have used special generation to uniformly create arbitrary exponents, so the absolute errors were sometimes huge (over 1020superscript102010^{20}.) With this wide range of exponents, SGEMM actually did better (marginally) over the BF16s, but it’s still very comparable.

Figure 3: Same as Fig. [2](#S4.F2 "Figure 2 ‣ IV-A SGEMM and SGETRF using combined BF16 Datatypes ‣ IV Experimental Results ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations"), but with maximal exponent distribution (huge range).

Figure [3](#S4.F3 "Figure 3 ‣ IV-A SGEMM and SGETRF using combined BF16 Datatypes ‣ IV Experimental Results ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations") exaggerates the variance, and looking closely at the vertical axis, one sees that both methods are nearly identical even when the exponent range of the data is huge.

Finally, our last GEMM case study is when the exponents have a Gaussian distribution instead of a Uniform distribution. In this case, the exponent bits were set via calling the Vector Statistical Library with ”VSL RNG METHOD GAUSSIAN BOXMULLER”) in Intel MKL[[12](#bib.bib12)]. So the exponents could be wide, but statistically that was unlikely, to give us more of a medium range exponent distribution as opposed to the last two examples. Again, the SGEMM and BF16 results were separately compared against DGEMM’s answer like the other two cases. Figure [4](#S4.F4 "Figure 4 ‣ IV-A SGEMM and SGETRF using combined BF16 Datatypes ‣ IV Experimental Results ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations") contains these results.

Figure 4: Same as Fig. [2](#S4.F2 "Figure 2 ‣ IV-A SGEMM and SGETRF using combined BF16 Datatypes ‣ IV Experimental Results ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations"), but with Gaussian exponent distribution (medium range).

In Figure  [4](#S4.F4 "Figure 4 ‣ IV-A SGEMM and SGETRF using combined BF16 Datatypes ‣ IV Experimental Results ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations"), we see that this technique appears on average worse than SGEMM results, however the gap seems to be smaller than the wide-range case in Figure  [3](#S4.F3 "Figure 3 ‣ IV-A SGEMM and SGETRF using combined BF16 Datatypes ‣ IV Experimental Results ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations").

The next curve in Figure [5](#S4.F5 "Figure 5 ‣ IV-A SGEMM and SGETRF using combined BF16 Datatypes ‣ IV Experimental Results ‣ Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations") was one doing an entire FP32 LU decomposition (Gaussian Elimination), in one case using SGETRF ([[10](#bib.bib10)]) from Intel(R) MKL (which is based on FP32 GEMM) and in the other case using a SGETRF based on triplets of BF16s and six products. Because this curve shows both small range data and large range data, we simply things just by showing the ratio of the relative errors. In every case, the triplet of BF16s was more accurate. The comparison points were results from DGETRF on the same input data.

Figure 5: SGETRF vs BFLOAT16x3\_6 LU Decomposition: Element errors average improvement over a 100 runs for N×N𝑁𝑁N\times N square matrices with an extremely large range [−1.010,1.010]superscript1.010superscript1.010[-1.0^{10},1.0^{10}] and matrices with a small range [−1.0,1.0]1.01.0[-1.0,1.0]

### IV-B Iterative Refinement

For iterative refinement on solutions to A​x=b𝐴𝑥𝑏Ax=b with LU, using lower precision tends to work only for well-conditioned matrices, where the lower the precision, the more stringent conditioning is needed.

We ran 100 tests for unsymmetric dense matrices of order 50, setting the condition number and using a residual tolerance of c​o​n​d​(A)∗e​p​s𝑐𝑜𝑛𝑑𝐴𝑒𝑝𝑠cond(A)\*eps. When the condition number grows, it appears that using a single bfloat16 (as opposed to the triplet discussed elsewhere) instead of FP32 gets more and more risky.

| Precision | Condition # | % Converged | Ave. iterations |
| --- | --- | --- | --- |
| FP32 | 10 | 100 | 3.47 |
| BF16 | 10 | 45 | 39.3556 |
| FP16 | 10 | 90 | 16.2667 |
| FP32 | 100 | 100 | 2.67 |
| BF16 | 100 | 32 | 41.125 |
| FP16 | 100 | 91 | 16.989 |
| FP32 | 1000 | 100 | 2.49 |
| BF16 | 1000 | 29 | 47.0345 |
| FP16 | 1000 | 89 | 19.4831 |
| FP32 | 10000 | 100 | 2.39 |
| BF16 | 10000 | 21 | 48.4286 |
| FP16 | 10000 | 91 | 13.5604 |

For row and column diagonally dominant unsymmetric matrices trying to solve A​x=b𝐴𝑥𝑏Ax=b, one can also apply GMRES instead of iterative refinement, and use the LU decomposition in the lower precision from the last table as a pre-conditioner. We used the same tolerance as before and again 100 tests, but this time varied the sizes n𝑛n of the matrices instead of the condition number.

| Precision | n | % Converged | Ave. iterations |
| --- | --- | --- | --- |
| FP32 | 10 | 100 | 2.0 |
| BF16 | 10 | 100 | 6.59 |
| FP16 | 10 | 100 | 4.24 |
| FP32 | 50 | 100 | 2.0 |
| BF16 | 50 | 100 | 7.0 |
| FP16 | 50 | 100 | 5.0 |
| FP32 | 100 | 100 | 2.0 |
| BF16 | 100 | 100 | 7.0 |
| FP16 | 100 | 100 | 5.0 |

We see that if the matrix is diagonally dominant, then using GMRES with the LU as a pre-conditioner allows for faster convergence and the method is more reliable.

## V Performance Ramifications

We can only estimate performance at this early stage or rely on data reported
on NVIDIA hardware with FP16 inputs, but not BF16 as bare-metal programmable
BF16 hardware is not yet available.
Timing is broken down into three parts:
the conversion of data into BF16 parts (which has N2superscript𝑁2N^{2} complexity for SGEMM), the products
involved in the computation (which has N3superscript𝑁3N^{3} complexity for SGEMM), and the final additions
in the end to get the final answer (which are free as we assume FMA hardware and
we chain the products). While we studied the accuracy of the SGEMM and SGETRF, the target goal is accelerating mainly all compute bound dense linear algebra
functions in BLAS and LAPACK. Therefore, the aforementioned complexities are
always true and we can assume that the splitting can be hidden behind the computations on modern out-of-order/threaded hardware. That means the middle step, the low
precision partial matrix multiplications will dominate.

We know today that NVIDIA Volta has 10x more FLOPS in FP16 and it would be even
higher with BF16. The area of FP-FMA is dominated by the multiplier as it roughly
grows squared with mantissa size (and therefore also consumes a lot of power). That
means this area can be approximated for FP32 as 242=576superscript24257624^{2}=576 area-units where as BF16 requires only 82=64superscript82648^{2}=64 area-units. So BF16 is roughly 10×\times smaller using this first order approximation. Additionally,
machine learning pushes the hardware vendors to implement dataflow engines (e.g. NVIDIA’s
Tensorcores or Google’s TPU), also know as
systolic arrays, for efficient matrix computations with dense FLOPS. Therefore we can see that
8-32×\times more FLOPS than the classic FP32 FLOPS within the same silicon area are possible for the right matrix computations.

The presented approached matches FP32 accuracy for important dense linear
algebra routines with 6×\times more low-precision computations. This now opens a
wide range of optimization opportunities for hardware vendors. First FP32-like
dense linear algebra computation can be several times faster (when splitting can
be hidden):

|  |  |
| --- | --- |
| BF16 density over FP32 | projected Speed-Up over FP32 |
| 8 | ≤1.3=8/6absent1.386\leq 1.3=8/6 |
| 16 | ≤2.7=16/6absent2.7166\leq 2.7=16/6 |
| 32 | ≤5.2=32/6absent5.2326\leq 5.2=32/6 |

The performance results in [[5](#bib.bib5)] show that the assumptions made
here are correct. Similar Speed-Ups are also possible in iterative
refinement scenarios [[10](#bib.bib10)].

Apart from having faster “FP32” on general purpose hardware such as CPUs and/or
GPUs, it also means that deep learning optimized hardware, such as Google’s TPU
could be efficiently used for classic HPC which only requires FP32. Only the support
for splitting a FP32 number into multiple BF16 needs to be provided. There is no
need for native FP32 FMA units, a mixed precision BF16-FP32 FMA unit is sufficient. People have been proposing using mixed precision to refine other problems like eigenvalue problems for years such as in [[13](#bib.bib13)]. More recently,
there has been success with FP32 Eigenvalue solvers which are compute intensive
and are the bottleneck in quantum chemistry problems[[14](#bib.bib14)]. These applications
consume a huge fraction of large super-computers. Using the presented approach,
we can use BF16 hardware without FP32 support for computation with single
precision comparable accuracy.

## VI Conclusions

Lower precision units like BF16 and FP16 are starting to appear with accelerated performance due to machine learning pushes. Normally, FP32 is twice as fast as FP64, but a smaller precision may widen that performance gap. This means more scientists will wish to exploit the faster calculations. We expect BF16/FP16 systolic arrays to
provide 8-32×\times more compute potential than a classic FP32 vector compute engine.

Multiple combined BF16 have comparable accuracy (possibly better) when compared to FP32 and if a matrix-multiply can be implemented fast in terms of BF16, then it can be faster alternative to FP32’s matrix-multiply (SGEMM) as well. We have shown a line of sight to up to 5.2×\times faster dense linear algebra computations. Furthermore, nearly every processor is designed with FP32 these days, but this opens the door to an alternative; namely, if the processor has a fast BF16 or FP16 unit already, it may be able to emulate a lot of FP32 work, without providing extra FP32 FMA hardware. This alternative
is beneficial for deep learning optimized hardware.

Mixed precision computation such as iterative refinement is a surging area of research because scientists will want to exploit a much faster lower precision. If the bulk of the work can be done faster, then perhaps the overall problem can be done faster.

In general, people used to think “less precision per element” means less overall accuracy. This paper shows that folly in that thinking. Not only can, in some cases, a smaller precision unit be combined to achieve higher accuracy, but also refinement techniques can be developed that ultimately converge to higher accuracy. Since
these lower precision units allow for much denser packing on silicon, classic
higher precision compute units can be outperformed performance-wise while
still delivering high precision numeric results.

## References

* [1]

  “Tensorflow development summit,” March 30 2018.
* [2]

  *BFLOAT16 – Hardware Numerics Definition*.   Santa Clara, USA: Intel Corporation, 2018.
* [3]

  P. Micikevicius, S. Narang, J. Alben, G. F. Diamos, E. Elsen, D. García,
  B. Ginsburg, M. Houston, O. Kuchaiev, G. Venkatesh, and H. Wu, “Mixed
  precision training,” *CoRR*, vol. abs/1710.03740, 2017.
* [4]

  D. H. B. Yozo Hida, Xiaoye S Li, “Library for double-double and quad-double
  arithmetic,” 2007.
* [5]

  X. Cheng, A. Sorna, E. D’Azevedo, K. Wong, and S. Tomov, “Accelerating 2d fft:
  Exploit gpu tensor cores through mixed-precision,” 2018.
* [6]

  J. Dongarra, J. D. Croz, S. Hammarling, and I. Duff, “A set of level 3 basic
  linear algebra subprograms,” 1990.
* [7]

  E. Anderson, Z. Bai, C. Bischof, J. Demmel, J. Dongarra, J. D. Croz,
  A. Greenbaum, S. Hammarling, A. McKenney, and D. Sorenson, *LAPACK
  User’s Guide*.   Philadelphia, PA: SIAM
  Publications, 1992.
* [8]

  I. Yamazaki *et al.*, “Mixed-precision cholesky qr factorization and its
  case studies on multicore cpu with multiple gpus,” 2015, sIAM J. Sci.
  Comput., Volume 37, Issue 3, C307–C330.
* [9]

  E. Carson and N. Higham, “A new analysis of iterative refinement and its
  application to accurate solution of ill-conditioned sparse linear systems,”
  2017, sIAM J. SCI. COMPUT. Vol. 39, No. 6, pp. A2834–A2856.
* [10]

  A. Haidar, S. Tomov, J. Dongarra, and N. J. Higham, “Harnessing gpu tensor
  cores for fast fp16 arithmetic to speed up mixed-precision iterative
  refinement solvers,” 2018.
* [11]

  Barrett and B. et al, *Templates for the Solution of Linear Systems:
  Building Blocks for Iterative Methods*.   SIAM Publications, 1993.
* [12]

  *Intel Math Kernel Library. Reference Manual*.   Santa Clara, USA: Intel Corporation, iSBN 630813-054US.
* [13]

  J. Dongarra, C. Moler, and J. Wilkinson, “Improving the accuracy of computed
  eigenvalues and eigenvectors,” 1983, sIAM J. Numer. Anal. Vol. 20, No. 1.
* [14]

  A. Alvermann *et al.*, “Benefits from using mixed precision computations
  in the elpa-aeo and essex-ii eigensolver projects.”
