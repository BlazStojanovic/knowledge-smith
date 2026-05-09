---
source: paper
url: https://ai.stanford.edu/~ang/papers/icml04-l1l2.pdf
retrieved: 2026-05-06
title: Feature Selection, L1 vs. L2 Regularization, and Rotational Invariance
authors: Andrew Y. Ng
venue: ICML 2004
doi: 10.1145/1015330.1015435
license: unclear
extraction: docling 2.x (RapidOCR backend), 2026-05-06
---

## Andrew Y. Ng

Computer Science Department, Stanford University, Stanford, CA 94305, USA

## Abstract

We consider supervised learning in the presence of very many irrelevant features, and study two different regularization methods for preventing overfitting. Focusing on logistic regression, we show that using L 1 regularization of the parameters, the sample complexity (i.e., the number of training examples required to learn 'well,') grows only logarithmically in the number of irrelevant features. This logarithmic rate matches the best known bounds for feature selection, and indicates that L 1 regularized logistic regression can be effective even if there are exponentially many irrelevant features as there are training examples. We also give a lowerbound showing that any rotationally invariant algorithm-including logistic regression with L 2 regularization, SVMs, and neural networks trained by backpropagation-has a worst case sample complexity that grows at least linearly in the number of irrelevant features.

## 1. Introduction

We consider supervised learning in settings where there are many input features, but where there is a small subset of the features that is sufficient to approximate the target concept well.

In supervised learning settings with many input features, overfitting is usually a potential problem unless there is ample training data. For example, it is wellknown that for unregularized discriminative models fit via training-error minimization, sample complexity (i.e., the number of training examples needed to learn 'well') grows linearly with the VC dimension. Fur-

Appearing in Proceedings of the 21 st International Conference on Machine Learning , Banff, Canada, 2004. Copyright 2004 by the first author.

ther, the VC dimension for most models grows about linearly in the number of parameters (Vapnik, 1982), which typically grows at least linearly in the number of input features. Thus, unless the training set size is large relative to the dimension of the input, some special mechanism-such as regularization, which encourages the fitted parameters to be small-is usually needed to prevent overfitting.

In this paper, we focus on logistic regression, and study the behavior of two standard regularization methods when they are applied to problems with many irrelevant features. The first, L 1 regularization, uses a penalty term which encourages the sum of the absolute values of the parameters to be small. The second, L 2 regularization, encourages the sum of the squares of the parameters to be small. It has frequently been observed that L 1 regularization in many models causes many parameters to equal zero, so that the parameter vector is sparse. This makes it a natural candidate in feature selection settings, where we believe that many features should be ignored. For example, linear leastsquares regression with L 1 regularization is called the Lasso algorithm (Tibshirani, 1996), which is known to generally give sparse feature vectors. Another example of learning using L 1 regularization is found in (Zheng et al., 2004).

In this paper, we prove that for logistic regression with L 1 regularization, sample complexity grows only logarithmically in the number of irrelevant features (and at most polynomially in all other quantities of interest). Logistic regression with L 1 regularization is an appealing algorithm since it requires solving only a convex optimization problem. Further, the logarithmic dependence on the input dimension matches the best known bounds proved in various feature selection contexts (e.g., Ng, 1998; Ng &amp; Jordan, 2001; Littlestone, 1988; Helmbold et al., 1996; Kivinen &amp; Warmuth, 1994).

We also consider logistic regression with L 2 regularization. (E.g., Nigam et al., 1999). We show that this gives a rotationally invariant algorithm, and that any

## Feature selection, L 1 vs. L 2 regularization,

## and rotational invariance

ang@cs.stanford.edu rotationally invariant algorithm-which also includes SVMs, neural networks, and many other algorithmshas a worst case sample complexity that grows at least linearly in the number of irrelevant features, even if only a single feature is relevant. This suggests that these algorithms may not be effective in settings where only a few features are relevant, and the number of training examples is significantly smaller than the input dimension.

## 2. Preliminaries

We consider a supervised learning problem where we are given a set S = { ( x ( i ) , y ( i ) ) } i m =1 of m training examples drawn i.i.d. from some distribution D . Here, x ( i ) ∈ [ -1 , 1] n are the n -dimensional inputs, and y ( i ) ∈ { 0 , 1 } are the labels. For notational convenience, we assume that the last coordinate of the input vectors x ( i ) n = 1 always, so that the intercept term needs not be treated separately. We will focus on logistic regression, so our model will be

<!-- formula-not-decoded -->

where θ ∈ R n are the parameters of our model.

One way to describe regularized logistic regression is as the finding the parameters θ that solve following optimization problem:

<!-- formula-not-decoded -->

where R ( θ ) is a regularization term that is used to penalize large weights/parameters. If R ( θ ) ≡ 0, then this model is the standard, unregularized, logistic regression model with its parameters fit using the maximum likelihood criteria. If R ( θ ) = || θ || 1 = ∑ n i =1 | θ i | , then this is L 1 regularized logistic regression. If R ( θ ) = || θ || 2 2 = ∑ n i =1 θ 2 i , this is L 2 regularized logistic regression.

In the optimization problem in Equation (2), the parameter α ≥ 0 controls a tradeoff between fitting the data well, and having well-regularized/small parameters. In this paper, it will sometimes be useful to consider an alternative way of parameterizing this tradeoff. Specifically, we will also consider the constrained optimization problem:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For every solution θ to Equation (2) found using some particular value of α , there is some corresponding value of B in the optimization problem (3-4) that will give the same θ . Thus, these are two equivalent reparameterizations of the same problem. Readers familiar with convex analysis (Rockafellar, 1970) may also verify the equivalence between the two problems by noting that the Lagrangian for the constrained optimization (3-4) is exactly the objective in the optimization (2) (plus a constant that does not depend on θ ), where here α is the Lagrange multiplier. Thus, (3-4) may be solved by solving (2) for an appropriate α .

Because our logistic regression model is fit via (regularized) maximum likelihood, one natural metric for a fitted model's error is its negative loglikelihood (also called the 'logloss') on test data:

<!-- formula-not-decoded -->

Here, the subscript '( x, y ) ∼ D ' indicates that the expectation is with respect to a test example ( x, y ) drawn from D . Our main theoretical results regarding L 1 regularization will use this error metric. Given a dataset S , we also define the empirical logloss on S to be

<!-- formula-not-decoded -->

Sometimes, we will also be interested in the 0/1 misclassification error of our algorithm. We define

glyph[negationslash]

<!-- formula-not-decoded -->

where t is a threshold function ( t ( z ) = 1 if z ≥ 0 . 5, t ( z ) = 0 otherwise). The empirical 0/1 misclassification error ˆ ε m ( θ ) = ˆ ε m S ( θ ) is also defined analogously to be the fraction of examples in S that a model using parameter θ misclassifies.

It is straightforward to verify that, for the logistic regression model, we have ε l ( θ ) ≥ (log 2) · ε m ( θ ). Thus, an upper-bound on logloss also implies an upperbound on misclassification error, and a lower-bound on misclassification error (such as given in Section 4) also implies a lower-bound on logloss.

## 3. L 1 regularized logistic regression

We are interested in supervised learning problems where there is a very large number n of input features, but where there may be a small subset-say r glyph[lessmuch] n of them-that is sufficient to learn the target concept well. We will consider the following implementation of L 1 regularized logistic regression:

1. Split the data S into a training set S 1 consisting of the first (1 -γ ) m examples, and a hold-out cross

validation set S 2 consisting of the remaining γm examples.

2. For B = 0 , 1 , 2 , 4 , . . . , C ,

Fit a logistic regression model using the training set S 1 only, by solving the optimization problem (3-4) with the specified value of B . Call the resulting parameter vector θ B .

3. Among the θ B 's from Step 2, select and output the one with the lowest hold-out error on S 2 . I.e., pick θ = arg min i ∈{ 0 , 1 , 2 ,...,C } ˆ ε S 2 ( θ i )

Thus, this algorithm uses uses hold-out cross validation to select the regularization parameter B used in (3-4). In Step 3 of the algorithm, we did not exactly specify the error metric ˆ ε S 2 . If the goal is to minimize our expected logloss on the test data, it would make sense to use ˆ ε S 2 ( θ ) = ˆ ε l S 2 ( θ ) here. It will be this minimum logloss setting to which the theoretical results in this section apply. However, if the goal is to minimize 0/1 misclassification error, then it would also make sense to pick the θ i with the smallest misclassification error on the hold-out test set, and use ˆ ε S 2 ( θ ) = ˆ ε m S 2 ( θ ).

We want to show that if there is some hypothesis that attains low generalization error using only a small number r of features, then L 1 regularized logistic regression will attain performance that is (nearly) as good as that of this hypothesis, even if the training set is small.

Theorem 3.1: Let any glyph[epsilon1] &gt; 0 , δ &gt; 0 , C &gt; 0 , 0 &lt; γ &lt; 1 , K ≥ 1 , and m be fixed. Suppose there exists r indices 1 ≤ i 1 , i 2 , . . . , i r ≤ n , and a parameter vector θ ∗ ∈ R n such that only the r corresponding components of θ ∗ are non-zero, and | θ i j | ≤ K ( j = 1 , . . . , r ). Suppose further that C ≥ rK . Then, in order to guarantee that, with probability at least 1 -δ , the parameters ˆ θ output by our learning algorithm does nearly as well as θ ∗ , i.e., that

<!-- formula-not-decoded -->

it suffices that

<!-- formula-not-decoded -->

The main tools used to show this result are certain covering number bounds shown by (Bartlett, 1998; Zhang, 2002). The proof is given in Appendix A. This result shows that the sample complexity of our algorithmthat is, the number of training examples needed to learn 'well'-grows only logarithmically in the number of irrelevant features. Thus, logistic regression with L 1 regularization is capable of learning in problems even where the number of irrelevant features may be far larger than the training set size.

Space constraints preclude a full discussion, but we also note that C can be chosen automatically (as a function of m ) so that the same bound as stated above holds, but with the dependence on C removed. Further, by modifying the definition of p ( y | x ; θ ), it is straightforward to generalize this result to L 1 regularized versions of other models from the generalized linear model family (McCullagh &amp; Nelder, 1989), such as linear least squares regression.

## 4. Rotational invariance and L 2 regularization

Let M = { M ∈ R n × n | MM T = M T M = I, | M | = 1 } be the class of rotational matrices. 1 Thus, if x ∈ R n and M ∈ M , then Mx is x rotated through some angle around the origin. 2

Given a training set S = { ( x ( i ) , y ( i ) ) } i m =1 , we let MS = { ( Mx ( i ) , y ( i ) ) } i m =1 denote the training set with all the inputs rotated according to M . Given a learning algorithm L , we let L [ S ]( x ) denote the predicted label resulting from using the learning algorithm to train on a dataset S , and using the resulting hypothesis/classifier to make a prediction on x .

Definition 4.1: Given a (deterministic) learning algorithm L , we say that it is rotationally invariant if, for any training set S , rotational matrix M ∈ M , and test example x , we have that L [ S ]( x ) = L [ S ′ ]( x ′ ), where S ′ = MS, x ′ = Mx . More generally, if L is a stochastic learning algorithm so that its predictions are random, we say that it is rotationally invariant if, for any S, M, x , the predictions L [ S ]( x ) and L [ S ′ ]( x ′ ) have the same distribution.

Some readers familiar with logistic regression may already recognize that its L 2 regularized version is rotationally invariant. But for the sake of completeness, we will state and formally prove this here.

Proposition 4.2: L 2 regularized logistic regression (Equation 2, with α &gt; 0 ) is rotationally invariant.

Proof. Let any S, M, x be given, and let S ′ = MS , x ′ = Mx . Because M T M = I , we have 1 1+exp( -θ T x ) =

1 If we drop the condition that the determinant is | M | = 1, then we obtain the class of all orthogonal matrices, which may include a reflection as well as a rotation. (Strang, 1988) Using the more restrictive set as we do here leads to a slightly stronger theoretical result.

2 If we are using the convention (mentioned earlier) that x n = 1 always to handle the intercept term, then we may restrict attention to matrices M where M nn = 1 , M jn = 0 ( j &lt; n ), so that the final coordinate is not changed by M . This makes no difference to our results.

1 1+exp( -( Mθ ) T ( Mx )) , and thus p ( y | x ; θ ) = p ( y | Mx ; Mθ ). Further, R ( θ ) = θ T θ = ( Mθ ) T ( Mθ ) = R ( Mθ ). Define J ( θ ) = ∑ m i =1 log p ( y ( i ) | x ( i ) ; θ ) -αR ( θ ), and J ′ ( θ ) = ∑ m i =1 log p ( y ( i ) | Mx ( i ) ; θ ) -αR ( θ ). Let ˆ θ = arg max θ J ( θ ) be the parameters resulting from fitting L 2 regularized logistic regression to S . (Because α &gt; 0, the Hessian of J can be shown to be negative definite, and thus J has a unique maximum.) Similarly, let ˆ θ ′ = arg max θ J ′ ( θ ) be the parameters resulting from fitting to S ′ . By our previous argument, clearly J ( θ ) = J ′ ( Mθ ) for all θ . Thus, ˆ θ = argmax θ J ( θ ) = M -1 arg max θ J ′ ( θ ) = M -1 ˆ θ ′ , which implies ˆ θ ′ = M ˆ θ . Hence, L [ S ]( x ) = 1 / (1 + exp( -ˆ θ T x )) = 1 / (1 + exp( -( M ˆ θ ) T ( Mx ))) = 1 / (1 + exp( -( ˆ θ ′ ) T x ′ )) = L [ S ′ ]( x ′ ). glyph[square]

We also give, without proof, additional examples of rotationally invariant algorithms:

- SVMs using most kernels. 3
- Multilayer neural networks trained using backpropagation. 4
- Unregularized logistic regression. 5
- The perceptron algorithm.
- Any algorithm that uses PCA or ICA as a preprocessing step, by first re-representing the data in the basis formed by the top k principle components/independent components. 6
- Gaussian discriminant analysis (a generative learning algorithm which models p ( x | y ) with a multivariate normal distribution). 7

Examples of non-rotationally invariant algorithms include logistic regression with L 1 regularization, naive

3 Including the linear K ( x, z ) = x T z , polynomial K ( x, z ) = ( x T z + c ) d , or RBF (Gaussian) K ( x, z ) = exp( -|| x -z || 2 /σ 2 ) kernels, or any other kernel K ( x, z ) that can be written as a function of only x T x , x T z and z T z . Note also that the ' L 1 norm soft margin' formulation of SVMs uses a different, per-training example, L 1 penalty on the slack variables, and rotational invariance still holds.

4 Under the technical assumption that the weights are initialized, say, using independent samples from a Normal(0, glyph[epsilon1] ) distribution (or any other spherically symmetric distribution).

5 Here, we restrict attention to training sets where the maximization (2) has a unique optimum with α = 0. (If not, one can also use the limiting solution from α → 0 + with R ( θ ) = || θ || 2 2 , if the limit exists).

6 Assuming we do not preprocess the data for PCA by rescaling each input feature to have the same variance.

7 Assuming the model uses a full covariance matrix, so that the off-diagonal entries are allowed to be non-zero.

Bayes, decision trees that make only axis-aligned splits, Winnow (Littlestone, 1988), EG (Kivinen &amp; Warmuth, 1994), and most feature selection algorithms (Blum &amp; Langley, 1997; Kohavi &amp; John, 1997; Ng &amp; Jordan, 2001; Ng, 1998).

We now give a lower-bound on the worst-case sample complexity of feature selection using any rotationally invariant algorithm.

Theorem 4.3: Let L be any rotationally invariant learning algorithm, and let any 0 &lt; glyph[epsilon1] &lt; 1 / 8 , 0 &lt; δ &lt; 1 / 100 be fixed. Then there exists a learning problem D so that: (i) The labels are deterministically related to the inputs according to y = 1 if x 1 ≥ t , y = 0 otherwise for some t , and (ii) In order for L to attain glyph[epsilon1] or lower 0/1 misclassification error with probability at least 1 -δ , it is necessary that the training set size be at least

<!-- formula-not-decoded -->

Thus, for any rotationally invariant algorithm L , there exists at least one problem that should have been 'easy' in the sense that there is only one relevant feature ( x 1 ) and the labels are simply obtained by thresholding x 1 , but L requires a large number of training examples to learn it. Note that a good feature selection algorithm should be able to learn any target concept of this form using only O (log n ) training examples. (E.g., Ng, 1998, Littlestone, 1988.) However, L requires a number of training examples that's at least linear in the dimension of the input.

This suggests that rotationally invariant algorithms are unlikely to be effective feature selection algorithms, particularly in settings where only a small subset of the features are relevant, and the dimension of the input n is significantly larger than the training set size m .

The proof of this result is given in Appendix B, and uses ideas from the lower-bounds originally proved by (Ehrenfeucht et al., 1989; Vapnik, 1982). A related result was also shown by (Kivinen et al., 1995) for the perceptron learning algorithm. They point out that the perceptron is rotationally invariant, and that an adversary choosing the sequence of training examples can force it (or, more generally, any 'additive linear online prediction algorithm') to make Ω( n ) mistakes.

Remark. Support Vectors Machines have been proved to work well in extremely high dimensional input spaces, even infinite-dimensional ones, as long as the data is separated with a large margin γ . (Vapnik, 1998) Thus, it may seem surprising that we can show that SVMs perform poorly in the presence of high dimensional inputs (with many irrelevant features). To reconcile this, we note that while the margin does not shrink as extra irrelevant features are added, the diameter of the data (e.g., maximum distance between any two points measured in the L 2 -norm) grows with the number of irrelevant features, and it is actually the margin divided by the diameter that governs generalization performance. (Vapnik, 1998)

Figure 1. Experiment comparing logistic regression with L 1 regularization (blue solid lines; colors where available) vs. logistic regression with L 2 regularization (red dashed lines). Left column: One relevant feature. Middle column: Three relevant features. Right column: Exponentially-decaying relevance. Top row: Misclassification error with m=100. Middle row: Logloss error with m=100. Bottom row: Misclassification error with m=200.

<!-- image -->

## 5. Experiments

We now present some empirical results comparing logistic regression using L 1 and L 2 regularization. All results reported here are averages over at least 100 independent trials, and in each experiment, 30% of the data was used as hold-out data for selecting the regularization parameter. (Very similar results are ob- tained if the regularization parameters are tuned on test data.)

In the first experiment, we let the total number of features vary and let just a single feature be relevant. 8 Figure 1a shows the misclassification error of the two methods, when trained using 100 training examples. As we see, the results are dramatically different. Using L 1 regularization, logistic regression is extremely insensitive to the presence of irrelevant features. Note the scale on the horizontal axis: Even learning with just 100 examples in a 1000-dimensional input space, it is able to attain very low generalization error. In contrast, the error of logistic regression with L 2 regularization rapidly approaches 0.5.

8 Experimental details: Inputs were drawn from a multivariate normal distribution. For one relevant feature, the labels were generated using a logistic model with θ 1 = 10 (and all other θ i = 0). For three relevant features, we used θ 1 = θ 2 = θ 3 = 10 c 1 . For the third problem, we used θ i = (1 / 2) i -1 c 2 ( i ≥ 1), (The constants were c 1 = 1 / √ 3, c 2 = √ 75, which preserve the scaling of the problem so that Bayes error remains the same.) Results reporting logloss and 0/1 misclassification error used respectively ˆ ε S 2 = ˆ ε l S 2 and ˆ ε S 2 = ˆ ε m S 2 in the hold-out cross validation step.

Figure 1b shows the same experiment repeated with three relevant features. Figure 1c shows results from a third experiment where all the features contain some information about the output, but where the degree to which feature i is relevant decreases exponentially with i . Only the first few features have a significant effect on the output label, and to model the data well, it is sufficient to use only a very small number of features. Again, L 1 regularization is clearly superior as n becomes large.

Figures 1d-f repeat the same experiments, but here the logloss is plotted instead of misclassification error. Figures 1g-i show the same experiments repeated using 200 training examples. In all cases, logistic regression with L 1 regularization, as predicted by the theoretical results, exhibits a significantly higher tolerance to the presence of many irrelevant features.

## Acknowledgments

I give warm thanks to Pieter Abbeel, Chris Manning, Rajat Raina, Yoram Singer and Kristina Toutanova for helpful conversations. This work was supported by the Department of the Interior/DARPA under contract number NBCHD030010.

## References

- Anthony, M., &amp; Bartlett, P. (1999). Neural network learning: Theoretical foundations . Cambridge University Press.
- Bartlett, P. (1998). The sample complexity of pattern classification with neural networks: The size of the weights is more important than the size of the network. IEEE Transactions on Information Theory , 2 , 525-536.
- Blum, A., &amp; Langley, P. (1997). Selection of relevant features and examples in machine learning. Artificial Intelligence , 97 , 245-271.
- Ehrenfeucht, A., Haussler, D., Kearns, M., &amp; Valiant, L. (1989). A general lower bound on the number of examples needed for learning. Information and Computation , 82 , 247-261.
- Haussler, D. (1992). Decision-theoretic generalizations of the PAC model for neural networks and other applications. Information and Computation , 100 , 78-150.
- Kivinen, J., Warmuth, M., &amp; Auer, P. (1995). The percep-
- tron vs. winnow: Linear vs. logarithmic mistake bounds when few input variables are relevant. Proc. 8th Annual Conference on Computational Learning Theory (pp. 289-296).
- Kivinen, J., &amp; Warmuth, M. K. (1994). Exponentiated gradient versus gradient descent for linear predictors (Technical Report UCSC-CRL-94-16). Univ. of California Santa Cruz, Computer Research Laboratory.
- Kohavi, R., &amp; John, G. (1997). Wrappers for feature subset selection. Artificial Intelligence , 97 , 273-324.
- Littlestone, N. (1988). Learning quickly when irrelevant attributes abound: A new linear-threshold algorithm. Machine Learning , 2 , 285-318.
- McCullagh, P., &amp; Nelder, J. A. (1989). Generalized linear models (second edition) . Chapman and Hall.
- Ng, A. Y. (1998). On feature selection: Learning with exponentially many irrelevant features as training examples. Proceedings of the Fifteenth International Conference on Machine Learning (pp. 404-412). Morgan Kaufmann.
- Ng, A. Y., &amp; Jordan, M. I. (2001). Convergence rates of the voting gibbs classifier, with application to bayesian feature selection. Proceedings of the Eighteenth International Conference on Machine Learning . Morgan Kaufmann.
- Nigam, K., Lafferty, J., &amp; McCallum, A. (1999). Using maximum entropy for text classification. IJCAI-99 Workshop on ML for Information Filtering .
- Pollard, D. (1984). Empirical processes: Theory and applications . Springer-Verlag.
- Rockafellar, R. (1970). Convex analysis . Princeton Univ. Press.
- Strang, G. (1988). Linear algebra and its applications, 3rd ed. International Thomas Publishing.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. J. Royal. Statist. Soc B. , 58 , 267-288.
- Vapnik, V. (1982). Estimation of dependences based on empirical data . Springer-Verlag.
- Vapnik, V. N. (1998). Statistical learning theory . John Wiley &amp; Sons.
- Zhang, T. (2002). Covering number bounds of certain regularized linear function classes. Journal of Machine Learning Research , 527-550.
- Zheng, A. X., Jordan, M. I., Liblit, B., &amp; Aiken, A. (2004). Statistical debugging of sampled programs. Neural Information Processing Systems 16 .

## Appendix A: Proof of Theorem 3.1

Our proof of Theorem 3.1 is based on bounding the covering numbers of certain function classes. Due to space constraints, our proof is necessarily brief, but for highly readable introductions to covering numbers, see, e.g., (Anthony &amp; Bartlett, 1999; Haussler, 1992).

Let there be a class of functions F with some domain U and range [ -M,M ] ⊂ R . Given some set of points z (1) , . . . , z ( m ) ∈ U , we let F | z (1) ,...,z ( m ) = { [ f ( z (1) ) , . . . , f ( z ( m ) )]; f ∈ F} ⊆ [ -M,M ] m . We say that a set { v (1) , . . . , v ( k ) } ⊆ R m glyph[epsilon1] -covers F | z (1) ,...,z ( m ) in the p -norm if, for every u ∈ F | z (1) ,...,z ( m ) , there is some v ( i ) so that || u -v ( i ) || p ≤ m 1 /p glyph[epsilon1] . Here, || t || p = ( ∑ m i =1 | t i | p ) 1 /p . Define N p ( F , glyph[epsilon1], [ z 1 , . . . , z m ]) to be the size of the smallest set that glyph[epsilon1] -covers F | z 1 ,...,z m in the p -norm. Also, let N p ( F , glyph[epsilon1],m ) = sup z 1 ,...,z m N p ( F , glyph[epsilon1], [ z 1 , . . . , z m ]).

Let there be some distribution D over U , and define ε ( f ) = E z ∼ D [ f ( z )] . If z (1) , . . . , z ( m ) ∼ iid D , then (Pollard, 1984) showed that

<!-- formula-not-decoded -->

Further, (Zhang, 2002) shows that if G = { g : g ( x ) = θ T x, x ∈ R n , || θ || q ≤ a } is a class of linear functions parameterized by weights θ with q -norm bounded by a , and if the inputs x ∈ R n are also norm-bounded so that || x || p ≤ b , and further 1 /p + 1 /q = 1 (so the p -and q -norms and dual) with 2 ≤ p ≤ ∞ , then

<!-- formula-not-decoded -->

(A special case of this is also found in Bartlett, 1998.) Some other well-known properties of covering numbers (e.g., Anthony and Bartlett, 1999; Zhang, 2002; Haussler, 1992) include that

<!-- formula-not-decoded -->

and that given a class of functions G with domain R , if F is a class of functions R × Y ↦→ R defined according to F = { f g ( x, y ) = glyph[lscript] ( g ( x ) , y ) : g ∈ G , y ∈ { 0 , 1 }} , where glyph[lscript] ( · , y ) (for any fixed y and viewed a function of the first parameter only) is Lipschitz continuous with Lipschitz constant L , then

<!-- formula-not-decoded -->

We now give the main part of the proof. First, notice that the algorithm uses hold-out cross validation to select amongst the values B = 0 , 1 , 2 , 4 , . . . . Let ˆ B be the smallest value in { 0 , 1 , 2 , 4 , . . . } that is greater than or equal to rK . Notice therefore that rK ≤ ˆ B ≤ 2 rK . We will begin by considering the step in the algorithm where logistic regression was fit using the regularization parameter ˆ B . Specifically, let ˆ θ denote the parameter vector resulting from solving the optimization problem given by Equations (3-4) with B = ˆ B .

Let G = { g θ : [ -1 , 1] n ↦→ R : g θ ( x ) = θ T x, || θ || 1 ≤ ˆ B } be a class of linear functions parameterized by θ with L 1 -norm bounded by ˆ B . Using Equations (12,11), we have that

<!-- formula-not-decoded -->

(Recall our assumption in Section 2 that x ∈ [ -1 , 1] n , which implies || x || ∞ ≤ 1.) From Holder's inequality, we also have

<!-- formula-not-decoded -->

Now, let F be a class of functions f : R × Y ↦→ R defined according to F = { f θ ( x, y ) = glyph[lscript] ( g θ ( x ) , y ) : g θ ∈ G , y ∈ { 0 , 1 }} , where glyph[lscript] ( g ( x ) , 1) = -log 1 / (1 + exp( -g ( x ))), and glyph[lscript] ( g ( x ) , 0) = -log(1 -1 / (1 + exp( -g ( x )))). Thus, glyph[lscript] ( g ( x ) , y ) is the logloss suffered by the logistic regression model on an example where it predicts p ( y = 1 | x ) = 1 / (1 + exp( -g ( x )), and the correct label was y . It is straightforward to show that | d dt glyph[lscript] ( t, y ) | ≤ 1 for any y ∈ { 0 , 1 } . Thus, glyph[lscript] ( · , y ) is Lipschitz continuous with Lipschitz constant L = 1. Hence, combining Equations (13,14), we get

<!-- formula-not-decoded -->

It is also straightforward to show that | glyph[lscript] ( t, 1) | = | log 1 / (1+exp( -t )) | ≤ | t | +1 (and similarly for glyph[lscript] ( t, 0)). Together with Equation (15), this implies that

<!-- formula-not-decoded -->

Let m 1 = (1 -γ ) m be the number of examples the parameters ˆ θ were trained on in the inner-loop of the algorithm. (The remaining m 2 = γm examples were used for hold-out cross validation.) Recalling that N 1 ( F , glyph[epsilon1], [ z (1) , . . . , z ( m ) ])] ≤ N 1 ( F , glyph[epsilon1],m ) by definition, and putting together Equations (16,10,17) with M = ˆ B +1, we find that

<!-- formula-not-decoded -->

We would like for this probability to be small. By setting the right hand side to δ and solving for m 1 , we find that in order for the probability above to be upper-bounded by δ , it suffices that m 1 = Ω((log n ) · poly( ˆ B, 1 glyph[epsilon1] , log 1 δ )) = Ω((log n ) · poly( r, K, 1 glyph[epsilon1] , log 1 δ )), where to obtain the second equality we used the fact

(shown earlier) that rK ≤ ˆ B ≤ 2 rK . Since m 1 = (1 -γ ) m , if we treat (1 -γ ) as a constant that can be absorbed into the big-Ω notation, then to ensure the above holds, it suffices that

<!-- formula-not-decoded -->

To summarize, we have shown that if m satisfies Equation (19), then with probability 1 -δ , it will hold true that for all f ∈ F , we have that

<!-- formula-not-decoded -->

∣ ∣ By referring to the definitions of F and G , we see this would imply that for all θ : || θ || 1 ≤ ˆ B , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∣ ∣ In summary, we have shown that with a training set whose size has to be at most logarithmic in n and polynomial in all quantities of interest, with probability 1 -δ , we will have that ˆ ε l S 1 ( θ ) is a uniformly good estimate for ε l ( θ ). Now, recall that the parameter vector ˆ θ was found by solving ˆ θ = arg min θ : || θ || 1 ≤ ˆ B ˆ ε l S 1 ( θ ). Using Equation (22), a standard uniform convergence result 9 (e.g., Vapnik, 1982; Anthony and Bartlett, 1999) shows that minimizing ˆ ε l is nearly as good as minimizing ε l , and that in particular, Equation (22) implies

<!-- formula-not-decoded -->

where the second step used the fact that || θ ∗ || 1 ≤ ˆ B . Hence, we have shown that in Step 2 of the algorithm, we will find at least one parameter vector ˆ θ whose performance (generalization error as measured according to ε l ) is nearly as good as that of θ ∗ . In Step 3 of the algorithm, we use hold-out cross validation to select from the set of θ B 's found in Step 2. Using another entirely standard argument, it is straightforward to show that, with only a 'small' (at most polynomially large, and independent of n ) number of examples used in the hold-out set, we can ensure that with probability 1 -δ , the selected parameter vector will have performance at most glyph[epsilon1] worse that that of the best parameter vector in the set. The details of this step are glyph[epsilon1]

9 Specifically, that if | f ( θ ) -ˆ f ( θ ) | ≤ glyph[epsilon1] for all θ ∈ Θ, then f (arg min θ ∈ Θ ˆ f ( θ )) ≤ min θ ∈ Θ f ( θ ) + 2 glyph[epsilon1] .

omitted due to space, but is entirely standard and may be found in, e.g., (Vapnik, 1982; Anthony &amp; Bartlett, 1999). Putting this together with (23), we have shown that, with probability 1 -2 δ , the output θ satisfies

<!-- formula-not-decoded -->

Finally, replacing δ with δ/ 2 and glyph[epsilon1] with glyph[epsilon1]/ 3 everywhere in the proof shows the theorem. glyph[square]

## Appendix B: Proof of Theorem 4.3

glyph[negationslash]

Let L and glyph[epsilon1], δ be as given in the statement of the theorem. Consider the concept class of all linear separators in n dimensions, C = { h θ : h θ ( x ) = 1 { θ T x ≥ β } , θ = 0 } . (Here, we do not use the convention adopted previously that necessarily x n = 1. Also, 1 {·} is the indicator function, so that 1 { True } = 1, and 1 { False } = 0.) It is well-known that VC( C ) = n +1. (Vapnik, 1982)

From a standard PAC lower bound, there must therefore exist a distribution D X over the inputs, and a target concept h ∗ ∈ C , so that if D X is the input distribution, and the labels are given by y = h ∗ ( x ), then for L to attain glyph[epsilon1] or lower 0/1 misclassification error with probability at least 1 -δ , it is necessary that the training set size be at least m = Ω( n/glyph[epsilon1] ). (Results of this type have been proved by Vapnik, 1982 and Ehrenfeucht et al., 1989. The particular result stated here is also given in, e.g., Theorem 5.3 of Anthony and Bartlett, 1999).

Since h ∗ ∈ C is a linear target concept, it can be written h ∗ ( x ) = 1 { θ ∗ T x ≥ β ∗ } for some θ ∗ ∈ R n and β ∗ ∈ R . Because replacing ( θ ∗ , β ∗ ) with ( cθ ∗ , cβ ∗ ) for any positive constant c does not change anything, we may assume without loss of generality that || θ ∗ || 2 = 1.

Let M be any orthogonal matrix whose first row is θ ∗ T . Such a matrix must exist. (Strang, 1988) Thus, Mθ ∗ = [1 , 0 , . . . , 0] T = e 1 (the first basis vector). Further, by flipping the signs of any single row (other than the first row) of M if necessary, we may ensure that | M | = 1, and hence M ∈ M . Now, consider a learning problem where the input distribution is induced by sampling x ∼ D X , and then computing x ′ = Mx . Further, let the labels be given by y ′ = 1 { x ′ 1 ≥ β ∗ } . Because 1 { x ′ 1 ≥ β ∗ } = 1 { e T 1 x ′ ≥ β ∗ } = 1 { ( Mθ ∗ ) T ( Mx ) ≥ β ∗ } = 1 { ( θ ∗ ) T x ≥ β ∗ } = y , we therefore see that a learning problem with examples drawn from the ( x ′ , y ′ ) distribution is simply a rotated version of the problem with examples ( x, y ). But since L is rotationally invariant, its predictions on test sets under the original and rotated problems will be identical, and thus its generalization error, and sample complexity, must also be the same under either problem. glyph[square]