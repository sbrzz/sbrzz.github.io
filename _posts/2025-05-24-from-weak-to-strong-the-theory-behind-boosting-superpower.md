---
title: "From Weak to Strong: The Theory Behind Boosting’s Superpower"
categories:
  - Blog
tags:
  - Artificial Intelligence
  - Machine Learning
  - Learning Theory
  - Ensemble Learning
math: true
layout: single
classes: wide
---

How can we transform a collection of weak classifiers into a single strong one?
To answer this question, we turn to one of my favorites ML algorithm: Boosting.

This blog post wants to simply cover how and why this fascinating algorithm works, along with some theoretical insights.

First of all a bit of history. Boosting is not new (middle 90') but for sure it makes history in the ML community.
The fathers of this algorithm are Robert E. Schapire and Yoav Freund along with their collaborators.
Schapire and Freund authored the highly recommended book [Boosting Foundation and Algoriths][boostingbook] from which I inspired this work.

Let's jump into one version of the algorithm: Adaboost.

## Adaboost

Let $(x_1, y_1)...(x_m, y_m)$ your dataset, where $x_i \in X$ and $y_i \in \\{ -1, 1 \\}$. We work with a binary target for simplicity.

Adaboost is iterative, let be $D_t:X \rightarrow \mathbb{R}$  the distribution over the training samples at the $t\text{-th}$ iteration, $D_t \( x_i \)$ allows to assign the weight to each sample in the training set.

At the first iteration we assign equal weight to all samples in the dataset, e.g. $D_1\( x_i \) = \frac{1}{m}$.

Now the steps for each iteration t=1...T:

1. train a weak learner - using $D_t$ - and get a weak hypothesis $h_t:X \rightarrow \\{ -1, 1 \\}$
2. the hypothesis is chosen with the goal of minimize the - weighted - training error $\epsilon_t = Prob\[ h_t\( x_i\) \neq y_i \]$
3. choose the scale factor $\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$
4. update the weights for all samples in the training set: $\displaystyle D_{t+1} = \frac{D_t\(i\)\exp(-\alpha_t y_i h_t\(x_i\) )}{Z_t}$ where $Z_t$ is a normalization factor so that $D_{t+1}$ is a distribution.

After all the steps we have the final hypothesis for a sample $x \in X$, i.e. our *strong classifier*:

$$
\begin{align}H(x) = sign\left( \sum_{t=1}^{T}\alpha_t h_t(x) \right)\end{align}
$$

The weak learner goal is to find a hypothesis that is at least slightly better than random guessing. This principle, known as the *weak learning hypothesis*, is fundamental to the theory of Boosting.

In our case, since we deal with a binary classification problem then a classifier that guess random has a probability of $\frac{1}{2}$ of misclassifying a sample.
The *weak learning hypothesis* states that each hypothesis $h_t$ should be bounded away from $\frac{1}{2}$, so that $\epsilon_t$ is at most $\frac{1}{2} - \gamma$, with $\gamma$ a small positive constant.

Observations:
* $D_{t+1}$ is selected so that in the next iteration the weak learner will focus on the most difficult samples. In a few words: at each step we try to find a new hypothesis that covers the fails of the previous ones.
* The scale factor $\alpha_t$ measures the relevance of the hypothesis $h_t$ on the final choice $H(x)$.
* To take advantage of $D_t$ there are two basic options: *boosting by reweights* and *boosting by resample*. The first assumes the capacity of the weak learner to use the weighted samples (many ML algorithms can do it). The second deals with a creation of a new dataset by sampling the most difficult samples, no weights are used by the weak learner.

## A bound on the training error

If you are in the ML world you should know that fitting well the training set is - generally - a good sign of the capability of the model to solve part of the distribution (i.e. the training part). It doesn't mean to work well on the test set, but thas's a different story.

The Boosting theory come to us with a very interesting theorem: the training error has a superior bound, and it decreases as a function of the number of boosting iteration $T$.
If you think to this point, it is quite sensational! Indipendently on the weak classifier we can decrease the training error by pushing on the boosting iterations.

This is the formula on upper bound of the training error:

$$
Prob_{i \sim D_1}[H(x_i) \neq y_i] \leq \prod_{t=1}^{T} \sqrt{1-4\gamma_t^2} \leq \exp(-2\prod_{t=1}^{T}\gamma_t^2)
$$

Let's see how to obtain the first part $\prod_{t=1}^{T} \sqrt{1-4\gamma_t^2}$ in two points:

* Given $F(x) = \sum_{t=1}^{T} \alpha_t h_t(x)$ we unravel $D_{T+1}$:

$$
D_{T+1} = D_1(i) \frac{\exp(-y_i \alpha_1 h_1(x_i))}{Z_{1}} .... \frac{\exp(-y_i \alpha_T h_T(x_i))}{Z_{T}}
$$

$$
= D_1(i) \frac{\exp(-y_i \sum_{t=1}^{T} \alpha_t h_t(x_i) )}{\prod_{t=1}^{T} Z_t} = D_1(i) \frac{\exp(-y_i F(x))}{\prod_{t=1}^{T} Z_t} 
$$

Given the final hyphotesis $H(x) = sign(F(x))$, if $H(x) \neq y_i$ then $yF(x) \leq 0$ (remember that $y_i \in \\{-1,1\\})$.

We have that $\exp(-yF(x)) \geq 1$, by using the [indicator function][indfunc]:

$$
\mathbf{1}{\\{ H(x) \neq y\\}} \leq \exp(-yF(x))
$$

Now the training error can be written in this way:

$$
Prob_{i \sim D_1}[H(x_i) \neq y_i] = \sum_{i=1}^{m} D_1(i)\mathbf{1} (\\{ H(x) \neq y \\}) \leq \sum_{i=1}^{m} D_1(i)\exp(-y_iF(x_i)) \ldots
$$

$$
= \sum_{i=1}^{m} D_{t+1}(i) \prod_{t=1}^{T} Z_t = (\text{since} \sum_{i=1}^{m} D_{t+1}(i) \text{ is a distribution} ) \prod_{t=1}^{T} Z_t
$$

Let $\gamma_t = \frac{1}{2} - \epsilon_t$, by using the definition in (4) and given $\alpha_t$ as in (3):

$$
Z_t = \sum_{i=1}^{m} D_t(i) \exp(-\alpha_t y_i h_t(x_i)) = \sum_{i:y_i=h_t(xi)} D_t(i)\exp(-\alpha_t) + \sum_{i:y_i\neq h_t(xi)} D_t(i)\exp(\alpha_t) \ldots
$$

$$
= \exp(-\alpha_t) (1-\epsilon_t) + \exp(\alpha_t) \epsilon_t = \exp(-\alpha_t) \left(\frac{1}{2}+\gamma_t \right) + \exp(\alpha_t) \left(\frac{1}{2}-\gamma_t \right) = \sqrt{1-4\gamma_t^2}
$$


* For the second bound is it possible to apply the approximation $1+x \leq exp(x)$ on $\sqrt{1-4\gamma_t^2}$ to obtain $\exp(-2\prod_{t=1}^{T}\gamma_t^2)$.

An example: suppose $\gamma_t$ is 5%, so that each $h_t$ has no training error superior to 45% ($\frac{1}{2} - \gamma_t$). Therefore the theorem tell us that the combined classifier has error rate lower than $(\sqrt{1-4(0.05)^2})^T \approx (0.99)^T$. It decreases exponentially as a function of the number of boosting rounds!

## A bound on the generalization error

What is a ML algorithm without a low generalization error? It's garbage! Don't worry, there is also a theoretical bound on the generalization error. I don't want to go into detail of the theorem here otherwise the post lasts too long, but if your are interested read Chapter 4 of [Boosting Foundation and Algoriths][boostingbook]. The goal is to derive a bound by introducing assumptions about the dataset samples $(x_i, y_i)$ and the *complexity* of the hypothesis $h_t$, which are not required for the training error bound.

## Boosting in our days

Is it possible to take advantage of this algorithm in a world where Neural Nets - far away from slightly then better - dominate the scene? To me the answer is absolutely yes!
There are really difficult problems where Neural Nets perform poorly, just to cite the really interesting [Vesuvio Challenge Ink Detection][vc]. I saw Boosting applyied to this problem, but unfortunately can't find the blog discussion anymore.

## Finally

Try Boosting! Many open-source Python packages implement it:


- **[XGBoost](https://xgboost.readthedocs.io/en/stable/):** Fast, scalable gradient boosting (GBDT). [GitHub](https://github.com/dmlc/xgboost)
- **[LightGBM](https://lightgbm.readthedocs.io/):** Microsoft’s high-speed GBDT with histogram-based learning. [GitHub](https://github.com/microsoft/LightGBM)
- **[CatBoost](https://catboost.ai/en/docs/):** Yandex’s boosting library with native categorical feature support. [GitHub](https://github.com/catboost/catboost)
- **[Scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting):** Includes AdaBoost and Gradient Boosting implementations. [GitHub](https://github.com/scikit-learn/scikit-learn)
- **[NGBoost](https://ngboost.readthedocs.io/):** Probabilistic boosting using natural gradients. [GitHub](https://github.com/stanfordmlgroup/ngboost)
- **[BoostARoota](https://github.com/chasedehan/BoostARoota):** XGBoost-based feature selection tool.


[boostingbook]:https://direct.mit.edu/books/oa-monograph/5342/BoostingFoundations-and-Algorithms
[indfunc]:https://en.wikipedia.org/wiki/Indicator_function#Definition
[vc]:https://scrollprize.org/


<hr/>

<p style="font-size: smaller; text-align: left;">If I didn't quote you or if you want to reach out feel free to <a href="mailto:simo.brazzo@gmail.com">contact me</a>.</p>
<p style="font-size: smaller; text-align: left;">© [Simone Brazzo] [2025] - Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>  with the following additional restriction: this content can be only used to train open-source AI models, where training data, models weights, architectures and training procedures are publicly available.</p>

<hr/>

