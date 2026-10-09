
# glmmsel

[![R-CMD-check](https://github.com/ryan-thompson/glmmsel/workflows/R-CMD-check/badge.svg)](https://github.com/ryan-thompson/glmmsel/actions)
[![codecov](https://codecov.io/gh/ryan-thompson/glmmsel/branch/master/graph/badge.svg)](https://github.com/ryan-thompson/glmmsel/actions)

## Overview

An R package for generalised linear mixed model (GLMM) selection.
`glmmsel` uses an $\ell_0$ regulariser to simultaneously select fixed
and random effects. A hierarchical constraint is included that a random
effect cannot be selected unless its corresponding fixed effect is also
selected. Gaussian and binomial families are currently supported. See
[this paper](https://arxiv.org/abs/2506.20425) for more information.

## Installation

To install the latest version from GitHub, run the following code:

``` r
devtools::install_github('ryan-thompson/glmmsel')
```

## Usage

The `glmmsel()` function fits a sparse GLMM over a sequence of the
regularisation parameter $\lambda$, with different values yielding
different sparsity levels. The `cv.glmmsel()` function provides a
convenient method for automatically cross-validating $\lambda$.

``` r
library(glmmsel)

# Generate some clustered data
n <- 100 # Number of observations
m <- 4 # Number of clusters
p <- 5 # Number of predictors
s.fix <- 2 # Number of nonzero fixed effects
s.rand <- 1 # Number of nonzero random effects
x <- matrix(rnorm(n * p), n, p) # Predictor matrix
beta <- c(rep(1, s.fix), rep(0, p - s.fix)) # True fixed effects
u <- cbind(matrix(rnorm(m * s.rand), m, s.rand), matrix(0, m, p - s.rand)) # True random effects
cluster <- sample(1:m, n, replace = TRUE) # Cluster labels
xb <- rowSums(x * sweep(u, 2, beta, '+')[cluster, ]) # x %*% (beta + u) matrix
y <- rnorm(n, xb) # Response vector

# Fit the ℓ0 regularisation path
fit <- glmmsel(x, y, cluster)
coef(fit, lambda = 10)
```

    ##             [,1]     [,2]     [,3] [,4] [,5] [,6]
    ## [1,] -0.12410477 1.128917 1.053438    0    0    0
    ## [2,]  0.15875835 1.128917 1.053438    0    0    0
    ## [3,] -0.01088924 1.128917 1.053438    0    0    0
    ## [4,]  0.09670996 1.128917 1.053438    0    0    0

``` r
# Cross-validate the ℓ0 regularisation path
fit <- cv.glmmsel(x, y, cluster)
coef(fit)
```

    ##             [,1]     [,2]     [,3] [,4] [,5] [,6]
    ## [1,] -0.12410477 1.128917 1.053438    0    0    0
    ## [2,]  0.15875835 1.128917 1.053438    0    0    0
    ## [3,] -0.01088924 1.128917 1.053438    0    0    0
    ## [4,]  0.09670996 1.128917 1.053438    0    0    0

## Documentation

See the package
[vignette](https://CRAN.R-project.org/package=glmmsel/vignettes/vignette.html)
or [reference
manual](https://CRAN.R-project.org/package=glmmsel/glmmsel.pdf).
