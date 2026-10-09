## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 5
)

## -----------------------------------------------------------------------------
set.seed(1234)
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

## -----------------------------------------------------------------------------
library(glmmsel)
fit <- glmmsel(x, y, cluster)

## -----------------------------------------------------------------------------
fixef(fit)
ranef(fit)

## -----------------------------------------------------------------------------
coef(fit)

## -----------------------------------------------------------------------------
x.new <- x[1:3, ]
cluster.new <- cluster[1:3]
predict(fit, x.new, cluster.new)

## -----------------------------------------------------------------------------
fit <- cv.glmmsel(x, y, cluster)

## -----------------------------------------------------------------------------
coef(fit)
predict(fit, x.new, cluster.new)

## -----------------------------------------------------------------------------
y <- rbinom(n, 1, 1 / (1 + exp(- xb)))
fit <- cv.glmmsel(x, y, cluster, family = 'binomial')
coef(fit)

## -----------------------------------------------------------------------------
x <- 0.2 * matrix(rnorm(n * p), n, p) + 0.8 * matrix(rnorm(n), n, p)
xb <- rowSums(x * sweep(u, 2, beta, '+')[cluster, ])
y <- rnorm(n, xb)
fit <- cv.glmmsel(x, y, cluster)
coef(fit)
fit <- cv.glmmsel(x, y, cluster, local.search = TRUE)
coef(fit)

