# Generate some data
set.seed(1)
n <- 100
p <- 2
m <- 5
cluster <- rep(1:m, each = n / m)
beta <- rep(1, p)
b <- matrix(rnorm(m * p), m, p)
x <- matrix(rnorm(n * p), n, p)
e <- rnorm(n)
y <- x %*% beta + rowSums(x * b[cluster, ]) + e
df <- data.frame(y = y, x = x, cluster = cluster)

#==================================================================================================#
# Linear model
#==================================================================================================#

testthat::test_that('linear models are fitted correctly', {

  # Fit models
  fit <- glmmsel(x, y, rep(1, nrow(x)), lambda = 0)
  fit.target <- lm(y ~ x.1 + x.2, df)

  # Test coef
  beta <- coef(fit, lambda = 0)[1, ]
  beta.target <- as.vector(coef(fit.target))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, rep(1, nrow(x)), lambda = 0)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# Linear model without intercept
#==================================================================================================#

testthat::test_that('linear models without an intercept are fitted correctly', {

  # Fit models
  fit <- glmmsel(x, y, rep(1, nrow(x)), lambda = 0, intercept = FALSE)
  fit.target <- lm(y ~ 0 + x.1 + x.2, df)

  # Test coef
  beta <- coef(fit, lambda = 0)[1, ]
  beta.target <- as.vector(coef(fit.target))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, rep(1, nrow(x)), lambda = 0)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# Linear mixed model
#==================================================================================================#

testthat::test_that('linear mixed models are fitted correctly', {

  # Fit models
  fit <- glmmsel(x, y, cluster, lambda = 0)
  fit.target <- lme4::lmer(y ~ x.1 + x.2 + (x.1 + x.2 || cluster), df, REML = FALSE)

  # Test fixef
  beta <- fixef(fit, lambda = 0)
  beta.target <- as.vector(lme4::fixef(fit.target))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test ranef
  u <- ranef(fit, lambda = 0)
  u.target <- unname(as.matrix(lme4::ranef(fit.target)$cluster))
  testthat::expect_equal(u, u.target, tolerance = 1e-2)

  # Test coef
  beta <- coef(fit, lambda = 0)
  beta.target <- unname(as.matrix(coef(fit.target)$cluster))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, cluster, lambda = 0)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# Linear mixed model without intercept
#==================================================================================================#

testthat::test_that('linear mixed models without an intercept are fitted correctly', {

  # Fit models
  fit <- glmmsel(x, y, cluster, lambda = 0, intercept = FALSE)
  fit.target <- lme4::lmer(y ~ 0 + x.1 + x.2 + (0 + x.1 + x.2 || cluster), df, REML = FALSE)

  # Test fixef
  beta <- fixef(fit, lambda = 0)
  beta.target <- as.vector(lme4::fixef(fit.target))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test ranef
  u <- ranef(fit, lambda = 0)
  u.target <- unname(as.matrix(lme4::ranef(fit.target)$cluster))
  testthat::expect_equal(u, u.target, tolerance = 1e-2)

  # Test coef
  beta <- coef(fit, lambda = 0)
  beta.target <- unname(as.matrix(coef(fit.target)$cluster))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, cluster, lambda = 0)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# Linear mixed model without random intercept
#==================================================================================================#

testthat::test_that('linear mixed models without a random intercept are fitted correctly', {

    # Fit models
    fit <- glmmsel(x, y, cluster, lambda = 0, random.intercept = FALSE)
    fit.target <- lme4::lmer(y ~ x.1 + x.2 + (0 + x.1 + x.2 || cluster), df, REML = FALSE)

    # Test fixef
    beta <- fixef(fit, lambda = 0)
    beta.target <- as.vector(lme4::fixef(fit.target))
    testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

    # Test ranef
    u <- ranef(fit, lambda = 0)
    u.target <- unname(as.matrix(lme4::ranef(fit.target)$cluster))
    testthat::expect_equal(u, u.target, tolerance = 1e-2)

    # Test coef
    beta <- coef(fit, lambda = 0)
    beta.target <- unname(as.matrix(coef(fit.target)$cluster))
    testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

    # Test predict
    yhat <- predict(fit, x, cluster, lambda = 0)
    yhat.target <- as.vector(predict(fit.target, df))
    testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# First penalty
#==================================================================================================#

testthat::test_that('infinite regularisation with alpha = 1 works correctly', {

  # Fit models
  fit <- glmmsel(x, y, cluster, lambda = Inf, alpha = 1)
  fit.target <- lme4::lmer(y ~ (1 | cluster), df, REML = FALSE)

  # Test fixef
  beta <- fixef(fit, lambda = 0)
  beta.target <- c(as.vector(lme4::fixef(fit.target)), rep(0, p))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test ranef
  u <- ranef(fit, lambda = 0)
  u.target <- cbind(unname(as.matrix(lme4::ranef(fit.target)$cluster)), matrix(0, m, p))
  testthat::expect_equal(u, u.target, tolerance = 1e-2)

  # Test coef
  beta <- coef(fit, lambda = 0)
  beta.target <- cbind(unname(as.matrix(coef(fit.target)$cluster)), matrix(0, m, p))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, cluster, lambda = 0)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# Second penalty
#==================================================================================================#

testthat::test_that('infinite regularisation with alpha = 0 works correclty', {

  # Fit models
  fit <- glmmsel(x, y, cluster, lambda = Inf, alpha = 0)
  fit.target <- lme4::lmer(y ~ x.1 + x.2 + (1 | cluster), df, REML = FALSE)

  # Test fixef
  beta <- fixef(fit, lambda = 0)
  beta.target <- as.vector(lme4::fixef(fit.target))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test ranef
  u <- ranef(fit, lambda = 0)
  u.target <- cbind(unname(as.matrix(lme4::ranef(fit.target)$cluster)), matrix(0, m, p))
  testthat::expect_equal(u, u.target, tolerance = 1e-2)

  # Test coef
  beta <- coef(fit, lambda = 0)
  beta.target <- unname(as.matrix(coef(fit.target)$cluster))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, cluster, lambda = 0)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# Other tests
#==================================================================================================#

testthat::test_that('predict works correctly when clusters do not begin at zero', {

  df2 <- df
  df2$cluster <- df2$cluster + 100
  cluster2 <- cluster + 100

  # Fit models
  fit <- glmmsel(x, y, cluster2, lambda = 0)
  fit.target <- lme4::lmer(y ~ x.1 + x.2 + (x.1 + x.2 || cluster), df2, REML = FALSE)

  # Test predict
  yhat <- predict(fit, x, cluster2, lambda = 0)
  yhat.target <- as.vector(predict(fit.target, df2))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

testthat::test_that('predict works correctly on unobserved clusters', {

  df2 <- df
  df2$cluster <- df2$cluster + 100
  cluster2 <- cluster + 100

  # Fit models
  fit <- glmmsel(x, y, cluster, lambda = 0)
  fit.target <- lme4::lmer(y ~ x.1 + x.2 + (x.1 + x.2 || cluster), df, REML = FALSE)

  # Test predict
  yhat <- predict(fit, x, cluster2, lambda = 0)
  yhat.target <- as.vector(predict(fit.target, df2, allow.new.levels = TRUE))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# Standardisation off still works
#==================================================================================================#

testthat::test_that('linear mixed models are fitted correctly', {

  # Fit models
  fit <- glmmsel(x, y, cluster, lambda = 0, standardise = FALSE)
  fit.target <- lme4::lmer(y ~ x.1 + x.2 + (x.1 + x.2 || cluster), df, REML = FALSE)

  # Test fixef
  beta <- fixef(fit, lambda = 0)
  beta.target <- as.vector(lme4::fixef(fit.target))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test ranef
  u <- ranef(fit, lambda = 0)
  u.target <- unname(as.matrix(lme4::ranef(fit.target)$cluster))
  testthat::expect_equal(u, u.target, tolerance = 1e-2)

  # Test coef
  beta <- coef(fit, lambda = 0)
  beta.target <- unname(as.matrix(coef(fit.target)$cluster))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, cluster, lambda = 0)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# Pathwise computation
#==================================================================================================#

# Generate some data
n <- 300
s <- 5
p <- 10
m <- 5
cluster <- rep(1:m, each = n / m)
beta <- c(rep(1, s), rep(0, p - s))
b <- cbind(matrix(rnorm(m * s), m, s), matrix(0, m, p - s))
x <- matrix(rnorm(n * p), n, p)
e <- rnorm(n)
y <- x %*% beta + rowSums(x * b[cluster, ]) + e
df <- data.frame(y = y, x = x, cluster = cluster)

testthat::test_that('pathwise computation works correctly', {

  # Fit models
  fit <- glmmsel(x, y, cluster)
  fit.target <- lme4::lmer(y ~ x.1 + x.2 + x.3 + x.4 + x.5 + x.6 + x.7 + x.8 + x.9 + x.10 +
                           (x.1 + x.2 + x.3 + x.4 + x.5 + x.6 + x.7 + x.8 + x.9 + x.10 || cluster), df, REML = FALSE)

  # Test lambda.max
  testthat::expect_true(norm(fit$beta[, 1], '2') == 0)
  testthat::expect_true(norm(fit$beta[, 2], '2') != 0)

  # Test fixef
  beta <- fixef(fit, lambda = 0)
  beta.target <- as.vector(lme4::fixef(fit.target))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test ranef
  u <- ranef(fit, lambda = 0)
  u.target <- unname(as.matrix(lme4::ranef(fit.target)$cluster))
  testthat::expect_equal(u, u.target, tolerance = 1e-2)

  # Test coef
  beta <- coef(fit, lambda = 0)
  beta.target <- unname(as.matrix(coef(fit.target)$cluster))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, cluster, lambda = 0)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# Cross-validation
#==================================================================================================#

testthat::test_that('cross-validation works correctly', {

  # Fit models
  fit <- cv.glmmsel(x, y, cluster, alpha = 1)
  fit.target <- lme4::lmer(y ~ x.1 + x.2 + x.3 + x.4 + x.5 + (x.1 + x.2 + x.3 + x.4 + x.5 || cluster), df, REML = FALSE)

  # Test fixef
  beta <- fixef(fit)
  beta.target <- c(as.vector(lme4::fixef(fit.target)), rep(0, p - s))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test ranef
  u <- ranef(fit)
  u.target <- cbind(unname(as.matrix(lme4::ranef(fit.target)$cluster)), matrix(0, m, p - s))
  testthat::expect_equal(u, u.target, tolerance = 1e-2)

  # Test coef
  beta <- coef(fit)
  beta.target <- cbind(unname(as.matrix(coef(fit.target)$cluster)), matrix(0, m, p - s))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, cluster)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

testthat::test_that('cross-validation with local search works correctly', {

  # Fit models
  fit <- cv.glmmsel(x, y, cluster, alpha = 1, local.search = TRUE)
  fit.target <- lme4::lmer(y ~ x.1 + x.2 + x.3 + x.4 + x.5 + (x.1 + x.2 + x.3 + x.4 + x.5 || cluster), df, REML = FALSE)

  # Test fixef
  beta <- fixef(fit)
  beta.target <- c(as.vector(lme4::fixef(fit.target)), rep(0, p - s))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test ranef
  u <- ranef(fit)
  u.target <- cbind(unname(as.matrix(lme4::ranef(fit.target)$cluster)), matrix(0, m, p - s))
  testthat::expect_equal(u, u.target, tolerance = 1e-2)

  # Test coef
  beta <- coef(fit)
  beta.target <- cbind(unname(as.matrix(coef(fit.target)$cluster)), matrix(0, m, p - s))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, cluster)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

#==================================================================================================#
# Heuristics
#==================================================================================================#

testthat::test_that('screening works correctly', {

  # Fit models
  fit <- cv.glmmsel(x, y, cluster, alpha = 1, screen = 5)
  fit.target <- lme4::lmer(y ~ x.1 + x.2 + x.3 + x.4 + x.5 + (x.1 + x.2 + x.3 + x.4 + x.5 || cluster), df, REML = FALSE)

  # Test fixef
  beta <- fixef(fit)
  beta.target <- c(as.vector(lme4::fixef(fit.target)), rep(0, p - s))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test ranef
  u <- ranef(fit)
  u.target <- cbind(unname(as.matrix(lme4::ranef(fit.target)$cluster)), matrix(0, m, p - s))
  testthat::expect_equal(u, u.target, tolerance = 1e-2)

  # Test coef
  beta <- coef(fit)
  beta.target <- cbind(unname(as.matrix(coef(fit.target)$cluster)), matrix(0, m, p - s))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, cluster)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})

testthat::test_that('no sorting works correctly', {

  # Fit models
  fit <- cv.glmmsel(x, y, cluster, alpha = 1, sort = FALSE)
  fit.target <- lme4::lmer(y ~ x.1 + x.2 + x.3 + x.4 + x.5 + (x.1 + x.2 + x.3 + x.4 + x.5 || cluster), df, REML = FALSE)

  # Test fixef
  beta <- fixef(fit)
  beta.target <- c(as.vector(lme4::fixef(fit.target)), rep(0, p - s))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test ranef
  u <- ranef(fit)
  u.target <- cbind(unname(as.matrix(lme4::ranef(fit.target)$cluster)), matrix(0, m, p - s))
  testthat::expect_equal(u, u.target, tolerance = 1e-2)

  # Test coef
  beta <- coef(fit)
  beta.target <- cbind(unname(as.matrix(coef(fit.target)$cluster)), matrix(0, m, p - s))
  testthat::expect_equal(beta, beta.target, tolerance = 1e-2)

  # Test predict
  yhat <- predict(fit, x, cluster)
  yhat.target <- as.vector(predict(fit.target, df))
  testthat::expect_equal(yhat, yhat.target, tolerance = 1e-2)

})
