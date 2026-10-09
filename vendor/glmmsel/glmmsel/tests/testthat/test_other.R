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

testthat::test_that('x can be a data frame', {
  testthat::expect_no_error(glmmsel(data.frame(x), y, cluster))
  testthat::expect_no_error(cv.glmmsel(data.frame(x), y, cluster))
})

testthat::test_that('missing values are not allowed', {
  testthat::expect_error(glmmsel(rbind(x, NA), y, cluster))
  testthat::expect_error(glmmsel(x, rbind(y, NA), cluster))
})

testthat::test_that('y and x must have the same number of observations', {
  testthat::expect_error(glmmsel(x, rbind(y, 1), cluster))
})

testthat::test_that('incorrect specification of arguments is not allowed', {
  testthat::expect_error(glmmsel(x, y, cluster, nlambda = 0))
  testthat::expect_error(glmmsel(x, y, cluster, alpha = 2))
  testthat::expect_error(glmmsel(x, y, cluster, lambda.step = 2))
  testthat::expect_error(cv.glmmsel(x, y, cluster, nfold = 1))
  testthat::expect_error(cv.glmmsel(x, y, cluster, folds = rep(1, n + 1)))
})

testthat::test_that('manual specification of folds runs without error', {
  testthat::expect_no_error(cv.glmmsel(x, y, cluster, folds = sample(1:10, n, replace = TRUE)))
})

testthat::test_that('plot function returns a plot', {
  testthat::expect_s3_class(plot(glmmsel(x, y, cluster)), 'ggplot')
  testthat::expect_s3_class(plot(glmmsel(x, y, cluster), 1), 'ggplot')
  testthat::expect_s3_class(plot(cv.glmmsel(x, y, cluster)), 'ggplot')
})

testthat::test_that('when max.cd.iter is exceeded a warning is provided', {
  testthat::expect_warning(glmmsel(x, y, cluster, max.cd.iter = 0))
})

testthat::test_that('when max.ls.iter is exceeded a warning is provided', {
  testthat::expect_warning(glmmsel(x, y, cluster, max.ls.iter = 0))
})

testthat::test_that('when max.pql.iter is exceeded a warning is provided', {
  testthat::expect_warning(glmmsel(x, y, cluster, max.pql.iter = 0))
})
