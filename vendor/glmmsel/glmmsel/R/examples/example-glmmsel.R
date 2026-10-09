# Generate data
set.seed(1234)
n <- 100
m <- 4
p <- 10
s <- 5
x <- matrix(rnorm(n * p), n, p)
beta <- c(rep(1, s), rep(0, p - s))
u <- cbind(matrix(rnorm(m * s), m, s), matrix(0, m, p - s))
cluster <- sample(1:m, n, replace = TRUE)
xb <- rowSums(x * sweep(u, 2, beta, '+')[cluster, ])
y <- rnorm(n, xb)

# Fit sparse linear mixed model
fit <- glmmsel(x, y, cluster)
plot(fit)
fixef(fit, lambda = 10)
ranef(fit, lambda = 10)
coef(fit, lambda = 10)
predict(fit, x[1:3, ], cluster[1:3], lambda = 10)
