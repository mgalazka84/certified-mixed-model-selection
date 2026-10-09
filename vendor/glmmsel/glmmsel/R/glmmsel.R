#' @title Generalised linear mixed model selection

#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Fits the regularisation path for a sparse generalised linear mixed model (GLMM).
#'
#' @param x a predictor matrix
#' @param y a response vector
#' @param cluster a vector of length \code{nrow(x)} with the jth element identifying the cluster
#' that the jth observation belongs to
#' @param family the likelihood family to use; 'gaussian' for a continuous response or 'binomial'
#' for a binary response
#' @param local.search a logical indicating whether to perform local search after coordinate
#' descent; typically leads to higher quality solutions
#' @param max.nnz the maximum number of predictors ever allowed to be active
#' @param nlambda the number of regularisation parameters to evaluate when \code{lambda} is
#' computed automatically
#' @param lambda.step the step size taken when computing \code{lambda} from the data; should be a
#' value strictly between 0 and 1; larger values typically lead to a finer grid of subset sizes
#' @param lambda an optional vector of regularisation parameters
#' @param alpha the hierarchical parameter
#' @param intercept a logical indicating whether to include a fixed intercept
#' @param random.intercept a logical indicating whether to include a random intercept; applies
#' only when \code{intercept = TRUE}
#' @param standardise a logical indicating whether to scale the data to have unit root mean square;
#' all parameters are returned on the original scale of the data
#' @param eps the convergence tolerance; convergence is declared when the relative maximum
#' difference in consecutive parameter values is less than \code{eps}
#' @param max.cd.iter the maximum number of coordinate descent iterations allowed
#' @param max.ls.iter the maximum number of local search iterations allowed
#' @param max.bls.iter the maximum number of backtracking line search iterations allowed
#' @param t.init the initial value of the gradient step size during backtracking line search
#' @param t.scale the scaling parameter of the gradient step size during backtracking line search
#' @param max.pql.iter the maximum number of penalised quasi-likelihood iterations allowed
#' @param active.set a logical indicating whether to use active set updates; typically lowers the
#' run time
#' @param active.set.count the number of consecutive coordinate descent iterations in which a
#' subset should appear before running active set updates
#' @param sort a logical indicating whether to sort the coordinates before running coordinate
#' descent; typically leads to higher quality solutions
#' @param screen the number of predictors to keep after gradient screening; smaller values typically
#' lower the run time
#' @param warn a logical indicating whether to print a warning if the algorithms fail to
#' converge
#'
#' @return An object of class \code{glmmsel}; a list with the following components:
#' \item{beta0}{a vector of fixed intercepts}
#' \item{gamma0}{a vector of random intercept variances}
#' \item{beta}{a matrix of fixed slopes}
#' \item{gamma}{a matrix of random slope variances}
#' \item{u}{an array of random coefficient predictions}
#' \item{sigma2}{a vector of residual variances}
#' \item{loss}{a vector of loss function values}
#' \item{cd.iter}{a vector indicating the number of coordinate descent iterations for convergence}
#' \item{ls.iter}{a vector indicating the number of local search iterations for convergence}
#' \item{pql.iter}{a vector indicating the number of penalised quasi-likelihood iterations for
#' convergence}
#' \item{nnz}{a vector of the number of nonzeros}
#' \item{lambda}{a vector of regularisation parameters used for the fit}
#' \item{family}{the likelihood family used}
#' \item{clusters}{a vector of cluster identifiers}
#' \item{alpha}{the value of the hierarchical parameter used for the fit}
#' \item{intercept}{whether a fixed intercept is included in the model}
#' \item{random.intercept}{whether a random intercept is included in the model}
#'
#' @references Thompson, R., Wand, M. P., and Wang, J. J. J. (2025). 'Scalable subset selection in
#' linear mixed models'. arXiv: \href{https://arxiv.org/abs/2506.20425}{2506.20425}.
#'
#' @example R/examples/example-glmmsel.R
#'
#' @export

glmmsel <- \(
  x,
  y,
  cluster,
  family = c('gaussian', 'binomial'),
  local.search = FALSE,
  max.nnz = 100,
  nlambda = 100,
  lambda.step = 0.99,
  lambda = NULL,
  alpha = 0.8,
  intercept = TRUE,
  random.intercept = TRUE,
  standardise = TRUE,
  eps = 1e-4,
  max.cd.iter = 1e4,
  max.ls.iter = 100,
  max.bls.iter = 30,
  t.init = 1,
  t.scale = 0.5,
  max.pql.iter = 100,
  active.set = TRUE,
  active.set.count = 3,
  sort = TRUE,
  screen = 100,
  warn = TRUE
) {

  family <- match.arg(family)

  # Check data is valid
  if (!is.matrix(x)) x <- as.matrix(x)
  if (!is.matrix(y)) y <- as.matrix(y)
  if (anyNA(y)) stop('y contains NAs; remove or impute rows with missing values')
  if (anyNA(x)) stop('x contains NAs; remove or impute rows with missing values')
  attributes(y) <- list(dim = attributes(y)$dim)
  attributes(x) <- list(dim = attributes(x)$dim)
  if (nrow(x) != nrow(y)) stop('x and y must have same number of observations')

  # Check arguments are valid
  if (nlambda < 1) stop('nlambda must be at least 1')
  if (alpha < 0 || alpha > 1) stop('alpha must be between 0 and 1 (inclusive)')
  if (lambda.step <= 0 || lambda.step >= 1) stop('lambda.step must be between 0 and 1 (strictly)')

  # Save problem dimensions
  n <- nrow(x)
  p <- ncol(x)
  unique.cluster <- unique(cluster)

  # Standardise data
  if (standardise) {
    if (family == 'binomial') {
      y.s <- 1
    } else {
      y <- scale(y, center = FALSE)
      y.s <- attr(y, 'scaled:scale')
    }
    x <- scale(x, center = FALSE)
    x.s <- attr(x, 'scaled:scale')
  }

  # Set up regularisation sequence
  if (is.null(lambda)) {
    lambda <- rep(- 1, nlambda)
  }

  # Organise data into clusters
  x <- lapply(unique.cluster, \(i) x[cluster == i, , drop = FALSE])
  y <- lapply(unique.cluster, \(i) y[cluster == i])

  # Fit regularisation path
  result <- fitpath(x, y, family, local.search, max.nnz, lambda.step, lambda, alpha, intercept,
                    random.intercept, eps, max.cd.iter, max.ls.iter, max.bls.iter, t.init, t.scale,
                    max.pql.iter, active.set, active.set.count, sort, screen)

  # Rescale parameters
  if (standardise) {
    result$beta0 <- y.s * result$beta0
    result$beta <- y.s / x.s * result$beta
    result$gamma <- 1 / x.s ^ 2 * result$gamma
    if (intercept && random.intercept) {
      result$u <- sweep(result$u, 2, c(y.s, y.s / x.s), '*')
    } else {
      result$u <- sweep(result$u, 2, y.s / x.s, '*')
    }
    result$sigma2 <- y.s ^ 2 * result$sigma2;
  }

  # Warn if maximum iterations reached
  if (warn & any(result$cd.iter == max.cd.iter)) {
    warning('coordinate descent did not converge for at least one value of the regularisation
            parameter')
  }
  if (warn & any(result$ls.iter == max.ls.iter)) {
    warning('local search did not converge for at least one value of the regularisation parameter')
  }
  if (warn & any(result$pql.iter == max.pql.iter)) {
    warning('penalised quasi-likelihood did not converge for at least one value of the \
            regularisation parameter')
  }

  # Return fitted model
  result$beta0 <- as.numeric(result$beta0)
  result$gamma0 <- as.numeric(result$gamma0)
  result$sigma2 <-  as.numeric(result$sigma2)
  result$loss <- as.numeric(result$loss)
  result$cd.iter <- as.numeric(result$cd.iter)
  result$ls.iter <- as.numeric(result$ls.iter)
  result$pql.iter <- as.numeric(result$pql.iter)
  result$nnz <- as.numeric(result$nnz)
  result$lambda <- as.numeric(result$lambda)
  result$family <- family
  result$clusters <- unique.cluster
  result$alpha <- alpha
  result$intercept <- intercept
  result$random.intercept <- random.intercept
  class(result) <- 'glmmsel'
  return(result)

}

#==================================================================================================#
# Fixed effects function
#==================================================================================================#

#' @title Fixed effects function for \code{glmmsel} object
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Extracts fixed effects for a specified value of the regularisation parameter.
#'
#' @param object an object of class \code{glmmsel}
#' @param lambda a value of the regularisation parameter
#' @param ... any other arguments
#'
#' @return A matrix of fixed effects.
#'
#' @export

fixef.glmmsel <- \(object, lambda = NULL, ...) {

  if (is.null(lambda)) {
    if (object$intercept) {
      rbind(object$beta0, object$beta)
    } else {
      object$beta
    }
  } else {
    index <- which.min(abs(lambda - object$lambda))
    if (object$intercept) {
      c(object$beta0[index], object$beta[, index])
    } else {
      object$beta[, index]
    }
  }

}

#==================================================================================================#
# Random effects function
#==================================================================================================#

#' @title Random effects function for \code{glmmsel} object

#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Extracts random effects for a specified value of the regularisation parameter.
#'
#' @param object an object of class \code{glmmsel}
#' @param lambda a value of the regularisation parameter
#' @param ... any other arguments
#'
#' @return A matrix of random effects.
#'
#' @export

ranef.glmmsel <- \(object, lambda = NULL, ...) {

  if (is.null(lambda)) {
    object$u
  } else {
    index <- which.min(abs(lambda - object$lambda))
    result <- object$u[, , index, drop = FALSE]
    dim(result) <- dim(result)[1:2]
    result
  }

}

#==================================================================================================#
# Coefficient function
#==================================================================================================#

#' @title Coefficient function for \code{glmmsel} object
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Extracts coefficients for a specified value of the regularisation parameter.
#'
#' @param object an object of class \code{glmmsel}
#' @param lambda a value of the regularisation parameter
#' @param ... any other arguments
#'
#' @return An array of coefficients.
#'
#' @method coef glmmsel
#'
#' @export
#'
#' @importFrom stats "coef"

coef.glmmsel <- \(object, lambda = NULL, ...) {

  beta <- fixef(object, lambda)
  u <- ranef(object, lambda)
  if (is.null(lambda)) {
    if (object$intercept & !object$random.intercept) {
      dims <- dim(u)
      dims[2] <- dims[2] + 1
      result <- array(dim = dims)
      for (i in 1:dims[3]) {
        result[, 1, i] <- beta[1, i]
        result[, - 1, i] <- sweep(u[, , i, drop = FALSE], 2, beta[- 1, i], '+')
      }
      result
    } else {
      dims <- dim(u)
      result <- array(dim = dims)
      for (i in 1:dims[3]) {
        result[, , i] <- sweep(u[, , i, drop = FALSE], 2, beta[, i], '+')
      }
      result
    }
  } else {
    index <- which.min(abs(lambda - object$lambda))
    if (object$intercept & !object$random.intercept) {
      cbind(beta[1], sweep(u, 2, beta[- 1], '+'))
    } else {
      sweep(u, 2, beta, '+')
    }
  }

}

#==================================================================================================#
# Predict function
#==================================================================================================#

#' @title Predict function for \code{glmmsel} object
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Generates predictions for new data using a specified value of the regularisation
#' parameter.
#'
#' @param object an object of class \code{glmmsel}
#' @param x.new a matrix of new values for the predictors
#' @param cluster.new a vector identifying the clusters that the rows of \code{x.new} belong to
#' @param lambda a value of the regularisation parameter
#' @param ... any other arguments
#'
#' @return A matrix of predictions.
#'
#' @method predict glmmsel
#'
#' @export
#'
#' @importFrom stats "predict"

predict.glmmsel <- \(object, x.new, cluster.new, lambda = NULL, ...) {

  if (object$intercept) x.new <- cbind(1, x.new)
  if (is.null(lambda)) {
    cluster.coefs <- coef(object, lambda)
    beta <- fixef(object, lambda)
    cluster.coefs <- lapply(1:dim(cluster.coefs)[3], \(i) rbind(cluster.coefs[, , i], beta[, i]))
    ind <- match(cluster.new, object$clusters)
    ind[is.na(ind)] <- length(object$clusters) + 1
    yhat <- sapply(1:length(cluster.coefs), \(i) rowSums(x.new * cluster.coefs[[i]][ind, ]), simplify = FALSE)
    simplify2array(yhat, except = NULL)
  } else {
    cluster.coefs <- coef(object, lambda)
    beta <- fixef(object, lambda)
    cluster.coefs <- unname(rbind(cluster.coefs, beta))
    ind <- match(cluster.new, object$clusters)
    ind[is.na(ind)] <- length(object$clusters) + 1
    rowSums(x.new * cluster.coefs[ind, ])
  }

}

#==================================================================================================#
# Plot function
#==================================================================================================#

#' @title Plot function for \code{glmmsel} object
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Plots the coefficients as a function of the regularisation parameter.
#'
#' @param x an object of class \code{glmmsel}
#' @param cluster the cluster whose coefficients to plot
#' @param ... any other arguments
#'
#' @return A plot of the coefficient profiles.
#'
#' @method plot glmmsel
#'
#' @export
#'
#' @importFrom graphics "plot"

plot.glmmsel <- \(x, cluster = NULL, ...) {

  if (is.null(cluster)) {
    beta <- fixef(x)
  } else {
    beta <- coef(x)[cluster, , ]
  }
  beta <- beta[- 1, , drop = FALSE]
  df <- data.frame(beta = as.vector(beta), predictor = as.factor(seq_along(beta[, 1])),
                   lambda = rep(x$lambda, each = nrow(beta)))
  df <- df[df$beta != 0, ]
  p <- ggplot2::ggplot(df, ggplot2::aes_string('lambda', 'beta', col = 'predictor')) +
    ggplot2::geom_point()
  p

}
