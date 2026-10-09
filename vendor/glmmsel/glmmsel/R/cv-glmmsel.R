#' @title Cross-validated generalised linear mixed model selection
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Fits the regularisation path for a sparse generalised linear mixed model and then
#' cross-validates this path.
#'
#' @param x a predictor matrix
#' @param y a response vector
#' @param cluster a vector of length \code{nrow(x)} with the jth element identifying the cluster
#' that the jth observation belongs to
#' @param family the likelihood family to use; 'gaussian' for a continuous response or 'binomial'
#' for a binary response
#' @param lambda the regularisation parameter for the overlapping penalty on the fixed and random
#' slopes
#' @param nfold the number of cross-validation folds
#' @param folds an optional vector of length \code{nrow(x)} with the jth entry identifying the fold
#' that the jth observation belongs to
#' @param cv.loss an optional cross-validation loss-function to use; should accept a vector of
#' predicted values and a vector of actual values
#' @param interpolate a logical indicating whether to interpolate the \code{lambda} sequence for
#' the cross-validation fits
#' @param ... any other arguments for \code{glmmsel()}
#'
#' @return An object of class \code{cv.glmmsel}; a list with the following components:
#' \item{cv.mean}{a vector of cross-validation means}
#' \item{cv.sd}{a vector of cross-validation standard errors}
#' \item{lambda}{a vector of cross-validated regularisation parameters}
#' \item{lambda.min}{the value of \code{lambda} minimising \code{cv.mean}}
#' \item{fit}{the fit from running \code{glmmsel()} on the full data}
#'
#' @references Thompson, R., Wand, M. P., and Wang, J. J. J. (2025). 'Scalable subset selection in
#' linear mixed models'. arXiv: \href{https://arxiv.org/abs/2506.20425}{2506.20425}.
#'
#' @example R/examples/example-cv-glmmsel.R
#'
#' @export

cv.glmmsel <- \(
  x,
  y,
  cluster,
  family = c('gaussian', 'binomial'),
  lambda = NULL,
  nfold = 10,
  folds = NULL,
  cv.loss = NULL,
  interpolate = TRUE,
  ...
) {

  family <- match.arg(family)

  # Check data is valid
  if (!is.matrix(x)) x <- as.matrix(x)
  if (!is.matrix(y)) y <- as.matrix(y)

  # Check arguments are valid
  if (nfold < 2 | nfold > nrow(x)) {
    stop('nfold must be at least 2 and at most the number of rows in x')
  }
  if (!is.null(folds) & length(folds) != nrow(x)) {
    stop('length of folds must equal number of rows in x')
  }

  # Perform initial fit
  lambda.compute <- is.null(lambda)
  fit <- glmmsel(x, y, cluster, family, lambda = lambda, ...)
  lambda <- fit$lambda
  nlambda <- length(lambda)
  if (is.null(folds)) {
    if (family == 'gaussian') {
      folds <- sample(rep_len(1:nfold, nrow(x)))
    } else if (family == 'binomial') {
      folds <- integer(nrow(x))
      folds[y == 0] <- sample(rep_len(1:nfold, sum(y == 0)))
      folds[y == 1] <- sample(rep_len(1:nfold, sum(y == 1)))
    }
  } else {
    nfold <- length(unique(folds))
  }

  # Save cross-validation loss functions
  if (is.null(cv.loss)) {
    if (family == 'gaussian') {
      cv.loss <- \(xb, y) 0.5 * mean((y - xb) ^ 2)
    } else if (family == 'binomial') {
      cv.loss <- \(xb, y) {
        pi <- pmax(1e-5, pmin(1 - 1e-5, 1 / (1 + exp(- xb))))
        - mean(y * log(pi) + (1 - y) * log(1 - pi))
      }
    }
  }

  # If lambda was computed, use midpoints between consecutive lambdas in cross-validation
  lambda.cv <- lambda
  if (interpolate && lambda.compute && (length(lambda) > 2)) {
    lambda.cv[2:(nlambda - 1)] <- lambda[- c(1, nlambda)] + diff(lambda[- 1]) / 2
  }

  # Loop over folds
  cvf <- \(fold) {
    fold.ind <- which(folds == fold)
    x.train <- x[- fold.ind, , drop = FALSE]
    x.valid <- x[fold.ind, , drop = FALSE]
    y.train <- y[- fold.ind, , drop = FALSE]
    y.valid <- y[fold.ind, , drop = FALSE]
    cluster.train <- cluster[- fold.ind]
    cluster.valid <- cluster[fold.ind]
    fit.fold <- glmmsel(x.train, y.train, cluster.train, family, lambda = lambda.cv, ...)
    yhat <- predict(fit.fold, x.valid, cluster.valid)
    apply(yhat, 2, cv.loss, y.valid)
  }
  cv <- simplify2array(sapply(1:nfold, cvf), except = NULL)

  # Compose cross-validation results
  cv.mean <- apply(cv, 1, mean)
  cv.sd <- apply(cv, 1, stats::sd)
  lambda.min <- lambda[which.min(cv.mean)]

  # Return cross validated model
  result <- list(
    cv.mean = cv.mean,
    cv.sd = cv.sd,
    lambda = lambda,
    lambda.min = lambda.min,
    fit = fit
  )
  class(result) <- 'cv.glmmsel'
  return(result)

}

#==================================================================================================#
# Fixed effects function
#==================================================================================================#

#' @title Fixed effects function for \code{cv.glmmsel} object
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Extract fixed effects for a cross-validated value of the regularisation parameter.
#'
#' @param object an object of class \code{cv.glmmsel}
#' @param lambda a value of the regularisation parameter
#' @param ... any other arguments
#'
#' @return A matrix of fixed effects.
#'
#' @export

fixef.cv.glmmsel <- \(object, lambda = 'lambda.min', ...) {

  if (!is.null(lambda)) if (lambda == 'lambda.min') lambda <- object$lambda.min
  fixef(object$fit, lambda = lambda, ...)

}

#==================================================================================================#
# Random effects function
#==================================================================================================#

#' @title Random effects function for \code{cv.glmmsel} object
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Extract random effects for a cross-validated value of the regularisation parameter.
#'
#' @param object an object of class \code{cv.glmmsel}
#' @param lambda a value of the regularisation parameter
#' @param ... any other arguments
#'
#' @return A matrix of random effects.
#'
#' @export

ranef.cv.glmmsel <- \(object, lambda = 'lambda.min', ...) {

  if (!is.null(lambda)) if (lambda == 'lambda.min') lambda <- object$lambda.min
  ranef(object$fit, lambda = lambda, ...)

}

#==================================================================================================#
# Coefficient function
#==================================================================================================#

#' @title Coefficient function for \code{cv.glmmsel} object
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Extract cluster coefficients for a cross-validated value of the regularisation
#' parameter.
#'
#' @param object an object of class \code{cv.glmmsel}
#' @param lambda a value of the regularisation parameter
#' @param ... any other arguments
#'
#' @return An array of coefficients.
#'
#' @method coef cv.glmmsel
#'
#' @export
#'
#' @importFrom stats "coef"

coef.cv.glmmsel <- \(object, lambda = 'lambda.min', ...) {

  if (!is.null(lambda)) if (lambda == 'lambda.min') lambda <- object$lambda.min
  coef(object$fit, lambda = lambda, ...)

}

#==================================================================================================#
# Predict function
#==================================================================================================#

#' @title Predict function for \code{cv.glmmsel} object
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Generates predictions for new data using a cross-validated value of the
#' regularisation parameter.
#'
#' @param object an object of class \code{cv.glmmsel}
#' @param x.new a matrix of new values for the predictors
#' @param cluster.new a vector identifying the clusters that the rows of \code{x.new} belong to
#' @param lambda a value of the regularisation parameter
#' @param ... any other arguments
#'
#' @return A matrix of predictions.
#'
#' @method predict cv.glmmsel
#'
#' @export
#'
#' @importFrom stats "predict"

predict.cv.glmmsel <- \(object, x.new, cluster.new, lambda = 'lambda.min', ...) {

  if (!is.null(lambda)) if (lambda == 'lambda.min') lambda <- object$lambda.min
  predict(object$fit, x.new, cluster.new, lambda = lambda, ...)

}

#==================================================================================================#
# Plot function
#==================================================================================================#

#' @title Plot function for \code{cv.glmmsel} object
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Plot the cross-validation loss as a function of the regularisation parameter.
#'
#' @param x an object of class \code{cv.glmmsel}
#' @param ... any other arguments
#'
#' @return A plot of the cross-validation results.
#'
#' @method plot cv.glmmsel
#'
#' @export
#'
#' @importFrom graphics "plot"

plot.cv.glmmsel <- \(x, ...) {

  df <- data.frame(cv.mean = x$cv.mean, cv.sd = x$cv.sd, lambda = x$lambda)
  p <- ggplot2::ggplot(df, ggplot2::aes_string('lambda', 'cv.mean')) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::geom_errorbar(ggplot2::aes_string(ymin = 'cv.mean - cv.sd', ymax = 'cv.mean + cv.sd'))
  p

}
