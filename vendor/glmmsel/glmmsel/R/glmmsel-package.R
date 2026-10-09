## usethis namespace: start
#' @useDynLib glmmsel, .registration = TRUE
## usethis namespace: end
NULL

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
## usethis namespace: end
NULL

#' @title Fixed effects function
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Generic function for extracting fixed effects from model objects.
#'
#' @param object a model object
#' @param ... any other arguments
#'
#' @return Depends on the specific method implementation.
#'
#' @export

fixef <- function(object, ...) {
  UseMethod('fixef')
}

#' @title Random effects function
#'
#' @author Ryan Thompson <ryan.thompson-1@uts.edu.au>
#'
#' @description Generic function for extracting random effects from model objects.
#'
#' @param object a model object
#' @param ... any other arguments
#'
#' @return Depends on the specific method implementation.
#'
#' @export

ranef <- function(object, ...) {
  UseMethod('ranef')
}
