#include <RcppArmadillo.h>
#include "path.h"
#include "auxiliary.h"

typedef fit fit_type;

// Fit regularisation path

void path::run(cd& cd, ls &ls, pql &pql) {

  // Set up initialisation point

  data data(x, y, family, alpha, intercept, random_intercept);
  fit fit(data);

  // Loop over lambda values

  bool compute_lambda = lambda(0) < 0;
  arma::uword i;
  for (i = 0; i < nlambda; ++i) {

    // Update lambda value

    data.lambda = lambda(i);
    data.lambda_step = lambda_step;
    if (i == 0 && lambda(0) < 0) data.lambda = 1e6;//arma::datum::inf;

    // Reset iteration counts

    cd.iter = 0;
    ls.iter = 0;
    pql.iter = 0;

    // Run algorithms directly if Gaussian family, otherwise run via penalised quasi-likelihood

    if (family == "gaussian") {

      cd.run(fit, data);
      if (ls.local_search) ls.run(fit, data, cd);

    } else {

      pql.run(fit, data, cd, ls, y);

    }

    // Exit if model is saturated

    if (compute_lambda && (data.intercept + data.random_intercept + arma::sum(fit.beta != 0) +
        arma::sum(fit.gamma != 0) + 1 >= data.n)) {
      i--;
      break;
    }

    double resid_var = residvar(fit, data);
    if (compute_lambda && family == "binomial" && (resid_var < 0.01 || resid_var > 100)) {
      i--;
      break;
    }

    // Exit if maximum number of active predictors is exceeded

    int nnz_i = arma::sum(fit.active);
    if (compute_lambda && (nnz_i > max_nnz)) {
      i--;
      break;
    }

    // Save solution for current point in path

    beta0(i) = fit.beta0;
    gamma0(i) = fit.gamma0;
    beta.col(i) = fit.beta;
    gamma.col(i) = fit.gamma;
    u.slice(i) = blup(fit, data);
    sigma2(i) = resid_var;
    loss(i) = fit.loss;
    cd_iter(i) = cd.iter;
    ls_iter(i) = ls.iter;
    pql_iter(i) = pql.iter;
    nnz(i) = nnz_i;
    lambda(i) = data.lambda;

    // Exit if maximum number of active predictors is reached

    if (compute_lambda && (nnz_i == max_nnz)) {
      break;
    }

    // Exit if all predictors are active

    if (compute_lambda && (nnz_i == data.p)) {
      break;
    }

  }

  // Trim any empty fits

  if (i == nlambda) i--;
  if (i < nlambda - 1) {
    beta0.shed_rows(i + 1, nlambda - 1);
    gamma0.shed_rows(i + 1, nlambda - 1);
    beta.shed_cols(i + 1, nlambda - 1);
    gamma.shed_cols(i + 1, nlambda - 1);
    u.shed_slices(i + 1, nlambda - 1);
    sigma2.shed_rows(i + 1, nlambda - 1);
    loss.shed_rows(i + 1, nlambda - 1);
    cd_iter.shed_rows(i + 1, nlambda - 1);
    ls_iter.shed_rows(i + 1, nlambda - 1);
    pql_iter.shed_rows(i + 1, nlambda - 1);
    nnz.shed_rows(i + 1, nlambda - 1);
    lambda.shed_rows(i + 1, nlambda - 1);
  }

}
