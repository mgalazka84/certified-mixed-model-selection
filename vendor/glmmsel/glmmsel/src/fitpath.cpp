#include <RcppArmadillo.h>
#include "path.h"

// Provides R interface for fitting the regularisation path

// [[Rcpp::export]]

Rcpp::List fitpath(
    const arma::field<arma::mat>& x,
    const arma::field<arma::vec>& y,
    const std::string& family,
    const bool& local_search,
    const int& max_nnz,
    const double& lambda_step,
    arma::vec lambda,
    const double& alpha,
    const bool& intercept,
    const bool& random_intercept,
    const double& eps,
    const unsigned& max_cd_iter,
    const unsigned& max_ls_iter,
    const unsigned& max_bls_iter,
    const double& t_init,
    const double& t_scale,
    const unsigned& max_pql_iter,
    const bool& active_set,
    const unsigned& active_set_count,
    const bool& sort,
    const unsigned& screen
) {
  cd cd(eps, max_cd_iter, max_bls_iter, t_init, t_scale, active_set, active_set_count, sort, screen);
  ls ls(local_search, max_ls_iter);
  pql pql(max_pql_iter);
  path path(x, y, family, max_nnz, lambda_step, lambda, alpha, intercept, random_intercept);
  path.run(cd, ls, pql);
  return Rcpp::List::create(
    Rcpp::Named("beta0") = path.beta0,
    Rcpp::Named("gamma0") = path.gamma0,
    Rcpp::Named("beta") = path.beta,
    Rcpp::Named("gamma") = path.gamma,
    Rcpp::Named("u") = path.u,
    Rcpp::Named("sigma2") = path.sigma2,
    Rcpp::Named("loss") = path.loss,
    Rcpp::Named("cd.iter") = path.cd_iter,
    Rcpp::Named("ls.iter") = path.ls_iter,
    Rcpp::Named("pql.iter") = path.pql_iter,
    Rcpp::Named("nnz") = path.nnz,
    Rcpp::Named("lambda") = path.lambda
  );
}

