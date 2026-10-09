#ifndef path_H
#define path_H
#include "cd.h"
#include "ls.h"
#include "pql.h"

// Contains data and functions for fitting regularisation path

class path {

  public:

    const arma::field<arma::mat> x;
    const arma::field<arma::vec> y;
    const std::string family;
    const int max_nnz;
    const double lambda_step;
    arma::vec lambda;
    const double alpha;
    const bool intercept;
    const bool random_intercept;
    arma::vec beta0;
    arma::vec gamma0;
    arma::mat beta;
    arma::mat gamma;
    arma::cube u;
    arma::vec sigma2;
    arma::vec loss;
    arma::vec cd_iter;
    arma::vec ls_iter;
    arma::vec pql_iter;
    arma::vec nnz;
    unsigned nlambda;

  path(
    const arma::field<arma::mat>& x,
    const arma::field<arma::vec>& y,
    const std::string& family,
    const int& max_nnz,
    const double& lambda_step,
    arma::vec lambda,
    const double& alpha,
    const bool& intercept,
    const bool& random_intercept
  ) :
    x(x),
    y(y),
    family(family),
    max_nnz(max_nnz),
    lambda_step(lambda_step),
    lambda(lambda),
    alpha(alpha),
    intercept(intercept),
    random_intercept(random_intercept)
  {
    unsigned m = y.n_elem;
    unsigned p = x(0).n_cols;
    nlambda = lambda.n_elem;
    beta0 = arma::vec(nlambda);
    gamma0 = arma::vec(nlambda);
    beta = arma::mat(p, nlambda);
    gamma = arma::mat(p, nlambda);
    u = arma::cube(m, (intercept && random_intercept ? 1 : 0) + p, nlambda);
    sigma2 = arma::vec(nlambda);
    loss = arma::vec(nlambda);
    cd_iter = arma::vec(nlambda);
    ls_iter = arma::vec(nlambda);
    pql_iter = arma::vec(nlambda);
    nnz = arma::vec(nlambda);
  };

  void run(cd &cd, ls &ls, pql &pql);

};

#endif
