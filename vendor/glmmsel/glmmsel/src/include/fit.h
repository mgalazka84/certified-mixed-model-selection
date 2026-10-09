#ifndef fit_H
#define fit_H
#include "data.h"

// fit struct contains model fit

struct fit {

  public:

    double beta0;
    double gamma0;
    arma::vec beta;
    arma::vec gamma;
    arma::field<arma::vec> r;
    arma::field<arma::mat> v_inv;
    arma::vec v_logdet;
    double loss;
    arma::uvec active;

    fit() : beta0(), gamma0(), beta(), gamma(), r(), v_inv(), v_logdet(), loss(), active() {};

    fit(
      const data& data
    ) {
      beta0 = 0;
      gamma0 = 0;
      beta = arma::vec(data.p, arma::fill::zeros);
      gamma = arma::vec(data.p, arma::fill::zeros);
      r = arma::field<arma::vec>(data.m);
      v_inv = arma::field<arma::mat>(data.m);
      v_logdet = arma::vec(data.m);
      double rt_vinv_r = 0;
      for (arma::uword i = 0; i < data.m; ++i) {
        r(i) = data.y(i);
        arma::mat v = arma::eye(data.x(i).n_rows, data.x(i).n_rows);
        v_inv(i) = v;
        v_logdet(i) = 0;
        rt_vinv_r += arma::dot(r(i).t() * v_inv(i), r(i));
      }
      loss = arma::sum(v_logdet) + data.n * std::log(rt_vinv_r);
      active = arma::uvec(data.p, arma::fill::zeros);
    };

};

#endif
