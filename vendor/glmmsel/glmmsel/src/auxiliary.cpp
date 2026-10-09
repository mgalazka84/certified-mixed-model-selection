#include <RcppArmadillo.h>
#include "auxiliary.h"

// Loss function

double loss(const fit& fit, const data& data) {
  double rt_vinv_r = 0;
  for (arma::uword i = 0; i < data.m; ++i) {
    rt_vinv_r += arma::dot(fit.r(i).t() * fit.v_inv(i), fit.r(i));
  }
  return arma::sum(fit.v_logdet) + data.n * std::log(rt_vinv_r);
}

// Intercept gradient function

arma::vec grad_intercept(const fit& fit, const data& data) {
  double rt_vinv_r = 0;
  double sum_vinv_r = 0;
  double sum_vinv_r2 = 0;
  double sum_vinv = 0;
  for (arma::uword i = 0; i < data.m; ++i) {
    arma::vec vinv_r = fit.v_inv(i) * fit.r(i);
    rt_vinv_r += arma::dot(fit.r(i).t(), vinv_r);
    double sum_vinv_r_i = arma::sum(vinv_r);
    sum_vinv_r += sum_vinv_r_i;
    if (data.random_intercept) {
      sum_vinv_r2 += std::pow(sum_vinv_r_i, 2);
      sum_vinv += arma::accu(fit.v_inv(i));
    }
  }
  double beta0_grad = - 2 * sum_vinv_r / (rt_vinv_r / data.n);
  double gamma0_grad = sum_vinv - sum_vinv_r2 / (rt_vinv_r / data.n);
  return arma::vec {beta0_grad, gamma0_grad};
}

// Slope gradient function

arma::vec grad_slope(const fit& fit, const data& data, const arma::uword& k) {
  double rt_vinv_r = 0;
  double xt_vinv_r = 0;
  double xt_vinv_r2 = 0;
  double xt_vinv_x = 0;
  for (arma::uword i = 0; i < data.m; ++i) {
    arma::vec vinv_r = fit.v_inv(i) * fit.r(i);
    rt_vinv_r += arma::dot(fit.r(i).t(), vinv_r);
    double xt_vinv_r_i = arma::dot(data.x(i).col(k).t(), vinv_r);
    xt_vinv_r += xt_vinv_r_i;
    xt_vinv_r2 += std::pow(xt_vinv_r_i, 2);
    xt_vinv_x += arma::dot(data.x(i).col(k).t() * fit.v_inv(i), data.x(i).col(k));
  }
  double beta_grad = - 2 * xt_vinv_r / (rt_vinv_r / data.n);
  double gamma_grad = xt_vinv_x - xt_vinv_r2 / (rt_vinv_r / data.n);
  return arma::vec {beta_grad, gamma_grad};
}

// Threshold function

void threshold(double& beta, double& gamma, const double& lambda, const double& alpha) {
  double beta2 = std::pow(beta, 2);
  double gamma2 = std::pow(gamma, 2);
  if ((beta2 < 2 * lambda * alpha) && (beta2 + gamma2 < 2 * lambda)) {
    beta = 0;
    gamma = 0;
  } else if (gamma2 < 2 * lambda * (1 - alpha)) {
    gamma = 0;
  }
}

// Best linear unbiased predictor function

arma::mat blup(const fit& fit, const data& data) {
  arma::mat u(data.m, (data.intercept && data.random_intercept ? 1 : 0) + data.p,
              arma::fill::zeros);
  for (arma::uword i = 0; i < data.m; ++i) {
    arma::vec vinv_r = fit.v_inv(i) * fit.r(i);
    if (data.intercept && data.random_intercept) {
      u(i, 0) = fit.gamma0 * arma::sum(vinv_r);
      for (arma::uword j = 0; j < data.p; ++j) {
        u(i, j + 1) = fit.gamma(j) * arma::dot(data.x(i).col(j), vinv_r);
      }
    } else {
      for (arma::uword j = 0; j < data.p; ++j) {
        u(i, j) = fit.gamma(j) * arma::dot(data.x(i).col(j), vinv_r);
      }
    }
  }
  return u;
}


// Residual variance function

double residvar(const fit& fit, const data& data) {
  double rt_vinv_r = 0;
  for (arma::uword i = 0; i < data.m; ++i) {
    rt_vinv_r += arma::dot(fit.r(i).t() * fit.v_inv(i), fit.r(i));
  }
  return rt_vinv_r / data.n;
}
