#include <RcppArmadillo.h>
#include "pql.h"
#include "auxiliary.h"

// Fit via penalised quasi-likelihood

void pql::run(fit& fit, data& data, cd& cd, ls& ls, const arma::field<arma::vec>& y_orig) {

  if (data.family == "binomial") {

    // Run updates

    while (iter < max_iter) {

      double loss_old = fit.loss;

      // Compute best linear unbiased predictors

      arma::mat u = blup(fit, data);

      // double sum_square_y = 0;

      arma::vec w;

      for (arma::uword i = 0; i < data.m; i++) {

        // Compute working weights and response

        arma::vec eta;
        if (data.intercept && data.random_intercept) {
          eta = fit.beta0 + u.row(i)(0) + data.x(i) * (fit.beta + u.submat(i, 1, i, data.p).t());
        } else {
          eta = fit.beta0 + data.x(i) * (fit.beta + u.row(i).t());
        }
        arma::vec pi = arma::clamp(1 / (1 + arma::exp(- eta)), 1e-5, 1 - 1e-5);
        w = pi % (1 - pi);
        arma::vec y = eta + (y_orig(i) - pi) / w;

        // Update quantities that depend on working weights and response

        fit.r(i) -= data.y(i) - y;
        arma::mat v = data.x(i) * arma::diagmat(fit.gamma) * data.x(i).t() +
          fit.gamma0 + arma::diagmat(1 / w);
        fit.v_inv(i) = arma::inv(v);
        fit.v_logdet(i) = arma::log_det_sympd(v);
        data.y(i) = y;
        // sum_square_y += arma::dot(y, y);

      }

      // Standardise working response

      // double std_y = std::sqrt(sum_square_y / data.n);
      // fit.beta0 /= std_y;
      // fit.beta /= std_y;
      // for (arma::uword i = 0; i < data.m; i++) {
      //   fit.r(i) /= std_y;
      //   data.y(i) /= std_y;
      // }
      fit.loss = loss(fit, data);

      // Run coordinate descent and local search

      cd.run(fit, data);
      if (ls.local_search) ls.run(fit, data, cd);
      iter++;

      // Unstandardise working response

      // fit.beta0 *= std_y;
      // fit.beta *= std_y;
      // for (arma::uword i = 0; i < data.m; i++) {
      //   fit.r(i) *= std_y;
      //   data.y(i) *= std_y;
      // }
      // fit.loss = loss(fit, data);

      // Exit if converged

      // std::cout << "lambda: " << data.lambda << " nnz: " <<  arma::sum(fit.active) << " w: " << arma::sum(w) << std::endl;

      if (std::abs(fit.loss - loss_old) <= eps * fit.loss) break;

    }

  }

}
