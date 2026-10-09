#include <RcppArmadillo.h>
#include "ls.h"
#include "auxiliary.h"

typedef fit fit_type;

// Local search algorithm

void ls::run(fit& fit, data& data, cd& cd) {

  while (iter < max_iter) {

    arma::uvec active = arma::find(fit.active);
    arma::uvec inactive = arma::find(1 - fit.active);
    bool improved = false;
    unsigned ninactive = inactive.size();
    if (ninactive == 0) break;

    // Loop over active set

    for (arma::uword k : active) {

      // Save objective function from current fit

      double obj_k = fit.loss + data.alpha * data.lambda +
        (1 - data.alpha) * data.lambda * (fit.gamma(k) != 0);

      // Remove predictor k from the current fit

      fit_type fit_mk = fit;
      fit_mk.beta(k) = 0;
      fit_mk.gamma(k) = 0;
      double delta_beta = - fit.beta(k);
      double delta_gamma = - fit.gamma(k);
      for (arma::uword i = 0; i < data.m; i++) fit_mk.r(i) -= data.x(i).col(k) * delta_beta;
      if (delta_gamma != 0) {
        for (arma::uword i = 0; i < data.m; i++) {
          arma::mat vinv_x = fit_mk.v_inv(i) * data.x(i).col(k);
          double xt_vinv_x = arma::as_scalar(data.x(i).col(k).t() * vinv_x);
          fit_mk.v_inv(i) -= vinv_x * vinv_x.t() / (1 / delta_gamma + xt_vinv_x);
          fit_mk.v_logdet(i) += std::log(1 + delta_gamma * xt_vinv_x);
        }
      }
      fit_mk.loss = loss(fit_mk, data);
      fit_mk.active(k) = 0;

      // Sort inactive predictors by their gradients

      arma::vec grad_norm = arma::vec(ninactive);
      for (arma::uword i = 0; i < ninactive; i++) {
        arma::uword s = inactive(i);
        grad_norm(i) = arma::norm(grad_slope(fit_mk, data, s));
      }
      arma::uvec order = arma::stable_sort_index(grad_norm, "descend");
      unsigned top_k = std::round(data.p * top_k_prop);
      top_k = std::min(std::max(top_k, top_k_min), ninactive);
      arma::uvec top_inactive = inactive.rows(order.rows(0, top_k - 1));

      // Loop over inactive set

      double best_obj = obj_k;
      fit_type best_fit;

      for (arma::uword s : top_inactive) {

        unsigned iter = 0;
        fit_type fit_s = fit_mk;

        // Compute partial minimiser

        while (iter < 50) {
          cd.max_delta = 0;
          cd.max_par = 1;
          cd.update_slope(fit_s, data,  s);
          iter++;
          if (cd.max_delta < eps * cd.max_par) break;
        }

        // If minimiser is active, check if it improves on incumbent minimiser

        if (fit_s.active(s) == 1) {
          double obj_s = fit_s.loss + data.alpha * data.lambda +
            (1 - data.alpha) * data.lambda * (fit_s.gamma(s) != 0);
          if (obj_s < best_obj) {
            best_fit = fit_s;
            best_obj = obj_s;
          }
        }

      }

      // Update solution if there was an improvement

      if (best_obj < obj_k) {
        fit = best_fit;
        cd.run(fit, data);
        improved = true;
        break;
      }

    }

    iter++;

    // Exit if no improvement

    if (!improved) break;

  }

}
