#include <RcppArmadillo.h>
#include "cd.h"
#include "auxiliary.h"

typedef fit fit_type;

// Regularisation parameter calculation

void cd::compute_lambda(const fit& fit, data& data, const arma::mat& grad,
                        const arma::uvec& inactive) {

  // Initialise lambda max

  double lambda = 0;

  // Loop through predictors

  for (arma::uword k : inactive) {

    // Compute gradients

    arma::vec grad_k = grad.col(k);

    // Create space for new parameters and auxiliary quantities

    double beta_new;
    double gamma_new;
    double lambda_k;

    // Loop through candidate step sizes

    double t = t_init;
    for (arma::uword iter = 1; iter <= max_bls_iter; ++iter) {

      fit_type fit_new = fit;

      // Take proximal gradient descent step

      beta_new = - t * grad_k(0);
      gamma_new = std::max(- t * grad_k(1), 0.0);
      double lambda_k_1 = std::pow(beta_new, 2) / (2 * data.alpha * t);
      double lambda_k_2 = (std::pow(beta_new, 2) + std::pow(gamma_new, 2)) / (2 * t);
      lambda_k = std::max(lambda_k_1, lambda_k_2);
      threshold(beta_new, gamma_new, t * lambda_k * 0.99, data.alpha);

      // Compute auxiliary quantities from new parameter values

      for (arma::uword i = 0; i < data.m; i++) {
        fit_new.r(i) -= data.x(i).col(k) * beta_new;
        arma::mat vinv_x = fit_new.v_inv(i) * data.x(i).col(k);
        double xt_vinv_x = arma::as_scalar(data.x(i).col(k).t() * vinv_x);
        fit_new.v_inv(i) -= vinv_x * vinv_x.t() / (1 / gamma_new + xt_vinv_x);
        fit_new.v_logdet(i) += std::log(1 + gamma_new * xt_vinv_x);
      }

      // Update loss function value

      fit_new.loss = loss(fit_new, data);

      // Check if stopping criterion is satisfied

      double grad_delta = grad_k(0) * beta_new + grad_k(1) * gamma_new;
      double delta2 = std::pow(beta_new, 2) + std::pow(gamma_new, 2);
      if (fit_new.loss > fit.loss + grad_delta + delta2 / (2 * t)) t *= t_scale;
      else break;

    }

    // Update lambda max

    // std::cout << lambda_k << std::endl;

    if (lambda_k > lambda) lambda = lambda_k;

  }

  data.lambda = lambda * data.lambda_step;

}

// Coordinate descent update for intercept

void cd::update_intercept(fit& fit, const data& data) {

  // Compute gradients

  arma::vec grad = grad_intercept(fit, data);

  // Create space for new parameters and auxiliary quantities

  fit_type fit_new;
  double delta_beta0;
  double delta_gamma0;

  // Loop through candidate step sizes

  double t = t_init;
  for (arma::uword iter = 1; iter <= max_bls_iter; ++iter) {

    fit_new = fit;

    // Take gradient descent step

    fit_new.beta0 = fit.beta0 - t * grad(0);
    fit_new.gamma0 = std::max(fit.gamma0 - t * grad(1), 0.0);
    delta_beta0 = fit_new.beta0 - fit.beta0;
    delta_gamma0 = fit_new.gamma0 - fit.gamma0;

    // Compute auxiliary quantities from new parameter values

    for (arma::uword i = 0; i < data.m; i++) {
      fit_new.r(i) -= delta_beta0;
      arma::vec sum_vinv_1 = sum(fit_new.v_inv(i), 1);
      double sum_vinv = sum(sum_vinv_1);
      fit_new.v_inv(i) -= sum_vinv_1 * sum_vinv_1.t() / (1 / delta_gamma0 + sum_vinv);
      fit_new.v_logdet(i) += std::log(1 + delta_gamma0 * sum_vinv);
    }
    fit_new.loss = loss(fit_new, data);

    // Check if stopping criterion is satisfied

    double grad_delta = grad(0) * delta_beta0 + grad(1) * delta_gamma0;
    double delta2 = std::pow(delta_beta0, 2) + std::pow(delta_gamma0, 2);
    if (fit_new.loss > fit.loss + grad_delta + delta2 / (2 * t)) t *= t_scale;
    else break;

  }

  // Update parameters and auxiliary quantities

  fit = fit_new;

  double delta_infnorm = std::max(std::abs(delta_beta0), std::abs(delta_gamma0));
  if (delta_infnorm > max_delta) max_delta = delta_infnorm;
  double par_infnorm = std::max(std::abs(fit.beta0), std::abs(fit.gamma0));
  if (par_infnorm > max_par) max_par = par_infnorm;

}

// Coordinate descent update for slopes

void cd::update_slope(fit& fit, const data& data, const arma::uword& k) {

  // Compute gradients

  arma::vec grad = grad_slope(fit, data, k);

  // Create space for new parameters and auxiliary quantities

  fit_type fit_new;
  double delta_beta;
  double delta_gamma;

  // Loop through candidate step sizes

  double t = t_init;
  for (arma::uword iter = 1; iter <= max_bls_iter; ++iter) {

    fit_new = fit;

    // Take proximal gradient descent step

    double beta_new = fit.beta(k) - t * grad(0);
    double gamma_new = std::max(fit.gamma(k) - t * grad(1), 0.0);
    threshold(beta_new, gamma_new, t * data.lambda, data.alpha);
    delta_beta = beta_new - fit.beta(k);
    delta_gamma = gamma_new - fit.gamma(k);

    // Update quantities related to beta

    if (delta_beta != 0) {
      fit_new.beta(k) = beta_new;
      for (arma::uword i = 0; i < data.m; i++) fit_new.r(i) -= data.x(i).col(k) * delta_beta;
    }

    // Update quantities related to gamma

    if (delta_gamma != 0) {
      fit_new.gamma(k) = gamma_new;
      for (arma::uword i = 0; i < data.m; i++) {
        arma::mat vinv_x = fit_new.v_inv(i) * data.x(i).col(k);
        double xt_vinv_x = arma::as_scalar(data.x(i).col(k).t() * vinv_x);
        fit_new.v_inv(i) -= vinv_x * vinv_x.t() / (1 / delta_gamma + xt_vinv_x);
        fit_new.v_logdet(i) += std::log(1 + delta_gamma * xt_vinv_x);
      }
    }

    // Update loss function value

    if ((delta_beta != 0) || (delta_gamma != 0)) fit_new.loss = loss(fit_new, data);
    fit_new.active(k) = (beta_new != 0) || (gamma_new != 0);

    // Check if stopping criterion is satisfied

    double grad_delta = grad(0) * delta_beta + grad(1) * delta_gamma;
    double delta2 = std::pow(delta_beta, 2) + std::pow(delta_gamma, 2);
    if (fit_new.loss > fit.loss + grad_delta + delta2 / (2 * t)) t *= t_scale;
    else break;

  }

  // Update parameters and auxiliary quantities

  fit = fit_new;

  double delta_infnorm = std::max(std::abs(delta_beta), std::abs(delta_gamma));
  if (delta_infnorm > max_delta) max_delta = delta_infnorm;
  double par_infnorm = std::max(std::abs(fit.beta(k)), std::abs(fit.gamma(k)));
  if (par_infnorm > max_par) max_par = par_infnorm;

}

// Coordinate descent algorithm

void cd::run(fit& fit, data& data) {

  // Compute initial gradients

  arma::mat grad = arma::mat(2, data.p);
  for (arma::uword k = 0; k < data.p; ++k) grad.col(k) = grad_slope(fit, data, k);

  // Compute lambda

  if (data.lambda < 0) {
    arma::uvec inactive = arma::find(1 - fit.active);
    compute_lambda(fit, data, grad, inactive);
  }

  // Sort coordinates

  arma::uvec order;
  if (sort) {
    arma::vec grad_norm = arma::vec(data.p);
    for (arma::uword k = 0; k < data.p; ++k) grad_norm(k) = arma::norm(grad.col(k));
    order = arma::stable_sort_index(grad_norm, "descend");
  }
  else order = arma::linspace<arma::uvec>(0, data.p - 1, data.p);

  // Screen coordinates

  arma::uvec strong;
  arma::uvec weak;
  unsigned nnz = sum(fit.active);
  bool doscreen = sort & (nnz + screen < data.p);
  if (doscreen) {
    strong = order.rows(0, nnz + screen - 1);
    weak = order.rows(nnz + screen, data.p - 1);
  } else {
    strong = order;
  }

  // Run updates

  unsigned stable_count = 0;

  while (iter < max_iter) {

    // std::cout << fit.beta << std::endl;

    max_delta = 0;
    max_par = 1;
    arma::uvec active_old = fit.active;

    // Update strong set

    if (data.intercept) update_intercept(fit, data);
    for (arma::uword k : strong) update_slope(fit, data, k);
    iter++;

    // Check convergence on strong set

    if (max_delta < eps * max_par) {

      // If converged, check weak set for violations

      if (doscreen) {

        active_old = fit.active;
        for (arma::uword k : weak) update_slope(fit, data, k);
        iter++;
        arma::uvec violate = fit.active - active_old;

        if (arma::any(violate)) {

          // Move any violations from weak set to strong set

          weak.shed_rows(arma::find(violate.rows(weak)));
          strong.insert_rows(0, arma::find(violate));

        } else {

          break;

        }

      } else {

        break;

      }

    } else {

      // If not converged, check active set for stabilisation

      if (active_set) {

        if (arma::any(fit.active != active_old)) stable_count = 0;
        else stable_count++;

        if (stable_count == active_set_count - 1) {

          arma::uvec active = arma::find(fit.active);

          while (iter < max_iter) {

            max_delta = 0;
            max_par = 1;

            // Update active set

            if (data.intercept) update_intercept(fit, data);
            for (arma::uword k : active) update_slope(fit, data, k);
            iter++;

            // Check convergence on active set

            if (max_delta < eps * max_par) break;

          }

        }

      }

    }

  }

}
