#ifndef cd_H
#define cd_H
#include "fit.h"
#include "data.h"

class cd {

  public:

    const double eps;
    const unsigned max_iter;
    const unsigned max_bls_iter;
    const double t_init;
    const double t_scale;
    const bool active_set;
    const unsigned active_set_count;
    const bool sort;
    const unsigned screen;
    unsigned iter;
    double max_delta;
    double max_par;

    cd(
      const double& eps,
      const unsigned& max_iter,
      const unsigned& max_bls_iter,
      const double& t_init,
      const double& t_scale,
      const bool& active_set,
      const unsigned& active_set_count,
      const bool& sort,
      const unsigned& screen
    ) :
      eps(eps),
      max_iter(max_iter),
      max_bls_iter(max_bls_iter),
      t_init(t_init),
      t_scale(t_scale),
      active_set(active_set),
      active_set_count(active_set_count),
      sort(sort),
      screen(screen)
    {};

    void run(fit& fit, data& data);
    void update_slope(fit& fit, const data& data, const arma::uword& k);
    void compute_lambda(const fit& fit, data& data, const arma::mat& grad,
                        const arma::uvec& inactive);
    void update_intercept(fit& fit, const data& data);

};

#endif
