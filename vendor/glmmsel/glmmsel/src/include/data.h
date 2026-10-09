#ifndef data_H
#define data_H

// data struct contains model data

struct data {

  public:

    const arma::field<arma::mat> x;
    arma::field<arma::vec> y;
    const std::string family;
    double lambda_step;
    const double alpha;
    const bool intercept;
    const bool random_intercept;
    double lambda;
    unsigned m;
    int n;
    unsigned p;

    data(
      const arma::field<arma::mat>& x,
      arma::field<arma::vec> y,
      const std::string& family,
      const double& alpha,
      const bool& intercept,
      const bool& random_intercept
    ) :
      x(x),
      y(y),
      family(family),
      alpha(alpha),
      intercept(intercept),
      random_intercept(random_intercept)
    {
      m = y.n_elem;
      n = 0;
      for (arma::uword i = 0; i < m; i++) n += y(i).n_elem;
      p = x(0).n_cols;
    };

};

#endif
