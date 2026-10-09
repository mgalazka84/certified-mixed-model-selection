#ifndef auxiliary_H
#define auxiliary_H
#include "fit.h"
#include "data.h"

double loss(
    const fit& fit,
    const data& data
);

arma::vec grad_intercept(
    const fit& fit,
    const data& data
);

arma::vec grad_slope(
    const fit& fit,
    const data& data,
    const arma::uword& k
);

void threshold(
    double& beta,
    double& gamma,
    const double& lambda,
    const double& alpha
);

arma::mat blup(
    const fit& fit,
    const data& data
);

double residvar(
    const fit& fit,
    const data& data
);

#endif
