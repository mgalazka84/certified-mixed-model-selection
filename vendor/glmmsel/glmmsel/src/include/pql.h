#ifndef pql_H
#define pql_H
#include "cd.h"
#include "ls.h"

class pql {

public:

  const double eps = 1e-4;
  const unsigned max_iter;
  unsigned iter;

  pql(
    const unsigned& max_iter
  ) :
    max_iter(max_iter) {};

  void run(fit& fit, data& data, cd& cd, ls& ls, const arma::field<arma::vec>& y_orig);

};

#endif
