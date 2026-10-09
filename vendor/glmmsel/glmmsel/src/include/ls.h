#ifndef ls_H
#define ls_H
#include "cd.h"

class ls {

  public:

    bool local_search;
    unsigned iter;
    const unsigned max_iter;
    const unsigned top_k_min = 10;
    const double top_k_prop = 0.05;
    const double eps = 1e-2;

    ls(
      const bool& local_search,
      const unsigned& max_iter
    ) :
      local_search(local_search),
      max_iter(max_iter) {};

    void run(fit& fit, data& data, cd& cd);

};

#endif
