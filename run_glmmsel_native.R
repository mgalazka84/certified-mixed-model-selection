# Native glmmsel 1.0.3 fits with explicit group-held-out tuning.
# The input training file contains no outer-test outcomes or features.
suppressPackageStartupMessages(library(glmmsel))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3, as.character(packageVersion('glmmsel')) == '1.0.3')
d <- fromJSON(args[1])
outdir <- args[3]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
x <- as.matrix(d$x)
y <- as.numeric(d$y)
cluster <- as.character(d$cluster)
folds <- as.integer(d$folds)
stopifnot(nrow(x) == length(y), length(y) == length(cluster), all(is.finite(x)),
          all(is.finite(y)), identical(sort(unique(folds)), 1:3))
warnings_saved <- list()
settings <- list(family = 'gaussian', local.search = TRUE, nlambda = 100,
                 alpha = 0.8, intercept = TRUE, random.intercept = TRUE,
                 standardise = TRUE, eps = 1e-6, max.cd.iter = 10000,
                 max.ls.iter = 100, max.bls.iter = 30, warn = TRUE)
compact <- function(fit) {
  fit[c('beta0', 'gamma0', 'beta', 'gamma', 'sigma2', 'loss', 'cd.iter',
        'ls.iter', 'pql.iter', 'nnz', 'lambda', 'alpha')]
}
admissible <- function(fit) {
  is.finite(fit$loss) & is.finite(fit$sigma2) & fit$sigma2 > 0 &
    fit$cd.iter < settings$max.cd.iter & fit$ls.iter < settings$max.ls.iter &
    apply(is.finite(fit$beta), 2, all) & apply(is.finite(fit$gamma), 2, all)
}
fit_one <- function(xx, yy, cc, stage, lambda = NULL) {
  withCallingHandlers(
    do.call(glmmsel, c(list(x = xx, y = yy, cluster = cc, lambda = lambda), settings)),
    warning = function(w) {
      warnings_saved[[length(warnings_saved) + 1]] <<-
        list(stage = stage, message = conditionMessage(w))
      invokeRestart('muffleWarning')
    }
  )
}
started <- proc.time()[['elapsed']]
answer <- tryCatch({
  full <- fit_one(x, y, cluster, 'full_training')
  lambda <- full$lambda
  stopifnot(length(lambda) >= 2, all(is.finite(lambda)), all(lambda >= 0))
  inner <- vector('list', 3)
  cv <- matrix(NA_real_, nrow = length(lambda), ncol = 3)
  eligible <- admissible(full)
  for (k in 1:3) {
    train <- folds != k
    valid <- folds == k
    stopifnot(!any(cluster[valid] %in% cluster[train]))
    fit <- fit_one(x[train, , drop = FALSE], y[train], cluster[train],
                   paste0('inner_', k), lambda)
    stopifnot(identical(fit$lambda, lambda))
    predicted <- predict(fit, x[valid, , drop = FALSE], cluster[valid])
    cv[, k] <- colMeans((predicted - y[valid])^2)
    eligible <- eligible & admissible(fit) & is.finite(cv[, k])
    inner[[k]] <- fit
  }
  if (!any(eligible)) stop('No finite, converged common penalty candidate')
  cv_mean <- rowMeans(cv)
  choice_loss <- cv_mean
  choice_loss[!eligible] <- Inf
  # First minimizer on the descending path: deterministic tie handling.
  index <- which.min(choice_loss)
  beta <- as.numeric(fixef(full, lambda[index]))
  elapsed <- proc.time()[['elapsed']] - started
  result <- list(status = 'complete_native', pipeline_seconds = elapsed,
                 selected_index = index, selected_lambda = lambda[index],
                 cv_mean_mse = cv_mean[index], eligible = eligible,
                 cv_fold_mse = cv, settings = settings,
                 beta0 = beta[1], beta = beta[-1],
                 gamma0 = full$gamma0[index], gamma = full$gamma[, index],
                 sigma2 = full$sigma2[index],
                 scaled_profile_loss = full$loss[index],
                 response_scale_squared = sum(y^2)/(length(y)-1),
                 full_path = compact(full), inner_paths = lapply(inner, compact))
  # Outer predictor values are read only after all fitting and tuning end.
  pred_input <- fromJSON(args[2])
  xp <- as.matrix(pred_input$x)
  if (length(pred_input$cluster)) {
    cp <- as.character(pred_input$cluster)
    stopifnot(!any(cp %in% cluster))
    result$predictions <- as.numeric(predict(full, xp, cp, lambda[index]))
  } else result$predictions <- numeric(0)
  saveRDS(list(full = full, inner = inner, selected_index = index),
          file.path(outdir, 'native_fits.rds'))
  result
}, error = function(e) {
  list(status = 'native_error', reason = conditionMessage(e),
       pipeline_seconds = proc.time()[['elapsed']] - started)
})
answer$warnings <- warnings_saved
answer$environment <- list(R = R.version.string,
                           glmmsel = as.character(packageVersion('glmmsel')),
                           Rcpp = as.character(packageVersion('Rcpp')),
                           RcppArmadillo = as.character(packageVersion('RcppArmadillo')),
                           BLAS = extSoftVersion()[['BLAS']],
                           threads = Sys.getenv(c('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                                  'MKL_NUM_THREADS')))
write_json(answer, file.path(outdir, 'native.json'), auto_unbox = TRUE,
           pretty = TRUE, digits = 17, na = 'null', matrix = 'rowmajor')
writeLines(capture.output(sessionInfo()), file.path(outdir, 'R_sessionInfo.txt'))
cat(toJSON(list(status = answer$status, seconds = answer$pipeline_seconds,
                warnings = length(warnings_saved)), auto_unbox = TRUE), '\n')
