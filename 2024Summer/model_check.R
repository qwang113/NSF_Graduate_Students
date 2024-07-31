schools <- read.csv(here::here("nsf_final_wide_car.csv"))
int_pred <- readRDS("D:/77/Research/temp/pred_all_int.Rda")
sep_pred <- readRDS("D:/77/Research/temp/pred_all_sep.Rda")
ingarch_pred <- readRDS("D:/77/Research/temp/pred_all_ING.Rda")
single_esn_pred <- readRDS("D:/77/Research/temp/pred_all_single_esn.Rda")
ensemble_esn_pred <- readRDS("D:/77/Research/temp/pred_all_ensemble_esn.Rda")

int_mean <- apply(int_pred, c(1,2), mean)
sep_mean <- apply(sep_pred, c(1,2), mean)
ingarch_mean <- ingarch_pred[1,,]
single_esn_mean <- t(single_esn_pred)
ensemble_esn_mean <- apply(ensemble_esn_pred, c(1,2), mean)

alpha <- 0.05
int_up <- apply(int_pred, c(1,2), quantile, 1-alpha/2)
int_low <- apply(int_pred, c(1,2), quantile, alpha/2)

sep_up <- apply(sep_pred, c(1,2), quantile, 1-alpha/2)
sep_low <- apply(sep_pred, c(1,2), quantile, alpha/2)

ingarch_up <- ingarch_pred[2,,]
ingarch_low <- ingarch_pred[3,,]

ensemble_up <- apply(ensemble_esn_pred, c(1,2), quantile, 1-alpha/2)
ensemble_low <- apply(ensemble_esn_pred, c(1,2), quantile, alpha/2)


for (years in years_to_pred) {
  # Set up hypeparameters for ESN
  Xin <- Xpred <- model.matrix( ~ factor(state) -1, data = schools)
  Yin <- schoolsM[,1:years]
  
  # Generate H
  H <- ESN_expansion(Xin = state_idx, Yin = Yin, Xpred = state_idx, nh=nh, nu=nu, aw=aw, pw=pw, au=au, pu=pu, eps = eps)
  
  # Number of times to repeat
  n <- ncol(Yin)
  # Repeat the matrix and bind by rows
  repeated_state <- do.call(rbind, replicate(n, state_idx, simplify = FALSE))
  design_mat <- cbind(H$train_h, repeated_state)
  
  
  # Input Data
  nh <- dim(H$train_h)[2]
  ns <- dim(state_idx)[2]
  y_tr <- as.vector(Yin)
  # Posterior samples holder
  tilde_eta <- matrix(NA, ncol = total_samples, nrow = ncol(design_mat))
  sig_xi_inv <- rep(NA, total_samples)
  sep_eta_pred <- matrix(NA, nrow = nrow(schoolsM), ncol = total_samples)
  
  # Bayesian - separate fitting model ---------------------------------------------------------------------------
  for (idx in 1:total_samples) {
    print(idx)
    for (i in 1:nrow(schoolsM)) {
      # curr_idx_h <- seq(from = i, to = i + nrow(schoolsM)*(nrow(H$train_h)/nrow(schoolsM) - 1), by = nrow(schoolsM))
      curr_idx_h <- i + (0:(years - 2))*nrow(schoolsM)
      curr_H <- H$train_h[curr_idx_h,]
      curr_H_sep <- rbind(curr_H,alpha^{-1/2}*diag(rep(sig_eta_inv,nh)))
      curr_y <- Yin[i,1:(years-1)]
      curr_alpha <- matrix(c(curr_y+eps, rep(alpha,nh)), ncol = 1)
      curr_kappa <- matrix(c(rep(1, ncol(Yin)-1), rep(alpha, nh)), ncol = 1)
      curr_pos_eta <- rCMLG(H = curr_H_sep, alpha = curr_alpha, kappa = curr_kappa)
      sep_eta_pred[i,idx] <- exp(H$pred_h[i,] %*% curr_pos_eta)
    }
  }
  pred_all_sep[years-min(years_to_pred)+1,,] <- sep_eta_pred
  for(j in 1:reps){
    H <-  ESN_expansion(Xin = state_idx, Yin = Yin, Xpred = state_idx, nh=nh, nu=nu, aw=aw, pw=pw, au=au, pu=pu, eps = eps)
    fit_fesn_ens <- glmnet(x = H$train_h, y = y_tr, family = "poisson", alpha = 1, lambda = lambda)
    pred_all_ensemble_esn[years - min(years_to_pred) + 1,,j] <- predict(fit_fesn_ens, newx = H$pred_h, type = "response")
  }
}
