rm(list = ls())
library(tidyverse)
library(glmnet)
library(tscount)

schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])

rCMLG <- function(H=matrix(rnorm(6),3), alpha=c(1,1,1), kappa=c(1,1,1)){
  ## This function will simulate from the cMLG distribution
  m <- length(kappa)
  w <- log(rgamma(m, shape=alpha, rate=kappa))
  return(as.numeric(solve(t(H)%*%H)%*%t(H)%*%w))
}
pos_sig_xi <- function(sig_xi, xi, alpha){
  ns <- length(xi)
  loglike <- -log(1+(sig_xi/10000)^2) + ns/2 * log(1/sig_xi) + alpha^(1/2)*1/sig_xi*sum(xi)- alpha * sum(exp(alpha^(-1/2)*1/sig_xi*xi))
  return(loglike)
}

ESN_expansion <- function(Xin, Yin, Xpred, nh=120, nu=0.8, aw=0.1, pw=0.1, au=0.1, pu=0.1){
  ## Fit
  p <- ncol(Xin)
  W <- matrix(runif(nh*nh, min=-aw, max=aw), nrow=nh) * matrix(rbinom(nh*nh,1,1-pw), nrow=nh)
  W <- (nu/max(abs(eigen(W, only.values=T)$values))) * W
  U <- matrix(runif(nh*p, min=-au, max=au), nrow=nh) * matrix(rbinom(nh*p,1,1-pu), nrow=nh)
  Uy <- matrix(runif(nh, min = -au, max = au), nrow = nh) * matrix(rbinom(nh,1,1-pu), ncol = 1)
  H <- matrix(NA, nrow=nrow(Xin), ncol=nh)
  tmp <- tanh(Xin %*% t(U))
  H <- tmp
  for(i in 2:ncol(Yin)){
    tmp_new <- tanh(tmp%*%W + Xin%*%t(U) + matrix(Yin[,i-1], ncol = 1 ) %*% t(Uy) ) 
    tmp <- tmp_new
    H <- rbind(H, tmp_new)
  }
  Hpred <- tanh(H[(nrow(H)-nrow(tmp)+1):nrow(H), ]%*%W + Xin%*%t(U) + matrix( Yin[,ncol(Yin)], ncol = 1 ) %*% t(Uy)) 
  return(list("train_h" = H, "pred_h" = Hpred))
}
# MCMC parameters
total_samples <- 1000
burn = 500
thin = 2
alpha = 1000
years_to_pred <- 46:50
pred_all_int <- array(NA, dim = c(length(years_to_pred), nrow(schoolsM),total_samples))
pred_all_sep <- array(NA, dim = c(length(years_to_pred), nrow(schoolsM),total_samples))
for (years in years_to_pred) {
  # Set up hypeparameters for ESN
  Xin <- Xpred <- model.matrix( ~ factor(state) -1, data = schools)
  Yin <- schoolsM[,1:years]
  nh = 120
  nu = 0.9
  aw = au = 0.1
  pw = pu = 0.1
  reps = 1
  state_idx <- model.matrix( ~ factor(state) -1, data = schools)
  
  # Generate H
  H <- ESN_expansion(Xin = state_idx, Yin = Yin, Xpred = state_idx, nh=nh, nu=nu, aw=aw, pw=pw, au=au, pu=pu)
  
  # Number of times to repeat
  n <- ncol(Yin)
  # Repeat the matrix and bind by rows
  repeated_state <- do.call(rbind, replicate(n, state_idx, simplify = FALSE))
  design_mat <- cbind(H$train_h, repeated_state)
  
  #Hyper-parameters
  
  sig_eta_inv = 100
  eps = 1 # Avoid underflow, avoid log(0)
  
  # Input Data
  nh <- dim(H$train_h)[2]
  ns <- dim(state_idx)[2]
  y_tr <- as.vector(Yin)
  # Posterior samples holder
  tilde_eta <- matrix(NA, ncol = total_samples, nrow = ncol(design_mat))
  sig_xi_inv <- rep(NA, total_samples)
  sep_eta_pred <- matrix(NA, nrow = nrow(schoolsM), ncol = total_samples)
  
  # Separate fitting model ---------------------------------------------------------------------------
  for (idx in 1:total_samples) {
    print(idx)
    for (i in 1:nrow(schoolsM)) {
      curr_idx_h <- seq(from = i, to = i + nrow(schoolsM)*(nrow(H$train_h)/nrow(schoolsM) - 1), by = nrow(schoolsM))
      curr_H <- H$train_h[curr_idx_h,]
      curr_H_sep <- rbind(curr_H,alpha^{-1/2}*diag(rep(sig_eta_inv,nh)))
      curr_y <- Yin[i,]
      curr_alpha <- matrix(c(curr_y, rep(alpha,nh)), ncol = 1)
      curr_kappa <- matrix(c(rep(1, ncol(Yin)), rep(alpha, nh)), ncol = 1)
      curr_pos_eta <- rCMLG(H = curr_H_sep, alpha = curr_alpha, kappa = curr_kappa)
      sep_eta_pred[i,idx] <- exp(H$pred_h[i,] %*% curr_pos_eta)
    }
  }
  
  int_eta_pred <- matrix(NA, nrow = nrow(schoolsM), ncol = total_samples)
  pred_all_sep[years-min(years_to_pred)+1,,] <- sep_eta_pred
  # Random Effect Model ----------------------------------------------------------------------------------------  
  
  # Initialization
  curr_eta <- matrix(0,nrow = dim(tilde_eta)[1], ncol = 1)
  curr_sig_xi_inv <- 0.1
  curr_idx <- 1
  save_idx <- 0
  while (save_idx < total_samples) {
    # Define current H_eta
    curr_H_eta <- rbind(design_mat, alpha^{-1/2}*diag(c(rep(sig_eta_inv,nh), rep(curr_sig_xi_inv,ns))))
    # Define current alpha_eta
    curr_alpha_eta <- matrix(c(y_tr+eps,rep(alpha,nh+ns)))
    # Define current kappa_eta
    curr_kappa_eta <- c(rep(1,nrow(design_mat)), rep(alpha,nh+ns))
    # Sample tilde eta
    curr_eta <- rCMLG(H = curr_H_eta, alpha = curr_alpha_eta, kappa = curr_kappa_eta)
    
    # Propose a sigma_xi
    d <- min(0.5, curr_sig_xi_inv)
    temp_sig_xi_inv <- runif(1, min = curr_sig_xi_inv-d, max = curr_sig_xi_inv+d)
    # Metropolis-hastings
    prev_loglike <- pos_sig_xi(sig_xi =  1/curr_sig_xi_inv, xi = curr_eta[(length(curr_eta)-ns+1):length(curr_eta)], alpha = alpha)
    curr_loglike <- pos_sig_xi(sig_xi = 1/temp_sig_xi_inv, xi = curr_eta[(length(curr_eta)-ns+1):length(curr_eta)], alpha = alpha)
    p_trans <- exp(curr_loglike - prev_loglike)
    l <- runif(1)
    if(p_trans > l){
      curr_sig_xi_inv <- temp_sig_xi_inv
    }
    curr_idx <- curr_idx + 1 
    # Save current values after burn and thin
    if( (curr_idx > burn) & ((curr_idx - burn)%%thin == 0) ){
      save_idx <- save_idx + 1
      tilde_eta[,save_idx] <- curr_eta
      sig_xi_inv[save_idx] <- curr_sig_xi_inv
      pred_mat <- cbind(H$pred_h,state_idx)
      int_eta_pred[,save_idx] <- exp(pred_mat%*%curr_eta)
      pred_all_int[years-min(years_to_pred)+1,,] <- int_eta_pred
    } 
    print(curr_idx)
  }

}

saveRDS(pred_all_int, file="pred_all_int.Rda")
saveRDS(pred_all_sep, file="pred_all_sep.Rda")

# write.csv(tilde_eta, here::here("eta.csv"), row.names = FALSE)
# write.csv(sig_xi_inv, here::here("sig_xi_inv.csv"), row.names = FALSE)
# write.csv(H$pred_h,here::here("pred_h.csv"), row.names = FALSE)
# write.csv(int_eta_pred, here::here("int_eta_pred.csv"))
# write.csv(sep_eta_pred, here::here("sep_eta_pred.csv"))
