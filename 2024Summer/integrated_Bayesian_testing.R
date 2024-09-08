rm(list = ls())
library(tidyverse)
library(glmnet)
library(tscount)
library(Matrix)
set.seed(0)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])

rCMLG <- function(H=matrix(rnorm(6),3), alpha=c(1,1,1), kappa=c(1,1,1)){
  ## This function will simulate from the cMLG distribution
  m <- length(kappa)
  w <- Matrix(log(rgamma(m, shape=alpha, rate=kappa)), sparse = TRUE)
  sparse_H <- Matrix(H, sparse = TRUE)
  return(as.numeric(solve( crossprod(sparse_H) )%*%t(sparse_H)%*%w))
}

pos_sig_xi <- function(sig_xi, xi, alpha){
  ns <- length(xi)
  loglike <- -log(1+(sig_xi)^2) + ns * log(1/sig_xi) + alpha^(1/2)*1/sig_xi*sum(xi)- alpha * sum(exp(alpha^(-1/2)*1/sig_xi*xi))
  return(loglike)
}


ESN_expansion <- function(Xin, Yin, Xpred, nh=100, nu=0.8, aw=0.1, pw=0.1, au=0.1, pu=0.1, eps = 1){
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
    tmp_new <- tanh(tmp%*%W + Xin%*%t(U) + matrix( log(Yin[,i-1] + eps), ncol = 1 ) %*% t(Uy) ) 
    tmp <- tmp_new
    H <- rbind(H, tmp_new)
  }
  Hpred <- tanh(H[(nrow(H)-nrow(tmp)+1):nrow(H), ]%*%W + Xin%*%t(U) + matrix( log(Yin[,ncol(Yin)] + eps), ncol = 1 ) %*% t(Uy)) 
  return(list("train_h" = H, "pred_h" = Hpred))
}


state_idx <- model.matrix( ~ factor(state) -1, data = schools)

# MCMC parameters
total_samples <- 1000
alpha = 1000
years_to_pred <- 46:50
burn = 500
thin = 2


#Hyper-parameters
sig_eta_inv = 0.01
eps = 1 # Avoid underflow, avoid log(0)

# ESN Parameters
nh = 120
nu = 0.9
aw = au = 0.1
pw = pu = 0.1
reps = 1000


# Initialization
pred_all_int <- array(NA, dim = c(length(years_to_pred), nrow(schoolsM),total_samples))


for (years in years_to_pred) {
  # Set up hypeparameters for ESN
  Xin <- Xpred <- model.matrix( ~ factor(state) -1, data = schools)
  Yin <- schoolsM[,1:(years-1)]
  
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
  
  # Posterior sample boxes
  tilde_eta <- matrix(NA, ncol = total_samples, nrow = ncol(design_mat))
  sig_xi_inv <- rep(NA, total_samples)
  sep_eta_pred <- matrix(NA, nrow = nrow(schoolsM), ncol = total_samples)
  
  # Bayesian - Integrated random Effect Model ----------------------------------------------------------------------------------------  
  # Initialization
  curr_eta <- matrix(0,nrow = dim(tilde_eta)[1], ncol = 1)
  curr_sig_xi_inv <- 0.2
  curr_idx <- 1
  save_idx <- 0
  int_eta_pred <- matrix(NA, nrow = nrow(schoolsM), ncol = total_samples)
  
  while (save_idx < total_samples) {
    # Define current H_eta
    curr_H_eta <- rbind(design_mat, alpha^(-1/2)*diag(c(rep(sig_eta_inv,nh), rep(curr_sig_xi_inv,ns))))
    # Define current alpha_eta
    curr_alpha_eta <- matrix(c(y_tr+eps,rep(alpha,nh+ns)))
    # Define current kappa_eta
    curr_kappa_eta <- c(rep(1,nrow(design_mat)), rep(alpha,nh+ns))
    # Sample tilde eta
    curr_eta <- rCMLG(H = curr_H_eta, alpha = curr_alpha_eta, kappa = curr_kappa_eta)
    
    # Propose a sigma_xi
    d <- min(0.5, 1/curr_sig_xi_inv)
    temp_sig_xi_inv <- 1/runif(1, min = 1/curr_sig_xi_inv-d, max = 1/curr_sig_xi_inv+d)
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
      int_eta_pred[,save_idx] <- rpois (nrow(schoolsM),exp(pred_mat%*%curr_eta))
      print(mean((int_eta_pred[,save_idx] - schoolsM[,years])^2))
      pred_all_int[years-min(years_to_pred)+1,,] <- int_eta_pred
      plot(sig_xi_inv[1:save_idx])
    } 
    print(curr_idx)
  }
}

saveRDS(pred_all_int, file="pred_all_int.Rda")

