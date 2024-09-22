rm(list = ls())
library(Matrix)
library(tidyverse)
library(glmnet)
library(tscount)
library(ggplot2)
library(pgdraw)
set.seed(0)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
# %>% filter(state%in%c("CA","OH","TX","WI","IL"))
schoolsM <- as.matrix(schools[,10:59])



ESN_expansion <- function(Xin, Yin, Xpred, nh=120, nu=0.8, aw=0.1, pw=0.1, au=0.1, pu=0.1, eps = 1){
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
    tmp_new <- tanh(tmp%*%W + matrix( log(Yin[,i-1] + eps), ncol = 1 ) %*% t(Uy) ) 
    tmp <- tmp_new
    H <- rbind(H, tmp_new)
  }
  Hpred <- tanh(H[(nrow(H)-nrow(tmp)+1):nrow(H), ]%*%W  + matrix( log(Yin[,ncol(Yin)] + eps), ncol = 1 ) %*% t(Uy)) 
  return(list("train_h" = H, "pred_h" = Hpred))
}

state_idx <- model.matrix( ~ factor(state) - 1, data = schools)
school_idx <- model.matrix( ~ factor(UNITID) - 1, data = schools)


# MCMC parameters
total_samples <- 1000
burn = 100
thin = 2
years_to_pred = 46:50
alpha_eta = 0.001
beta_eta = 0.001
alpha_xi = 0.001
beta_xi = 0.001
eps = 1 # Avoid underflow, avoid log(0)


# ESN Parameters
nh = 30
nu = 0.9
aw = au = 0.01
pw = pu = 0.1
ns = length(unique(schools$state))
N = length(unique(schools$UNITID))

# Initialization

pred_all_randslp <- array(NA, dim = c(length(years_to_pred), nrow(schoolsM),total_samples))

for(years in years_to_pred){
  
  # Set up hypeparameters for ESN
  Xin <- Xpred <- school_idx
  Yin <- schoolsM[,(1:(years-1))]
  
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
  sig_xi <- rep(NA, total_samples)
  sep_eta_pred <- matrix(NA, nrow = nrow(schoolsM), ncol = total_samples)
  random_slope_pred <- matrix(NA, nrow = nrow(schoolsM), ncol = total_samples)
  # Bayesian - Random Slope Model ----------------------------------------------------------------------------------------  
  
  pb <- txtProgressBar(min = 0, max = nrow(H$train_h), style = 3)
  # Initialize the transformed matrix
  transformed_H <- matrix(NA, nrow = nrow(H$train_h), ncol = nh * ns)
  # Loop through each row and update the progress bar
  print("Transforming H to a huge matrix")
  for (j in 1:nrow(transformed_H)) {
    transformed_H[j, ] <- as.vector(outer(H$train_h[j, ], repeated_state[j, ], "*"))
    setTxtProgressBar(pb, j)  # Update the progress bar
  }
  # Bind the random intercept
  design_here <- cbind(transformed_H, repeated_state)
  # Block matrix inversion, save computation
  sparse_design <- Matrix(design_here, sparse = TRUE)
  tilde_eta_rs <- matrix(NA, ncol = total_samples, nrow = ncol(design_here))
  sig_xi <- rep(NA, total_samples)
  sig_eta <- rep(NA, total_samples)
  rr <- rep(NA, total_samples)
  curr_eta <- matrix(0,nrow = dim(design_here)[2], ncol = 1)
  curr_sig_xi <- .1
  curr_sig_eta <- .1
  curr_omega <- rep(1, length(Yin))

  prior_mu_eta <- rep(0, dim(design_here)[2])
  curr_r = 10
  curr_idx <- 1
  save_idx <- 0
  
  while (save_idx < total_samples) {
    # Sample current omega
    b_it = as.vector(Yin) + curr_r
    kappa_it = curr_r - b_it/2
    curr_psi = sparse_design %*% curr_eta
    curr_omega = pgdraw(b = b_it, c = as.vector(curr_psi))
    
    # Sample current eta
    curr_B <- Matrix(diag(c(rep(curr_sig_eta, nh*ns), rep(curr_sig_xi, ns))),sparse = TRUE)
    pos_sigma_eta <- solve(t(sparse_design)%*%(sparse_design * curr_omega)  + solve(curr_B))
    pos_mu_eta <- pos_sigma_eta%*%(t(sparse_design)%*%kappa_it)
    
    L <- chol(pos_sigma_eta)
    sth <- rnorm(length(pos_mu_eta))
    curr_eta <- t(L) %*% sth + pos_mu_eta
    # curr_eta <- as.vector(mvtnorm::rmvnorm(1, mean = pos_mu_eta, sigma = as.matrix(pos_sigma_eta)))
    
    # Propose a sigma_eta
    alpha_eta_pos = ns*nh/2+alpha_eta
    beta_eta_pos = sum(curr_eta[1:(nh*ns)]^2/2)+beta_eta
    curr_sig_eta = 1/rgamma(1, shape = alpha_eta_pos,rate = beta_eta_pos)
    
    
    # Propose a sigma_xi
    alpha_xi_pos = ns/2+alpha_xi
    beta_xi_pos = sum(curr_eta[(nh*ns+1):length(curr_eta)]^2/2)+beta_xi
    curr_sig_xi = 1/rgamma(1, shape = alpha_xi_pos,rate = beta_xi_pos)
    
    
    # Propose an r
    # Prior: discrete uniform distribution [1:20]
    # if(curr_r == 1 ){
    #   new_r <- 2
    #   curr_llh <- sum(lgamma(as.vector(Yin+curr_r))) - length(as.vector(Yin))*lgamma(curr_r) 
    #   + curr_r*sum(sparse_design%*%curr_eta) - curr_r*sum(log(1+exp(sparse_design%*%curr_eta)))
    #   new_llh <- sum(lgamma(as.vector(Yin+new_r))) - length(as.vector(Yin))*lgamma(new_r)
    #   + new_r*sum(sparse_design%*%curr_eta) - new_r*sum(log(1+exp(sparse_design%*%curr_eta)))
    #   
    #   p_trans <- exp(new_llh - curr_llh) * 0.5
    # }else if(curr_r == 20 ){
    #   new_r <- 19
    #   curr_llh <- sum(lgamma(as.vector(Yin+curr_r))) - length(as.vector(Yin))*lgamma(curr_r)
    #   + curr_r*sum(sparse_design%*%curr_eta) - curr_r*sum(log(1+exp(sparse_design%*%curr_eta)))
    #   
    #   new_llh <- sum(lgamma(as.vector(Yin+new_r))) - length(as.vector(Yin))*lgamma(new_r)
    #   + new_r*sum(sparse_design%*%curr_eta) - new_r*sum(log(1+exp(sparse_design%*%curr_eta)))
    #   p_trans <- exp(new_llh - curr_llh) * 0.5
    # }else{
    #   walker = runif(1)
    #   new_r = ifelse(walker>0.5, curr_r+1,curr_r-1)
    #   curr_llh <- sum(lgamma(as.vector(Yin+curr_r))) - length(as.vector(Yin))*lgamma(curr_r)
    #   + curr_r*sum(sparse_design%*%curr_eta) - curr_r*sum(log(1+exp(sparse_design%*%curr_eta)))
    #   
    #   new_llh <- sum(lgamma(as.vector(Yin+new_r))) - length(as.vector(Yin))*lgamma(new_r)
    #   + new_r*sum(sparse_design%*%curr_eta) - new_r*sum(log(1+exp(sparse_design%*%curr_eta)))
    #   p_trans <- exp(new_llh - curr_llh)
    # }
    # 
    # l = runif(1)
    # curr_r = ifelse(l < p_trans, new_r, curr_r)
    
    curr_idx <- curr_idx + 1 
    # Save current values after burn and thin
    if( (curr_idx > burn) & ((curr_idx - burn)%%thin == 0) ){
      save_idx <- save_idx + 1
      tilde_eta_rs[,save_idx] <- as.vector(curr_eta)
      sig_xi[save_idx] <- curr_sig_xi
      sig_eta[save_idx] <- curr_sig_eta
      rr[save_idx] <- curr_r
      
      # Initialize the transformed matrix
      transformed_H_pred <- matrix(NA, nrow = nrow(H$pred_h), ncol = nh * ns)
      # Loop through each row and update the progress bar
      for (j in 1:nrow(transformed_H_pred)) {
        transformed_H_pred[j, ] <- as.vector(outer(H$pred_h[j, ], repeated_state[j, ], "*"))
      }
      pred_here <- cbind(transformed_H_pred, state_idx)
      curr_p <- as.numeric(exp(pred_here%*%curr_eta)) / (1+as.numeric(exp(pred_here%*%curr_eta)))
      random_slope_pred[,save_idx] <- curr_r * (1-curr_p)/curr_p
      # pred_all_randslp[years-min(years_to_pred)+1,,] <- random_slope_pred
      
      par(mfrow = c(5,1), mar = c(2,2,2,2))
      # plot(x = 1:save_idx, y = sig_xi_inv[1:save_idx], type = 'l', main = "sig xi inv", xlab = "")
      plot(x = 1:save_idx, y = sig_xi[1:save_idx], type = 'l', main = "sig xi", xlab = "")
      # plot(x = 1:save_idx, y = sig_eta_inv[1:save_idx], type = 'l', main = "sig eta inv", xlab = "")
      plot(x = 1:save_idx, y = sig_eta[1:save_idx], type = 'l', main = "sig eta", xlab = "")
      plot(x = 1:save_idx, y = rr[1:save_idx], type = 'l', main = "r", xlab = "")
      
    } 
    pred <- random_slope_pred[,save_idx]
    true_value <- schoolsM[,years]
    mse <- mean((pred-true_value)^2)
    print(paste(years,curr_idx,mse))
    
  }
  pred_all_randslp[years-min(years_to_pred)+1,,] <- random_slope_pred
}



saveRDS(pred_all_randslp, file="pred_all_randsl_sch.Rda")
