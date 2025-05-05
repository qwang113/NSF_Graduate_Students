rm(list = ls())
library(Matrix)
library(tidyverse)
library(glmnet)
library(tscount)
library(ggplot2)
set.seed(0)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
# %>% filter(state%in%c("CA","OH","TX","WI","IL"))
schoolsM <- as.matrix(schools[,10:59])

rCMLG <- function(H=matrix(rnorm(6),3), alpha=c(1,1,1), kappa=c(1,1,1)){
  ## This function will simulate from the cMLG distribution
  m <- length(kappa)
  w <- log(rgamma(m, shape=alpha, rate=kappa))
  sparse_H <- Matrix(H, sparse = TRUE)
  return(as.numeric(solve( t(sparse_H)%*%sparse_H )%*%t(sparse_H)%*%w))
}

pos_sig_xi <- function(sig_xi, xi, alpha){
  ns <- length(xi)
  loglike <- -log(1+(sig_xi/100)^2) + ns * log(1/sig_xi) + alpha^(1/2)*1/sig_xi*sum(xi)- alpha * sum(exp(alpha^(-1/2)*1/sig_xi*xi))
  return(loglike)
}

pos_sig_eta <- function(sig_eta, eta_trunc, alpha){
  n_eta <- length(eta_trunc)
  loglike <- -log(1+(sig_eta/100)^2) + n_eta * log(1/sig_eta) + alpha^(1/2)*1/sig_eta*sum(eta_trunc) - 
    alpha * sum(exp(alpha^(-1/2)*1/sig_eta*eta_trunc))
  return(loglike)
}



ESN_expansion <- function(Xin = matrix(0), Yin, Xpred, nh=100, nu=0.8, aw=0.1, pw=0.1, au=0.1, pu=0.1, eps = 1){
  ## Fit
  p <- ncol(Xin)
  W <- matrix(runif(nh*nh, min=-aw, max=aw), nrow=nh) * matrix(rbinom(nh*nh,1,1-pw), nrow=nh)
  W <- (nu/max(abs(eigen(W, only.values=T)$values))) * W
  U <- matrix(runif(nh*p, min=-au, max=au), nrow=nh) * matrix(rbinom(nh*p,1,1-pu), nrow=nh)
  Uy <- matrix(runif(nh, min = -au, max = au), nrow = nh) * matrix(rbinom(nh,1,1-pu), ncol = 1)
  H <- tmp <- matrix(0, nrow = nrow(schoolsM), ncol=nh)
  for(i in 2:ncol(Yin)){
    tmp_new <- tanh(tmp%*%W + matrix(log(Yin[,i-1] + eps), ncol = 1 ) %*% t(Uy) ) 
    tmp <- tmp_new
    H <- rbind(H, tmp_new)
  }
  Hpred <- tanh(H[(nrow(H)-nrow(tmp)+1):nrow(H), ]%*%W + matrix( log(Yin[,ncol(Yin)] + eps), ncol = 1 ) %*% t(Uy)) 
  return(list("train_h" = H[-c(1:nrow(Xin)),], "pred_h" = Hpred))
}

state_idx <- model.matrix( ~ factor(state) -1, data = schools)


# MCMC parameters
total_samples <- 1000
burn = 500
thin = 2
alpha = 1000
eps = 1 # Avoid underflow, avoid log(0)


# ESN Parameters
nh = 30
nu = 0.9
aw = au = 0.01
pw = pu = 0.1
ns = length(unique(schools$state))

# Initialization
years = 51
# Set up hypeparameters for ESN
Xin <- Xpred <- model.matrix( ~ factor(state) -1, data = schools)
Yin <- schoolsM[,(1:(years-1))]

# Generate H
H <- ESN_expansion(Xin = state_idx, Yin = Yin, Xpred = state_idx, nh=nh, nu=nu, aw=aw, pw=pw, au=au, pu=pu, eps = eps)

# Number of times to repeat
n <- ncol(Yin[,-1])
# Repeat the matrix and bind by rows
repeated_state <- do.call(rbind, replicate(n, state_idx, simplify = FALSE))
design_mat <- cbind(H$train_h, repeated_state)

# Input Data
nh <- dim(H$train_h)[2]
ns <- dim(state_idx)[2]
y_tr <- as.vector(Yin[,-1])

# Posterior sample boxes
tilde_eta <- matrix(NA, ncol = total_samples, nrow = ncol(design_mat))
sig_xi_inv <- rep(NA, total_samples)
sep_eta_pred <- matrix(NA, nrow = nrow(schoolsM), ncol = total_samples)
insample_pred <- matrix(NA, nrow = length(schoolsM[,-1]), ncol = total_samples)
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
design_here <- cbind(transformed_H, repeated_state)
# Block matrix inversion, save computation
sparse_design <- Matrix(design_here, sparse = TRUE)
# A <- crossprod(sparse_design)
# A_inv <- solve(A)
# Initialization
tilde_eta_rs <- matrix(NA, ncol = total_samples, nrow = ncol(design_here))
sig_xi_inv <- rep(NA, total_samples)
sig_eta_inv <- rep(NA, total_samples)
curr_eta <- matrix(0,nrow = dim(design_here)[2], ncol = 1)
curr_sig_xi_inv <- 0.1
curr_sig_eta_inv <- 0.0001
curr_idx <- 1
save_idx <- 0

while (save_idx < total_samples) {
  # Define current H_eta
  DIAG <- Matrix(diag(c(rep(curr_sig_eta_inv, nh*ns), rep(curr_sig_xi_inv, ns))),sparse = TRUE)
  curr_H_eta <- rbind(sparse_design, alpha^{-1/2}*DIAG)
  # curr_H_eta <- rbind(design_here, alpha^{-1/2}*diag(c(rep(curr_sig_eta_inv,nh*ns), rep(curr_sig_xi_inv,ns))))
  # Define current alpha_eta
  curr_alpha_eta <- matrix(c(y_tr+eps,rep(alpha,(nh+1)*ns)))
  # Define current kappa_eta
  curr_kappa_eta <- c(rep(1,nrow(design_here)), rep(alpha,(nh+1)*ns))
  
  # B <- 1/alpha * Matrix(diag(c(rep(curr_sig_eta_inv^2, nh*ns), rep(curr_sig_xi_inv^2, ns))),sparse = TRUE)
  # # Sample tilde eta
  # m <- length(curr_kappa_eta)
  # w <- Matrix(log(rgamma(m, shape=curr_alpha_eta, rate=curr_kappa_eta)))
  # S_inv <- A_inv - A_inv %*% solve(solve(B) + A_inv) %*% A_inv
  curr_eta <- rCMLG(H = curr_H_eta, alpha = curr_alpha_eta, kappa = curr_kappa_eta)
  
  # curr_eta <- rCMLG(H = curr_H_eta, alpha = curr_alpha_eta, kappa = curr_kappa_eta)
  
  # Propose a sigma_xi
  # d <- min(0.5, 1/curr_sig_xi_inv)
  # temp_sig_xi_inv <- 1/runif(1, min = 1/curr_sig_xi_inv-d, max = 1/curr_sig_xi_inv+d)
  temp_sig_xi_inv <- exp(rnorm(1,mean = log(curr_sig_xi_inv), sd = 0.1))
  # Metropolis-hastings
  prev_loglike <- pos_sig_xi(sig_xi =  1/curr_sig_xi_inv, xi = curr_eta[(length(curr_eta)-ns+1):length(curr_eta)], alpha = alpha)
  curr_loglike <- pos_sig_xi(sig_xi = 1/temp_sig_xi_inv, xi = curr_eta[(length(curr_eta)-ns+1):length(curr_eta)], alpha = alpha)
  p_trans <- exp(curr_loglike - prev_loglike)
  l <- runif(1)
  if(p_trans > l){
    curr_sig_xi_inv <- temp_sig_xi_inv
  }
  
  # Propose a sigma_eta
  d <- min(1, 1/curr_sig_eta_inv)
  # temp_sig_eta_inv <- 1/runif(1, min = 1/curr_sig_eta_inv-d, max = 1/curr_sig_eta_inv+d)
  temp_sig_eta_inv <- exp(rnorm(1,mean = log(curr_sig_eta_inv), sd = 0.01))
  # Metropolis-hastings
  prev_loglike <- pos_sig_eta(sig_eta =  1/curr_sig_eta_inv, eta_trunc = curr_eta[1:(ns*nh)], alpha = alpha)
  curr_loglike <- pos_sig_eta(sig_eta = 1/temp_sig_eta_inv, eta_trunc = curr_eta[1:(ns*nh)], alpha = alpha)
  p_trans <- exp(curr_loglike - prev_loglike)
  l <- runif(1)
  if(p_trans > l){
    curr_sig_eta_inv <- temp_sig_eta_inv
  }
  
  curr_idx <- curr_idx + 1 
  # Save current values after burn and thin
  if( (curr_idx > burn) & ((curr_idx - burn)%%thin == 0) ){
    save_idx <- save_idx + 1
    tilde_eta_rs[,save_idx] <- as.vector(curr_eta)
    sig_xi_inv[save_idx] <- curr_sig_xi_inv
    sig_eta_inv[save_idx] <- curr_sig_eta_inv
    
    
    insample_pred[,save_idx] <- as.numeric(exp(design_here%*%curr_eta)) 
    # pred_all_randslp[years-min(years_to_pred)+1,,] <- random_slope_pred
    
    par(mfrow = c(4,1), mar = c(2,2,2,2))
    plot(x = 1:save_idx, y = sig_xi_inv[1:save_idx], type = 'l', main = "sig xi inv", xlab = "")
    plot(x = 1:save_idx, y = 1/sig_xi_inv[1:save_idx], type = 'l', main = "sig xi", xlab = "")
    plot(x = 1:save_idx, y = sig_eta_inv[1:save_idx], type = 'l', main = "sig eta inv", xlab = "")
    plot(x = 1:save_idx, y = 1/sig_eta_inv[1:save_idx], type = 'l', main = "sig eta", xlab = "")
  } 
  pred <- insample_pred[,save_idx]
  true_value <- as.vector(schoolsM[,-1])
  mse <- mean((pred-true_value)^2)
  print(paste(years,curr_idx,mse))
}

pred_mean<- matrix(apply(insample_pred, 1, mean), nrow = nrow(schoolsM))
pred_res <- pred_mean - schoolsM[,-1]
xt_var <- pred_mean
st <- pred_res/sqrt(xt_var)
var(as.vector(st))
saveRDS(insample_pred, file="insample_pois.Rda")
