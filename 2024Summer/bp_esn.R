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

Xin <- Xpred <- model.matrix( ~ factor(state) -1, data = schools)
Yin <- schoolsM[,1:40]
nh=120
nu = 0.35
aw = pw = au = pu = 0.1
reps = 1
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
state_idx <- model.matrix( ~ factor(state) -1, data = schools)

# Generate H
H <- ESN_expansion(Xin = state_idx, Yin = schoolsM[,1:40], Xpred = state_idx, nh=120, nu=0.8, aw=0.1, pw=0.1, au=0.1, pu=0.1)

print(H)
# Number of times to repeat
n <- ncol(Yin)
# Repeat the matrix and bind by rows
repeated_state <- do.call(rbind, replicate(n, state_idx, simplify = FALSE))
design_mat <- cbind(H$train_h, repeated_state)
pos_sig_xi <- function(sig_xi, xi, alpha){
  ns <- length(xi)
  loglike <- -log(1+sig_xi^2) + ns/2 * log(sig_xi) + alpha^(1/2)*sig_xi*sum(xi)- alpha * sum(exp(alpha^(-1/2)*sig_xi^(-1)*xi))
  return(loglike)
}

# specify the prior distribution

# MCMC parameters
total_samples <- 1000
burn = 500
thin = 2
alpha = 1000

#Hyper-parameters

sig_eta = 10
eps = 1 # Avoid underflow, avoid log(0)

# Input Data
nh <- dim(H$train_h)[2]
ns <- dim(state_idx)[2]
y_tr <- as.vector(Yin)
# Posterior samples holder
tilde_eta <- matrix(NA, ncol = total_samples, nrow = ncol(design_mat))
sig_xi <- rep(NA, total_samples)
# Initialization
curr_eta <- matrix(0,nrow = dim(tilde_eta)[1], ncol = 1)
curr_sig_xi <- 0.1
curr_idx <- 1
save_idx <- 0

while (save_idx < total_samples) {
  # Define current H_eta
  curr_H_eta <- rbind(design_mat, diag(c(rep(sig_eta,nh), rep(curr_sig_xi,ns))))
  # Define current alpha_eta
  curr_alpha_eta <- matrix(c(y_tr+eps,rep(sqrt(alpha),nh+ns)))
  # Define current kappa_eta
  curr_kappa_eta <- c(rep(1,nrow(design_mat)), rep(alpha,nh+ns))
  # Sample tilde eta
  curr_eta <- rCMLG(H = curr_H_eta, alpha = curr_alpha_eta, kappa = curr_kappa_eta)
  
  # Propose a sigma_xi
  temp_sig_xi <- exp(rnorm(1, mean = log(curr_sig_xi)))
  # Metropolis-hastings
  prev_loglike <- pos_sig_xi(curr_sig_xi, xi = curr_eta[(length(curr_eta)-ns+1):length(curr_eta)], alpha = alpha)
  curr_loglike <- pos_sig_xi(temp_sig_xi, xi = curr_eta[(length(curr_eta)-ns+1):length(curr_eta)], alpha = alpha)
  p_trans <- exp(curr_loglike - prev_loglike)
  l <- runif(1)
  if(p_trans > l){
    curr_sig_xi <- temp_sig_xi
  }
  curr_idx <- curr_idx + 1 
  # Save current values after burn and thin
  if( (curr_idx > burn) & ((curr_idx - burn)%%thin == 0) ){
    save_idx <- save_idx + 1
    tilde_eta[,save_idx] <- curr_eta
    sig_xi[save_idx] <- curr_sig_xi
  } 
  print(curr_idx)
}
write.csv(tilde_eta, here::here("eta.csv"), row.names = FALSE)
write.csv(sig_xi, here::here("sig_xi.csv"), row.names = FALSE)
write.csv(H$pred_h,here::here("pred_h.csv"), row.names = FALSE)
